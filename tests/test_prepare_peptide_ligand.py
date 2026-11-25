"""Tests for turning an aligned peptide into a chemically validated ligand.

The OXT geometry tests use hand-built coordinates whose answers are
trigonometry. The chemistry tests need RDKit and the real `spec_0574` design,
and skip without them, because asserting that bond-order transfer works is
only meaningful against a real predicted structure.

The central assertion is the one that costs something: that bond-order
transfer **fails** on BoltzGen's output as it stands, and succeeds once the
C-terminal oxygen is placed. That is the blocker stated as a test rather than
as prose.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import gemmi  # noqa: E402
import numpy as np  # noqa: E402

import peptide_geometry  # noqa: E402
import prepare_peptide_ligand as ppl  # noqa: E402
import prepare_peptide_pose as ppp  # noqa: E402

REAL_CIF = (
    REPOSITORY_ROOT
    / "project_private"
    / "example_run"
    / "murc_1750_snapshot"
    / "final_ranked_designs"
    / "final_46_designs"
    / "rank0017_spec_0574.cif"
)
REAL_RECEPTOR = (
    REPOSITORY_ROOT
    / "project_private"
    / "epicac_receptor_validation"
    / "receptors"
    / "murc_anp_nomn_try1_prepared.pdb"
)
SEQUENCE = "SRGSGTSF"
REAL_DATA = REAL_CIF.is_file() and REAL_RECEPTOR.is_file()
JUSTIFICATION = "test fixture; not a scientific decision"


def write_aligned_peptide(directory: Path, mirror: bool = False) -> Path:
    """Write the real aligned `spec_0574` peptide into `directory`.

    With `mirror`, every z coordinate is negated, which inverts every chiral
    centre. That gives a structure whose stereochemistry must be rejected, and
    it is derived from real coordinates rather than invented.
    """
    structure = peptide_geometry.read_structure(REAL_CIF)
    peptide_chain = peptide_geometry.identify_chain_by_sequence(
        structure[0], SEQUENCE
    )
    design_protein = [c for c in structure[0] if c.name == "B"][0]
    reference = peptide_geometry.read_structure(REAL_RECEPTOR)
    reference_chain = [c for c in reference[0] if c.name == "A"][0]
    transform, _ = peptide_geometry.superpose_protein(
        design_protein, reference_chain
    )
    placed = ppp.aligned_peptide_structure(peptide_chain, transform, "spec_0574")
    if mirror:
        for chain in placed[0]:
            for residue in chain:
                for atom in residue:
                    atom.pos = gemmi.Position(
                        atom.pos.x, atom.pos.y, -atom.pos.z
                    )
    path = directory / ("mirrored.pdb" if mirror else "aligned.pdb")
    placed.write_pdb(str(path))
    return path


def single_residue_structure(
    name: str = "ALA", with_oxt: bool = False
) -> gemmi.Structure:
    """One residue with a well-defined carboxyl plane, built by hand.

    CA, C and O are placed non-collinearly so the carboxyl bisector is
    defined. The expected OXT direction is therefore computable by hand.
    """
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("P")
    residue = gemmi.Residue()
    residue.name = name
    residue.seqid = gemmi.SeqId(1, " ")
    atoms = [
        ("N", "N", (-2.5, 0.0, 0.0)),
        ("CA", "C", (-1.5, 0.0, 0.0)),
        ("C", "C", (0.0, 0.0, 0.0)),
        ("O", "O", (0.6, 1.1, 0.0)),
        ("CB", "C", (-1.5, 1.5, 0.0)),
    ]
    if with_oxt:
        atoms.append(("OXT", "O", (0.6, -1.1, 0.0)))
    for atom_name, element, position in atoms:
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(*position)
        residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    return structure


class OxtPlacementTests(unittest.TestCase):
    """Geometry only; no RDKit needed."""

    def test_oxt_is_placed_at_the_stated_bond_length(self):
        structure = single_residue_structure()
        record = ppl.place_c_terminal_oxt(structure)
        self.assertTrue(record["placed"])
        self.assertEqual(record["bond_length_angstrom"], 1.25)
        residue = structure[0][0][0]
        carbon = np.asarray(residue["C"][0].pos.tolist())
        oxt = np.asarray(residue["OXT"][0].pos.tolist())
        self.assertAlmostEqual(float(np.linalg.norm(oxt - carbon)), 1.25, places=6)

    def test_oxt_sits_at_about_120_degrees_to_both_neighbours(self):
        record = ppl.place_c_terminal_oxt(single_residue_structure())
        self.assertAlmostEqual(record["angle_ca_c_oxt_degrees"], 120.0, delta=1.5)
        self.assertAlmostEqual(record["angle_o_c_oxt_degrees"], 120.0, delta=1.5)

    def test_the_two_angles_are_equal_because_the_bisector_is_used(self):
        record = ppl.place_c_terminal_oxt(single_residue_structure())
        self.assertAlmostEqual(
            record["angle_ca_c_oxt_degrees"],
            record["angle_o_c_oxt_degrees"],
            places=6,
        )

    def test_oxt_lies_in_the_carboxyl_plane(self):
        # CA, C and O are all at z = 0, so OXT must be too.
        structure = single_residue_structure()
        ppl.place_c_terminal_oxt(structure)
        oxt = structure[0][0][0]["OXT"][0].pos
        self.assertAlmostEqual(oxt.z, 0.0, places=9)

    def test_an_existing_oxt_is_not_moved_or_duplicated(self):
        structure = single_residue_structure(with_oxt=True)
        with self.assertRaises(ppl.LigandPreparationError) as raised:
            ppl.place_c_terminal_oxt(structure)
        self.assertIn("already has OXT", str(raised.exception))

    def test_a_missing_carbonyl_oxygen_is_fatal(self):
        structure = single_residue_structure()
        residue = structure[0][0][0]
        for index, atom in enumerate(residue):
            if atom.name == "O":
                del residue[index]
                break
        with self.assertRaises(ppl.LigandPreparationError) as raised:
            ppl.place_c_terminal_oxt(structure)
        self.assertIn("carboxyl plane is undefined", str(raised.exception))

    def test_collinear_backbone_atoms_are_fatal_rather_than_arbitrary(self):
        # With CA, C and O collinear the bisector vanishes, and any direction
        # would be an arbitrary choice.
        structure = single_residue_structure()
        residue = structure[0][0][0]
        residue["O"][0].pos = gemmi.Position(1.2, 0.0, 0.0)
        residue["CA"][0].pos = gemmi.Position(-1.5, 0.0, 0.0)
        with self.assertRaises(ppl.LigandPreparationError) as raised:
            ppl.place_c_terminal_oxt(structure)
        self.assertIn("collinear", str(raised.exception))

    def test_the_record_states_that_the_atom_was_constructed(self):
        record = ppl.place_c_terminal_oxt(single_residue_structure())
        self.assertIn("constructed, not predicted", record["caveat"])

    def test_several_peptide_chains_are_refused(self):
        structure = single_residue_structure()
        extra = gemmi.Chain("Q")
        residue = gemmi.Residue()
        residue.name = "ALA"
        residue.seqid = gemmi.SeqId(1, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(20.0, 0.0, 0.0)
        residue.add_atom(atom)
        extra.add_residue(residue)
        structure[0].add_chain(extra)
        with self.assertRaises(ppl.LigandPreparationError) as raised:
            ppl.place_c_terminal_oxt(structure)
        self.assertIn("exactly one peptide chain", str(raised.exception))


@unittest.skipUnless(ppl.RDKIT_AVAILABLE, "RDKit not importable")
@unittest.skipUnless(REAL_DATA, "real refolded design or receptor not present")
class BondOrderTransferTests(unittest.TestCase):
    """The blocker, and its remedy, as tests."""

    def test_transfer_fails_on_boltzgen_output_as_it_stands(self):
        # BoltzGen omits OXT, so the structure cannot be matched to its own
        # sequence. This is the finding that makes --place-c-terminal-oxt
        # mandatory rather than cosmetic.
        with tempfile.TemporaryDirectory() as temporary:
            aligned = write_aligned_peptide(Path(temporary))
            with self.assertRaises(ppl.LigandPreparationError) as raised:
                ppl.assign_bond_orders(aligned, SEQUENCE)
            message = str(raised.exception)
            self.assertIn("55 heavy atoms", message)
            self.assertIn("56", message)
            self.assertIn("OXT", message)

    def test_the_diagnostic_names_both_molecular_formulae(self):
        with tempfile.TemporaryDirectory() as temporary:
            aligned = write_aligned_peptide(Path(temporary))
            with self.assertRaises(ppl.LigandPreparationError) as raised:
                ppl.assign_bond_orders(aligned, SEQUENCE)
            message = str(raised.exception)
            # One oxygen apart: O12 in the structure, O13 in the template.
            self.assertIn("O12", message)
            self.assertIn("O13", message)

    def test_transfer_succeeds_once_oxt_is_placed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aligned = write_aligned_peptide(directory)
            structure = peptide_geometry.read_structure(aligned)
            ppl.place_c_terminal_oxt(structure)
            repaired = directory / "with_oxt.pdb"
            structure.write_pdb(str(repaired))
            molecule = ppl.assign_bond_orders(repaired, SEQUENCE)
            self.assertEqual(molecule.GetNumAtoms(), 56)

    def test_a_sequence_that_does_not_match_the_structure_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aligned = write_aligned_peptide(directory)
            structure = peptide_geometry.read_structure(aligned)
            ppl.place_c_terminal_oxt(structure)
            repaired = directory / "with_oxt.pdb"
            structure.write_pdb(str(repaired))
            with self.assertRaises(ppl.LigandPreparationError):
                ppl.assign_bond_orders(repaired, "SRGSGTSW")


@unittest.skipUnless(ppl.RDKIT_AVAILABLE, "RDKit not importable")
@unittest.skipUnless(REAL_DATA, "real refolded design or receptor not present")
class EndToEndLigandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp())
        cls.aligned = write_aligned_peptide(cls.directory)
        cls.record = ppl.prepare_ligand(
            cls.aligned,
            SEQUENCE,
            "spec_0574",
            "physiological",
            JUSTIFICATION,
            place_oxt=True,
            out_dir=cls.directory / "ligand",
        )

    def test_the_ligand_validates_and_an_sdf_is_written(self):
        self.assertTrue(self.record["validated"])
        self.assertEqual(self.record["chemical_validation_failures"], [])
        self.assertTrue(Path(self.record["outputs"]["sdf"]).is_file())

    def test_heavy_atom_coordinates_are_preserved_exactly(self):
        # The guarantee the whole pilot rests on: the molecule handed to Meeko
        # carries BoltzGen's coordinates, not re-generated ones.
        coordinates = self.record["heavy_atom_coordinates"]
        self.assertTrue(coordinates["coordinates_preserved"])
        self.assertEqual(coordinates["heavy_atoms_compared"], 56)
        self.assertEqual(coordinates["unmatched"], [])
        self.assertLessEqual(coordinates["maximum_deviation_angstrom"], 0.002)

    def test_the_physiological_state_carries_a_net_charge_of_plus_one(self):
        # N-terminal amine +1, ARG2 guanidinium +1, C-terminal carboxyl -1.
        ionisation = self.record["ionisation"]
        self.assertEqual(ionisation["net_formal_charge"], 1)
        groups = {c["group"] for c in ionisation["changes"]}
        self.assertEqual(
            groups,
            {
                "N-terminal amine",
                "C-terminal carboxyl",
                "arginine guanidinium",
            },
        )

    def test_the_guanidinium_charge_is_placed_on_the_double_bonded_nitrogen(self):
        arginine = [
            c
            for c in self.record["ionisation"]["changes"]
            if c["group"] == "arginine guanidinium"
        ][0]
        self.assertEqual(arginine["residue"], "ARG2")
        self.assertIn(arginine["atom"], ("NH1", "NH2"))
        self.assertEqual(arginine["formal_charge"], 1)

    def test_stereochemistry_is_confirmed_by_cip_codes_from_3d(self):
        # An independent route to peptide_geometry's signed-volume result.
        stereochemistry = self.record["stereochemistry"]
        self.assertTrue(stereochemistry["all_l_configuration"])
        self.assertEqual(stereochemistry["mismatches"], [])
        # Seven centres: six alpha carbons with side chains (the two glycines
        # have none), plus threonine's beta carbon.
        self.assertEqual(stereochemistry["centres_checked"], 7)

    def test_threonine_is_checked_at_both_of_its_centres(self):
        codes = self.record["stereochemistry"]["observed_cip_codes"]
        self.assertEqual(codes["THR6/CA"], "S")
        self.assertEqual(codes["THR6/CB"], "R")

    def test_hydrogens_are_added_because_meeko_will_not_add_them(self):
        hydrogens = self.record["hydrogens"]
        self.assertEqual(hydrogens["total_atoms_with_hydrogen"], 108)
        self.assertIn("does not add hydrogens", hydrogens["reason"])

    def test_the_torsion_count_is_reported_with_its_caveat(self):
        torsions = self.record["torsional_degrees_of_freedom"]
        self.assertEqual(torsions["rdkit_rotatable_bonds_strict"], 25)
        self.assertIn("not Meeko's TORSDOF", torsions["caveat"])
        # The misattribution this project previously carried, corrected.
        self.assertIn("AutoDock 4", torsions["relevance"])

    def test_the_atom_mapping_covers_every_atom_including_hydrogens(self):
        mapping = self.record["atom_mapping"]
        self.assertEqual(len(mapping), 108)
        heavy = [m for m in mapping if m["element"] != "H"]
        self.assertEqual(len(heavy), 56)
        named = [m for m in heavy if m.get("atom_name") == "OXT"]
        self.assertEqual(len(named), 1)

    def test_the_oxt_placement_is_recorded_as_idealised(self):
        oxt = self.record["c_terminal_oxt"]
        self.assertTrue(oxt["placed"])
        self.assertEqual(oxt["residue"], "PHE8")
        self.assertIn("idealised", oxt["geometry"])

    def test_the_record_and_its_checksums_are_written(self):
        self.assertEqual(len(self.record["inputs"]["aligned_peptide_pdb_sha256"]), 64)
        self.assertEqual(len(self.record["outputs"]["sdf_sha256"]), 64)
        written = json.loads(Path(self.record["outputs"]["record"]).read_text())
        self.assertEqual(written["design_id"], "spec_0574")


@unittest.skipUnless(ppl.RDKIT_AVAILABLE, "RDKit not importable")
@unittest.skipUnless(REAL_DATA, "real refolded design or receptor not present")
class RefusalTests(unittest.TestCase):
    def test_preparation_without_oxt_placement_fails_rather_than_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aligned = write_aligned_peptide(directory)
            with self.assertRaises(ppl.LigandPreparationError):
                ppl.prepare_ligand(
                    aligned,
                    SEQUENCE,
                    "spec_0574",
                    "physiological",
                    JUSTIFICATION,
                    place_oxt=False,
                    out_dir=directory / "ligand",
                )

    def test_an_empty_protonation_justification_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aligned = write_aligned_peptide(directory)
            with self.assertRaises(ppl.LigandPreparationError) as raised:
                ppl.prepare_ligand(
                    aligned,
                    SEQUENCE,
                    "spec_0574",
                    "physiological",
                    "   ",
                    place_oxt=True,
                    out_dir=directory / "ligand",
                )
            self.assertIn("recorded decision", str(raised.exception))

    def test_mirrored_coordinates_are_rejected_and_no_sdf_is_written(self):
        # Every chiral centre inverted. The pilot must refuse to prepare a
        # D-peptide silently, and must not leave a usable file behind.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            mirrored = write_aligned_peptide(directory, mirror=True)
            record = ppl.prepare_ligand(
                mirrored,
                SEQUENCE,
                "mirrored",
                "neutral",
                JUSTIFICATION,
                place_oxt=True,
                out_dir=directory / "ligand",
            )
            self.assertFalse(record["validated"])
            self.assertTrue(record["chemical_validation_failures"])
            self.assertIn("stereochemistry", record["chemical_validation_failures"][0])
            self.assertIsNone(record["outputs"]["sdf"])
            self.assertFalse((directory / "ligand" / "mirrored_ligand.sdf").is_file())

    def test_the_neutral_state_carries_no_formal_charge(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aligned = write_aligned_peptide(directory)
            record = ppl.prepare_ligand(
                aligned,
                SEQUENCE,
                "spec_0574",
                "neutral",
                JUSTIFICATION,
                place_oxt=True,
                out_dir=directory / "ligand",
            )
            self.assertEqual(record["ionisation"]["net_formal_charge"], 0)
            self.assertEqual(record["ionisation"]["changes"], [])
            self.assertIn("dictionary state", record["ionisation"]["limitation"])

    def test_writing_twice_into_one_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aligned = write_aligned_peptide(directory)
            for attempt in range(2):
                if attempt == 0:
                    ppl.prepare_ligand(
                        aligned, SEQUENCE, "spec_0574", "neutral",
                        JUSTIFICATION, True, directory / "ligand",
                    )
                else:
                    with self.assertRaises(ppl.LigandPreparationError) as raised:
                        ppl.prepare_ligand(
                            aligned, SEQUENCE, "spec_0574", "neutral",
                            JUSTIFICATION, True, directory / "ligand",
                        )
                    self.assertIn("Refusing to overwrite", str(raised.exception))

    def test_an_unknown_ionisation_state_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            aligned = write_aligned_peptide(directory)
            with self.assertRaises(ppl.LigandPreparationError):
                ppl.prepare_ligand(
                    aligned, SEQUENCE, "spec_0574", "ph_9",
                    JUSTIFICATION, True, directory / "ligand",
                )


@unittest.skipUnless(ppl.RDKIT_AVAILABLE, "RDKit not importable")
class IonisationRefusalTests(unittest.TestCase):
    """Residues whose state pH alone cannot settle."""

    def test_histidine_and_cysteine_are_named_as_undecidable(self):
        self.assertIn("HIS", ppl.IONISATION_REFUSED)
        self.assertIn("CYS", ppl.IONISATION_REFUSED)
        self.assertIn("tautomer", ppl.IONISATION_REFUSED["HIS"])
        self.assertIn("8.3", ppl.IONISATION_REFUSED["CYS"])

    def test_a_histidine_peptide_is_refused_rather_than_assumed(self):
        from rdkit import Chem

        molecule = Chem.MolFromSequence("AHA")
        self.assertIsNotNone(molecule)
        with self.assertRaises(ppl.LigandPreparationError) as raised:
            ppl.apply_physiological_ionisation(molecule)
        self.assertIn("tautomer", str(raised.exception))

    def test_a_cysteine_peptide_is_refused(self):
        from rdkit import Chem

        with self.assertRaises(ppl.LigandPreparationError) as raised:
            ppl.apply_physiological_ionisation(Chem.MolFromSequence("ACA"))
        self.assertIn("CYS", str(raised.exception).upper())


class ExpectedConfigurationTableTests(unittest.TestCase):
    """The stereochemistry reference table, which is easy to get wrong."""

    def test_every_standard_residue_appears(self):
        self.assertEqual(len(ppl.EXPECTED_ALPHA_CIP), 20)

    def test_glycine_has_no_chiral_alpha_carbon(self):
        self.assertIsNone(ppl.EXPECTED_ALPHA_CIP["GLY"])

    def test_cysteine_is_the_exception_at_r(self):
        # L-cysteine is (R) at CA because sulphur outranks the carboxyl
        # carbon. Every other L residue is (S).
        self.assertEqual(ppl.EXPECTED_ALPHA_CIP["CYS"], "R")
        others = [
            code
            for name, code in ppl.EXPECTED_ALPHA_CIP.items()
            if name not in ("GLY", "CYS")
        ]
        self.assertEqual(set(others), {"S"})

    def test_only_threonine_and_isoleucine_have_a_second_centre(self):
        self.assertEqual(set(ppl.EXPECTED_BETA_CIP), {"THR", "ILE"})
        self.assertEqual(ppl.EXPECTED_BETA_CIP["THR"], "R")
        self.assertEqual(ppl.EXPECTED_BETA_CIP["ILE"], "S")


if __name__ == "__main__":
    unittest.main()
