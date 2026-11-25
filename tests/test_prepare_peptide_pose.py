"""Tests for placing a BoltzGen refolded peptide into the receptor frame.

The connectivity, terminus and transform checks are tested on hand-built
geometry whose answers are arithmetic. The end-to-end behaviour is tested
against the real `spec_0574` refolded complex and the real prepared receptor,
and those tests skip when the files are absent.

The real-data tests assert numbers that were cross-checked against an
independent route: the 463-residue fit and 0.4443 A RMSD, the 55 heavy atoms,
and the 3547 + 31 receptor partition all agree with what the receptor manifest
and `peptide_geometry` report separately.
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
BOX_30A = REPOSITORY_ROOT / "examples" / "box_murc_uma_chainE_30A.json"

REAL_DATA = REAL_CIF.is_file() and REAL_RECEPTOR.is_file()


def residue(name: str, seqid: int, atoms) -> gemmi.Residue:
    out = gemmi.Residue()
    out.name = name
    out.seqid = gemmi.SeqId(seqid, " ")
    for atom_name, element, position in atoms:
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(*position)
        out.add_atom(atom)
    return out


def glycine_pair(c_to_n: float):
    """Two glycines whose C-N separation is exactly `c_to_n` Angstroms.

    Placed along x so the bond length is readable from the coordinates. CA
    atoms are offset in y so the CA-CA distance is not degenerate with the
    bond being measured.
    """
    first = residue(
        "GLY",
        1,
        [
            ("N", "N", (-2.0, 0.0, 0.0)),
            ("CA", "C", (-1.0, 1.0, 0.0)),
            ("C", "C", (0.0, 0.0, 0.0)),
            ("O", "O", (0.0, -1.2, 0.0)),
        ],
    )
    second = residue(
        "GLY",
        2,
        [
            ("N", "N", (c_to_n, 0.0, 0.0)),
            ("CA", "C", (c_to_n + 1.0, 1.0, 0.0)),
            ("C", "C", (c_to_n + 2.0, 0.0, 0.0)),
            ("O", "O", (c_to_n + 2.0, -1.2, 0.0)),
        ],
    )
    return [first, second]


class PeptideBondTests(unittest.TestCase):
    def test_an_ideal_bond_is_inside_the_window(self):
        findings = ppp.peptide_bond_findings(glycine_pair(1.329))
        self.assertEqual(findings["measured_bonds"], 1)
        self.assertEqual(findings["bonds"][0]["c_to_n_angstrom"], 1.329)
        self.assertEqual(findings["bonds"][0]["deviation_from_ideal_angstrom"], 0.0)
        self.assertEqual(findings["bonds_outside_window"], [])
        self.assertTrue(findings["connectivity_intact"])

    def test_a_short_bond_outside_the_window_is_reported(self):
        findings = ppp.peptide_bond_findings(glycine_pair(1.10))
        self.assertEqual(findings["bonds_outside_window"], ["GLY1-GLY2"])
        self.assertEqual(findings["chain_breaks"], [])
        # Outside the window but still bonded: connectivity is intact.
        self.assertTrue(findings["connectivity_intact"])

    def test_a_broken_chain_is_reported_as_a_break(self):
        findings = ppp.peptide_bond_findings(glycine_pair(4.0))
        self.assertEqual(findings["chain_breaks"], ["GLY1-GLY2"])
        self.assertFalse(findings["connectivity_intact"])

    def test_a_bond_just_under_the_break_threshold_is_not_a_break(self):
        findings = ppp.peptide_bond_findings(glycine_pair(1.79))
        self.assertEqual(findings["chain_breaks"], [])
        self.assertEqual(findings["bonds_outside_window"], ["GLY1-GLY2"])

    def test_a_missing_backbone_atom_makes_the_bond_unmeasurable(self):
        pair = glycine_pair(1.329)
        stripped = residue("GLY", 2, [("CA", "C", (1.0, 1.0, 0.0))])
        findings = ppp.peptide_bond_findings([pair[0], stripped])
        self.assertEqual(findings["measured_bonds"], 0)
        self.assertEqual(len(findings["unmeasurable_bonds"]), 1)
        self.assertIn("C or N absent", findings["unmeasurable_bonds"][0])
        self.assertFalse(findings["connectivity_intact"])

    def test_bond_count_is_one_fewer_than_the_residue_count(self):
        residues = glycine_pair(1.329) + [
            residue(
                "GLY",
                3,
                [
                    ("N", "N", (2.7, 0.0, 0.0)),
                    ("CA", "C", (3.7, 1.0, 0.0)),
                    ("C", "C", (4.7, 0.0, 0.0)),
                ],
            )
        ]
        findings = ppp.peptide_bond_findings(residues)
        self.assertEqual(findings["expected_bonds"], 2)
        self.assertEqual(findings["measured_bonds"], 2)

    def test_a_single_residue_has_no_bonds_and_is_not_an_error(self):
        findings = ppp.peptide_bond_findings(glycine_pair(1.329)[:1])
        self.assertEqual(findings["expected_bonds"], 0)
        self.assertIsNone(findings["minimum_angstrom"])
        self.assertTrue(findings["connectivity_intact"])

    def test_the_minimum_and_maximum_span_every_bond(self):
        residues = [
            residue("GLY", 1, [("N", "N", (0, 0, 0)), ("C", "C", (1.0, 0, 0))]),
            residue("GLY", 2, [("N", "N", (2.3, 0, 0)), ("C", "C", (3.3, 0, 0))]),
            residue("GLY", 3, [("N", "N", (4.7, 0, 0)), ("C", "C", (5.7, 0, 0))]),
        ]
        findings = ppp.peptide_bond_findings(residues)
        self.assertEqual(findings["minimum_angstrom"], 1.3)
        self.assertEqual(findings["maximum_angstrom"], 1.4)


class TerminusTests(unittest.TestCase):
    def test_a_missing_oxt_requires_a_decision(self):
        residues = glycine_pair(1.329)
        findings = ppp.terminus_findings(residues)
        self.assertFalse(findings["c_terminal_has_oxt"])
        self.assertTrue(findings["termini_require_a_decision"])
        self.assertIn("incomplete", findings["reason"])

    def test_a_present_oxt_needs_no_decision(self):
        residues = glycine_pair(1.329)
        oxt = gemmi.Atom()
        oxt.name = "OXT"
        oxt.element = gemmi.Element("O")
        oxt.pos = gemmi.Position(4.5, -1.0, 0.0)
        residues[1].add_atom(oxt)
        findings = ppp.terminus_findings(residues)
        self.assertTrue(findings["c_terminal_has_oxt"])
        self.assertFalse(findings["termini_require_a_decision"])

    def test_the_termini_are_named_by_residue(self):
        findings = ppp.terminus_findings(glycine_pair(1.329))
        self.assertEqual(findings["n_terminal_residue"], "GLY1")
        self.assertEqual(findings["c_terminal_residue"], "GLY2")

    def test_an_empty_residue_list_is_fatal(self):
        with self.assertRaises(ppp.PoseError):
            ppp.terminus_findings([])


class TransformRecordTests(unittest.TestCase):
    def test_the_identity_transform_moves_nothing(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        record = ppp.transform_record(gemmi.Transform(), points, points)
        self.assertEqual(record["rotation_angle_degrees"], 0.0)
        self.assertEqual(record["translation_norm_angstrom"], 0.0)
        self.assertEqual(record["peptide_displacement_angstrom"]["maximum"], 0.0)

    def test_a_pure_translation_is_reported_as_zero_rotation(self):
        transform = gemmi.Transform()
        transform.vec.x = 3.0
        transform.vec.y = 4.0
        before = np.asarray([[0.0, 0.0, 0.0]])
        after = np.asarray([[3.0, 4.0, 0.0]])
        record = ppp.transform_record(transform, before, after)
        self.assertEqual(record["rotation_angle_degrees"], 0.0)
        # 3-4-5 triangle.
        self.assertEqual(record["translation_norm_angstrom"], 5.0)
        self.assertEqual(record["peptide_displacement_angstrom"]["mean"], 5.0)

    def test_a_ninety_degree_rotation_is_recovered(self):
        transform = gemmi.Transform()
        # Rotation by 90 degrees about z: trace is 0 + 0 + 1 = 1, so the angle
        # is arccos((1 - 1) / 2) = 90 degrees.
        transform.mat.fromlist([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        points = np.asarray([[1.0, 0.0, 0.0]])
        record = ppp.transform_record(transform, points, np.asarray([[0.0, 1.0, 0.0]]))
        self.assertAlmostEqual(record["rotation_angle_degrees"], 90.0, places=3)

    def test_displacement_statistics_cover_every_atom(self):
        before = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        after = np.asarray([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        record = ppp.transform_record(gemmi.Transform(), before, after)
        displacement = record["peptide_displacement_angstrom"]
        self.assertEqual(displacement["minimum"], 1.0)
        self.assertEqual(displacement["maximum"], 3.0)
        self.assertEqual(displacement["mean"], 2.0)


class BlockingIssueTests(unittest.TestCase):
    """A clash must never block; an incomplete chemistry must."""

    def base_record(self):
        return {
            "atoms_complete": True,
            "all_calpha_L": True,
            "d_configured_residues": [],
            "peptide_bonds": {"chain_breaks": [], "unmeasurable_bonds": []},
            "termini": {"termini_require_a_decision": False},
        }

    def test_an_intact_design_has_no_blocking_issues(self):
        self.assertEqual(ppp.blocking_issues(self.base_record()), [])

    def test_missing_atoms_block(self):
        record = self.base_record()
        record["atoms_complete"] = False
        self.assertIn("heavy atoms missing", ppp.blocking_issues(record)[0])

    def test_a_d_amino_acid_blocks_and_names_the_residue(self):
        record = self.base_record()
        record["all_calpha_L"] = False
        record["d_configured_residues"] = [4]
        issues = ppp.blocking_issues(record)
        self.assertIn("non-L alpha carbon", issues[0])
        self.assertIn("4", issues[0])

    def test_a_chain_break_blocks(self):
        record = self.base_record()
        record["peptide_bonds"]["chain_breaks"] = ["GLY5-THR6"]
        self.assertIn("chain broken", ppp.blocking_issues(record)[0])

    def test_a_missing_oxt_blocks_ligand_preparation(self):
        record = self.base_record()
        record["termini"]["termini_require_a_decision"] = True
        self.assertIn("C-terminal carboxyl incomplete", ppp.blocking_issues(record)[0])

    def test_several_issues_are_all_reported(self):
        record = self.base_record()
        record["atoms_complete"] = False
        record["termini"]["termini_require_a_decision"] = True
        self.assertEqual(len(ppp.blocking_issues(record)), 2)


@unittest.skipUnless(REAL_DATA, "real refolded design or receptor not present")
class RealDesignTests(unittest.TestCase):
    """End-to-end on the actual `spec_0574` design."""

    @classmethod
    def setUpClass(cls):
        cls.record = ppp.place_peptide(
            REAL_CIF,
            "SRGSGTSF",
            REAL_RECEPTOR,
            "A",
            "spec_0574",
            box_record=json.loads(BOX_30A.read_text()),
        )

    def test_the_peptide_is_identified_by_sequence_as_chain_a(self):
        self.assertEqual(self.record["chain_assignment"]["peptide_chain"], "A")
        self.assertEqual(
            self.record["chain_assignment"]["design_receptor_chain"], "B"
        )
        self.assertEqual(self.record["residues"], 8)
        self.assertEqual(self.record["peptide_heavy_atoms"], 55)

    def test_the_wrong_sequence_is_fatal_rather_than_matching_by_length(self):
        with self.assertRaises(peptide_geometry.GeometryError):
            ppp.place_peptide(
                REAL_CIF, "SRGSGTSW", REAL_RECEPTOR, "A", "spec_0574"
            )

    def test_the_protein_fit_uses_463_residues_at_sub_angstrom_rmsd(self):
        fit = self.record["protein_superposition"]
        self.assertEqual(fit["matched_ca_atoms"], 463)
        self.assertEqual(fit["skipped_residues"], [])
        self.assertAlmostEqual(fit["rmsd_angstrom"], 0.4443, places=3)

    def test_the_reference_partition_separates_protein_from_anp(self):
        partition = self.record["reference_partition"]
        self.assertEqual(partition["amino_acid_atoms"], 3547)
        self.assertEqual(partition["non_amino_acid_components"], {"ANP": 31})

    def test_the_peptide_is_intact_and_all_l_configured(self):
        self.assertTrue(self.record["atoms_complete"])
        self.assertTrue(self.record["all_calpha_L"])
        self.assertEqual(self.record["d_configured_residues"], [])
        self.assertTrue(self.record["peptide_bonds"]["connectivity_intact"])
        self.assertTrue(self.record["structure_intact"])

    def test_all_seven_peptide_bonds_are_measured_and_none_is_broken(self):
        bonds = self.record["peptide_bonds"]
        self.assertEqual(bonds["expected_bonds"], 7)
        self.assertEqual(bonds["measured_bonds"], 7)
        self.assertEqual(bonds["chain_breaks"], [])
        self.assertEqual(bonds["unmeasurable_bonds"], [])

    def test_the_missing_c_terminal_oxt_blocks_ligand_preparation(self):
        # BoltzGen refolded output carries no OXT, so the terminal carboxyl is
        # incomplete. The design is structurally intact but not yet preparable.
        self.assertFalse(self.record["termini"]["c_terminal_has_oxt"])
        self.assertTrue(self.record["structure_intact"])
        self.assertFalse(self.record["ready_for_ligand_preparation"])
        self.assertEqual(len(self.record["blocking_issues"]), 1)
        self.assertIn("OXT", self.record["blocking_issues"][0])

    def test_the_peptide_lies_entirely_within_the_recorded_30_angstrom_box(self):
        containment = self.record["box_containment"]
        self.assertTrue(containment["fully_contained"])
        self.assertEqual(containment["atoms_inside"], 55)
        self.assertEqual(containment["heavy_atoms"], 55)

    def test_protein_and_anp_contacts_are_reported_separately(self):
        targets = [c["target"] for c in self.record["contacts"]]
        self.assertEqual(len(targets), 2)
        self.assertIn("amino-acid residues only", targets[0])
        self.assertIn("ANP", targets[1])
        protein, anp = self.record["contacts"]
        self.assertEqual(protein["target_heavy_atoms"], 3547)
        self.assertEqual(anp["target_heavy_atoms"], 31)

    def test_the_closest_protein_and_anp_approaches_are_distinct_atoms(self):
        protein, anp = self.record["contacts"]
        # Measured independently: 1.527 A to HIS348 NE2, 1.579 A to ANP O2G.
        self.assertAlmostEqual(protein["minimum_distance_angstrom"], 1.527, places=3)
        self.assertAlmostEqual(anp["minimum_distance_angstrom"], 1.5788, places=3)
        self.assertEqual(
            protein["closest_pair"]["second"]["label"], "A/HIS348/NE2"
        )
        self.assertEqual(anp["closest_pair"]["second"]["label"], "A/ANP603/O2G")

    def test_pooling_protein_and_anp_would_change_the_contact_count(self):
        # The conflation this module prevents, quantified on the real design:
        # 4 protein pairs and 1 ANP pair below 2.0 A, which pool to 5.
        strict = ppp.place_peptide(
            REAL_CIF,
            "SRGSGTSF",
            REAL_RECEPTOR,
            "A",
            "spec_0574",
            contact_cutoff=2.0,
        )
        protein, anp = strict["contacts"]
        self.assertEqual(protein["contacts_below_cutoff"], 4)
        self.assertEqual(anp["contacts_below_cutoff"], 1)

    def test_the_clash_set_is_dominated_by_one_arginine_side_chain(self):
        summary = self.record["clash_summary"]
        residues = summary["peptide_residues_involved"]
        self.assertIn("ARG2", residues)
        self.assertGreater(residues["ARG2"], summary["total_clashes"] // 2)

    def test_exactly_one_backbone_clash_prevents_a_side_chain_only_remedy(self):
        # The decisive result for restrained relaxation: almost every clash is
        # on a side chain, but one backbone overlap remains, so holding the
        # backbone fixed cannot resolve the pose entirely.
        summary = self.record["clash_summary"]
        self.assertEqual(summary["backbone_clashes"], 1)
        self.assertFalse(summary["side_chain_only"])
        self.assertTrue(summary["peptide_backbone_involved"])
        offender = summary["backbone_clashing_pairs"][0]
        self.assertEqual(offender["peptide_atom"], "A/GLY5/O")
        self.assertEqual(offender["target_atom"], "A/GLY28/CA")

    def test_the_transform_is_a_single_rigid_motion_recorded_in_full(self):
        transform = self.record["transform"]
        matrix = np.asarray(transform["rotation_matrix"], dtype=float)
        # A rotation matrix: orthogonal with determinant +1. A reflection
        # would silently invert the peptide's chirality.
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=6)

    def test_input_checksums_are_recorded_for_both_inputs(self):
        inputs = self.record["inputs"]
        self.assertEqual(len(inputs["refolded_cif_sha256"]), 64)
        self.assertEqual(len(inputs["reference_structure_sha256"]), 64)

    def test_software_versions_are_recorded(self):
        software = self.record["software"]
        self.assertIn("gemmi", software)
        self.assertIn("numpy", software)


@unittest.skipUnless(REAL_DATA, "real refolded design or receptor not present")
class WrittenOutputTests(unittest.TestCase):
    """The aligned coordinates must be preserved exactly and provably."""

    def place_and_write(self, directory: Path):
        structure = peptide_geometry.read_structure(REAL_CIF)
        peptide_chain = peptide_geometry.identify_chain_by_sequence(
            structure[0], "SRGSGTSF"
        )
        design_protein = [c for c in structure[0] if c.name == "B"][0]
        reference = peptide_geometry.read_structure(REAL_RECEPTOR)
        reference_chain = [c for c in reference[0] if c.name == "A"][0]
        transform, _ = peptide_geometry.superpose_protein(
            design_protein, reference_chain
        )
        record = ppp.place_peptide(
            REAL_CIF, "SRGSGTSF", REAL_RECEPTOR, "A", "spec_0574"
        )
        written = ppp.write_outputs(
            record, peptide_chain, transform, directory, "spec_0574"
        )
        return written, transform, peptide_chain

    def test_the_written_peptide_holds_all_55_heavy_atoms(self):
        with tempfile.TemporaryDirectory() as temporary:
            written, _, _ = self.place_and_write(Path(temporary))
            placed = gemmi.read_structure(written["aligned_peptide_pdb"])
            atoms = sum(len(r) for c in placed[0] for r in c)
            residues = sum(len(c) for c in placed[0])
            self.assertEqual(atoms, 55)
            self.assertEqual(residues, 8)

    def test_the_written_coordinates_equal_the_transformed_originals(self):
        # The guarantee that matters: what lands on disk is the BoltzGen
        # coordinates under one rigid transform, not a re-derived structure.
        with tempfile.TemporaryDirectory() as temporary:
            written, transform, peptide_chain = self.place_and_write(
                Path(temporary)
            )
            placed = gemmi.read_structure(written["aligned_peptide_pdb"])
            expected = [
                transform.apply(atom.pos)
                for res in peptide_geometry.amino_acid_residues(peptide_chain)
                for atom in res
            ]
            actual = [atom.pos for c in placed[0] for r in c for atom in r]
            self.assertEqual(len(expected), len(actual))
            for want, got in zip(expected, actual):
                # PDB stores three decimal places, so agreement is to 0.001 A.
                self.assertAlmostEqual(want.x, got.x, places=3)
                self.assertAlmostEqual(want.y, got.y, places=3)
                self.assertAlmostEqual(want.z, got.z, places=3)

    def test_residue_names_numbering_and_atom_names_survive_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            written, _, peptide_chain = self.place_and_write(Path(temporary))
            placed = gemmi.read_structure(written["aligned_peptide_pdb"])
            original = [
                (r.name, r.seqid.num, [a.name for a in r])
                for r in peptide_geometry.amino_acid_residues(peptide_chain)
            ]
            produced = [
                (r.name, r.seqid.num, [a.name for a in r])
                for c in placed[0]
                for r in c
            ]
            self.assertEqual(original, produced)

    def test_the_record_names_its_output_and_its_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            written, _, _ = self.place_and_write(Path(temporary))
            record = json.loads(Path(written["pose_record"]).read_text())
            self.assertIn("reference receptor frame", record["outputs"]["coordinate_frame"])
            self.assertIn("no minimisation", record["outputs"]["coordinate_frame"])
            self.assertEqual(
                len(record["outputs"]["aligned_peptide_pdb_sha256"]), 64
            )

    def test_writing_twice_into_one_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.place_and_write(Path(temporary))
            with self.assertRaises(ppp.PoseError) as raised:
                self.place_and_write(Path(temporary))
            self.assertIn("Refusing to overwrite", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
