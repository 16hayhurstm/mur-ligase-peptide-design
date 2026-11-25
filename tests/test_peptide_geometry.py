#!/usr/bin/env python3
"""Checks for refolded-design geometry validation.

Three kinds of check, kept separate:

- **Pure geometry**, on hand-built coordinates whose answers are known by
  construction: box containment, distances, and the chirality determinant
  including a deliberately mirrored residue.
- **Gemmi-backed**, on small synthetic mmCIF fixtures written here, for chain
  identification by sequence, atom completeness and the protein-only
  superposition.
- **Real structure**, on `data/1P3D.cif`, which pins the empirical chirality
  convention. That test skips if the file is absent rather than failing, since
  structures are not tracked in Git.

None of these validates docking, binding, or the ANP representation.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

import gemmi  # noqa: E402
import peptide_geometry as pg  # noqa: E402

ONE_P3D = REPO / "data" / "1P3D.cif"

# An L-alanine-like local frame. The determinant
# (N-CA) x (C-CA) . (CB-CA) is positive here, matching the convention
# calibrated on 1P3D; negating one coordinate axis mirrors it to D.
L_ALANINE = {
    "N": (-0.525, 1.362, 0.000),
    "CA": (0.000, 0.000, 0.000),
    "C": (1.520, 0.000, 0.000),
    "O": (2.200, 1.030, 0.000),
    "CB": (-0.510, -0.785, -1.200),
}


def mirrored(atoms):
    """Mirror through the z=0 plane, inverting configuration."""
    return {name: (x, y, -z) for name, (x, y, z) in atoms.items()}


def build_residue(name, seqid, atoms):
    residue = gemmi.Residue()
    residue.name = name
    residue.seqid = gemmi.SeqId(seqid, " ")
    for atom_name, (x, y, z) in atoms.items():
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(atom_name[0])
        atom.pos = gemmi.Position(x, y, z)
        residue.add_atom(atom)
    return residue


def glycine_atoms(offset):
    """Backbone-only glycine, translated along x so residues do not overlap."""
    return {
        "N": (offset - 0.5, 1.3, 0.0),
        "CA": (offset, 0.0, 0.0),
        "C": (offset + 1.5, 0.0, 0.0),
        "O": (offset + 2.2, 1.0, 0.0),
    }


def write_structure(path, chains):
    """Write a minimal mmCIF with the given {chain_name: [residues]}."""
    structure = gemmi.Structure()
    structure.name = path.stem
    model = gemmi.Model("1")
    for chain_name, residues in chains.items():
        chain = gemmi.Chain(chain_name)
        for residue in residues:
            chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(path))


def alanine_chain(count, start=1, shift=0.0, mirror=False):
    """A chain of `count` alanine residues on a non-collinear path.

    The residues are deliberately not collinear: a superposition of collinear
    point sets is rotationally degenerate and yields a NaN RMSD, which would
    make these fixtures test nothing.
    """
    residues = []
    for i in range(count):
        atoms = mirrored(L_ALANINE) if mirror else dict(L_ALANINE)
        dx, dy, dz = shift + 4.0 * i, 1.5 * (i % 2), 0.8 * (i % 3)
        placed = {
            name: (x + dx, y + dy, z + dz) for name, (x, y, z) in atoms.items()
        }
        residues.append(build_residue("ALA", start + i, placed))
    return residues


class ChiralityConventionTests(unittest.TestCase):
    """Positive determinant is L; the convention must be pinned, not assumed."""

    def test_l_frame_is_positive_and_classified_L(self):
        residue = build_residue("ALA", 1, L_ALANINE)
        configuration, determinant = pg.calpha_configuration(residue)
        self.assertEqual(configuration, "L")
        self.assertGreater(determinant, 0.0)

    def test_mirrored_frame_is_classified_D(self):
        residue = build_residue("ALA", 1, mirrored(L_ALANINE))
        configuration, determinant = pg.calpha_configuration(residue)
        self.assertEqual(configuration, "D")
        self.assertLess(determinant, 0.0)

    def test_glycine_has_no_centre_rather_than_failing(self):
        residue = build_residue("GLY", 1, glycine_atoms(0.0))
        configuration, determinant = pg.calpha_configuration(residue)
        self.assertIsNone(configuration)
        self.assertIsNone(determinant)

    def test_absent_backbone_atom_gives_no_verdict(self):
        atoms = dict(L_ALANINE)
        del atoms["N"]
        residue = build_residue("ALA", 1, atoms)
        self.assertEqual(pg.calpha_configuration(residue), (None, None))

    @unittest.skipUnless(ONE_P3D.is_file(), "data/1P3D.cif is not present")
    def test_every_cb_bearing_residue_in_1p3d_is_L(self):
        structure = pg.read_structure(ONE_P3D)
        verdicts = []
        for chain in structure[0]:
            for residue in pg.amino_acid_residues(chain):
                configuration, _ = pg.calpha_configuration(residue)
                if configuration is not None:
                    verdicts.append(configuration)
        self.assertGreater(len(verdicts), 500, "expected many chiral residues")
        self.assertEqual(
            set(verdicts), {"L"}, "a deposited protein must be all-L under this convention"
        )


class CrystallographicCopyTests(unittest.TestCase):
    """Pooling copies fabricates contacts; the chain filter must prevent it.

    1P3D holds two copies of the assembly. The numbers asserted here were
    measured from the deposited file and are what make the defect visible:
    pooled, UMA's nearest ANP atom is 3.02 A away, which is a crystal-packing
    artefact rather than a contact within either site.
    """

    @unittest.skipUnless(ONE_P3D.is_file(), "data/1P3D.cif is not present")
    def test_pooling_both_copies_doubles_every_component(self):
        model = pg.read_structure(ONE_P3D)[0]
        self.assertEqual(len(pg.component_positions(model, "ANP")), 62)
        self.assertEqual(len(pg.component_positions(model, "UMA")), 98)
        self.assertEqual(len(pg.component_positions(model, "MN")), 4)

    @unittest.skipUnless(ONE_P3D.is_file(), "data/1P3D.cif is not present")
    def test_restricting_to_one_author_chain_halves_the_counts(self):
        model = pg.read_structure(ONE_P3D)[0]
        # Both UMA copies sit in author chain A and B respectively in 1P3D's
        # author numbering; restricting to one must not return everything.
        single = pg.component_positions(model, "UMA", ["A"])
        self.assertEqual(len(single), 49)
        self.assertLess(len(single), len(pg.component_positions(model, "UMA")))

    @unittest.skipUnless(ONE_P3D.is_file(), "data/1P3D.cif is not present")
    def test_the_two_copies_have_genuinely_different_uma_anp_distances(self):
        """Copy A gives 3.96 A and copy B gives 3.02 A, measured within each copy.

        The 3.96 A figure reproduces the value independently recorded for this
        structure elsewhere in the project, which is a cross-check on this
        distance code against a number it did not produce.
        """
        model = pg.read_structure(ONE_P3D)[0]
        within = {
            chain: pg.minimum_distance(
                pg.component_positions(model, "UMA", [chain]),
                pg.component_positions(model, "ANP", [chain]),
                3.5,
            )["minimum_angstrom"]
            for chain in ("A", "B")
        }
        self.assertAlmostEqual(within["A"], 3.9619, places=3)
        self.assertAlmostEqual(within["B"], 3.0176, places=3)

    @unittest.skipUnless(ONE_P3D.is_file(), "data/1P3D.cif is not present")
    def test_pooling_reports_the_wrong_copys_geometry(self):
        """Pooling answers for whichever copy happens to be closer.

        The two copies are ~89 A apart, so pooling does not invent a contact
        across them. What it does is report copy B's 3.02 A while copy A, the
        copy actually being assessed, is 3.96 A. The figure is real but belongs
        to the wrong site, which is why the chain filter exists.
        """
        model = pg.read_structure(ONE_P3D)[0]
        uma_a = pg.component_positions(model, "UMA", ["A"])
        pooled = pg.minimum_distance(uma_a, pg.component_positions(model, "ANP"), 3.5)
        matched = pg.minimum_distance(
            uma_a, pg.component_positions(model, "ANP", ["A"]), 3.5
        )
        cross = pg.minimum_distance(
            uma_a, pg.component_positions(model, "ANP", ["B"]), 3.5
        )
        self.assertAlmostEqual(matched["minimum_angstrom"], 3.9619, places=3)
        self.assertGreater(cross["minimum_angstrom"], 80.0)
        # Restricting the UMA side to copy A already removes copy B's shorter
        # value, so the pooled ANP search agrees here; the hazard is pooling
        # the side being assessed.
        self.assertAlmostEqual(pooled["minimum_angstrom"], 3.9619, places=3)
        both_pooled = pg.minimum_distance(
            pg.component_positions(model, "UMA"),
            pg.component_positions(model, "ANP"),
            3.5,
        )
        self.assertAlmostEqual(both_pooled["minimum_angstrom"], 3.0176, places=3)

    @unittest.skipUnless(ONE_P3D.is_file(), "data/1P3D.cif is not present")
    def test_assess_design_defaults_the_component_search_to_one_chain(self):
        """The default must be a single copy, not every copy in the file."""
        import inspect

        signature = inspect.signature(pg.assess_design)
        self.assertIn("component_author_chains", signature.parameters)
        self.assertIsNone(
            signature.parameters["component_author_chains"].default,
            "the default must be None so assess_design can fall back to the "
            "reference chain rather than pooling copies",
        )

    @unittest.skipUnless(ONE_P3D.is_file(), "data/1P3D.cif is not present")
    def test_the_recorded_24a_box_contains_exactly_one_uma_copy(self):
        model = pg.read_structure(ONE_P3D)[0]
        box = json.loads(
            (REPO / "examples" / "box_murc_uma_chainE_24A.json").read_text()
        )
        pooled = pg.box_containment(pg.component_positions(model, "UMA"), box)
        self.assertEqual(pooled["atoms_inside"], 49)
        self.assertEqual(pooled["heavy_atoms"], 98)
        # The box was built from one UMA copy, so that copy must be contained.
        single = pg.box_containment(
            pg.component_positions(model, "UMA", ["A"]), box
        )
        self.assertTrue(single["fully_contained"])


class AtomCompletenessTests(unittest.TestCase):
    def test_complete_alanine_passes(self):
        chain = gemmi.Chain("P")
        for residue in alanine_chain(2):
            chain.add_residue(residue)
        findings = pg.inspect_peptide_residues(chain)
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(f.acceptable for f in findings))
        self.assertEqual(findings[0].missing_atoms, [])

    def test_missing_side_chain_atom_is_reported_not_raised(self):
        atoms = dict(L_ALANINE)
        del atoms["CB"]
        chain = gemmi.Chain("P")
        chain.add_residue(build_residue("ALA", 1, atoms))
        findings = pg.inspect_peptide_residues(chain)
        self.assertEqual(findings[0].missing_atoms, ["CB"])
        self.assertFalse(findings[0].acceptable)

    def test_oxt_is_permitted_and_not_flagged_unexpected(self):
        atoms = dict(L_ALANINE)
        atoms["OXT"] = (2.0, -1.0, 0.0)
        chain = gemmi.Chain("P")
        chain.add_residue(build_residue("ALA", 1, atoms))
        findings = pg.inspect_peptide_residues(chain)
        self.assertEqual(findings[0].unexpected_atoms, [])
        self.assertTrue(findings[0].acceptable)

    def test_non_standard_residue_is_noted_not_silently_accepted(self):
        chain = gemmi.Chain("P")
        chain.add_residue(build_residue("MSE", 1, L_ALANINE))
        findings = pg.inspect_peptide_residues(chain)
        self.assertIn("not a standard amino acid", findings[0].note)
        self.assertFalse(findings[0].acceptable)

    def test_a_d_residue_is_not_acceptable(self):
        chain = gemmi.Chain("P")
        chain.add_residue(build_residue("ALA", 1, mirrored(L_ALANINE)))
        findings = pg.inspect_peptide_residues(chain)
        self.assertEqual(findings[0].configuration, "D")
        self.assertFalse(findings[0].acceptable)


class SequenceIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.model = gemmi.Model("1")
        peptide = gemmi.Chain("B")
        for residue in alanine_chain(3):
            peptide.add_residue(residue)
        receptor = gemmi.Chain("A")
        for residue in alanine_chain(5, start=10, shift=50.0):
            receptor.add_residue(residue)
        self.model.add_chain(receptor)
        self.model.add_chain(peptide)

    def test_chain_is_found_by_exact_sequence(self):
        chain = pg.identify_chain_by_sequence(self.model, "AAA")
        self.assertEqual(chain.name, "B")

    def test_identification_does_not_use_chain_length(self):
        # The five-residue chain is selected when its sequence is asked for,
        # even though a shorter chain exists: length plays no part.
        chain = pg.identify_chain_by_sequence(self.model, "AAAAA")
        self.assertEqual(chain.name, "A")

    def test_no_match_is_fatal_and_lists_what_was_present(self):
        with self.assertRaises(pg.GeometryError) as caught:
            pg.identify_chain_by_sequence(self.model, "WWWW")
        self.assertIn("No chain matches", str(caught.exception))
        self.assertIn("AAA", str(caught.exception))

    def test_ambiguous_match_is_fatal(self):
        duplicate = gemmi.Chain("C")
        for residue in alanine_chain(3, start=30, shift=100.0):
            duplicate.add_residue(residue)
        self.model.add_chain(duplicate)
        with self.assertRaises(pg.GeometryError) as caught:
            pg.identify_chain_by_sequence(self.model, "AAA")
        self.assertIn("several chains", str(caught.exception))

    def test_empty_sequence_is_refused(self):
        with self.assertRaises(pg.GeometryError):
            pg.identify_chain_by_sequence(self.model, "   ")


class SuperpositionTests(unittest.TestCase):
    """Only the protein is fitted, and the fit must be honest about its basis."""

    def test_identical_chains_fit_at_zero_rmsd(self):
        a = gemmi.Chain("A")
        b = gemmi.Chain("A")
        for residue in alanine_chain(5):
            a.add_residue(residue)
        for residue in alanine_chain(5):
            b.add_residue(residue)
        _, report = pg.superpose_protein(a, b)
        self.assertEqual(report["matched_ca_atoms"], 5)
        self.assertAlmostEqual(report["rmsd_angstrom"], 0.0, places=3)

    def test_a_pure_translation_is_recovered_exactly(self):
        mobile = gemmi.Chain("A")
        reference = gemmi.Chain("A")
        for residue in alanine_chain(5, shift=17.0):
            mobile.add_residue(residue)
        for residue in alanine_chain(5):
            reference.add_residue(residue)
        transform, report = pg.superpose_protein(mobile, reference)
        self.assertAlmostEqual(report["rmsd_angstrom"], 0.0, places=3)
        moved = transform.apply(gemmi.Position(17.0, 0.0, 0.0))
        self.assertAlmostEqual(moved.x, 0.0, places=3)

    def test_residue_name_mismatch_is_skipped_and_reported(self):
        mobile = gemmi.Chain("A")
        reference = gemmi.Chain("A")
        for residue in alanine_chain(4):
            mobile.add_residue(residue)
        residues = alanine_chain(4)
        residues[1].name = "GLY"
        for residue in residues:
            reference.add_residue(residue)
        _, report = pg.superpose_protein(mobile, reference)
        self.assertEqual(report["matched_ca_atoms"], 3)
        self.assertEqual(len(report["skipped_residues"]), 1)
        self.assertIn("ALA vs GLY", report["skipped_residues"][0])

    def test_too_few_corresponding_residues_is_fatal(self):
        mobile = gemmi.Chain("A")
        reference = gemmi.Chain("A")
        for residue in alanine_chain(2):
            mobile.add_residue(residue)
        for residue in alanine_chain(2, start=90):
            reference.add_residue(residue)
        with self.assertRaises(pg.GeometryError) as caught:
            pg.superpose_protein(mobile, reference)
        self.assertIn("at least", str(caught.exception))


class BoxContainmentTests(unittest.TestCase):
    BOX = {"centre_angstrom": [0.0, 0.0, 0.0], "size_angstrom": [10.0, 10.0, 10.0]}

    def test_points_inside_are_fully_contained(self):
        points = np.array([[0.0, 0.0, 0.0], [4.0, -4.0, 4.0]])
        result = pg.box_containment(points, self.BOX)
        self.assertTrue(result["fully_contained"])
        self.assertEqual(result["fraction_inside"], 1.0)
        self.assertEqual(result["per_axis_overshoot_angstrom"], [0.0, 0.0, 0.0])

    def test_a_point_on_the_face_counts_as_inside(self):
        points = np.array([[5.0, 0.0, 0.0]])
        self.assertTrue(pg.box_containment(points, self.BOX)["fully_contained"])

    def test_partial_containment_is_quantified(self):
        points = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        result = pg.box_containment(points, self.BOX)
        self.assertFalse(result["fully_contained"])
        self.assertEqual(result["atoms_inside"], 1)
        self.assertEqual(result["fraction_inside"], 0.5)
        self.assertAlmostEqual(result["per_axis_overshoot_angstrom"][0], 4.0)
        # 18 A about this centre would be needed on x to contain both.
        self.assertAlmostEqual(
            result["required_size_about_this_centre_angstrom"][0], 18.0
        )

    def test_malformed_box_records_are_refused(self):
        for bad in (
            {"centre_angstrom": [0, 0, 0]},
            {"centre_angstrom": [0, 0], "size_angstrom": [1, 1, 1]},
            {"centre_angstrom": [0, 0, 0], "size_angstrom": [0, 1, 1]},
        ):
            with self.assertRaises(pg.GeometryError):
                pg.box_containment(np.zeros((1, 3)), bad)


class DistanceTests(unittest.TestCase):
    def test_minimum_and_contact_count_use_a_strict_cutoff(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        result = pg.minimum_distance(a, b, contact_cutoff=4.0)
        self.assertAlmostEqual(result["minimum_angstrom"], 3.0)
        self.assertEqual(result["contacts_below_cutoff"], 1)

    def test_a_distance_equal_to_the_cutoff_is_not_a_contact(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[4.0, 0.0, 0.0]])
        result = pg.minimum_distance(a, b, contact_cutoff=4.0)
        self.assertEqual(result["contacts_below_cutoff"], 0)

    def test_empty_component_yields_no_distance_rather_than_zero(self):
        result = pg.minimum_distance(np.zeros((1, 3)), np.zeros((0, 3)), 4.0)
        self.assertIsNone(result["minimum_angstrom"])
        self.assertIn("empty", result["note"])


class EndToEndTests(unittest.TestCase):
    """A synthetic refolded complex and reference, assessed as a whole."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Reference receptor: 6 alanines, plus an ANP-like component.
        anp = build_residue("ANP", 500, {"P": (3.0, 0.0, 0.0)})
        write_structure(
            self.tmp / "reference.cif",
            {"A": alanine_chain(6) + [anp]},
        )
        # Refolded complex: the same receptor translated, plus a 3-mer peptide.
        write_structure(
            self.tmp / "design.cif",
            {
                "A": alanine_chain(6, shift=25.0),
                "B": alanine_chain(3, start=200, shift=25.0),
            },
        )
        self.box = {
            "centre_angstrom": [4.0, 0.0, 0.0],
            "size_angstrom": [30.0, 30.0, 30.0],
        }

    def assess(self, **kwargs):
        return pg.assess_design(
            self.tmp / "design.cif",
            kwargs.pop("sequence", "AAA"),
            self.tmp / "reference.cif",
            "A",
            box_record=kwargs.pop("box", self.box),
            design_id="demo",
            **kwargs,
        )

    def test_record_identifies_peptide_and_protein_chains_correctly(self):
        record = self.assess()
        self.assertEqual(record["peptide_chain"], "B")
        self.assertEqual(record["protein_chain"], "A")
        self.assertEqual(record["residues"], 3)
        self.assertTrue(record["usable_for_preparation"])
        self.assertTrue(record["atoms_complete"])
        self.assertTrue(record["all_calpha_L"])
        self.assertEqual(record["d_configured_residues"], [])

    def test_protein_fit_removes_the_translation_so_distances_are_meaningful(self):
        record = self.assess()
        self.assertAlmostEqual(
            record["protein_superposition"]["rmsd_angstrom"], 0.0, places=3
        )
        self.assertEqual(record["protein_superposition"]["matched_ca_atoms"], 6)
        # The peptide was built on top of the receptor's first three residues,
        # so after the protein fit it sits within a few Angstroms of ANP at
        # x = 3. A large value here would mean the transform was not applied.
        self.assertLess(record["distances"]["ANP"]["minimum_angstrom"], 10.0)

    def test_box_containment_is_reported_against_the_supplied_record(self):
        record = self.assess()
        containment = record["box_containment"]
        self.assertTrue(containment["fully_contained"])
        self.assertEqual(containment["box_size_angstrom"], [30.0, 30.0, 30.0])

    def test_a_small_box_reports_partial_containment_without_failing(self):
        record = self.assess(
            box={"centre_angstrom": [0.0, 0.0, 0.0], "size_angstrom": [2.0, 2.0, 2.0]}
        )
        self.assertFalse(record["box_containment"]["fully_contained"])
        # Box fit is not a preparation blocker; it is a separate finding.
        self.assertTrue(record["usable_for_preparation"])

    def test_a_d_residue_makes_the_design_unusable_and_is_listed(self):
        write_structure(
            self.tmp / "d_design.cif",
            {
                "A": alanine_chain(6, shift=25.0),
                "B": alanine_chain(3, start=200, shift=25.0, mirror=True),
            },
        )
        record = pg.assess_design(
            self.tmp / "d_design.cif",
            "AAA",
            self.tmp / "reference.cif",
            "A",
            design_id="mirror",
        )
        self.assertFalse(record["all_calpha_L"])
        self.assertFalse(record["usable_for_preparation"])
        self.assertEqual(record["d_configured_residues"], [1, 2, 3])

    def test_unknown_reference_chain_is_fatal(self):
        with self.assertRaises(pg.GeometryError):
            pg.assess_design(
                self.tmp / "design.cif",
                "AAA",
                self.tmp / "reference.cif",
                "Z",
            )

    def test_absent_structure_is_fatal(self):
        with self.assertRaises(pg.GeometryError):
            pg.assess_design(
                self.tmp / "missing.cif", "AAA", self.tmp / "reference.cif", "A"
            )


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        anp = build_residue("ANP", 500, {"P": (3.0, 0.0, 0.0)})
        write_structure(self.tmp / "reference.cif", {"A": alanine_chain(6) + [anp]})
        write_structure(
            self.tmp / "design.cif",
            {
                "A": alanine_chain(6, shift=25.0),
                "B": alanine_chain(3, start=200, shift=25.0),
            },
        )

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO / "python" / "peptide_geometry.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_cli_emits_a_record_and_exits_zero_when_usable(self):
        out = self.tmp / "record.json"
        result = self.run_cli(
            str(self.tmp / "design.cif"),
            "--designed-sequence", "AAA",
            "--reference", str(self.tmp / "reference.cif"),
            "--json-out", str(out),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertTrue(record["usable_for_preparation"])
        self.assertEqual(json.loads(out.read_text())["peptide_chain"], "B")

    def test_cli_exits_two_on_a_sequence_that_matches_nothing(self):
        result = self.run_cli(
            str(self.tmp / "design.cif"),
            "--designed-sequence", "WWWW",
            "--reference", str(self.tmp / "reference.cif"),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("assessment failed", result.stderr)

    def test_cli_refuses_to_overwrite_a_record(self):
        out = self.tmp / "record.json"
        out.write_text("keep")
        result = self.run_cli(
            str(self.tmp / "design.cif"),
            "--designed-sequence", "AAA",
            "--reference", str(self.tmp / "reference.cif"),
            "--json-out", str(out),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(out.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
