"""Tests for the toolkit-independent parts of ligand verification.

What is tested here, and what is not
------------------------------------
`verify_ligand_rdkit.py` is split so that everything except the chemistry
perception can be executed without a toolkit installed. These tests cover that
part, in full:

- the index-to-name correspondence derived from coordinates, including every
  way it can be ambiguous;
- element and connectivity validation under that correspondence;
- the per-centre stereochemistry comparison, including the carbon/phosphorus
  separation and the "investigate, do not assume inversion" classification;
- the heavy-atom coordinate check after hydrogen addition;
- the failure record, and the downstream validation of a verification record.

**None of this exercises RDKit.** Molecule reading, `AssignStereochemistryFrom3D`,
`rdCIPLabeler.AssignCIPLabels`, `AddHs(addCoords=True)` and `MolToMolFile` are
reached only through the thin adapter in that module, which is **not executed
anywhere in this suite**: RDKit is not installed on the development machine.
There is deliberately **no RDKit stub** — stubbing stereochemistry perception
would assert chemistry that nothing here can check, and could hide an
interface error rather than reveal one.

The fixtures below are `MoleculeSnapshot` objects built by hand, with
coordinates chosen so the expected answers are derivable by reading them. They
are not molecules and nothing here establishes any chemical fact.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import analyse_redocking  # noqa: E402
import box_geometry  # noqa: E402
import receptor_provenance as provenance  # noqa: E402
import verify_ligand_rdkit as verify  # noqa: E402


# A four-atom fixture: a carbon, two oxygens and a nitrogen at known places.
REFERENCE = {
    "C1": (0.0, 0.0, 0.0),
    "O1": (1.5, 0.0, 0.0),
    "O2": (0.0, 1.5, 0.0),
    "N1": (0.0, 0.0, 1.5),
}
REFERENCE_ELEMENTS = {"C1": "C", "O1": "O", "O2": "O", "N1": "N"}
DICTIONARY_BONDS = [
    ("C1", "O1", "DOUB"),
    ("C1", "O2", "SING"),
    ("C1", "N1", "SING"),
]


def snapshot(order=("C1", "O1", "O2", "N1"), *, positions=None,
             elements=None, bonds=None, extra_hydrogens=0, charge=0,
             cip=()):
    """Build a snapshot whose atoms appear in `order`.

    `order` is what makes the correspondence tests meaningful: the file order
    is varied while the coordinates stay attached to their names, so a mapping
    derived from coordinates must still be right and a mapping assumed from
    order must be wrong.
    """
    names = list(order)
    coordinates = positions or {name: REFERENCE[name] for name in names}
    symbols = elements or {name: REFERENCE_ELEMENTS[name] for name in names}
    index = {name: position for position, name in enumerate(names)}
    atom_positions = [coordinates[name] for name in names]
    atom_elements = [symbols[name] for name in names]
    for number in range(extra_hydrogens):
        atom_positions.append((9.0 + number, 9.0, 9.0))
        atom_elements.append("H")
    bond_set = set()
    for first, second, order_name in (bonds if bonds is not None
                                      else DICTIONARY_BONDS):
        rdkit_order = {"SING": "SINGLE", "DOUB": "DOUBLE",
                       "TRIP": "TRIPLE"}.get(order_name, order_name)
        bond_set.add((index[first], index[second], rdkit_order))
    return verify.MoleculeSnapshot(
        elements=tuple(atom_elements), positions=tuple(atom_positions),
        bonds=frozenset(bond_set), n_atoms_total=len(atom_elements),
        formal_charge=charge,
        cip_codes=tuple((index[name], code) for name, code in cip))


class CorrespondenceTests(unittest.TestCase):
    """The mapping must be derived from coordinates, never assumed from order."""

    def test_file_order_matching_the_reference_maps_correctly(self):
        mapping = verify.correspondence_from_coordinates(snapshot(), REFERENCE)
        self.assertEqual(mapping, {0: "C1", 1: "O1", 2: "O2", 3: "N1"})

    def test_a_reordered_file_still_maps_correctly(self):
        """This is the point of deriving the mapping rather than assuming it."""
        mapping = verify.correspondence_from_coordinates(
            snapshot(order=("N1", "O2", "C1", "O1")), REFERENCE)
        self.assertEqual(mapping, {0: "N1", 1: "O2", 2: "C1", 3: "O1"})
        # An index-order assumption would have produced the wrong names.
        self.assertNotEqual([mapping[i] for i in sorted(mapping)],
                            list(REFERENCE))

    def test_hydrogens_are_ignored_by_the_correspondence(self):
        mapping = verify.correspondence_from_coordinates(
            snapshot(extra_hydrogens=3), REFERENCE)
        self.assertEqual(len(mapping), 4)
        self.assertNotIn(4, mapping)

    def test_a_different_heavy_atom_count_is_refused(self):
        with self.assertRaises(verify.VerificationError) as caught:
            verify.correspondence_from_coordinates(
                snapshot(order=("C1", "O1", "O2"),
                         bonds=[("C1", "O1", "DOUB"), ("C1", "O2", "SING")]),
                REFERENCE)
        self.assertIn("bijection", str(caught.exception))

    def test_an_atom_matching_nothing_is_refused_with_the_nearest_distance(self):
        moved = dict(REFERENCE)
        moved_positions = {name: REFERENCE[name] for name in REFERENCE}
        moved_positions["N1"] = (0.0, 0.0, 5.0)
        with self.assertRaises(verify.VerificationError) as caught:
            verify.correspondence_from_coordinates(
                snapshot(positions=moved_positions), moved)
        message = str(caught.exception)
        self.assertIn("matches no reference atom", message)
        self.assertIn("nearest", message)

    def test_two_atoms_claiming_one_reference_name_is_refused(self):
        duplicated = {name: REFERENCE[name] for name in REFERENCE}
        duplicated["O2"] = REFERENCE["O1"]
        with self.assertRaises(verify.VerificationError) as caught:
            verify.correspondence_from_coordinates(
                snapshot(positions=duplicated), REFERENCE)
        self.assertIn("not a", str(caught.exception))

    def test_coincident_reference_atoms_are_refused_as_ambiguous(self):
        coincident = dict(REFERENCE)
        coincident["O2"] = REFERENCE["O1"]
        with self.assertRaises(verify.VerificationError) as caught:
            verify.correspondence_from_coordinates(snapshot(), coincident)
        self.assertIn("ambiguous", str(caught.exception))

    def test_the_tolerance_is_applied(self):
        shifted = {name: REFERENCE[name] for name in REFERENCE}
        shifted["O1"] = (1.5015, 0.0, 0.0)
        with self.assertRaises(verify.VerificationError):
            verify.correspondence_from_coordinates(
                snapshot(positions=shifted), REFERENCE, tolerance=0.001)
        mapping = verify.correspondence_from_coordinates(
            snapshot(positions=shifted), REFERENCE, tolerance=0.002)
        self.assertEqual(mapping[1], "O1")


class ElementAndConnectivityTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = snapshot()
        self.mapping = verify.correspondence_from_coordinates(
            self.snapshot, REFERENCE)

    def test_matching_elements_report_no_problem(self):
        self.assertEqual(
            verify.validate_elements(self.mapping, self.snapshot,
                                     REFERENCE_ELEMENTS), [])

    def test_an_element_disagreement_is_reported(self):
        wrong = dict(REFERENCE_ELEMENTS, N1="C")
        problems = verify.validate_elements(self.mapping, self.snapshot, wrong)
        self.assertEqual(len(problems), 1)
        self.assertIn("N1", problems[0])

    def test_matching_connectivity_agrees(self):
        report = verify.validate_connectivity(self.mapping, self.snapshot,
                                              DICTIONARY_BONDS)
        self.assertTrue(report["agrees"])
        self.assertEqual(report["n_molecule_heavy_bonds"], 3)
        self.assertEqual(report["n_dictionary_bonds"], 3)

    def test_a_missing_bond_is_reported(self):
        report = verify.validate_connectivity(
            self.mapping, snapshot(bonds=DICTIONARY_BONDS[:2]),
            DICTIONARY_BONDS)
        self.assertFalse(report["agrees"])
        self.assertEqual(report["missing_bonds"], ["C1-N1"])
        self.assertEqual(report["unexpected_bonds"], [])

    def test_an_unexpected_bond_is_reported(self):
        report = verify.validate_connectivity(
            self.mapping,
            snapshot(bonds=DICTIONARY_BONDS + [("O1", "O2", "SING")]),
            DICTIONARY_BONDS)
        self.assertFalse(report["agrees"])
        self.assertEqual(report["unexpected_bonds"], ["O1-O2"])
        self.assertEqual(report["missing_bonds"], [])

    def test_a_bond_order_difference_is_reported_separately(self):
        """A wrong order is a different failure from a wrong skeleton."""
        report = verify.validate_connectivity(
            self.mapping,
            snapshot(bonds=[("C1", "O1", "SING"), ("C1", "O2", "SING"),
                            ("C1", "N1", "SING")]),
            DICTIONARY_BONDS)
        self.assertFalse(report["agrees"])
        self.assertEqual(report["missing_bonds"], [])
        self.assertEqual(report["unexpected_bonds"], [])
        self.assertEqual(len(report["bond_order_differences"]), 1)
        self.assertIn("C1-O1", report["bond_order_differences"][0])

    def test_bonds_to_hydrogens_are_skipped(self):
        with_hydrogen = snapshot(extra_hydrogens=1)
        bonds = set(with_hydrogen.bonds) | {(0, 4, "SINGLE")}
        hydrated = verify.MoleculeSnapshot(
            elements=with_hydrogen.elements, positions=with_hydrogen.positions,
            bonds=frozenset(bonds), n_atoms_total=5)
        report = verify.validate_connectivity(self.mapping, hydrated,
                                              DICTIONARY_BONDS)
        self.assertTrue(report["agrees"])


# A plan state in the shape `prepare_ligand.py stereo-plan` writes.
PLAN_STATE = {
    "formal_charge": -3,
    "must_match": [
        {"atom": "C21", "element": "C", "declared_configuration": "S",
         "fatal_if_mismatched": True},
        {"atom": "C18", "element": "C", "declared_configuration": "R",
         "fatal_if_mismatched": True},
    ],
    "not_stereogenic_at_this_state": [
        {"atom": "PA", "element": "P", "declared_configuration": "R",
         "fatal_if_mismatched": False},
        {"atom": "PB", "element": "P", "declared_configuration": "R",
         "fatal_if_mismatched": False},
    ],
}


class RawMolfileBondTests(unittest.TestCase):
    """The molfile's own bond orders, read without any toolkit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, bonds, names=("C1", "O1", "O2", "N1"), dimensional="3D"):
        index = {name: position for position, name in enumerate(names, start=1)}
        lines = ["fixture", "  murligpd" + "0" * 10 + dimensional, "comment",
                 f"{len(names):>3}{len(bonds):>3}  0  0  0  0  0  0  0  0999 V2000"]
        for position, name in enumerate(names):
            lines.append(f"{float(position):>10.4f}{0.0:>10.4f}{1.0:>10.4f} "
                         f"{name[0]:<3} 0  0  0  0  0  0  0  0  0  0  0")
        codes = {"SING": 1, "DOUB": 2, "TRIP": 3, "AROM": 4}
        for first, second, order in bonds:
            lines.append(f"{index[first]:>3}{index[second]:>3}"
                         f"{codes[order]:>3}  0  0  0  0")
        lines.append("M  END")
        path = self.root / "fixture.mol"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_matching_kekule_orders_agree(self):
        path = self.write(DICTIONARY_BONDS)
        report = verify.validate_raw_molfile_bonds(
            path, ("C1", "O1", "O2", "N1"), DICTIONARY_BONDS)
        self.assertTrue(report["agrees"])
        self.assertEqual(report["n_molfile_bonds"], 3)
        self.assertEqual(report["order_differences"], [])

    def test_a_changed_bond_order_in_the_file_is_caught(self):
        """The file must say what the dictionary says, bond for bond."""
        path = self.write([("C1", "O1", "SING"), ("C1", "O2", "SING"),
                           ("C1", "N1", "SING")])
        report = verify.validate_raw_molfile_bonds(
            path, ("C1", "O1", "O2", "N1"), DICTIONARY_BONDS)
        self.assertFalse(report["agrees"])
        self.assertEqual(len(report["order_differences"]), 1)
        self.assertIn("C1-O1", report["order_differences"][0])
        self.assertIn("dictionary DOUB", report["order_differences"][0])
        self.assertIn("molfile SING", report["order_differences"][0])

    def test_a_missing_bond_in_the_file_is_caught(self):
        path = self.write(DICTIONARY_BONDS[:2])
        report = verify.validate_raw_molfile_bonds(
            path, ("C1", "O1", "O2", "N1"), DICTIONARY_BONDS)
        self.assertFalse(report["agrees"])
        self.assertEqual(report["missing_bonds"], ["C1-N1"])

    def test_an_atom_count_disagreement_is_refused(self):
        path = self.write(DICTIONARY_BONDS)
        with self.assertRaises(verify.VerificationError) as caught:
            verify.validate_raw_molfile_bonds(
                path, ("C1", "O1", "O2"), DICTIONARY_BONDS)
        self.assertIn("declares 4 atoms", str(caught.exception))

    def test_a_malformed_counts_line_is_refused(self):
        path = self.root / "bad.mol"
        path.write_text("t\nl2\nl3\nnot a counts line\n")
        with self.assertRaises(verify.VerificationError) as caught:
            verify.validate_raw_molfile_bonds(path, ("C1",), [])
        self.assertIn("counts line", str(caught.exception))

    def test_the_generated_header_declares_three_dimensions(self):
        """Columns 21-22 of line 2 carry the dimensional code.

        The first Epicac run warned that the molecule was tagged 2D while
        carrying non-zero Z coordinates, because free text occupied those
        columns.
        """
        path = self.write(DICTIONARY_BONDS)
        line = path.read_text().splitlines()[1]
        self.assertEqual(line[20:22], "3D")


class OrderNormalisationTests(unittest.TestCase):
    """The two bond-order vocabularies must not be confused with chemistry."""

    def test_rdkit_and_ccd_tokens_normalise_together(self):
        for rdkit_token, ccd_token in (("SINGLE", "SING"), ("DOUBLE", "DOUB"),
                                       ("TRIPLE", "TRIP"),
                                       ("AROMATIC", "AROM")):
            self.assertEqual(verify.normalise_order(rdkit_token),
                             verify.normalise_order(ccd_token))

    def test_different_orders_stay_different(self):
        self.assertNotEqual(verify.normalise_order("SINGLE"),
                            verify.normalise_order("DOUB"))
        self.assertNotEqual(verify.normalise_order("AROMATIC"),
                            verify.normalise_order("SING"))

    def test_connectivity_comparison_is_vocabulary_insensitive(self):
        """Both sides may speak either vocabulary without a false mismatch."""
        current = snapshot()
        mapping = verify.correspondence_from_coordinates(current, REFERENCE)
        rdkit_style = [("C1", "O1", "DOUBLE"), ("C1", "O2", "SINGLE"),
                       ("C1", "N1", "SINGLE")]
        self.assertTrue(verify.validate_connectivity(
            mapping, current, rdkit_style)["agrees"])
        self.assertTrue(verify.validate_connectivity(
            mapping, current, DICTIONARY_BONDS)["agrees"])

    def test_an_aromatic_expectation_against_a_kekule_molecule_disagrees(self):
        """Normalisation must not make AROMATIC and SINGLE interchangeable.

        This is the asymmetric case the Epicac failure exposed. Treating the
        two as equal would have made that error disappear while hiding real
        differences, which is why the fix is to perceive both sides
        identically rather than to widen this comparison.
        """
        current = snapshot()
        mapping = verify.correspondence_from_coordinates(current, REFERENCE)
        aromatic_expectation = [("C1", "O1", "AROM"), ("C1", "O2", "SING"),
                                ("C1", "N1", "SING")]
        report = verify.validate_connectivity(mapping, current,
                                              aromatic_expectation)
        self.assertFalse(report["agrees"])
        self.assertEqual(len(report["bond_order_differences"]), 1)


class StereoComparisonTests(unittest.TestCase):
    def test_agreement_on_every_centre_passes(self):
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C21": "S", "C18": "R", "PA": "R", "PB": "R"}, True)
        self.assertTrue(result["passes"])
        self.assertEqual(result["fatal_mismatches"], [])
        self.assertEqual(result["requires_investigation"], [])
        self.assertEqual(result["n_agreeing"], 4)

    def test_carbon_and_phosphorus_are_reported_separately(self):
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C21": "S", "C18": "R", "PA": "S", "PB": "R"}, True)
        self.assertEqual([entry["atom"] for entry
                          in result["carbon_and_other_centres"]],
                         ["C21", "C18"])
        self.assertEqual([entry["atom"] for entry
                          in result["phosphorus_and_sulphur_centres"]],
                         ["PA", "PB"])

    def test_a_carbon_mismatch_is_fatal(self):
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C21": "R", "C18": "R", "PA": "R", "PB": "R"}, True)
        self.assertFalse(result["passes"])
        self.assertEqual(result["fatal_mismatches"], ["C21"])
        self.assertIn("C21", result["requires_investigation"])

    def test_a_phosphorus_difference_is_recorded_but_not_fatal(self):
        """Not silently discarded, and not treated as the wrong stereoisomer."""
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C21": "S", "C18": "R", "PA": "S", "PB": "R"}, True)
        self.assertTrue(result["passes"])
        self.assertEqual(result["fatal_mismatches"], [])
        self.assertIn("PA", result["requires_investigation"])
        entry = next(item for item in result["phosphorus_and_sulphur_centres"]
                     if item["atom"] == "PA")
        self.assertFalse(entry["agrees"])
        self.assertEqual(entry["assigned_from_3d"], "S")
        self.assertIn("NOT a physical inversion", entry["finding"])

    def test_a_difference_is_not_called_an_inversion_when_coordinates_verified(self):
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C21": "R", "C18": "R", "PA": "R", "PB": "R"}, True)
        finding = next(entry["finding"] for entry
                       in result["carbon_and_other_centres"]
                       if entry["atom"] == "C21")
        self.assertIn("NOT a physical inversion", finding)
        self.assertIn("priority", finding)

    def test_without_a_coordinate_check_inversion_cannot_be_excluded(self):
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C21": "R", "C18": "R", "PA": "R", "PB": "R"}, False)
        finding = next(entry["finding"] for entry
                       in result["carbon_and_other_centres"]
                       if entry["atom"] == "C21")
        self.assertIn("cannot be excluded", finding)

    def test_a_missing_code_on_a_must_match_centre_needs_investigation(self):
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C18": "R", "PA": "R", "PB": "R"}, True)
        self.assertIn("C21", result["requires_investigation"])
        self.assertEqual(result["fatal_mismatches"], ["C21"])
        entry = next(item for item in result["carbon_and_other_centres"]
                     if item["atom"] == "C21")
        self.assertIsNone(entry["assigned_from_3d"])
        self.assertIn("no CIP code", entry["finding"])

    def test_a_missing_code_on_a_non_stereogenic_centre_is_consistent(self):
        result = verify.compare_stereocentres(
            PLAN_STATE, {"C21": "S", "C18": "R"}, True)
        self.assertTrue(result["passes"])
        self.assertEqual(result["requires_investigation"], [])
        entry = next(item for item in result["phosphorus_and_sulphur_centres"]
                     if item["atom"] == "PA")
        self.assertIn("consistent with the plan", entry["finding"])

    def test_every_planned_centre_appears_in_the_result(self):
        """Nothing is dropped, whatever the outcome."""
        result = verify.compare_stereocentres(PLAN_STATE, {}, True)
        reported = {entry["atom"] for entry
                    in result["carbon_and_other_centres"]
                    + result["phosphorus_and_sulphur_centres"]}
        self.assertEqual(reported, {"C21", "C18", "PA", "PB"})

    def test_the_phosphorus_plan_for_the_minus2_state_is_honoured(self):
        """At -2 the plan moves PB to must_match, marked non-fatal."""
        minus2 = {
            "formal_charge": -2,
            "must_match": [
                {"atom": "C21", "element": "C", "declared_configuration": "S",
                 "fatal_if_mismatched": True},
                {"atom": "PB", "element": "P", "declared_configuration": "R",
                 "fatal_if_mismatched": False},
            ],
            "not_stereogenic_at_this_state": [
                {"atom": "PA", "element": "P", "declared_configuration": "R",
                 "fatal_if_mismatched": False},
            ],
        }
        result = verify.compare_stereocentres(
            minus2, {"C21": "S", "PB": "S", "PA": "R"}, True)
        self.assertTrue(result["passes"], "a non-fatal PB difference must not fail")
        self.assertIn("PB", result["requires_investigation"])
        entry = next(item for item in result["phosphorus_and_sulphur_centres"]
                     if item["atom"] == "PB")
        self.assertTrue(entry["expected_to_match"])
        self.assertFalse(entry["fatal_if_mismatched"])


class HydrogenCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.before = snapshot()
        self.mapping = verify.correspondence_from_coordinates(
            self.before, REFERENCE)

    def test_unmoved_heavy_atoms_pass(self):
        after = snapshot(extra_hydrogens=4)
        report = verify.compare_heavy_coordinates(
            self.before, after, self.mapping)
        self.assertTrue(report["preserved"])
        self.assertEqual(report["max_deviation_angstrom"], 0.0)
        self.assertEqual(report["n_heavy_atoms_compared"], 4)

    def test_a_moved_heavy_atom_is_caught(self):
        moved = {name: REFERENCE[name] for name in REFERENCE}
        moved["O2"] = (0.0, 1.6, 0.0)
        after = snapshot(positions=moved, extra_hydrogens=4)
        report = verify.compare_heavy_coordinates(
            self.before, after, self.mapping)
        self.assertFalse(report["preserved"])
        self.assertAlmostEqual(report["max_deviation_angstrom"], 0.1, places=6)
        self.assertEqual(report["max_deviation_atom"], "O2")

    def test_losing_a_heavy_atom_raises(self):
        truncated = verify.MoleculeSnapshot(
            elements=("C", "O"), positions=((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)),
            bonds=frozenset(), n_atoms_total=2)
        with self.assertRaises(verify.VerificationError) as caught:
            verify.compare_heavy_coordinates(self.before, truncated,
                                             self.mapping)
        self.assertIn("atom order was not preserved", str(caught.exception))


class FailureRecordTests(unittest.TestCase):
    def test_a_failure_record_never_says_passed(self):
        class Arguments:
            state = "modelled"
            ligand_manifest = Path("ligands/uma.ligand.json")
            stereo_plan = Path("ligands/uma.stereo_plan.json")

        record = verify.failure_record(Arguments(), "the charge was 0, not -3")
        self.assertEqual(record["result"], "failed")
        self.assertEqual(record["schema"], verify.VERIFICATION_SCHEMA)
        self.assertIsNone(record["outputs"]["hydrogenated_molfile"])
        self.assertIn("the charge was 0", record["failure_reason"])
        self.assertNotIn("passed", json.dumps(record["outputs"]))


class DownstreamRecordValidationTests(unittest.TestCase):
    """A record must be validated, not merely present."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "uma.ligand.json"
        self.manifest.write_text(json.dumps({"schema": "mur-ligase-ligand/1"}))
        self.manifest_hash = box_geometry.file_sha256(self.manifest)

    def record(self, **overrides):
        base = {
            "schema": verify.VERIFICATION_SCHEMA,
            "result": "passed",
            "state": "modelled",
            "tools": {"rdkit": "2026.3.3"},
            "inputs": {"ligand_manifest_sha256": self.manifest_hash},
            "checks": {"stereochemistry_before_hydrogens":
                       {"fatal_mismatches": [], "requires_investigation": []}},
        }
        base.update(overrides)
        path = self.root / "uma.verification.json"
        path.write_text(json.dumps(base))
        return path

    def test_no_record_is_not_satisfied(self):
        status = analyse_redocking.validated_stereo_verification(
            None, self.manifest)
        self.assertFalse(status["satisfied"])
        self.assertIn("no record supplied", status["reason"])

    def test_a_valid_passing_record_is_satisfied(self):
        status = analyse_redocking.validated_stereo_verification(
            self.record(), self.manifest)
        self.assertTrue(status["satisfied"], status["reason"])
        self.assertEqual(status["record_result"], "passed")
        self.assertEqual(status["rdkit_version"], "2026.3.3")

    def test_a_failed_record_is_not_satisfied(self):
        path = self.record(result="failed",
                           failure_reason="the charge was 0, not -3")
        status = analyse_redocking.validated_stereo_verification(
            path, self.manifest)
        self.assertFalse(status["satisfied"])
        self.assertIn("not 'passed'", status["reason"])
        self.assertIn("the charge was 0", status["reason"])

    def test_a_record_with_a_fatal_mismatch_is_not_satisfied(self):
        path = self.record(checks={"stereochemistry_before_hydrogens":
                                   {"fatal_mismatches": ["C21"]}})
        status = analyse_redocking.validated_stereo_verification(
            path, self.manifest)
        self.assertFalse(status["satisfied"])
        self.assertIn("C21", status["reason"])

    def test_a_record_for_another_ligand_is_not_satisfied(self):
        """Identity, not just existence: the hash must match."""
        path = self.record(inputs={"ligand_manifest_sha256": "f" * 64})
        status = analyse_redocking.validated_stereo_verification(
            path, self.manifest)
        self.assertFalse(status["satisfied"])
        self.assertIn("different ligand manifest", status["reason"])

    def test_a_record_without_a_manifest_hash_is_not_satisfied(self):
        path = self.record(inputs={})
        status = analyse_redocking.validated_stereo_verification(
            path, self.manifest)
        self.assertFalse(status["satisfied"])
        self.assertIn("no ligand manifest hash", status["reason"])

    def test_a_record_cannot_be_tied_without_a_ligand_manifest(self):
        status = analyse_redocking.validated_stereo_verification(
            self.record(), None)
        self.assertFalse(status["satisfied"])
        self.assertIn("cannot be tied", status["reason"])

    def test_an_arbitrary_json_file_is_refused(self):
        path = self.root / "not_a_record.json"
        path.write_text(json.dumps({"result": "passed"}))
        status = analyse_redocking.validated_stereo_verification(
            path, self.manifest)
        self.assertFalse(status["satisfied"])
        self.assertIn("not a usable verification record", status["reason"])

    def test_centres_requiring_investigation_are_carried_but_do_not_block(self):
        path = self.record(checks={"stereochemistry_before_hydrogens": {
            "fatal_mismatches": [], "requires_investigation": ["PA"]}})
        status = analyse_redocking.validated_stereo_verification(
            path, self.manifest)
        self.assertTrue(status["satisfied"], status["reason"])
        self.assertEqual(status["centres_requiring_investigation"], ["PA"])


# Inline fixtures, so the guard tests depend on no file outside the
# repository. The earlier version required project_private/ccd_downloads/,
# which is git-excluded and exists only on the development Mac, so these
# tests silently skipped on the machine that matters.
GUARD_CCD = """data_LIG
_chem_comp.id LIG
_chem_comp.name "GUARD TEST ACID"
_chem_comp.formula "C2 H4 O3"
_chem_comp.pdbx_formal_charge 0
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_stereo_config
LIG C1 C 0 N N
LIG C2 C 0 N N
LIG O1 O 0 N N
LIG O2 O 0 N N
LIG O3 O 0 N N
LIG H1 H 0 N N
LIG HO2 H 0 N N
LIG HO3 H 0 N N
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
LIG C1 C2 SING N N
LIG C2 O1 DOUB N N
LIG C2 O2 SING N N
LIG C1 O3 SING N N
LIG C1 H1 SING N N
LIG O2 HO2 SING N N
LIG O3 HO3 SING N N
#
"""

GUARD_CIF = """data_guard
_entry.id GRD
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N N  . GLY A 1 1 ? 0.000 0.000 0.000 1.00 10.00 1 A 1
ATOM   2 C CA . GLY A 1 1 ? 1.000 0.000 0.000 1.00 10.00 1 A 1
HETATM 3 C C1 . LIG B 2 . ? 0.000 0.000 0.000 1.00 10.00 601 A 1
HETATM 4 C C2 . LIG B 2 . ? 1.500 0.000 0.000 1.00 10.00 601 A 1
HETATM 5 O O1 . LIG B 2 . ? 2.200 1.000 0.000 1.00 10.00 601 A 1
HETATM 6 O O2 . LIG B 2 . ? 2.200 -1.000 0.000 1.00 10.00 601 A 1
HETATM 7 O O3 . LIG B 2 . ? -0.700 1.200 0.000 1.00 10.00 601 A 1
"""


class InputGuardTests(unittest.TestCase):
    """The input-identity guards, which run before RDKit is required.

    These execute the real CLI end to end and stop at the missing-RDKit
    boundary, so they test the gating without touching chemistry. Every input
    is written here, so they need nothing outside the repository and run on
    any machine with Gemmi.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import gemmi  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("needs Gemmi for the dictionary reader")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.dictionary = cls.root / "LIG.cif"
        cls.dictionary.write_text(GUARD_CCD)
        cls.structure = cls.root / "guard.cif"
        cls.structure.write_text(GUARD_CIF)
        cls.run_script(ROOT / "python" / "ligand_reference.py",
                       cls.structure, "LIG", cls.root / "lig",
                       "--ligand-canonical-chain", "B",
                       "--component-dictionary", cls.dictionary)
        for prefix, atoms, reason in (("lig_a", "O2,O3", "two acids"),
                                      ("lig_b", "O2", "one acid")):
            cls.run_script(ROOT / "python" / "prepare_ligand.py", "build",
                           cls.root / "lig.reference.json", cls.root / prefix,
                           "--component-dictionary", cls.dictionary,
                           "--ph", "7.4", "--deprotonate", atoms,
                           "--protonation-justification", reason)
        cls.run_script(ROOT / "python" / "prepare_ligand.py", "stereo-plan",
                       cls.root / "lig_a.ligand.json",
                       "--out", cls.root / "lig_a.stereo_plan.json",
                       "--state", "other=O2")
        cls.run_script(ROOT / "python" / "prepare_ligand.py", "stereo-plan",
                       cls.root / "lig_b.ligand.json",
                       "--out", cls.root / "lig_b.stereo_plan.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def run_script(script, *args, expect_success=True):
        import subprocess
        finished = subprocess.run(
            [sys.executable, str(script), *[str(value) for value in args]],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "python")})
        if expect_success and finished.returncode != 0:
            raise AssertionError(finished.stdout + finished.stderr)
        return finished

    def verify(self, manifest, plan, state, prefix):
        return self.run_script(ROOT / "python" / "verify_ligand_rdkit.py",
                               self.root / manifest, self.root / plan, state,
                               "--out-prefix", self.root / prefix,
                               expect_success=False)

    def test_a_plan_from_another_manifest_is_refused(self):
        finished = self.verify("lig_a.ligand.json", "lig_b.stereo_plan.json",
                               "modelled", "g1")
        self.assertIn("built from a different revision", finished.stderr)

    def test_a_state_whose_charge_contradicts_the_manifest_is_refused(self):
        finished = self.verify("lig_a.ligand.json", "lig_a.stereo_plan.json",
                               "other", "g2")
        self.assertIn("do not describe the same protonation", finished.stderr)

    def test_an_unknown_state_is_refused_and_names_the_available_ones(self):
        finished = self.verify("lig_a.ligand.json", "lig_a.stereo_plan.json",
                               "nonexistent", "g3")
        self.assertIn("has no state 'nonexistent'", finished.stderr)
        self.assertIn("modelled", finished.stderr)

    def test_an_edited_molfile_is_refused(self):
        molfile = self.root / "lig_a_ligand.mol"
        original = molfile.read_bytes()
        molfile.write_bytes(original + b"\n")
        try:
            finished = self.verify("lig_a.ligand.json",
                                   "lig_a.stereo_plan.json", "modelled", "g4")
            self.assertIn("has changed since the manifest was written",
                          finished.stderr)
        finally:
            molfile.write_bytes(original)

    def test_the_guards_pass_and_the_run_proceeds_to_the_toolkit(self):
        """With every input consistent, the guards are not what stops the run.

        Without RDKit the run stops at the toolkit requirement and writes a
        failed record with no output. With RDKit it proceeds into the
        chemistry, which this suite does not assert anything about.
        """
        finished = self.verify("lig_a.ligand.json", "lig_a.stereo_plan.json",
                               "modelled", "g5")
        record = json.loads((self.root / "g5.verification.json").read_text())
        for guard in ("built from a different revision",
                      "do not describe the same protonation",
                      "has no state", "has changed since the manifest"):
            self.assertNotIn(guard, finished.stderr)
        if not RDKIT_AVAILABLE:
            self.assertIn("RDKit is not available", finished.stderr)
            self.assertEqual(record["result"], "failed")
            self.assertIsNone(record["outputs"]["hydrogenated_molfile"])


try:
    import rdkit  # noqa: F401
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@unittest.skipUnless(RDKIT_AVAILABLE, "REQUIRES RDKIT: not run on the Mac")
class RdkitRepresentationTests(unittest.TestCase):
    """REQUIRES RDKIT. The aromaticity regression the first Epicac run hit.

    These tests were written on a machine without RDKit and have therefore
    **never been executed** at the time of writing. They run on Epicac and
    should be read as the first execution of this comparison, not as an
    already-passing guarantee.

    The fixture is a uracil-like ring written in Kekule form, which is the
    representation the Chemical Component Dictionary uses for UMA's uracil and
    the one that RDKit aromatises on sanitisation.
    """

    def setUp(self):
        try:
            import chemical_component  # noqa: F401
        except ImportError:
            self.skipTest("needs Gemmi for the dictionary reader")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    @staticmethod
    def uracil_component(bonds=None):
        """A pyrimidinedione ring in Kekule form, as a dictionary component."""
        import chemical_component
        import chemical_graph
        names = ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"]
        elements = {"N1": "N", "C2": "C", "N3": "N", "C4": "C", "C5": "C",
                    "C6": "C", "O2": "O", "O4": "O"}
        default = [("N1", "C2", "SING"), ("C2", "N3", "SING"),
                   ("N3", "C4", "SING"), ("C4", "C5", "SING"),
                   ("C5", "C6", "DOUB"), ("C6", "N1", "SING"),
                   ("C2", "O2", "DOUB"), ("C4", "O4", "DOUB")]
        graph = chemical_graph.graph_from_bond_names(
            names, [elements[name] for name in names], bonds or default)
        return chemical_component.ChemicalComponent(
            identifier="URA", name="test ring", formula="C4 H4 N2 O2",
            dictionary_formal_charge=0, graph=graph, elements=elements,
            hydrogens_on={"N1": ("HN1",), "N3": ("HN3",)},
            stereocentres={}, aromatic_atoms=(), n_hydrogens=4,
            source_path="in-memory fixture")

    @staticmethod
    def ring_positions():
        """A planar hexagon with the two carbonyl oxygens placed outward."""
        import math
        positions = {}
        for number, name in enumerate(["N1", "C2", "N3", "C4", "C5", "C6"]):
            angle = math.radians(60 * number)
            positions[name] = (1.39 * math.cos(angle), 1.39 * math.sin(angle),
                               0.0)
        for name, source in (("O2", "C2"), ("O4", "C4")):
            x, y, _ = positions[source]
            length = math.hypot(x, y)
            positions[name] = (x * (length + 1.23) / length,
                               y * (length + 1.23) / length, 0.0)
        return positions

    def test_a_kekule_ring_agrees_with_itself_after_identical_sanitisation(self):
        """The regression: Kekule input must not disagree with its own source.

        RDKit may aromatise the ring. As long as both sides are built from the
        same connection table and sanitised the same way, the comparison must
        report agreement, because the only difference was representational.
        """
        component = self.uracil_component()
        positions = self.ring_positions()
        expected_mol, expected_snapshot, expected_index = \
            verify.expected_molecule_from_dictionary(component, {}, positions)
        expected_names = {position: name
                          for name, position in expected_index.items()}
        perceived = [(expected_names[first], expected_names[second], order)
                     for first, second, order in expected_snapshot.bonds]
        # The loaded molecule here is the same construction, standing in for
        # MolFromMolFile on the Kekule molfile: both go through SanitizeMol.
        mapping = verify.correspondence_from_coordinates(
            expected_snapshot, positions)
        report = verify.validate_connectivity(mapping, expected_snapshot,
                                              perceived)
        self.assertTrue(report["agrees"], report["bond_order_differences"])

    def test_the_old_comparison_would_have_disagreed(self):
        """Confirms the fixture reproduces the failure mode it is testing.

        Comparing the perceived molecule against the raw dictionary orders is
        what the first Epicac run did. If RDKit aromatises this ring, that
        comparison disagrees; if it does not, this test is skipped rather than
        asserting something the toolkit does not do.
        """
        component = self.uracil_component()
        positions = self.ring_positions()
        _, expected_snapshot, expected_index = \
            verify.expected_molecule_from_dictionary(component, {}, positions)
        aromatic = [order for _, _, order in expected_snapshot.bonds
                    if order.upper() == "AROMATIC"]
        if not aromatic:
            self.skipTest("this RDKit build does not aromatise the fixture "
                          "ring, so the representation mismatch cannot be "
                          "reproduced with it")
        mapping = verify.correspondence_from_coordinates(
            expected_snapshot, positions)
        raw_dictionary = [
            (component.graph.names[first], component.graph.names[second], order)
            for pair, order in component.graph.bonds.items()
            for first, second in (tuple(pair),)]
        report = verify.validate_connectivity(mapping, expected_snapshot,
                                              raw_dictionary)
        self.assertFalse(report["agrees"])
        self.assertTrue(report["bond_order_differences"])

    def test_a_genuinely_changed_bond_is_still_rejected(self):
        """The negative test: identical perception must not hide real changes.

        The expected molecule keeps the dictionary's C5=C6; the molecule under
        test is built with that bond reduced to a single bond. Both are
        sanitised identically, so if the comparison still reports a
        difference, it is reporting chemistry rather than representation.
        """
        positions = self.ring_positions()
        reference_component = self.uracil_component()
        changed_component = self.uracil_component(bonds=[
            ("N1", "C2", "SING"), ("C2", "N3", "SING"), ("N3", "C4", "SING"),
            ("C4", "C5", "SING"), ("C5", "C6", "SING"), ("C6", "N1", "SING"),
            ("C2", "O2", "DOUB"), ("C4", "O4", "DOUB")])

        _, reference_snapshot, reference_index = \
            verify.expected_molecule_from_dictionary(
                reference_component, {}, positions)
        reference_names = {position: name
                           for name, position in reference_index.items()}
        reference_bonds = [
            (reference_names[first], reference_names[second], order)
            for first, second, order in reference_snapshot.bonds]

        _, changed_snapshot, _ = verify.expected_molecule_from_dictionary(
            changed_component, {}, positions)
        mapping = verify.correspondence_from_coordinates(changed_snapshot,
                                                         positions)
        report = verify.validate_connectivity(mapping, changed_snapshot,
                                              reference_bonds)
        self.assertFalse(report["agrees"],
                         "a changed C5-C6 bond order must still be reported")
        self.assertTrue(report["bond_order_differences"])

    def test_the_modelled_charges_reach_the_expected_molecule(self):
        """Protonation state and formal charges must be preserved, not dropped."""
        from rdkit import Chem
        component = self.uracil_component()
        positions = self.ring_positions()
        molecule, snapshot_built, _ = verify.expected_molecule_from_dictionary(
            component, {"O2": -1}, positions)
        self.assertEqual(Chem.GetFormalCharge(molecule), -1)
        self.assertEqual(snapshot_built.formal_charge, -1)


class RdkitAbsenceTests(unittest.TestCase):
    """The diagnostic given when RDKit is missing.

    This tests the absence path, which can only be observed where RDKit is
    absent, so it **skips wherever RDKit is installed** — including on Epicac.
    That skip is the expected outcome there and is not a gap: the presence
    path is covered by `RdkitRepresentationTests`, which skip here instead.
    One of the two always skips, whichever machine the suite runs on.
    """

    def test_a_missing_rdkit_gives_a_useful_diagnostic(self):
        if RDKIT_AVAILABLE:
            self.skipTest("RDKit is installed, so its absence cannot be "
                          "observed here; this is the expected skip on a "
                          "machine that has RDKit")
        with self.assertRaises(verify.VerificationError) as caught:
            verify.rdkit_version()
        message = str(caught.exception)
        self.assertIn("must run in an environment that provides it", message)
        self.assertIn("boltz", message)


if __name__ == "__main__":
    unittest.main()
