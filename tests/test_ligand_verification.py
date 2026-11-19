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


@unittest.skipUnless((ROOT / "data" / "1P3D.cif").is_file()
                     and (ROOT / "project_private" / "ccd_downloads"
                          / "UMA.cif").is_file(),
                     "needs the structure and the UMA dictionary entry")
class InputGuardTests(unittest.TestCase):
    """The input-identity guards, which run before RDKit is required.

    These execute the real CLI end to end against real files and stop at the
    missing-RDKit boundary, so they test the gating without touching
    chemistry. They are skipped where the inputs are not available.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import gemmi  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("needs Gemmi")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.dictionary = (ROOT / "project_private" / "ccd_downloads" / "UMA.cif")
        cls.run_script(ROOT / "python" / "ligand_reference.py",
                ROOT / "data" / "1P3D.cif", "UMA", cls.root / "uma",
                "--ligand-canonical-chain", "E",
                "--component-dictionary", cls.dictionary)
        for prefix, atoms, reason in (
                ("uma", "O2A,O2B,O20", "Provisional -3."),
                ("uma_minus2", "O2A,O20", "Sensitivity -2.")):
            cls.run_script(ROOT / "python" / "prepare_ligand.py", "build",
                    cls.root / "uma.reference.json", cls.root / prefix,
                    "--component-dictionary", cls.dictionary,
                    "--ph", "7.4", "--deprotonate", atoms,
                    "--protonation-justification", reason)
        cls.run_script(ROOT / "python" / "prepare_ligand.py", "stereo-plan",
                cls.root / "uma.ligand.json",
                "--out", cls.root / "uma.stereo_plan.json",
                "--state", "sensitivity_minus2=O2A,O20")
        cls.run_script(ROOT / "python" / "prepare_ligand.py", "stereo-plan",
                cls.root / "uma_minus2.ligand.json",
                "--out", cls.root / "uma_minus2.stereo_plan.json")

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
        finished = self.verify("uma.ligand.json",
                               "uma_minus2.stereo_plan.json", "modelled", "g1")
        self.assertIn("built from a different revision", finished.stderr)

    def test_a_state_whose_charge_contradicts_the_manifest_is_refused(self):
        finished = self.verify("uma.ligand.json", "uma.stereo_plan.json",
                               "sensitivity_minus2", "g2")
        self.assertIn("do not describe the same protonation", finished.stderr)

    def test_an_unknown_state_is_refused_and_names_the_available_ones(self):
        finished = self.verify("uma.ligand.json", "uma.stereo_plan.json",
                               "nonexistent", "g3")
        self.assertIn("has no state 'nonexistent'", finished.stderr)
        self.assertIn("modelled", finished.stderr)

    def test_an_edited_molfile_is_refused(self):
        molfile = self.root / "uma_ligand.mol"
        original = molfile.read_bytes()
        molfile.write_bytes(original + b"\n")
        try:
            finished = self.verify("uma.ligand.json", "uma.stereo_plan.json",
                                   "modelled", "g4")
            self.assertIn("has changed since the manifest was written",
                          finished.stderr)
        finally:
            molfile.write_bytes(original)

    def test_a_correct_pair_reaches_the_toolkit_requirement(self):
        """The guards pass, and the run stops only for the missing toolkit."""
        try:
            import rdkit  # noqa: F401
            self.skipTest("RDKit is installed; this boundary test needs it absent")
        except ImportError:
            pass
        finished = self.verify("uma.ligand.json", "uma.stereo_plan.json",
                               "modelled", "g5")
        self.assertIn("RDKit is not available", finished.stderr)
        record = json.loads(
            (self.root / "g5.verification.json").read_text())
        self.assertEqual(record["result"], "failed")
        self.assertIsNone(record["outputs"]["hydrogenated_molfile"])
        self.assertFalse((self.root / "g5_withH.mol").exists())


class RdkitAdapterTests(unittest.TestCase):
    """The RDKit path is not exercised; this records that fact explicitly."""

    def test_rdkit_is_absent_here_and_reports_so_usefully(self):
        try:
            import rdkit  # noqa: F401
        except ImportError:
            with self.assertRaises(verify.VerificationError) as caught:
                verify.rdkit_version()
            self.assertIn("must run in an environment that provides it",
                          str(caught.exception))
            return
        self.skipTest("RDKit is installed; the adapter is exercised on Epicac, "
                      "and this suite still does not verify its chemistry")


if __name__ == "__main__":
    unittest.main()
