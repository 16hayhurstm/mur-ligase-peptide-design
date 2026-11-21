"""Tests for manifest identity across the stages, and the RMSD comparison.

Scope, and what these tests do not establish
--------------------------------------------
Software correctness only. Two things are checked here:

- **manifest identity**: that `prepare_ligand.py check` no longer modifies the
  ligand manifest, that a manifest an earlier version *did* modify is still
  recognised as the verified one, and that a genuine change of ligand,
  protonation state, input coordinates or prepared PDBQT is still rejected.
  The full build to downstream-verification sequence is exercised through the
  real CLI against a hand-written PDBQT fixture.
- **the independent RMSD comparison**: that the synthetic cases are generated
  deterministically with the equivalences the modelled protonation state
  justifies, that this project's values meet the geometric expectations, and
  that the analysis stage validates the comparison record's contents rather
  than its existence.

The RDKit half of the comparison is **not** exercised here: RDKit is not a
dependency of these tests, and a stub toolkit is injected instead. Nothing
here establishes that RDKit agrees with this project, which is exactly what
the Epicac run is for. Nothing here establishes anything biological.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import chemical_graph as graph_module  # noqa: E402
import independent_rmsd  # noqa: E402
import receptor_provenance as provenance  # noqa: E402
import analyse_redocking  # noqa: E402

PREPARE_LIGAND = ROOT / "python" / "prepare_ligand.py"

try:
    import gemmi  # noqa: F401
    import chemical_component  # noqa: F401
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False


def write(path: Path, record: dict) -> str:
    """Write a record with the project's writer and return its digest."""
    provenance.write_manifest(path, record, overwrite=True)
    return provenance.file_digest_of(path)


def ligand_manifest(charge: int = -1, component: str = "LIG",
                    with_sentinel: bool = True) -> dict:
    """A minimal ligand manifest of the shape `build` writes."""
    record = {
        "schema": "mur-ligase-ligand/1",
        "component": component,
        "protonation_state": {"modelled_ph": 7.4,
                              "modelled_formal_charge": charge,
                              "deprotonated_atoms": ["O2"]},
        "reference": {"path": "reference.json"},
        "dictionary": {"path": "LIG.cif"},
    }
    if with_sentinel:
        record["preparation"] = {"performed": False}
    return record


class ManifestIdentityTests(unittest.TestCase):
    """The agreement rule, in isolation from the CLI."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.path = self.root / "lig.ligand.json"
        self.recorded = write(self.path, ligand_manifest())

    def test_an_unmodified_manifest_agrees_without_restoration(self):
        verdict = provenance.manifest_hash_agreement(self.path, self.recorded)
        self.assertTrue(verdict["agrees"])
        self.assertFalse(verdict["legacy_compatibility_used"])
        self.assertEqual(verdict["basis"], "the manifest as it stands")

    def test_a_legacy_modified_manifest_agrees_and_says_so(self):
        """The historical in-place rewrite must remain usable, and be visible.

        An earlier `check` replaced the build-stage sentinel with its results,
        changing the digest the verification had already recorded. Restoring
        that one block reproduces the digest exactly, so the manifest is the
        verified one; the verdict has to record that the compatibility path
        was taken rather than look like a clean match.
        """
        manifest = json.loads(self.path.read_text())
        manifest["preparation"] = {"performed": True, "torsdof": 3,
                                   "checked_utc": "2026-11-19T10:00:00Z"}
        write(self.path, manifest)
        verdict = provenance.manifest_hash_agreement(self.path, self.recorded)
        self.assertTrue(verdict["agrees"])
        self.assertTrue(verdict["legacy_compatibility_used"])
        self.assertIn("restored", verdict["basis"])

    def test_a_manifest_that_never_had_the_sentinel_agrees_when_stripped(self):
        """The other historical shape: the key absent before `check` ran."""
        path = self.root / "older.ligand.json"
        recorded = write(path, ligand_manifest(with_sentinel=False))
        manifest = json.loads(path.read_text())
        manifest["preparation"] = {"performed": True, "torsdof": 3}
        write(path, manifest)
        verdict = provenance.manifest_hash_agreement(path, recorded)
        self.assertTrue(verdict["agrees"])
        self.assertTrue(verdict["legacy_compatibility_used"])
        self.assertIn("absent", verdict["basis"])

    def test_a_changed_protonation_state_is_rejected(self):
        """The restoration must not launder a different modelled state."""
        manifest = ligand_manifest(charge=-2)
        manifest["preparation"] = {"performed": True, "torsdof": 3}
        write(self.path, manifest)
        verdict = provenance.manifest_hash_agreement(self.path, self.recorded)
        self.assertFalse(verdict["agrees"])
        self.assertIn("verified content has itself changed", verdict["reason"])

    def test_a_changed_ligand_is_rejected(self):
        manifest = ligand_manifest(component="ANP")
        manifest["preparation"] = {"performed": True}
        write(self.path, manifest)
        self.assertFalse(
            provenance.manifest_hash_agreement(self.path, self.recorded)["agrees"])

    def test_a_silent_edit_under_a_preparation_block_is_rejected(self):
        """Stripping the block must not hide a change made alongside it."""
        manifest = json.loads(self.path.read_text())
        manifest["preparation"] = {"performed": True}
        manifest["protonation_state"]["deprotonated_atoms"] = ["O1"]
        write(self.path, manifest)
        self.assertFalse(
            provenance.manifest_hash_agreement(self.path, self.recorded)["agrees"])

    def test_a_changed_manifest_without_the_block_is_rejected(self):
        """No legacy stage to account for means no restoration is attempted."""
        manifest = ligand_manifest()
        manifest.pop("preparation")
        manifest["component"] = "ANP"
        write(self.path, manifest)
        verdict = provenance.manifest_hash_agreement(self.path, self.recorded)
        self.assertFalse(verdict["agrees"])
        self.assertIn("no top-level", verdict["reason"])

    def test_a_missing_recorded_hash_is_not_treated_as_agreement(self):
        verdict = provenance.manifest_hash_agreement(self.path, None)
        self.assertFalse(verdict["agrees"])
        self.assertIn("no ligand manifest hash", verdict["reason"])

    def test_an_unreadable_manifest_is_reported_not_passed(self):
        verdict = provenance.manifest_hash_agreement(
            self.root / "absent.json", self.recorded)
        self.assertFalse(verdict["agrees"])


class StereoPrerequisiteTests(unittest.TestCase):
    """The analysis gate that the in-place rewrite used to break."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.manifest = self.root / "lig.ligand.json"
        self.recorded = write(self.manifest, ligand_manifest())
        self.verification = self.root / "lig.verification.json"
        write(self.verification, {
            "schema": "mur-ligase-ligand-verification/1",
            "result": "passed", "state": "minus1",
            "tools": {"rdkit": "2024.03.5"},
            "inputs": {"ligand_manifest_sha256": self.recorded},
            "checks": {"stereochemistry_before_hydrogens":
                       {"fatal_mismatches": [], "requires_investigation": []}},
        })

    def test_the_prerequisite_survives_a_legacy_in_place_rewrite(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["preparation"] = {"performed": True, "torsdof": 3}
        write(self.manifest, manifest)
        status = analyse_redocking.validated_stereo_verification(
            self.verification, self.manifest)
        self.assertTrue(status["satisfied"], status["reason"])
        self.assertTrue(
            status["ligand_manifest_identity"]["legacy_compatibility_used"])

    def test_the_prerequisite_still_rejects_a_different_ligand(self):
        write(self.manifest, ligand_manifest(component="ANP"))
        status = analyse_redocking.validated_stereo_verification(
            self.verification, self.manifest)
        self.assertFalse(status["satisfied"])
        self.assertIn("different ligand manifest", status["reason"])

    def test_a_failed_verification_never_satisfies_the_prerequisite(self):
        write(self.verification, {
            "schema": "mur-ligase-ligand-verification/1",
            "result": "failed", "failure_reason": "stereo mismatch",
            "inputs": {"ligand_manifest_sha256": self.recorded},
            "checks": {},
        })
        status = analyse_redocking.validated_stereo_verification(
            self.verification, self.manifest)
        self.assertFalse(status["satisfied"])


@unittest.skipUnless(GEMMI_AVAILABLE, "Gemmi is required")
class PreparationRecordTests(unittest.TestCase):
    """The complete build to downstream-verification sequence, through the CLI.

    The PDBQT is hand-written, so this exercises the verification logic and
    says nothing about a real preparation tool.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.ccd = self.root / "LIG.cif"
        self.ccd.write_text(COMPONENT_CIF)
        self.reference = self.root / "lig.reference.json"
        write(self.reference, {
            "schema": "mur-ligase-ligand-reference/1",
            "component": "LIG",
            "source_structure": {"path": "synthetic.cif"},
            "selection": {"canonical_chain": "B"},
            "outputs": {"reference_pdb": str(self.root / "lig.reference.pdb")},
            "atom_mapping": {
                name: {"element": element,
                       "coordinates_angstrom": list(position)}
                for name, element, position in REFERENCE_ATOMS},
        })

    def run_script(self, script, *args, success=True):
        finished = subprocess.run(
            [sys.executable, str(script), *[str(a) for a in args]],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(finished.returncode == 0, success,
                         finished.stdout + finished.stderr)
        return finished

    def build(self):
        self.run_script(PREPARE_LIGAND, "build", self.reference,
                        self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "pKa below pH 7.4")
        return self.root / "built.ligand.json"

    @staticmethod
    def pdbqt_text(atoms, charge_total, torsdof=2):
        """A minimal PDBQT whose heavy-atom names match the reference."""
        share = charge_total / len(atoms)
        lines = ["ROOT"]
        for serial, (name, position) in enumerate(atoms.items(), start=1):
            element = "OA" if name.startswith("O") else "C"
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} LIG A   1    "
                f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                f"  1.00  0.00    {share:+6.3f} {element:<2s}")
        lines.append("ENDROOT")
        lines.append(f"TORSDOF {torsdof}")
        return "\n".join(lines) + "\n"

    def reference_coordinates(self):
        return {name: position for name, _, position in REFERENCE_ATOMS}

    def test_check_writes_a_separate_record_and_leaves_the_manifest_alone(self):
        manifest_path = self.build()
        before = provenance.file_digest_of(manifest_path)
        pdbqt = self.root / "lig.pdbqt"
        pdbqt.write_text(self.pdbqt_text(self.reference_coordinates(), -1.0))
        self.run_script(PREPARE_LIGAND, "check", manifest_path, pdbqt,
                        "--out", self.root / "lig.preparation.json")
        self.assertEqual(provenance.file_digest_of(manifest_path), before,
                         "the check stage must not modify the manifest")
        record = json.loads((self.root / "lig.preparation.json").read_text())
        self.assertEqual(record["schema"], "mur-ligase-ligand-preparation/1")
        self.assertEqual(record["result"], "passed")
        self.assertEqual(record["inputs"]["ligand_manifest_sha256"], before)
        self.assertEqual(record["inputs"]["prepared_pdbqt_sha256"],
                         provenance.file_digest_of(pdbqt))

    def test_the_preparation_record_is_tied_to_the_prepared_pdbqt(self):
        """Editing the PDBQT after the check must not match the record."""
        manifest_path = self.build()
        pdbqt = self.root / "lig.pdbqt"
        pdbqt.write_text(self.pdbqt_text(self.reference_coordinates(), -1.0))
        self.run_script(PREPARE_LIGAND, "check", manifest_path, pdbqt,
                        "--out", self.root / "lig.preparation.json")
        recorded = json.loads(
            (self.root / "lig.preparation.json").read_text()
        )["inputs"]["prepared_pdbqt_sha256"]
        pdbqt.write_text(pdbqt.read_text() + "REMARK edited after the check\n")
        self.assertNotEqual(recorded, provenance.file_digest_of(pdbqt))

    def test_moved_input_coordinates_still_fail_the_check(self):
        manifest_path = self.build()
        moved = {name: (x + 1.0, y, z)
                 for name, (x, y, z) in self.reference_coordinates().items()}
        pdbqt = self.root / "moved.pdbqt"
        pdbqt.write_text(self.pdbqt_text(moved, -1.0))
        finished = self.run_script(
            PREPARE_LIGAND, "check", manifest_path, pdbqt,
            "--out", self.root / "moved.preparation.json", success=False)
        self.assertIn("must not move the heavy atoms", finished.stderr)
        record = json.loads(
            (self.root / "moved.preparation.json").read_text())
        self.assertEqual(record["result"], "failed",
                         "a failed check must record its failure, not pass")

    def test_a_contradicted_protonation_state_still_fails_the_check(self):
        manifest_path = self.build()
        pdbqt = self.root / "neutral.pdbqt"
        pdbqt.write_text(self.pdbqt_text(self.reference_coordinates(), 0.0))
        finished = self.run_script(
            PREPARE_LIGAND, "check", manifest_path, pdbqt,
            "--out", self.root / "neutral.preparation.json", success=False)
        self.assertIn("did not apply the protonation state", finished.stderr)

    def test_the_sequence_leaves_the_stereo_prerequisite_satisfied(self):
        """build, verify, check, then the downstream gate: end to end.

        This is the sequence the in-place rewrite broke. The verification
        record is hand-written here, standing in for verify_ligand_rdkit.py,
        which needs RDKit.
        """
        manifest_path = self.build()
        verified_digest = provenance.file_digest_of(manifest_path)
        verification = self.root / "lig.verification.json"
        write(verification, {
            "schema": "mur-ligase-ligand-verification/1",
            "result": "passed", "state": "minus1",
            "tools": {"rdkit": "2024.03.5"},
            "inputs": {"ligand_manifest_sha256": verified_digest},
            "checks": {"stereochemistry_before_hydrogens":
                       {"fatal_mismatches": [], "requires_investigation": []}},
        })
        pdbqt = self.root / "lig.pdbqt"
        pdbqt.write_text(self.pdbqt_text(self.reference_coordinates(), -1.0))
        self.run_script(PREPARE_LIGAND, "check", manifest_path, pdbqt,
                        "--out", self.root / "lig.preparation.json")

        status = analyse_redocking.validated_stereo_verification(
            verification, manifest_path)
        self.assertTrue(status["satisfied"], status["reason"])
        self.assertFalse(
            status["ligand_manifest_identity"]["legacy_compatibility_used"],
            "a current run needs no compatibility path")


# ---------------------------------------------------------------------------
# The independent RMSD comparison
# ---------------------------------------------------------------------------

def phosphate_graph():
    """A fixture with two centres: one fully deprotonated, one not.

    PA carries three terminal oxygens, all deprotonated, so they are
    equivalent. PB carries two, one of which keeps its proton, so they are
    not. The second centre is what the negative case is built from.
    """
    names = ["PA", "O1A", "O2A", "O3A", "PB", "O1B", "O2B", "C1"]
    elements = ["P", "O", "O", "O", "P", "O", "O", "C"]
    bonds = [("PA", "O1A", "DOUB"), ("PA", "O2A", "SING"),
             ("PA", "O3A", "SING"), ("PA", "C1", "SING"),
             ("C1", "PB", "SING"), ("PB", "O1B", "DOUB"),
             ("PB", "O2B", "SING")]
    return graph_module.graph_from_bond_names(names, elements, bonds)


PHOSPHATE_COORDINATES = {
    "PA": (0.0, 0.0, 0.0),
    "O1A": (1.5, 0.0, 0.0),
    "O2A": (-0.5, 1.4, 0.0),
    "O3A": (-0.5, -1.4, 0.0),
    "C1": (0.0, 0.0, 1.8),
    "PB": (0.0, 1.6, 2.9),
    "O1B": (1.4, 1.6, 3.6),
    "O2B": (-1.4, 1.6, 3.6),
}


class SyntheticCaseTests(unittest.TestCase):
    """Case generation, and this project's side of the comparison."""

    def setUp(self):
        self.base = phosphate_graph()
        # O2B keeps its proton at this state, so PB's oxygens are inequivalent.
        self.protonated = ["O2B"]
        self.groups = independent_rmsd.terminal_oxygen_groups(
            self.base, self.protonated)
        self.plan = independent_rmsd.negative_case_plan(self.base, self.groups)
        self.graph, _ = graph_module.normalise_resonance(
            self.base, hydrogen_bearing=self.protonated)
        self.permutations = graph_module.automorphisms(self.graph)
        self.cases = independent_rmsd.synthetic_cases(
            PHOSPHATE_COORDINATES, self.groups, self.plan)

    def test_the_groups_record_which_oxygens_the_state_makes_equivalent(self):
        by_centre = {group["centre"]: group for group in self.groups}
        self.assertTrue(by_centre["PA"]["all_deprotonated"])
        self.assertFalse(by_centre["PB"]["all_deprotonated"])
        self.assertEqual(by_centre["PB"]["protonated"], ["O2B"])

    def test_the_case_set_is_deterministic(self):
        again = independent_rmsd.synthetic_cases(PHOSPHATE_COORDINATES,
                                                 self.groups, self.plan)
        self.assertEqual([case.label for case in self.cases],
                         [case.label for case in again])
        self.assertEqual([case.coordinates for case in self.cases],
                         [case.coordinates for case in again])

    def test_the_case_set_covers_the_required_kinds(self):
        labels = [case.label for case in self.cases]
        self.assertEqual(labels[:3],
                         ["unchanged", "translation_small", "translation_large"])
        self.assertIn("swap_PA", labels)
        self.assertIn("negative_swap_PB", labels)
        self.assertNotIn("swap_PB", labels,
                         "PB's oxygens are inequivalent at this state")

    def test_a_rigid_translation_moves_every_atom_by_the_same_distance(self):
        """The geometric expectation the comparison leans on."""
        for case in self.cases:
            if case.kind != "rigid_translation":
                continue
            distances = {
                round(sum((p - q) ** 2 for p, q in zip(
                    case.coordinates[name], PHOSPHATE_COORDINATES[name])) ** 0.5, 9)
                for name in PHOSPHATE_COORDINATES}
            self.assertEqual(len(distances), 1)
            self.assertAlmostEqual(distances.pop(), case.expected_angstrom,
                                   places=9)

    def test_our_rmsd_meets_the_geometric_expectation_on_every_case(self):
        for case in self.cases:
            if case.expected_angstrom is None:
                continue
            value, _ = independent_rmsd.our_rmsd(
                self.graph, PHOSPHATE_COORDINATES, case.coordinates,
                self.permutations)
            self.assertAlmostEqual(value, case.expected_angstrom, places=9,
                                   msg=case.label)

    def test_an_equivalent_oxygen_swap_is_zero_and_names_the_exchange(self):
        case = next(c for c in self.cases if c.label == "swap_PA")
        value, exchanged = independent_rmsd.our_rmsd(
            self.graph, PHOSPHATE_COORDINATES, case.coordinates,
            self.permutations)
        self.assertAlmostEqual(value, 0.0, places=9)
        self.assertEqual(set(exchanged), set(case.atoms_exchanged))

    def test_the_negative_case_is_not_zero(self):
        """The protonation rule: swapping O1B and O2B is a real change.

        If this came out as zero, the measure would be treating a protonated
        and an unprotonated oxygen as interchangeable, and a wrong pose could
        pass. The expected value is the separation over the square root of the
        atom count, since only two atoms moved.
        """
        case = next(c for c in self.cases
                    if c.label == "negative_swap_PB")
        value, _ = independent_rmsd.our_rmsd(
            self.graph, PHOSPHATE_COORDINATES, case.coordinates,
            self.permutations)
        gap = independent_rmsd.separation(PHOSPHATE_COORDINATES, "O1B", "O2B")
        expected = (2 * gap ** 2 / len(PHOSPHATE_COORDINATES)) ** 0.5
        self.assertGreater(value, 0.1)
        self.assertAlmostEqual(value, expected, places=9)

    def test_a_fully_deprotonated_state_has_no_protonation_negative_case(self):
        """It must be recorded as not applicable, with a reason, not demanded.

        The -3 UMA state deprotonates every terminal oxygen, so no protonated
        and unprotonated pair exists to exchange. That is a property of the
        state. The element-swap case still applies, so the state is not left
        without a negative case.
        """
        groups = independent_rmsd.terminal_oxygen_groups(self.base, [])
        plan = independent_rmsd.negative_case_plan(self.base, groups)
        self.assertFalse(plan["protonation_oxygen_swap"]["applicable"])
        self.assertIn("property of the state",
                      plan["protonation_oxygen_swap"]["reason"])
        self.assertTrue(plan["element_swap"]["applicable"])

        cases = independent_rmsd.synthetic_cases(PHOSPHATE_COORDINATES,
                                                 groups, plan)
        self.assertFalse([c for c in cases
                          if c.kind == "protonation_oxygen_swap"])
        self.assertEqual(len([c for c in cases if c.kind == "element_swap"]), 1)

    def test_the_protonation_case_is_required_where_it_applies(self):
        """At the -2-like state the pair exists, so it must be generated."""
        self.assertTrue(self.plan["protonation_oxygen_swap"]["applicable"])
        self.assertEqual(
            sorted(self.plan["protonation_oxygen_swap"]["atoms"]),
            ["O1B", "O2B"])
        self.assertIn("negative_swap_PB",
                      [case.label for case in self.cases])

    def test_the_element_swap_case_is_inequivalent_at_every_state(self):
        case = next(c for c in self.cases if c.kind == "element_swap")
        self.assertEqual(case.equivalence_class, "inequivalent")
        value, _ = independent_rmsd.our_rmsd(
            self.graph, PHOSPHATE_COORDINATES, case.coordinates,
            self.permutations)
        self.assertGreater(value, 0.1)


class ComparisonRecordTests(unittest.TestCase):
    """The record assembled from both sides, with a stub toolkit.

    The stub stands in for RDKit, which is not a dependency of these tests.
    Agreement here is therefore agreement with a stub: it exercises the record
    and its verdict, and establishes nothing about RDKit.
    """

    def setUp(self):
        self.base = phosphate_graph()
        self.protonated = ["O2B"]
        self.groups = independent_rmsd.terminal_oxygen_groups(
            self.base, self.protonated)
        self.plan = independent_rmsd.negative_case_plan(self.base, self.groups)
        self.graph, _ = graph_module.normalise_resonance(
            self.base, hydrogen_bearing=self.protonated)
        self.permutations = graph_module.automorphisms(self.graph)
        self.cases = independent_rmsd.synthetic_cases(
            PHOSPHATE_COORDINATES, self.groups, self.plan)

    def rows(self, toolkit):
        return independent_rmsd.compare(
            self.cases, PHOSPHATE_COORDINATES, self.graph, self.permutations,
            toolkit)

    def agreeing_toolkit(self):
        """A stub that returns exactly what this project returns."""
        def toolkit(case):
            value, _ = independent_rmsd.our_rmsd(
                self.graph, PHOSPHATE_COORDINATES, case.coordinates,
                self.permutations)
            return value
        return toolkit

    def record(self, rows, equivalence_agree=True):
        return independent_rmsd.assemble_record(
            state="minus1", component="LIG",
            inputs={"ligand_manifest": "lig.ligand.json",
                    "ligand_manifest_sha256": "0" * 64},
            protonation_state={"modelled_formal_charge": -1},
            tools={"rdkit": "stub"}, groups=self.groups,
            equivalence={"agree": equivalence_agree}, rows=rows,
            plan=self.plan)

    def test_an_agreeing_comparison_passes_and_labels_itself_synthetic(self):
        record = self.record(self.rows(self.agreeing_toolkit()))
        self.assertEqual(record["result"], "passed", record["failure_reasons"])
        self.assertTrue(record["checks"]["all_cases_agree"])
        self.assertTrue(record["checks"]["negative_case_distinguished"])
        self.assertTrue(record["checks"]["analytic_expectations_met"])
        self.assertIn("not docking results", record["content_class"])
        self.assertFalse(
            record["measure"]["our_permutations_supplied_to_the_toolkit"])

    def test_a_disagreeing_toolkit_fails_the_record(self):
        def toolkit(case):
            return 42.0
        record = self.record(self.rows(toolkit))
        self.assertEqual(record["result"], "failed")
        self.assertTrue(any("disagree" in reason
                            for reason in record["failure_reasons"]))

    def test_a_toolkit_that_zeroes_the_negative_case_fails_the_record(self):
        """The failure that matters: a wrongly permitted oxygen exchange."""
        def toolkit(case):
            if case.kind in independent_rmsd.NEGATIVE_KINDS:
                return 0.0
            value, _ = independent_rmsd.our_rmsd(
                self.graph, PHOSPHATE_COORDINATES, case.coordinates,
                self.permutations)
            return value
        record = self.record(self.rows(toolkit))
        self.assertEqual(record["result"], "failed")
        self.assertFalse(record["checks"]["negative_case_distinguished"])
        self.assertTrue(any("came out as zero" in reason
                            for reason in record["failure_reasons"]))

    def test_an_unevaluated_case_fails_rather_than_being_skipped(self):
        def toolkit(case):
            return None
        record = self.record(self.rows(toolkit))
        self.assertEqual(record["result"], "failed")
        self.assertTrue(any("evaluated no value" in reason
                            for reason in record["failure_reasons"]))

    def test_a_diagnostic_disagreement_does_not_invalidate_the_selection(self):
        """A non-selected setting describes a different equivalence rule.

        Its disagreement must be reported prominently and must not fail a
        selected configuration that passes every required check on its own.
        """
        def toolkit(case):
            value, _ = independent_rmsd.our_rmsd(
                self.graph, PHOSPHATE_COORDINATES, case.coordinates,
                self.permutations)
            # The diagnostic setting disagrees on the justified oxygen swap,
            # which is what the unsymmetrised setting would really do.
            diagnostic = (3.0 if case.kind == "equivalent_oxygen_swap"
                          else value)
            return {independent_rmsd.SELECTED_VARIANT: value,
                    independent_rmsd.UNSYMMETRISED_VARIANT: diagnostic}
        record = self.record(self.rows(toolkit))
        self.assertEqual(record["result"], "passed", record["failure_reasons"])
        self.assertTrue(record["checks"]["all_cases_agree"])
        self.assertGreater(record["diagnostics"]["n_disagreements"], 0)
        self.assertIn("not binding", record["diagnostics"]["status"])
        self.assertEqual(record["checks"]["diagnostic_disagreements"],
                         record["diagnostics"]["n_disagreements"])

    def test_a_diagnostic_zero_on_a_negative_case_is_reported_not_blocking(self):
        def toolkit(case):
            value, _ = independent_rmsd.our_rmsd(
                self.graph, PHOSPHATE_COORDINATES, case.coordinates,
                self.permutations)
            diagnostic = (0.0 if case.kind in independent_rmsd.NEGATIVE_KINDS
                          else value)
            return {independent_rmsd.SELECTED_VARIANT: value,
                    independent_rmsd.UNSYMMETRISED_VARIANT: diagnostic}
        record = self.record(self.rows(toolkit))
        self.assertEqual(record["result"], "passed", record["failure_reasons"])
        self.assertTrue(record["checks"]["negative_case_distinguished"])
        self.assertTrue(
            record["checks"]["diagnostic_permitted_an_inequivalent_mapping"])
        self.assertFalse(
            record["checks"]["selected_permitted_an_inequivalent_mapping"])

    def test_the_selected_configuration_is_recorded_with_its_justification(self):
        record = self.record(self.rows(self.agreeing_toolkit()))
        selected = record["measure"]["selected_configuration"]
        self.assertEqual(selected["variant"], independent_rmsd.SELECTED_VARIANT)
        self.assertTrue(selected["symmetrize_conjugated_terminal_groups"])
        self.assertIn("intent", selected["selected_on"])
        self.assertIn("only this configuration", selected["binding"])

    def test_a_toolkit_option_that_permits_the_swap_fails_the_record(self):
        """The symmetrisation risk, exercised through the variant results.

        RDKit's `symmetrizeConjugatedTerminalGroups` defaults to true, which
        can make CalcRMS exchange a conjugated terminal group's oxygens
        regardless of protonation. If it does that to the protonation case,
        the two measures are not computing the same quantity and the record
        must say so rather than average it away. A stub cannot establish what
        real RDKit does; this only checks that the record reacts correctly
        when a variant returns zero.
        """
        def toolkit(case):
            value, _ = independent_rmsd.our_rmsd(
                self.graph, PHOSPHATE_COORDINATES, case.coordinates,
                self.permutations)
            variants = {independent_rmsd.SELECTED_VARIANT: value,
                        independent_rmsd.UNSYMMETRISED_VARIANT: value}
            if case.kind == "protonation_oxygen_swap":
                variants[independent_rmsd.SELECTED_VARIANT] = 0.0
            return variants
        record = self.record(self.rows(toolkit))
        self.assertEqual(record["result"], "failed")
        self.assertTrue(
            any("permits a mapping" in reason
                for reason in record["failure_reasons"]),
            record["failure_reasons"])
        self.assertTrue(
            record["checks"]["selected_permitted_an_inequivalent_mapping"])
        self.assertTrue(any("needs a decision" in reason
                            for reason in record["failure_reasons"]))

    def test_a_not_applicable_protonation_case_does_not_fail_the_record(self):
        """The -3 state: no protonated terminal oxygen, so no such pair.

        The record must still pass on the strength of the element-swap case,
        and must carry the reason the protonation case does not apply.
        """
        groups = independent_rmsd.terminal_oxygen_groups(self.base, [])
        plan = independent_rmsd.negative_case_plan(self.base, groups)
        graph, _ = graph_module.normalise_resonance(self.base,
                                                    hydrogen_bearing=[])
        permutations = graph_module.automorphisms(graph)
        cases = independent_rmsd.synthetic_cases(PHOSPHATE_COORDINATES,
                                                 groups, plan)

        def toolkit(case):
            value, _ = independent_rmsd.our_rmsd(
                graph, PHOSPHATE_COORDINATES, case.coordinates, permutations)
            return value
        rows = independent_rmsd.compare(cases, PHOSPHATE_COORDINATES, graph,
                                        permutations, toolkit)
        record = independent_rmsd.assemble_record(
            state="minus3", component="LIG",
            inputs={"ligand_manifest_sha256": "0" * 64},
            protonation_state={"modelled_formal_charge": -3},
            tools={"rdkit": "stub"}, groups=groups,
            equivalence={"agree": True}, rows=rows, plan=plan)
        self.assertEqual(record["result"], "passed", record["failure_reasons"])
        self.assertTrue(record["checks"]["negative_case_distinguished"])
        protonation = record["negative_cases"]["protonation_oxygen_swap"]
        self.assertFalse(protonation["applicable"])
        self.assertIn("no centre has both", protonation["reason"])
        self.assertEqual(record["checks"]["negative_kinds_applicable"],
                         ["element_swap"])

    def test_disagreeing_equivalences_fail_the_record(self):
        record = self.record(self.rows(self.agreeing_toolkit()),
                             equivalence_agree=False)
        self.assertEqual(record["result"], "failed")
        self.assertFalse(
            record["checks"]["equivalences_independently_verified"])


class IndependentRmsdPrerequisiteTests(unittest.TestCase):
    """The analysis stage must validate the record, not its existence."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.manifest = self.root / "lig.ligand.json"
        self.recorded = write(self.manifest, ligand_manifest())
        self.path = self.root / "lig.independent_rmsd.json"

    def comparison_record(self, **overrides):
        record = {
            "schema": "mur-ligase-independent-rmsd/1",
            "result": "passed",
            "failure_reasons": [],
            "content_class": "synthetic software validation",
            "state": "minus1",
            "tools": {"rdkit": "2024.03.5"},
            "inputs": {"ligand_manifest_sha256": self.recorded},
            "tolerance_angstrom": 1e-6,
            "measure": {"superposition": "none"},
            "cases": [{"label": "unchanged"}],
            "checks": {"all_cases_agree": True,
                       "negative_case_distinguished": True,
                       "equivalences_independently_verified": True,
                       "max_absolute_difference_angstrom": 0.0},
        }
        record.update(overrides)
        return record

    def validate(self, record):
        write(self.path, record)
        return analyse_redocking.validated_independent_rmsd(
            self.path, self.manifest)

    def test_a_valid_record_satisfies_the_prerequisite(self):
        status = self.validate(self.comparison_record())
        self.assertTrue(status["satisfied"], status["reason"])

    def test_a_missing_record_does_not(self):
        status = analyse_redocking.validated_independent_rmsd(None,
                                                              self.manifest)
        self.assertFalse(status["satisfied"])

    def test_mere_existence_does_not_satisfy_the_prerequisite(self):
        """The defect this replaces: a file used to be enough."""
        status = self.validate(self.comparison_record(
            result="failed", failure_reasons=["the implementations disagree"]))
        self.assertFalse(status["satisfied"])
        self.assertIn("not 'passed'", status["reason"])

    def test_a_record_without_the_negative_case_does_not_satisfy_it(self):
        status = self.validate(self.comparison_record(
            checks={"all_cases_agree": True,
                    "negative_case_distinguished": False,
                    "equivalences_independently_verified": True}))
        self.assertFalse(status["satisfied"])
        self.assertIn("negative case", status["reason"])

    def test_a_fitted_rmsd_does_not_satisfy_the_prerequisite(self):
        status = self.validate(self.comparison_record(
            measure={"superposition": "Kabsch least squares"}))
        self.assertFalse(status["satisfied"])
        self.assertIn("different quantity", status["reason"])

    def test_assumed_equivalences_do_not_satisfy_the_prerequisite(self):
        status = self.validate(self.comparison_record(
            checks={"all_cases_agree": True,
                    "negative_case_distinguished": True,
                    "equivalences_independently_verified": False}))
        self.assertFalse(status["satisfied"])
        self.assertIn("equivalences", status["reason"])

    def test_a_record_for_a_different_ligand_does_not_satisfy_it(self):
        write(self.manifest, ligand_manifest(component="ANP"))
        status = self.validate(self.comparison_record())
        self.assertFalse(status["satisfied"])
        self.assertIn("different ligand manifest", status["reason"])

    def test_a_legacy_modified_manifest_still_satisfies_it(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["preparation"] = {"performed": True, "torsdof": 3}
        write(self.manifest, manifest)
        status = self.validate(self.comparison_record())
        self.assertTrue(status["satisfied"], status["reason"])
        self.assertTrue(
            status["ligand_manifest_identity"]["legacy_compatibility_used"])

    def test_the_wrong_schema_is_refused(self):
        status = self.validate(self.comparison_record(
            schema="mur-ligase-ligand/1"))
        self.assertFalse(status["satisfied"])


REFERENCE_ATOMS = (
    ("C1", "C", (0.0, 0.0, 0.0)),
    ("C2", "C", (1.5, 0.0, 0.0)),
    ("O1", "O", (2.2, 1.0, 0.0)),
    ("O2", "O", (2.2, -1.0, 0.0)),
    ("O3'", "O", (-0.7, 1.2, 0.0)),
)

COMPONENT_CIF = """data_LIG
_chem_comp.id LIG
_chem_comp.name "TEST ACID"
_chem_comp.formula "C2 H4 O2"
_chem_comp.pdbx_formal_charge 0
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_stereo_config
LIG C1 C 0 N N
LIG C2 C 0 N R
LIG O1 O 0 N N
LIG O2 O 0 N N
LIG "O3'" O 0 N N
LIG H1 H 0 N N
LIG HO2 H 0 N N
LIG "HO3'" H 0 N N
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
LIG C1 "O3'" SING N N
LIG C1 H1 SING N N
LIG O2 HO2 SING N N
LIG "O3'" "HO3'" SING N N
#
"""

if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# CLI orchestration of the comparison
# ---------------------------------------------------------------------------

class StubConformer:
    def __init__(self, positions):
        self._positions = [tuple(p) for p in positions]

    def GetAtomPosition(self, index):
        x, y, z = self._positions[index]
        return type("Point", (), {"x": x, "y": y, "z": z})()

    def SetAtomPosition(self, index, point):
        self._positions[index] = (point.x, point.y, point.z)


class StubAtom:
    def __init__(self, index):
        self._index = index

    def GetIdx(self):
        return self._index

    def GetPropsAsDict(self):
        # A molfile carries no atom names, which is the real case: the code
        # must fall back to indices rather than invent names.
        return {}


class StubMol:
    """A molecule just rich enough for the orchestration under test.

    Its self-matches are hand-declared, not derived from this project's
    automorphism code, so the stub does not smuggle our own answer back in.
    """

    def __init__(self, positions, self_matches):
        self._conformer = StubConformer(positions)
        self._self_matches = tuple(self_matches)

    def GetNumAtoms(self):
        return len(self._conformer._positions)

    def GetConformer(self):
        return self._conformer

    def GetAtoms(self):
        return [StubAtom(i) for i in range(self.GetNumAtoms())]

    def GetSubstructMatches(self, other, uniquify=True, useChirality=False,
                            maxMatches=1000):
        return self._self_matches[:maxMatches]

    def copy(self):
        return StubMol(list(self._conformer._positions), self._self_matches)


def install_stub_rdkit(molecule, symmetric_permutation):
    """Install a fake `rdkit` package and return a teardown callable.

    **This stub validates wiring only.** It does not validate RDKit chemistry:
    it is not RDKit, it perceives nothing, and its CalcRMS minimises over a
    permutation set written by hand in the test. Nothing it reports is
    evidence about what real RDKit does with UMA.
    """
    import math
    import types

    def calc_rms(probe, reference, symmetrizeConjugatedTerminalGroups=True):
        first = reference.GetConformer()._positions
        second = probe.GetConformer()._positions
        candidates = [tuple(range(len(first)))]
        if symmetrizeConjugatedTerminalGroups:
            candidates.append(tuple(symmetric_permutation))
        best = None
        for permutation in candidates:
            total = sum(
                sum((a - b) ** 2 for a, b in zip(first[i], second[permutation[i]]))
                for i in range(len(first)))
            value = math.sqrt(total / len(first))
            best = value if best is None else min(best, value)
        return best

    rdkit = types.ModuleType("rdkit")
    chem = types.ModuleType("rdkit.Chem")
    rdmolalign = types.ModuleType("rdkit.Chem.rdMolAlign")
    allchem = types.ModuleType("rdkit.Chem.AllChem")
    geometry = types.ModuleType("rdkit.Geometry")
    rdbase = types.ModuleType("rdkit.rdBase")

    rdbase.rdkitVersion = "stub-0.0"
    rdmolalign.CalcRMS = calc_rms
    chem.MolFromMolFile = lambda path, removeHs=True, sanitize=True: molecule
    chem.Mol = lambda other: other.copy()
    chem.MolFromSmiles = lambda smiles: molecule
    chem.AddHs = lambda mol: mol
    # One shared rank for the symmetric pair, distinct ranks elsewhere.
    moved = [i for i, target in enumerate(symmetric_permutation) if i != target]

    def canonical_rank_atoms(mol, breakTies=False):
        ranks = list(range(mol.GetNumAtoms()))
        for index in moved:
            ranks[index] = -1
        return ranks

    chem.CanonicalRankAtoms = canonical_rank_atoms
    chem.rdMolAlign = rdmolalign
    chem.AllChem = allchem
    allchem.EmbedMolecule = lambda mol, randomSeed=0: 0
    geometry.Point3D = lambda x, y, z: type(
        "Point3D", (), {"x": x, "y": y, "z": z})()
    rdkit.Chem = chem
    rdkit.rdBase = rdbase
    rdkit.Geometry = geometry

    modules = {
        "rdkit": rdkit, "rdkit.Chem": chem,
        "rdkit.Chem.rdMolAlign": rdmolalign,
        "rdkit.Chem.AllChem": allchem,
        "rdkit.Geometry": geometry, "rdkit.rdBase": rdbase,
    }
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)

    def teardown():
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return teardown


@unittest.skipUnless(GEMMI_AVAILABLE, "Gemmi is required")
class ComparisonOrchestrationTests(unittest.TestCase):
    """The CLI path through `_run`, which a compare()-only test cannot cover.

    This is the regression guard for the crash at the `compare()` call site:
    `_run` passed a parameter name that no longer existed, which every test
    of `compare()` in isolation missed because none of them went through
    `_run`.

    **RDKit is stubbed.** These tests validate orchestration and the
    selected-versus-diagnostic wiring. They validate **no RDKit chemistry**
    whatsoever: the stub is not RDKit, and its values prove nothing about
    what real RDKit computes for UMA.
    """

    def setUp(self):
        import argparse
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.argparse = argparse

        self.ccd = self.root / "LIG.cif"
        self.ccd.write_text(COMPONENT_CIF)
        component = chemical_component.read_component(self.ccd)
        self.names = list(component.graph.names)

        coordinates = {name: position
                       for name, _, position in REFERENCE_ATOMS}
        positions = [coordinates[name] for name in self.names]

        # O1 and O2 are C2's terminal oxygens; both are deprotonated at this
        # state, so exchanging them is the justified swap. Declared by hand.
        first, second = self.names.index("O1"), self.names.index("O2")
        permutation = list(range(len(self.names)))
        permutation[first], permutation[second] = second, first
        self.permutation = permutation
        self.molecule = StubMol(positions,
                                [tuple(range(len(self.names))),
                                 tuple(permutation)])

        self.manifest = self.root / "lig.ligand.json"
        write(self.manifest, {
            "schema": "mur-ligase-ligand/1",
            "component": "LIG",
            "protonation_state": {"modelled_ph": 7.4,
                                  "modelled_formal_charge": -1,
                                  "deprotonated_atoms": ["O2"]},
            "reference": {"path": str(self.root / "lig.reference.json")},
            "dictionary": {"path": str(self.ccd)},
            "preparation": {"performed": False},
        })
        self.molfile = self.root / "lig_withH.mol"
        self.molfile.write_text("stub molfile; the stub reader ignores it\n")
        self.verification = self.root / "lig.verification.json"
        write(self.verification, {
            "schema": "mur-ligase-ligand-verification/1",
            "result": "passed", "state": "minus1",
            "tools": {"rdkit": "stub-0.0"},
            "inputs": {
                "ligand_manifest_sha256": provenance.file_digest_of(
                    self.manifest)},
            "outputs": {
                "hydrogenated_molfile": str(self.molfile),
                "hydrogenated_molfile_sha256": provenance.file_digest_of(
                    self.molfile)},
            "checks": {"stereochemistry_before_hydrogens":
                       {"fatal_mismatches": [], "requires_investigation": []}},
        })
        self.addCleanup(
            install_stub_rdkit(self.molecule, self.permutation))

    def run_comparison(self, **overrides):
        options = {
            "ligand_manifest": self.manifest,
            "verification_record": self.verification,
            "out": self.root / "comparison.json",
            "component_dictionary": self.ccd,
            "tolerance": independent_rmsd.AGREEMENT_TOLERANCE_ANGSTROM,
            "overwrite": True,
        }
        options.update(overrides)
        independent_rmsd._run(self.argparse.Namespace(**options))
        return json.loads((options["out"]).read_text())

    def test_the_cli_path_runs_and_writes_a_record(self):
        """The regression: `_run` must reach `compare` with a valid argument."""
        record = self.run_comparison()
        self.assertEqual(record["schema"], "mur-ligase-independent-rmsd/1")
        self.assertEqual(record["result"], "passed", record["failure_reasons"])

    def test_the_selected_configuration_reaches_the_verdict(self):
        """Not just present in the record: actually the one judged."""
        record = self.run_comparison()
        selected = record["measure"]["selected_configuration"]["variant"]
        self.assertEqual(selected, independent_rmsd.SELECTED_VARIANT)
        for row in record["cases"]:
            self.assertEqual(row["toolkit_variant_judged"], selected,
                             f"{row['label']} was judged under another "
                             f"configuration than the record claims binding")
            self.assertIn(selected, row["toolkit_angstrom_by_variant"])

    def test_the_diagnostic_variant_is_recorded_and_not_binding(self):
        """The unsymmetrised setting disagrees on the justified swap.

        The stub's CalcRMS minimises over the hand-declared pair only when
        symmetrising, so the diagnostic setting scores the oxygen separation
        where this project scores zero. The record must pass on the selected
        configuration and report the disagreement.
        """
        record = self.run_comparison()
        self.assertEqual(record["result"], "passed", record["failure_reasons"])
        diagnostics = record["diagnostics"]
        self.assertIn(independent_rmsd.UNSYMMETRISED_VARIANT,
                      diagnostics["variants"])
        self.assertGreater(diagnostics["n_disagreements"], 0)
        self.assertTrue(any(entry["case"].startswith("swap_")
                            for entry in diagnostics["disagreements"]))
        self.assertIn("not binding", diagnostics["status"])

    def test_the_case_set_and_negative_applicability_come_through(self):
        record = self.run_comparison()
        labels = [row["label"] for row in record["cases"]]
        self.assertEqual(labels[:3],
                         ["unchanged", "translation_small", "translation_large"])
        self.assertTrue(any(label.startswith("swap_") for label in labels))
        negative = record["negative_cases"]
        # Nothing is both protonated and unprotonated at one centre here, as
        # in the -3 state, so that case must be not applicable with a reason.
        self.assertFalse(negative["protonation_oxygen_swap"]["applicable"])
        self.assertIn("property of the state",
                      negative["protonation_oxygen_swap"]["reason"])
        self.assertTrue(negative["element_swap"]["applicable"])
        self.assertTrue(record["checks"]["negative_case_distinguished"])

    def test_the_self_matches_are_not_claimed_as_calcrms_mappings(self):
        record = self.run_comparison()
        matches = record["measure"]["toolkit_options"]["self_matches"]
        self.assertIn("NOT verified", matches["relationship_to_calcrms"])
        self.assertFalse(matches["reached_match_limit"])
        self.assertTrue(matches["enumeration_complete"])

    def test_a_toolkit_missing_the_selected_configuration_is_refused(self):
        """Guard against the class of bug this test file was added for.

        If the selected configuration is absent from the toolkit's results,
        the comparison must refuse rather than quietly record every case as
        unevaluated.
        """
        cases = [independent_rmsd.SyntheticCase(
            label="unchanged", kind="identity", coordinates={"C1": (0, 0, 0)},
            expected_angstrom=0.0, equivalence_class="equivalent",
            justification="fixture")]
        graph = graph_module.graph_from_bond_names(["C1"], ["C"], [])
        with self.assertRaises(independent_rmsd.ComparisonError) as caught:
            independent_rmsd.compare(
                cases, {"C1": (0.0, 0.0, 0.0)}, graph, [(0,)],
                lambda case: {"some_other_variant": 0.0})
        self.assertIn("selected configuration", str(caught.exception))
