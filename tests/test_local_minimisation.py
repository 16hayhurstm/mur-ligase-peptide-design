"""Tests for the local-minimisation diagnostic.

**These tests use a stub Vina.** They validate the script's provenance gates,
its correspondence self-check, the no-superposition RMSD path and the contents
of the record. They say nothing whatever about Vina's real scoring function,
real minimisation behaviour, or the chemistry of any receptor: that evidence
can only come from running the script on Epicac against the real installed
Vina.

The scientific risks the tests target:

- a run that proceeds although the input pose does not match the deposited
  reference, which would make any reported RMSD uninterpretable;
- an atom map borrowed from a different prepared PDBQT;
- silently overwriting an existing result;
- a record that labels Vina's score components without evidence;
- a record that could be mistaken for a passing positive control.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_minimisation as diagnostic  # noqa: E402
import receptor_provenance as provenance  # noqa: E402
from test_manifest_provenance import COMPONENT_CIF, REFERENCE_ATOMS  # noqa: E402


def write(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


# Slot holding the unbound reference in Vina's score() array. Duplicated here
# as a plain constant rather than imported, so this module does not depend on
# the rescoring module it is also a fixture for.
REFERENCE_INDEX = 7


class StubVina:
    """A deterministic stand-in for `vina.Vina`.

    It reports fixed scores and writes back the coordinates it was given,
    optionally displaced, so the surrounding logic can be exercised without
    Vina installed. It does not minimise anything.
    """

    instances: list["StubVina"] = []
    displacement = 0.0
    before = (-9.5, -9.0, -0.5)
    after = (-10.25, -9.7, -0.55)

    def __init__(self, sf_name="vina", seed=0, cpu=1):
        self.sf_name, self.seed, self.cpu = sf_name, seed, cpu
        self.ligand_path = None
        self.maps = None
        StubVina.instances.append(self)

    def set_receptor(self, path):
        self.receptor_path = path

    def set_ligand_from_file(self, path):
        self.ligand_path = Path(path)

    def compute_vina_maps(self, center, box_size):
        self.maps = (list(center), list(box_size))

    # Stub internal energies, keyed on the ligand file's stem, so a test can
    # give two poses different internal strain and see the two reference
    # conventions diverge exactly as Vina's do.
    internal_energy: dict[str, float] = {}
    default_internal = 1.0
    # conf_independent divisor, 1 + w * N_rot. A fixed stub value: the real
    # one depends on the ligand's rotatable-bond count.
    conf_divisor = 2.0

    def _internal(self):
        stem = self.ligand_path.stem if self.ligand_path else ""
        return StubVina.internal_energy.get(stem, StubVina.default_internal)

    def score(self, unbound_energy=None):
        """Stub score. Mirrors Vina's reference-energy arithmetic.

        total = (inter + intra - reference) / conf_divisor, with the reference
        defaulting to this pose's own internal energy, and component 7 holding
        the reference actually used, as v1.2.7 does.
        """
        if len(StubVina.before) <= REFERENCE_INDEX:
            # Short form used by the minimisation tests, which do not exercise
            # the reference conventions at all.
            return list(StubVina.before)
        inter, intra = StubVina.before[1], self._internal()
        reference = self._internal() if unbound_energy is None else unbound_energy
        total = (inter + intra - reference) / StubVina.conf_divisor
        components = list(StubVina.before)
        components[0] = round(total, 3)
        components[5] = intra
        components[REFERENCE_INDEX] = reference
        return components

    def optimize(self, max_steps=0):
        self.max_steps = max_steps
        return list(StubVina.after)

    def write_pose(self, path, overwrite=False):
        target = Path(path)
        if target.exists() and not overwrite:
            raise RuntimeError("refusing to overwrite")
        lines = []
        for line in self.ligand_path.read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38]) + StubVina.displacement
                line = f"{line[:30]}{x:8.3f}{line[38:]}"
            lines.append(line)
        target.write_text("\n".join(lines) + "\n")


def install_stub_vina(test):
    module = types.ModuleType("vina")
    module.Vina = StubVina
    module.__version__ = "1.2.7-stub"
    previous = sys.modules.get("vina")
    sys.modules["vina"] = module
    StubVina.instances.clear()
    # A non-zero default, because a stub that reports a changed energy while
    # returning identical coordinates is the pathological case the script
    # refuses; see WritePoseSanityTests, which sets it to zero deliberately.
    StubVina.displacement = 0.25
    StubVina.after = (-10.25, -9.7, -0.55)

    def restore():
        if previous is None:
            sys.modules.pop("vina", None)
        else:
            sys.modules["vina"] = previous
    test.addCleanup(restore)


class DiagnosticFixture(unittest.TestCase):
    """A complete synthetic provenance chain for the diagnostic to consume."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        install_stub_vina(self)

        self.ccd = self.root / "LIG.cif"
        self.ccd.write_text(COMPONENT_CIF)
        self.coordinates = {name: position for name, _, position in REFERENCE_ATOMS}

        self.reference = self.root / "lig.reference.json"
        write(self.reference, {
            "schema": "mur-ligase-ligand-reference/1",
            "component": "LIG",
            "source_structure": {"path": "synthetic.cif", "sha256": "a" * 64},
            "selection": {"canonical_chain": "B"},
            "outputs": {"reference_pdb": str(self.root / "lig.reference.pdb")},
            "atom_mapping": {
                name: {"element": element, "coordinates_angstrom": list(position)}
                for name, element, position in REFERENCE_ATOMS},
        })

        self.ligand_manifest = self.root / "lig.ligand.json"
        write(self.ligand_manifest, {
            "schema": "mur-ligase-ligand/1",
            "component": "LIG",
            "protonation_state": {
                "modelled_ph": 7.4,
                "modelled_formal_charge": -1,
                "deprotonated_atoms": ["O2"],
                "justification": "synthetic fixture",
            },
            "preparation": {"performed": False},
        })

        self.ligand = self.root / "ligands" / "lig.pdbqt"
        self.ligand.parent.mkdir(parents=True, exist_ok=True)
        self.ligand.write_text(self.pdbqt_text(self.coordinates))

        # The hydrogenated molfile the verification stage writes and Meeko is
        # given. The atom map names this file, not the manifest's pre-hydrogen
        # outputs.molfile.
        self.molfile = self.root / "lig_withH.mol"
        self.molfile.write_text("synthetic hydrogenated molfile\n")

        self.verification = self.root / "lig.verification.json"
        write(self.verification, {
            "schema": "mur-ligase-ligand-verification/1",
            "result": "passed",
            "inputs": {
                "ligand_manifest": str(self.ligand_manifest),
                "ligand_manifest_sha256": provenance.file_digest_of(
                    self.ligand_manifest),
            },
            "outputs": {
                "hydrogenated_molfile": str(self.molfile),
                "hydrogenated_molfile_sha256": provenance.file_digest_of(
                    self.molfile),
            },
        })

        # The real atom-map schema: it names the prepared PDBQT and the input
        # molfile, and carries NO ligand manifest hash, because the builder
        # never sees the manifest.
        self.atom_map = self.root / "lig.atom_map.json"
        write(self.atom_map, {
            "schema": "mur-ligase-ligand-atom-map/1",
            "inputs": {
                "prepared_pdbqt": str(self.ligand),
                "prepared_pdbqt_sha256": provenance.file_digest_of(self.ligand),
                "input_molfile": str(self.molfile),
                "input_molfile_sha256": provenance.file_digest_of(self.molfile),
            },
            "serial_to_reference_name": {
                str(serial): name for serial, name
                in enumerate(sorted(self.coordinates), start=1)},
        })

        # The preparation record is the anchor: the only record naming the
        # manifest, the prepared PDBQT and the atom-map record together.
        self.preparation = self.root / "lig.preparation.json"
        self.write_preparation()

    def write_preparation(self, **overrides):
        inputs = {
            "ligand_manifest": str(self.ligand_manifest),
            "ligand_manifest_sha256": provenance.file_digest_of(
                self.ligand_manifest),
            "prepared_pdbqt": str(self.ligand),
            "prepared_pdbqt_sha256": provenance.file_digest_of(self.ligand),
            "atom_map_record": str(self.atom_map),
            "atom_map_record_sha256": provenance.file_digest_of(self.atom_map),
        }
        inputs.update(overrides.pop("inputs", {}))
        record = {
            "schema": "mur-ligase-ligand-preparation/1",
            "result": "passed",
            "failure_reasons": [],
            "component": "LIG",
            "inputs": inputs,
        }
        record.update(overrides)
        write(self.preparation, record)

        self.box_record = self.root / "box.json"
        write(self.box_record, {
            "schema": "mur-ligase-docking-box/1",
            "centre_mode": "bounds-midpoint",
            "centre_angstrom": [0.0, 0.0, 0.0],
            "size_angstrom": [30.0, 30.0, 30.0],
            "selection": {"protein_author_chain_id": "A",
                          "pocket_cutoff_angstrom": 5.0,
                          "pocket_author_residue_ids": ["10", "11"]},
            "source_structure": {"path": "synthetic.cif", "sha256": "a" * 64},
        })

        self.receptor = self.root / "receptor.pdbqt"
        self.receptor.write_text("ATOM      1  N   ALA A  10      "
                                 "  1.000   1.000   1.000  1.00  0.00    "
                                 "+0.000 N \n")
        self.receptor_manifest = self.root / "receptor.manifest.json"
        write(self.receptor_manifest, {
            "schema": provenance.MANIFEST_SCHEMA,
            "receptor_condition": "apo",
            "selections": {"protein_author_chain_id": "A",
                           "pocket_cutoff_angstrom": 5.0,
                           "pocket_author_residue_ids": ["10", "11"]},
            "source_structure": {"path": "synthetic.cif", "sha256": "a" * 64},
            "outputs": {"receptor_pdbqt": str(self.receptor)},
            "checks": {"pocket_complete": True},
            "unresolved_decisions": ["metal omission is an approximation"],
        })

    def pdbqt_text(self, coordinates) -> str:
        lines = ["ROOT"]
        for serial, name in enumerate(sorted(coordinates), start=1):
            position = coordinates[name]
            element = "OA" if name.startswith("O") else "C"
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} LIG A   1    "
                f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                f"  1.00  0.00    +0.000 {element:<2s}")
        lines.append("ENDROOT")
        lines.append("TORSDOF 2")
        return "\n".join(lines) + "\n"

    def arguments(self, **overrides):
        import argparse
        values = dict(
            receptor=self.receptor, ligand=self.ligand,
            output_dir=self.root / "out",
            box_record=self.box_record,
            receptor_manifest=self.receptor_manifest,
            reference=self.reference,
            component_dictionary=self.ccd,
            ligand_manifest=self.ligand_manifest,
            atom_map=self.atom_map,
            preparation_record=self.preparation,
            stereo_verification=self.verification,
            independent_rmsd=None,
            cpu=2, seed=1, max_steps=0,
            input_tolerance=diagnostic.INPUT_POSE_TOLERANCE_ANGSTROM,
            check_only=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)


class PrerequisiteTests(DiagnosticFixture):

    def test_a_fresh_output_directory_is_required(self):
        existing = self.root / "already"
        existing.mkdir()
        (existing / "result.json").write_text("{}")
        with self.assertRaises(diagnostic.DiagnosticError) as caught:
            diagnostic.fresh_output_directory(existing, self.ligand)
        self.assertIn("not empty", str(caught.exception))

    def test_the_output_may_not_sit_inside_the_ligand_directory(self):
        with self.assertRaises(diagnostic.DiagnosticError):
            diagnostic.fresh_output_directory(
                self.ligand.parent / "out", self.ligand)

    def refuses(self, fragment):
        with self.assertRaises(diagnostic.DiagnosticError) as caught:
            diagnostic.run_diagnostic(self.arguments(
                output_dir=diagnostic.fresh_output_directory(
                    self.root / "out", self.ligand)))
        self.assertIn(fragment, str(caught.exception))
        self.assertEqual(StubVina.instances, [],
                         "Vina must not be constructed once a provenance "
                         "gate has failed")

    def test_an_atom_map_naming_a_different_pdbqt_is_refused(self):
        """The atom map's own PDBQT hash must be the file being scored."""
        record = json.loads(self.atom_map.read_text())
        record["inputs"]["prepared_pdbqt_sha256"] = "b" * 64
        write(self.atom_map, record)
        self.write_preparation()      # re-anchor, so only this field is wrong
        self.refuses("prepared PDBQT, as named by the atom map")

    def test_an_atom_map_file_the_preparation_did_not_record_is_refused(self):
        """Substituting the atom-map file must not go unnoticed."""
        record = json.loads(self.atom_map.read_text())
        record["serial_to_reference_name"]["1"] = "O1"
        write(self.atom_map, record)   # preparation still names the old digest
        self.refuses("atom-map record does not match")

    def test_a_prepared_pdbqt_the_preparation_did_not_record_is_refused(self):
        shifted = {name: (x + 0.5, y, z)
                   for name, (x, y, z) in self.coordinates.items()}
        self.ligand.write_text(self.pdbqt_text(shifted))
        self.refuses("prepared PDBQT does not match")

    def test_a_missing_recorded_hash_is_refused_rather_than_skipped(self):
        """An absent hash must fail, never pass for want of evidence."""
        self.write_preparation(inputs={"atom_map_record_sha256": None})
        self.refuses("This check is not optional")

    def test_a_failed_preparation_record_is_refused(self):
        self.write_preparation(result="failed",
                               failure_reasons=["charge sum disagreed"])
        self.refuses("not 'passed'")

    def test_a_molfile_the_verification_did_not_write_is_refused(self):
        record = json.loads(self.atom_map.read_text())
        record["inputs"]["input_molfile_sha256"] = "c" * 64
        write(self.atom_map, record)
        self.write_preparation()
        self.refuses("built from a different molfile")

    def test_the_molfile_cross_check_is_recorded_as_skipped_without_a_record(self):
        """Absent evidence is reported as absent, not as agreement."""
        record = diagnostic.run_diagnostic(self.arguments(
            output_dir=diagnostic.fresh_output_directory(
                self.root / "out", self.ligand),
            stereo_verification=None))
        check = record["inputs"]["hydrogenated_molfile_cross_check"]
        self.assertFalse(check["performed"])
        self.assertNotIn("agrees", check)

    def test_a_drifted_ligand_manifest_is_refused(self):
        """The protonation state is read from this manifest, so identity binds."""
        manifest = json.loads(self.ligand_manifest.read_text())
        manifest["protonation_state"]["deprotonated_atoms"] = ["O1"]
        write(self.ligand_manifest, manifest)
        with self.assertRaises(diagnostic.DiagnosticError) as caught:
            diagnostic.run_diagnostic(self.arguments(
                output_dir=diagnostic.fresh_output_directory(
                    self.root / "out", self.ligand)))
        message = str(caught.exception)
        self.assertIn("is not the one", message)
        self.assertIn("lig.preparation.json was built on", message)
        self.assertIn("protonation state comes from this manifest", message)
        self.assertEqual(StubVina.instances, [])

    def test_a_box_disagreeing_with_the_receptor_is_refused(self):
        record = json.loads(self.box_record.read_text())
        record["selection"]["protein_author_chain_id"] = "B"
        write(self.box_record, record)
        with self.assertRaises(diagnostic.DiagnosticError) as caught:
            diagnostic.run_diagnostic(self.arguments(
                output_dir=diagnostic.fresh_output_directory(
                    self.root / "out", self.ligand)))
        self.assertIn("disagree", str(caught.exception))


class InputPoseCheckTests(DiagnosticFixture):
    """The guard that makes a reported RMSD interpretable."""

    def test_a_displaced_input_pose_aborts_before_vina_is_called(self):
        shifted = {name: (x + 1.5, y, z)
                   for name, (x, y, z) in self.coordinates.items()}
        self.ligand.write_text(self.pdbqt_text(shifted))
        # Re-anchor every provenance hash, so the only thing wrong is the
        # geometry: this must be caught by the coordinate check itself, not
        # incidentally by a digest mismatch.
        record = json.loads(self.atom_map.read_text())
        record["inputs"]["prepared_pdbqt_sha256"] = \
            provenance.file_digest_of(self.ligand)
        write(self.atom_map, record)
        self.write_preparation()

        with self.assertRaises(diagnostic.DiagnosticError) as caught:
            diagnostic.run_diagnostic(self.arguments(
                output_dir=diagnostic.fresh_output_directory(
                    self.root / "out", self.ligand)))
        message = str(caught.exception)
        self.assertIn("from the deposited", message)
        self.assertIn("Nothing was scored", message)
        self.assertEqual(StubVina.instances, [],
                         "Vina must not be constructed once the input pose "
                         "fails to match the reference")

    def test_the_matching_input_pose_records_a_zero_rmsd(self):
        record = diagnostic.run_diagnostic(self.arguments(
            output_dir=diagnostic.fresh_output_directory(
                self.root / "out", self.ligand)))
        self.assertLess(record["input_pose_check"]["rmsd_angstrom"], 1e-6)
        self.assertTrue(record["input_pose_check"]["passed"])


class RecordContentTests(DiagnosticFixture):

    def completed_record(self, **overrides):
        return diagnostic.run_diagnostic(self.arguments(
            output_dir=diagnostic.fresh_output_directory(
                self.root / "out", self.ligand), **overrides))

    def test_the_record_denies_being_a_positive_control(self):
        record = self.completed_record()
        self.assertFalse(record["experiment"]["is_independent_redocking"])
        self.assertFalse(record["experiment"]["counts_towards_2A_positive_control"])
        self.assertIn("NOT independent redocking",
                      record["experiment"]["statement"])

    def test_the_rmsd_applies_no_superposition(self):
        """A rigid displacement must survive as the RMSD, not be fitted away."""
        StubVina.displacement = 0.75
        record = self.completed_record()
        self.assertAlmostEqual(
            record["geometry"]["rmsd_to_deposited_angstrom"], 0.75, places=3)
        self.assertAlmostEqual(
            record["geometry"]["drift_from_input_angstrom"], 0.75, places=3)
        self.assertEqual(record["measure"]["superposition"], "none")

    def test_score_components_are_recorded_without_invented_labels(self):
        record = self.completed_record()
        before = record["scores"]["before_minimisation"]
        self.assertEqual(before["raw_components"], list(StubVina.before))
        self.assertEqual(before["total_kcal_per_mol"], StubVina.before[0])
        self.assertEqual(before["n_components"], 3)
        self.assertIn("docstring", before["component_labels_source"])
        self.assertIn("Stub score", before["installed_score_docstring"])

    def test_both_scores_and_the_pose_file_are_recorded(self):
        record = self.completed_record()
        self.assertEqual(record["result"], "completed")
        self.assertEqual(
            record["scores"]["after_minimisation"]["total_kcal_per_mol"],
            StubVina.after[0])
        pose_file = Path(record["geometry"]["minimised_pose_file"])
        self.assertTrue(pose_file.is_file())
        self.assertEqual(record["geometry"]["minimised_pose_sha256"],
                         provenance.file_digest_of(pose_file))
        self.assertEqual(record["geometry"]["n_heavy_atoms"],
                         len(self.coordinates))

    def test_the_box_and_receptor_condition_are_recorded(self):
        record = self.completed_record()
        self.assertEqual(record["box"]["size_angstrom"], [30.0, 30.0, 30.0])
        self.assertEqual(record["inputs"]["receptor_condition"], "apo")
        self.assertIn("metal omission is an approximation",
                      record["inputs"]["receptor_unresolved_decisions"])
        self.assertEqual(record["box"]["ligand_heavy_atoms_inside"],
                         len(self.coordinates))

    def test_the_protonation_basis_comes_from_the_manifest(self):
        record = self.completed_record()
        self.assertIn("deprotonated ['O2']", record["measure"]["protonation_basis"])
        self.assertIn("pH 7.4", record["measure"]["protonation_basis"])

    def test_check_only_validates_without_scoring(self):
        record = self.completed_record(check_only=True)
        self.assertEqual(record["result"], "checks_passed_nothing_scored")
        self.assertNotIn("scores", record)
        self.assertEqual(StubVina.instances, [])


class WritePoseSanityTests(DiagnosticFixture):
    """The false negative that would matter most.

    If `write_pose` emitted the input coordinates rather than the optimised
    ones, the RMSD would read as a perfectly stable native pose while nothing
    had been minimised. The script refuses that combination.
    """

    def test_a_changed_score_with_an_unchanged_pose_is_refused(self):
        StubVina.displacement = 0.0          # write_pose returns the input
        StubVina.after = (-10.25, -9.7, -0.55)   # but the energy moved
        with self.assertRaises(diagnostic.DiagnosticError) as caught:
            diagnostic.run_diagnostic(self.arguments(
                output_dir=diagnostic.fresh_output_directory(
                    self.root / "out", self.ligand)))
        self.assertIn("appears not to emit the optimised coordinates",
                      str(caught.exception))

    def test_an_unchanged_score_with_an_unchanged_pose_is_allowed(self):
        """A genuine no-op minimisation is a legitimate result, not an error."""
        StubVina.displacement = 0.0
        self.addCleanup(setattr, StubVina, "after", StubVina.after)
        StubVina.after = StubVina.before
        record = diagnostic.run_diagnostic(self.arguments(
            output_dir=diagnostic.fresh_output_directory(
                self.root / "out", self.ligand)))
        self.assertEqual(record["geometry"]["drift_from_input_angstrom"], 0.0)
        self.assertEqual(record["result"], "completed")


class PoseReadingTests(DiagnosticFixture):

    def test_a_multi_pose_file_is_refused(self):
        multi = self.root / "multi.pdbqt"
        single = self.pdbqt_text(self.coordinates)
        multi.write_text(f"MODEL 1\n{single}ENDMDL\nMODEL 2\n{single}ENDMDL\n")
        names = {serial: name for serial, name
                 in enumerate(sorted(self.coordinates), start=1)}
        with self.assertRaises(diagnostic.DiagnosticError) as caught:
            diagnostic.pose_from_pdbqt(multi, names)
        self.assertIn("exactly one", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
