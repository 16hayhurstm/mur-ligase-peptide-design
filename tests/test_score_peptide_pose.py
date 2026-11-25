"""Tests for the Vina peptide scoring stage.

Vina itself is not exercised: these tests cover the parts that can be wrong
without Vina noticing — box validation, PDBQT reading, the TORSDOF line, and
the displacement measurement that decides whether a local optimisation left
the predicted backbone alone.

The displacement tests matter most. A pose comparison done on file order
rather than atom identity reports motion that did not happen, and a single
RMSD hides a backbone that moved while side chains did not.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import score_peptide_pose as sps  # noqa: E402

BOX_30A = REPOSITORY_ROOT / "examples" / "box_murc_uma_chainE_30A.json"


def pdbqt(atoms, torsdof: int | None = None) -> str:
    """Build a minimal PDBQT from (serial, name, resname, resnum, x, y, z).

    Columns follow the PDB specification, which PDBQT shares up to column 54.
    """
    lines = ["ROOT"]
    for serial, name, resname, resnum, x, y, z in atoms:
        # Columns matter here: record 1-6, serial 7-11, atom name 13-16,
        # altloc 17, residue name 18-20, chain 22, residue number 23-26, and
        # the coordinates in 31-54. An off-by-one shifts the residue name into
        # the altloc column, which is how the first version of this helper
        # produced atoms named "ER" with altloc "S".
        name_field = name if len(name) == 4 else f" {name:<3}"
        lines.append(
            f"ATOM  {serial:>5} {name_field}{'':1}{resname:>3} "
            f"{'P':1}{resnum:>4}{'':1}   "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00    +0.000 C "
        )
    lines.append("ENDROOT")
    if torsdof is not None:
        lines.append(f"TORSDOF {torsdof}")
    return "\n".join(lines) + "\n"


def write_pdbqt(directory: Path, name: str, atoms, torsdof=None) -> Path:
    path = directory / name
    path.write_text(pdbqt(atoms, torsdof))
    return path


BASE_ATOMS = [
    (1, "N", "SER", 1, 25.0, -6.0, 48.0),
    (2, "CA", "SER", 1, 26.0, -6.0, 48.0),
    (3, "C", "SER", 1, 27.0, -6.0, 48.0),
    (4, "O", "SER", 1, 27.0, -7.0, 48.0),
    (5, "CB", "SER", 1, 26.0, -5.0, 48.0),
    (6, "OG", "SER", 1, 26.0, -4.0, 48.0),
]


class BoxRecordTests(unittest.TestCase):
    def test_the_recorded_30_angstrom_box_loads(self):
        centre, size, record = sps.load_box_record(BOX_30A)
        self.assertEqual(size, [30.0, 30.0, 30.0])
        self.assertEqual(len(centre), 3)
        self.assertEqual(record["schema"], "mur-ligase-docking-box/1")

    def test_a_box_without_a_centre_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "box.json"
            path.write_text(json.dumps({"size_angstrom": [10, 10, 10]}))
            with self.assertRaises(sps.ScoringError) as raised:
                sps.load_box_record(path)
            self.assertIn("centre_angstrom", str(raised.exception))

    def test_a_zero_dimension_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "box.json"
            path.write_text(
                json.dumps(
                    {"centre_angstrom": [0, 0, 0], "size_angstrom": [10, 0, 10]}
                )
            )
            with self.assertRaises(sps.ScoringError):
                sps.load_box_record(path)

    def test_a_two_component_centre_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "box.json"
            path.write_text(
                json.dumps(
                    {"centre_angstrom": [0, 0], "size_angstrom": [10, 10, 10]}
                )
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.load_box_record(path)
            self.assertIn("three values", str(raised.exception))

    def test_an_unreadable_box_record_is_refused(self):
        with self.assertRaises(sps.ScoringError) as raised:
            sps.load_box_record(Path("/nonexistent/box.json"))
        self.assertIn("cannot read", str(raised.exception))


class TorsionCountTests(unittest.TestCase):
    def test_the_torsdof_line_is_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pdbqt(Path(temporary), "l.pdbqt", BASE_ATOMS, torsdof=25)
            self.assertEqual(sps.pdbqt_torsion_count(path), 25)

    def test_a_file_without_torsdof_returns_none_rather_than_zero(self):
        # Zero would be a claim that the ligand is rigid, which is different
        # from not knowing.
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pdbqt(Path(temporary), "l.pdbqt", BASE_ATOMS)
            self.assertIsNone(sps.pdbqt_torsion_count(path))

    def test_a_malformed_torsdof_returns_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "l.pdbqt"
            path.write_text(pdbqt(BASE_ATOMS) + "TORSDOF many\n")
            self.assertIsNone(sps.pdbqt_torsion_count(path))


class PoseReadingTests(unittest.TestCase):
    def test_atoms_are_keyed_by_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pdbqt(Path(temporary), "l.pdbqt", BASE_ATOMS)
            positions = sps.pose_positions(path)
            self.assertEqual(len(positions), 6)
            names = {key.atom_name for key in positions}
            self.assertEqual(names, {"N", "CA", "C", "O", "CB", "OG"})

    def test_an_unreadable_pose_is_refused(self):
        with self.assertRaises(sps.ScoringError):
            sps.pose_positions(Path("/nonexistent/pose.pdbqt"))


class DisplacementTests(unittest.TestCase):
    def test_an_unmoved_pose_reports_zero_everywhere(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", BASE_ATOMS)
            after = write_pdbqt(directory, "b.pdbqt", BASE_ATOMS)
            report = sps.displacement_report(before, after)
            self.assertEqual(report["all_heavy_atoms"]["maximum_angstrom"], 0.0)
            self.assertEqual(report["backbone"]["rmsd_angstrom"], 0.0)
            self.assertEqual(report["side_chain"]["rmsd_angstrom"], 0.0)

    def test_a_side_chain_only_relaxation_leaves_the_backbone_at_zero(self):
        # The result the pilot is looking for: side chains moved, backbone did
        # not. A single pooled RMSD would report motion and hide which part.
        moved = [
            atom if atom[1] not in ("CB", "OG") else
            (atom[0], atom[1], atom[2], atom[3], atom[4], atom[5] + 1.0, atom[6])
            for atom in BASE_ATOMS
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", BASE_ATOMS)
            after = write_pdbqt(directory, "b.pdbqt", moved)
            report = sps.displacement_report(before, after)
            self.assertEqual(report["backbone"]["maximum_angstrom"], 0.0)
            self.assertEqual(report["backbone"]["atoms"], 4)
            self.assertEqual(report["side_chain"]["maximum_angstrom"], 1.0)
            self.assertEqual(report["side_chain"]["atoms"], 2)

    def test_a_backbone_movement_is_reported_distinctly(self):
        moved = [
            atom if atom[1] != "CA" else
            (atom[0], atom[1], atom[2], atom[3], atom[4], atom[5], atom[6] + 2.0)
            for atom in BASE_ATOMS
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", BASE_ATOMS)
            after = write_pdbqt(directory, "b.pdbqt", moved)
            report = sps.displacement_report(before, after)
            self.assertEqual(report["backbone"]["maximum_angstrom"], 2.0)
            self.assertEqual(report["side_chain"]["maximum_angstrom"], 0.0)
            self.assertEqual(
                report["largest_movements"][0]["atom"].endswith("CA"), True
            )
            self.assertTrue(report["largest_movements"][0]["backbone"])

    def test_pairing_is_by_identity_so_reordering_reports_no_motion(self):
        # If the comparison used file order, reversing the records would
        # report large spurious displacements.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", BASE_ATOMS)
            after = write_pdbqt(directory, "b.pdbqt", list(reversed(BASE_ATOMS)))
            report = sps.displacement_report(before, after)
            self.assertEqual(report["all_heavy_atoms"]["maximum_angstrom"], 0.0)

    def test_hydrogens_are_excluded_from_the_statistics(self):
        with_hydrogen = BASE_ATOMS + [(7, "HG", "SER", 1, 26.0, -3.0, 48.0)]
        moved = BASE_ATOMS + [(7, "HG", "SER", 1, 26.0, 5.0, 48.0)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", with_hydrogen)
            after = write_pdbqt(directory, "b.pdbqt", moved)
            report = sps.displacement_report(before, after)
            self.assertEqual(report["all_heavy_atoms"]["atoms"], 6)
            self.assertEqual(report["all_heavy_atoms"]["maximum_angstrom"], 0.0)

    def test_an_atom_present_in_only_one_pose_is_named(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", BASE_ATOMS)
            after = write_pdbqt(directory, "b.pdbqt", BASE_ATOMS[:-1])
            report = sps.displacement_report(before, after)
            self.assertEqual(len(report["atoms_only_in_before"]), 1)
            self.assertIn("OG", report["atoms_only_in_before"][0])
            self.assertEqual(report["atoms_only_in_after"], [])

    def test_oxt_counts_as_backbone(self):
        atoms = BASE_ATOMS + [(7, "OXT", "SER", 1, 28.0, -7.0, 48.0)]
        moved = BASE_ATOMS + [(7, "OXT", "SER", 1, 28.0, -7.0, 49.5)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", atoms)
            after = write_pdbqt(directory, "b.pdbqt", moved)
            report = sps.displacement_report(before, after)
            self.assertEqual(report["backbone"]["atoms"], 5)
            self.assertEqual(report["backbone"]["maximum_angstrom"], 1.5)

    def test_the_rmsd_is_a_root_mean_square_not_a_mean(self):
        # Two atoms moving 0 and 2 A: mean 1.0, RMSD sqrt(2) = 1.4142.
        moved = [
            (1, "N", "SER", 1, 25.0, -6.0, 48.0),
            (2, "CA", "SER", 1, 26.0, -6.0, 50.0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = write_pdbqt(directory, "a.pdbqt", BASE_ATOMS[:2])
            after = write_pdbqt(directory, "b.pdbqt", moved)
            report = sps.displacement_report(before, after)
            self.assertEqual(report["backbone"]["mean_angstrom"], 1.0)
            self.assertAlmostEqual(
                report["backbone"]["rmsd_angstrom"], 1.4142, places=3
            )


class BoxContainmentTests(unittest.TestCase):
    def test_a_pose_at_the_box_centre_is_contained(self):
        centre, _, record = sps.load_box_record(BOX_30A)
        atoms = [
            (1, "CA", "SER", 1, centre[0], centre[1], centre[2]),
            (2, "C", "SER", 1, centre[0] + 1.0, centre[1], centre[2]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pdbqt(Path(temporary), "l.pdbqt", atoms)
            containment = sps.pose_box_containment(path, record)
            self.assertTrue(containment["fully_contained"])
            self.assertIn("inside the box", containment["consequence"])

    def test_an_atom_outside_the_box_is_reported_with_its_consequence(self):
        centre, size, record = sps.load_box_record(BOX_30A)
        atoms = [
            (1, "CA", "SER", 1, centre[0], centre[1], centre[2]),
            (2, "C", "SER", 1, centre[0] + size[0], centre[1], centre[2]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pdbqt(Path(temporary), "l.pdbqt", atoms)
            containment = sps.pose_box_containment(path, record)
            self.assertFalse(containment["fully_contained"])
            self.assertEqual(containment["atoms_inside"], 1)
            self.assertIn("not a score of this pose", containment["consequence"])

    def test_a_pose_with_only_hydrogens_is_refused(self):
        atoms = [(1, "HA", "SER", 1, 25.0, -6.0, 48.0)]
        with tempfile.TemporaryDirectory() as temporary:
            path = write_pdbqt(Path(temporary), "l.pdbqt", atoms)
            _, _, record = sps.load_box_record(BOX_30A)
            with self.assertRaises(sps.ScoringError) as raised:
                sps.pose_box_containment(path, record)
            self.assertIn("no heavy atoms", str(raised.exception))


class RecordTests(unittest.TestCase):
    def test_the_two_results_are_kept_in_separate_fields(self):
        centre, size, box = sps.load_box_record(BOX_30A)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write_pdbqt(directory, "l.pdbqt", BASE_ATOMS, torsdof=25)
            receptor = write_pdbqt(directory, "r.pdbqt", BASE_ATOMS)
            record = sps.build_record(
                "spec_0574", ligand, receptor, BOX_30A, box, centre, size,
                seed=1, cpu=2,
                containment={"fully_contained": True},
                torsions=25,
                initial={"total_kcal_per_mol": -5.0},
                optimised={"total_kcal_per_mol": -7.0},
            )
            self.assertEqual(record["initial_pose"]["total_kcal_per_mol"], -5.0)
            self.assertEqual(
                record["after_local_optimisation"]["total_kcal_per_mol"], -7.0
            )
            self.assertEqual(record["schema"], "mur-ligase-peptide-score/1")

    def test_the_record_states_that_no_search_was_performed(self):
        centre, size, box = sps.load_box_record(BOX_30A)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write_pdbqt(directory, "l.pdbqt", BASE_ATOMS)
            record = sps.build_record(
                "d", ligand, ligand, BOX_30A, box, centre, size, 1, 2,
                {"fully_contained": True}, None, {}, None,
            )
            self.assertFalse(record["scoring_parameters"]["search_performed"])
            self.assertIn("dock() is never called", record["scoring_parameters"]["note"])

    def test_a_run_without_optimisation_records_none_not_an_empty_result(self):
        centre, size, box = sps.load_box_record(BOX_30A)
        with tempfile.TemporaryDirectory() as temporary:
            ligand = write_pdbqt(Path(temporary), "l.pdbqt", BASE_ATOMS)
            record = sps.build_record(
                "d", ligand, ligand, BOX_30A, box, centre, size, 1, 2,
                {"fully_contained": True}, 25, {"total_kcal_per_mol": -5.0}, None,
            )
            self.assertIsNone(record["after_local_optimisation"])

    def test_input_checksums_and_scoring_parameters_are_recorded(self):
        centre, size, box = sps.load_box_record(BOX_30A)
        with tempfile.TemporaryDirectory() as temporary:
            ligand = write_pdbqt(Path(temporary), "l.pdbqt", BASE_ATOMS)
            record = sps.build_record(
                "d", ligand, ligand, BOX_30A, box, centre, size, 7, 4,
                {"fully_contained": True}, 25, {}, None,
            )
            self.assertEqual(len(record["inputs"]["ligand_pdbqt_sha256"]), 64)
            self.assertEqual(len(record["inputs"]["box_record_sha256"]), 64)
            self.assertEqual(record["scoring_parameters"]["seed"], 7)
            self.assertEqual(record["scoring_parameters"]["cpu"], 4)

    def test_the_torsion_caveat_names_the_backbone_hazard(self):
        centre, size, box = sps.load_box_record(BOX_30A)
        with tempfile.TemporaryDirectory() as temporary:
            ligand = write_pdbqt(Path(temporary), "l.pdbqt", BASE_ATOMS)
            record = sps.build_record(
                "d", ligand, ligand, BOX_30A, box, centre, size, 1, 2,
                {"fully_contained": True}, 25, {}, None,
            )
            self.assertIn("phi and psi", record["ligand_torsions"]["caveat"])


class VinaAbsenceTests(unittest.TestCase):
    @unittest.skipIf(sps.VINA_AVAILABLE, "Vina is installed here")
    def test_a_missing_vina_is_a_blocker_and_says_so(self):
        with self.assertRaises(sps.ScoringError) as raised:
            sps.require_vina()
        message = str(raised.exception)
        self.assertIn("blocker", message)
        self.assertIn("not a skippable check", message)


if __name__ == "__main__":
    unittest.main()
