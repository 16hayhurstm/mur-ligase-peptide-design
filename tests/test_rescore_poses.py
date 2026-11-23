"""Tests for consistent pose rescoring.

**Stub Vina.** These validate extraction, chemical-identity preservation,
coordinate preservation, the provenance chain and the convention probe. They
establish nothing about Vina's scoring function.

The scientific risk they target is extraction: a pose that silently acquires
different coordinates, atom types, charges, serials or a rebuilt torsion tree
would be rescored as something other than what Vina produced, and the
comparison would be meaningless while looking entirely healthy.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rescore_poses as rescore  # noqa: E402
import receptor_provenance as provenance  # noqa: E402
from test_local_minimisation import (DiagnosticFixture, REFERENCE_INDEX,  # noqa: E402
                                     StubVina, write)

# An eight-component stub array, so the reference-energy slots exist. Index 1
# is the intermolecular term the stub uses; index 7 is overwritten with the
# reference actually applied.
EIGHT_COMPONENTS = [0.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def pose_block(coordinates, affinity, offset=0.0, torsdof=2):
    """A PDBQT MODEL block in Vina's output shape, with a torsion tree."""
    lines = [f"REMARK VINA RESULT:   {affinity:.3f}      0.000      0.000",
             "REMARK  7 active torsions:", "ROOT"]
    for serial, name in enumerate(sorted(coordinates), start=1):
        x, y, z = coordinates[name]
        element = "OA" if name.startswith("O") else "C"
        lines.append(
            f"ATOM  {serial:5d} {name:<4s} LIG A   1    "
            f"{x + offset:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00    -0.123 {element:<2s}")
    lines += ["ENDROOT", "BRANCH   1   2", "ENDBRANCH   1   2",
              f"TORSDOF {torsdof}"]
    return lines


class ExtractionFixture(DiagnosticFixture):
    """Reuses the validated provenance chain and adds a multi-model pose file."""

    def setUp(self):
        super().setUp()
        StubVina.before = list(EIGHT_COMPONENTS)
        StubVina.internal_energy = {}
        StubVina.default_internal = 1.0
        StubVina.conf_divisor = 2.0
        self.affinities = (-9.33, -9.25, -9.16)
        self.offsets = (0.0, 0.5, 1.25)
        blocks = [pose_block(self.coordinates, affinity, offset)
                  for affinity, offset in zip(self.affinities, self.offsets)]
        self.docked = self.root / "runs" / "poses" / "seed_1" / "lig_docked.pdbqt"
        self.docked.parent.mkdir(parents=True, exist_ok=True)
        self.docked.write_text("".join(
            f"MODEL {index}\n" + "\n".join(block) + "\nENDMDL\n"
            for index, block in enumerate(blocks, start=1)))

    def arguments(self, **overrides):
        import argparse
        values = dict(
            receptor=self.receptor, output_dir=self.root / "rescored",
            ligand=self.ligand, deposited=self.ligand, minimised=None,
            docked_rank=[f"1:{self.docked}", f"3:{self.docked}"],
            box_record=self.box_record,
            receptor_manifest=self.receptor_manifest,
            reference=self.reference,
            component_dictionary=self.ccd,
            ligand_manifest=self.ligand_manifest,
            atom_map=self.atom_map,
            preparation_record=self.preparation,
            stereo_verification=self.verification,
            reference_pose="docked_rank_1",
            cpu=2, seed=1, check_only=False,
        )
        values.update(overrides)
        return argparse.Namespace(**values)


class ExtractionTests(ExtractionFixture):

    def test_a_pose_is_extracted_byte_identically(self):
        destination = self.root / "rank1.pdbqt"
        record = rescore.extract_pose(self.docked, 1, destination)
        source_atoms = rescore.atom_lines(rescore.split_models(self.docked)[0])
        written_atoms = rescore.atom_lines(destination.read_text().splitlines())
        self.assertEqual(written_atoms, source_atoms)
        self.assertTrue(record["atom_records_identical_to_source"])
        self.assertEqual(record["rank_in_file"], 1)
        self.assertEqual(record["n_poses_in_file"], 3)

    def test_the_torsion_tree_and_charges_survive_extraction(self):
        destination = self.root / "rank2.pdbqt"
        rescore.extract_pose(self.docked, 2, destination)
        text = destination.read_text()
        for expected in ("ROOT", "ENDROOT", "BRANCH   1   2",
                         "ENDBRANCH   1   2", "TORSDOF 2", "-0.123"):
            self.assertIn(expected, text,
                          f"{expected!r} must survive extraction unaltered")
        self.assertNotIn("MODEL", text)
        self.assertNotIn("ENDMDL", text)

    def test_the_recorded_affinity_is_read_not_recomputed(self):
        for rank, affinity in enumerate(self.affinities, start=1):
            record = rescore.extract_pose(
                self.docked, rank, self.root / f"r{rank}.pdbqt")
            self.assertAlmostEqual(
                record["recorded_affinity_kcal_per_mol"], affinity, places=3)

    def test_each_rank_keeps_its_own_coordinates(self):
        """Rank 3 must not be written with rank 1's coordinates."""
        first = rescore.extract_pose(self.docked, 1, self.root / "a.pdbqt")
        third = rescore.extract_pose(self.docked, 3, self.root / "b.pdbqt")
        self.assertNotEqual(first["extracted_sha256"],
                            third["extracted_sha256"])
        third_x = [float(line[30:38]) for line
                   in rescore.atom_lines((self.root / "b.pdbqt").read_text()
                                         .splitlines())]
        expected = [position[0] + self.offsets[2]
                    for _, position in sorted(self.coordinates.items())]
        for got, want in zip(third_x, expected):
            self.assertAlmostEqual(got, want, places=3)

    def test_a_rank_beyond_the_file_is_refused(self):
        with self.assertRaises(rescore.RescoringError) as caught:
            rescore.extract_pose(self.docked, 9, self.root / "c.pdbqt")
        self.assertIn("rank 9 does not exist", str(caught.exception))

    def test_an_existing_destination_is_never_overwritten(self):
        destination = self.root / "taken.pdbqt"
        destination.write_text("earlier result\n")
        with self.assertRaises(rescore.RescoringError):
            rescore.extract_pose(self.docked, 1, destination)
        self.assertEqual(destination.read_text(), "earlier result\n")

    def test_a_single_pose_file_without_model_records_is_read(self):
        single = self.root / "single.pdbqt"
        single.write_text("\n".join(
            pose_block(self.coordinates, -8.5)) + "\n")
        self.assertEqual(len(rescore.split_models(single)), 1)

    def test_a_file_with_no_poses_is_refused(self):
        empty = self.root / "empty.pdbqt"
        empty.write_text("REMARK nothing here\n")
        with self.assertRaises(rescore.RescoringError):
            rescore.split_models(empty)


class AffinityReproductionTests(unittest.TestCase):
    """Replaces the earlier convention probe, whose inference was wrong.

    A score() component that does not match the recorded affinity is *not*
    evidence of a different reference convention. Both totals come from the
    same formula and differ only in which internal energy was supplied as the
    reference. The test is whether supplying the docking run's reference
    reproduces its affinity.
    """

    def test_agreement_within_tolerance_is_reported_as_reproduced(self):
        check = rescore.affinity_reproduction(-9.331, -9.329)
        self.assertTrue(check["reproduced"])
        self.assertAlmostEqual(check["residual_kcal_per_mol"], -0.002, places=6)
        self.assertIn("rounded to three decimals", check["tolerance_basis"])

    def test_a_large_residual_is_reported_as_not_reproduced(self):
        check = rescore.affinity_reproduction(-9.801, -9.161)
        self.assertFalse(check["reproduced"])
        self.assertAlmostEqual(check["residual_kcal_per_mol"], -0.640, places=6)

    def test_a_pose_without_a_recorded_affinity_is_not_checked(self):
        check = rescore.affinity_reproduction(-9.0, None)
        self.assertFalse(check["checked"])
        self.assertIn("no recorded affinity", check["reason"])

    def test_no_common_reference_score_means_no_check(self):
        check = rescore.affinity_reproduction(None, -9.161)
        self.assertFalse(check["checked"])

    def test_the_docstring_records_the_corrected_interpretation(self):
        text = rescore.affinity_reproduction.__doc__
        self.assertIn("That inference was wrong", text)
        self.assertIn("not evidence of a different convention", text)


class ReferenceConventionTests(ExtractionFixture):
    """The distinction the two conventions make, and why ranking can flip."""

    def setUp(self):
        super().setUp()
        # Rank 3 is internally strained relative to rank 1 by 1.0 kcal/mol,
        # the situation that demoted the near-native pose in the real run.
        StubVina.internal_energy = {"rank_1": 1.0, "rank_3": 2.0, "lig": 1.5}

    def test_the_reference_slot_is_confirmed_before_use(self):
        record = rescore.run(self.arguments())
        confirmation = record["reference_energy"]["confirmation"]
        self.assertEqual(confirmation["component_index"], REFERENCE_INDEX)
        self.assertIn("reproduced the same total", confirmation["confirmed_by"])
        self.assertAlmostEqual(confirmation["total_without_argument"],
                               confirmation["total_with_argument"], places=6)

    def test_an_unconfirmable_reference_slot_aborts(self):
        """If slot 7 is not the reference, nothing may be interpreted."""
        class Wrong(StubVina):
            def score(self, unbound_energy=None):
                components = list(EIGHT_COMPONENTS)
                components[0] = -5.0 if unbound_energy is None else -99.0
                return components
        sys.modules["vina"].Vina = Wrong
        self.addCleanup(setattr, sys.modules["vina"], "Vina", StubVina)
        with self.assertRaises(rescore.RescoringError) as caught:
            rescore.run(self.arguments())
        self.assertIn("is not the unbound reference", str(caught.exception))

    def test_the_reference_pose_scores_identically_under_both_conventions(self):
        """Its own reference *is* the shared one, so the two must agree."""
        record = rescore.run(self.arguments())
        row = next(r for r in record["poses"] if r["label"] == "docked_rank_1")
        specific = row["scores"]["pose_specific"]["total_kcal_per_mol"]
        shared = row["scores"]["common_reference"]["total_kcal_per_mol"]
        self.assertAlmostEqual(specific, shared, places=6)

    def test_a_strained_pose_diverges_between_the_conventions(self):
        record = rescore.run(self.arguments())
        row = next(r for r in record["poses"] if r["label"] == "docked_rank_3")
        specific = row["scores"]["pose_specific"]["total_kcal_per_mol"]
        shared = row["scores"]["common_reference"]["total_kcal_per_mol"]
        # strain 2.0 against reference 1.0, over conf_divisor 2.0
        self.assertAlmostEqual(shared - specific, 0.5, places=6)
        self.assertLess(specific, shared,
                        "the pose-specific total must be the more favourable "
                        "one when the pose is internally strained")

    def test_both_methods_are_retained_and_labelled(self):
        record = rescore.run(self.arguments())
        for row in record["poses"]:
            self.assertEqual(row["scores"]["pose_specific"]["method"],
                             "pose_specific")
            self.assertEqual(row["scores"]["common_reference"]["method"],
                             "common_reference")
        self.assertIn("neither_is_more_correct", record["scoring_methods"])
        self.assertIn("v1.2.7", record["scoring_methods"]["source"])

    def test_the_reference_must_come_from_a_requested_pose(self):
        with self.assertRaises(rescore.RescoringError) as caught:
            rescore.run(self.arguments(reference_pose="docked_rank_99"))
        self.assertIn("is not among the poses requested", str(caught.exception))

    def test_the_crystal_poses_are_scored_under_the_shared_reference_too(self):
        record = rescore.run(self.arguments())
        row = next(r for r in record["poses"] if r["label"] == "deposited")
        self.assertIn("common_reference", row["scores"])
        self.assertEqual(
            row["scores"]["common_reference"]["reference_kcal_per_mol"],
            record["reference_energy"]["reference_kcal_per_mol"])

    def test_the_uncertainty_note_no_longer_sets_a_ranking_threshold(self):
        record = rescore.run(self.arguments())
        note = record["uncertainty_note"]
        self.assertIn("NOT a calibrated threshold", note)
        self.assertIn("across different ligands", note)


class RescoringRunTests(ExtractionFixture):

    def test_every_pose_is_scored_under_identical_settings(self):
        record = rescore.run(self.arguments())
        self.assertEqual(len(record["poses"]), 3)
        labels = [row["label"] for row in record["poses"]]
        self.assertEqual(labels, ["deposited", "docked_rank_1", "docked_rank_3"])
        # One engine per pose, plus one built earlier to establish and
        # confirm the shared unbound reference.
        self.assertEqual(len(StubVina.instances), 4)
        boxes = {tuple(engine.maps[0] + engine.maps[1])
                 for engine in StubVina.instances}
        self.assertEqual(len(boxes), 1, "every pose must share one box setup")
        self.assertEqual({engine.sf_name for engine in StubVina.instances},
                         {"vina"})

    def test_all_heavy_atoms_are_required_in_every_pose(self):
        truncated = self.root / "short.pdbqt"
        block = pose_block(self.coordinates, -9.0)
        keep = [line for line in block
                if not line.startswith("ATOM") or " O1 " not in line]
        truncated.write_text("\n".join(keep) + "\n")
        with self.assertRaises(Exception) as caught:
            rescore.run(self.arguments(deposited=truncated, docked_rank=[],
                                       reference_pose="deposited"))
        self.assertRegex(str(caught.exception), "O1|heavy atom")

    def test_rmsd_is_measured_without_superposition(self):
        """Rank 3 is a rigid 1.25 A shift along x and must read as such."""
        record = rescore.run(self.arguments(docked_rank=[f"3:{self.docked}"],
                                           deposited=None,
                                           reference_pose="docked_rank_3"))
        row = record["poses"][0]
        self.assertAlmostEqual(row["rmsd_to_deposited_angstrom"], 1.25, places=3)
        self.assertEqual(record["measure"]["superposition"], "none")

    def test_the_record_states_that_poses_were_not_modified(self):
        record = rescore.run(self.arguments())
        self.assertFalse(record["experiment"]["poses_were_optimised"])
        self.assertFalse(record["experiment"]["poses_were_modified"])
        self.assertFalse(record["experiment"]["counts_towards_2A_positive_control"])

    def test_the_source_of_the_scoring_conventions_is_recorded(self):
        record = rescore.run(self.arguments())
        self.assertIn("vina.cpp", record["scoring_methods"]["source"])
        self.assertIn("unbound_energy", record["scoring_methods"]["source"])

    def test_check_only_validates_without_scoring(self):
        record = rescore.run(self.arguments(check_only=True))
        self.assertEqual(record["result"], "checks_passed_nothing_scored")
        self.assertEqual(StubVina.instances, [])
        for row in record["poses"]:
            self.assertNotIn("score", row)

    def test_a_pose_outside_the_box_is_refused(self):
        far = {name: (x + 500.0, y, z)
               for name, (x, y, z) in self.coordinates.items()}
        outside = self.root / "far.pdbqt"
        outside.write_text("\n".join(pose_block(far, -9.0)) + "\n")
        with self.assertRaises(rescore.RescoringError) as caught:
            rescore.run(self.arguments(deposited=outside, docked_rank=[],
                                       reference_pose="deposited"))
        self.assertIn("inside the recorded box", str(caught.exception))

    def test_the_provenance_chain_is_enforced_for_rescoring_too(self):
        self.write_preparation(result="failed", failure_reasons=["charge sum"])
        with self.assertRaises(Exception) as caught:
            rescore.run(self.arguments())
        self.assertIn("not 'passed'", str(caught.exception))

    def test_the_table_is_written_with_one_row_per_pose(self):
        record = rescore.run(self.arguments())
        table = self.root / "table.csv"
        rescore.write_table(table, record["poses"])
        lines = table.read_text().strip().splitlines()
        self.assertEqual(len(lines), 4)        # header plus three poses
        self.assertIn("docked_rank_1", lines[2])
        self.assertIn("-9.33", lines[2])       # the recorded affinity survives
        self.assertIn("label,rank_in_file", lines[0])


if __name__ == "__main__":
    unittest.main()
