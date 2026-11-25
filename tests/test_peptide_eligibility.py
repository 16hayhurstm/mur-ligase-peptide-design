#!/usr/bin/env python3
"""Checks for cross-cohort filter equivalence and eligibility.

The fixtures model the 46-versus-47 situation directly: two cohorts whose
passing counts differ by one, with the filter definitions varied one property
at a time so that each way of being non-equivalent is tested on its own. The
"CIF" files are empty or near-empty placeholders, because this module only
establishes that a recorded path exists and is non-empty; structural checks
belong to `peptide_geometry.py`.
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

import peptide_eligibility as pe  # noqa: E402

FILTERS = [
    "pass_has_x_filter",
    "pass_filter_rmsd_filter",
    "pass_filter_rmsd_design_filter",
]
THRESHOLDS = {"refolding_rmsd_threshold": 2.0, "ALA_fraction": 0.3}


def audit(passed, filters=None, rows=1703):
    return {
        "enabled_hard_filters": list(filters if filters is not None else FILTERS),
        "enabled_hard_filter_count": len(filters if filters is not None else FILTERS),
        "metrics_rows": rows,
        "pass_all_hard_filters_recorded": {
            "passed": passed,
            "failed": rows - passed,
            "unparseable": 0,
        },
    }


class FilterComparisonTests(unittest.TestCase):
    def cohorts(self, **overrides):
        base = [
            {
                "label": "anp",
                "enabled_hard_filters": FILTERS,
                "thresholds": dict(THRESHOLDS),
                "from_inverse_folded": True,
            },
            {
                "label": "noanp",
                "enabled_hard_filters": FILTERS,
                "thresholds": dict(THRESHOLDS),
                "from_inverse_folded": True,
            },
        ]
        base[1].update(overrides)
        return base

    def test_identical_definitions_are_equivalent(self):
        result = pe.compare_filter_definitions(self.cohorts())
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["reasons"], [])

    def test_a_differing_filter_set_is_not_equivalent_and_names_the_extra(self):
        extra = FILTERS + ["pass_CYS_fraction_filter"]
        result = pe.compare_filter_definitions(
            self.cohorts(enabled_hard_filters=extra)
        )
        self.assertFalse(result["equivalent"])
        self.assertFalse(result["same_filter_set"])
        self.assertEqual(
            result["filter_set_differences"]["noanp"]["extra"],
            ["pass_CYS_fraction_filter"],
        )

    def test_same_set_different_order_is_flagged_as_ranking_incomparable(self):
        reordered = [FILTERS[1], FILTERS[0], FILTERS[2]]
        result = pe.compare_filter_definitions(
            self.cohorts(enabled_hard_filters=reordered)
        )
        self.assertFalse(result["equivalent"])
        self.assertTrue(result["same_filter_set"])
        self.assertFalse(result["same_filter_order"])
        self.assertTrue(
            any("order differs" in r for r in result["reasons"]),
            result["reasons"],
        )

    def test_differing_thresholds_are_not_equivalent(self):
        result = pe.compare_filter_definitions(
            self.cohorts(thresholds={"refolding_rmsd_threshold": 2.5})
        )
        self.assertFalse(result["equivalent"])
        self.assertFalse(result["same_thresholds"])

    def test_unsupplied_thresholds_are_reported_as_unverified(self):
        result = pe.compare_filter_definitions(self.cohorts(thresholds=None))
        self.assertFalse(result["equivalent"])
        self.assertFalse(result["thresholds_supplied"])
        self.assertTrue(any("unverified" in r for r in result["reasons"]))

    def test_differing_from_inverse_folded_is_called_out_specifically(self):
        result = pe.compare_filter_definitions(
            self.cohorts(from_inverse_folded=False)
        )
        self.assertFalse(result["equivalent"])
        self.assertFalse(result["same_from_inverse_folded"])
        self.assertTrue(
            any("aliases a different metric" in r for r in result["reasons"]),
            result["reasons"],
        )


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.refold = self.tmp / "refold"
        self.refold.mkdir()

    def mapping(self, ids, present=None, empty=(), lengths=None):
        present = set(ids) if present is None else set(present)
        rows = []
        for index, design_id in enumerate(ids):
            name = f"{design_id}.cif"
            path = self.refold / name
            if design_id in present:
                path.write_text("" if design_id in empty else "data_x\n")
            length = (lengths or {}).get(design_id, 8 + index)
            rows.append(
                {
                    "id": design_id,
                    "file_name": name,
                    "designed_sequence": "A" * length,
                    "sequence_length": str(length),
                    "final_rank": str(index + 1),
                    "refold_cif": str(path),
                    "refold_cif_present": str(design_id in present),
                    "refold_cif_sha256": "",
                }
            )
        return rows

    def test_all_present_designs_are_eligible(self):
        rows = self.mapping(["d1", "d2", "d3"])
        result = pe.eligibility_for_cohort("anp", audit(3), rows)
        self.assertEqual(result["summary"]["eligible"], 3)
        self.assertEqual(result["summary"]["excluded"], 0)

    def test_absent_cif_excludes_the_design_with_a_reason(self):
        rows = self.mapping(["d1", "d2"], present=["d1"])
        result = pe.eligibility_for_cohort("anp", audit(2), rows)
        self.assertEqual(result["summary"]["eligible"], 1)
        self.assertEqual(result["excluded"][0]["id"], "d2")
        self.assertIn("absent", result["excluded"][0]["exclusion_reason"])

    def test_empty_cif_is_not_usable(self):
        rows = self.mapping(["d1"], empty=["d1"])
        result = pe.eligibility_for_cohort("anp", audit(1), rows)
        self.assertEqual(result["summary"]["eligible"], 0)
        self.assertIn("empty", result["excluded"][0]["exclusion_reason"])

    def test_a_cif_deleted_since_the_audit_is_detected(self):
        rows = self.mapping(["d1"])
        Path(rows[0]["refold_cif"]).unlink()
        result = pe.eligibility_for_cohort("anp", audit(1), rows)
        self.assertEqual(result["summary"]["eligible"], 0)
        self.assertIn("no longer present", result["excluded"][0]["exclusion_reason"])

    def test_mapping_disagreeing_with_the_audit_count_warns(self):
        rows = self.mapping(["d1", "d2"])
        result = pe.eligibility_for_cohort("anp", audit(47), rows)
        self.assertIn("warning", result["summary"])
        self.assertIn("47", result["summary"]["warning"])


class CombinedReportTests(unittest.TestCase):
    """The 46-versus-47 shape, end to end."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.helper = EligibilityTests()
        self.helper.tmp = self.tmp
        self.helper.refold = self.tmp / "refold"
        self.helper.refold.mkdir()

    def cohort(self, label, count, filters=None, first=0):
        ids = [f"{label}_{i:03d}" for i in range(first, first + count)]
        rows = self.helper.mapping(ids)
        return {
            "label": label,
            "audit": audit(count, filters=filters),
            "mapping": rows,
            "enabled_hard_filters": list(filters if filters is not None else FILTERS),
            "thresholds": dict(THRESHOLDS),
            "from_inverse_folded": True,
        }

    def test_forty_six_plus_forty_seven_is_ninety_three(self):
        report = pe.build_report(
            [self.cohort("anp", 46), self.cohort("noanp", 47, first=100)]
        )
        self.assertTrue(report["filter_comparison"]["equivalent"])
        self.assertEqual(report["total_eligible"], 93)
        self.assertEqual(report["eligible_by_cohort"], {"anp": 46, "noanp": 47})
        self.assertEqual(report["total_excluded"], 0)

    def test_length_distributions_are_reported_per_cohort(self):
        report = pe.build_report(
            [self.cohort("anp", 3), self.cohort("noanp", 3, first=100)]
        )
        self.assertEqual(
            set(report["sequence_length_distribution_by_cohort"]), {"anp", "noanp"}
        )
        self.assertEqual(sum(report["sequence_length_distribution"].values()), 6)

    def test_non_equivalent_filters_are_reported_not_raised(self):
        report = pe.build_report(
            [
                self.cohort("anp", 46),
                self.cohort(
                    "noanp", 47, filters=FILTERS + ["pass_GLY_fraction_filter"], first=100
                ),
            ]
        )
        self.assertFalse(report["filter_comparison"]["equivalent"])
        # The eligible set is still computed, so the difference can be assessed.
        self.assertEqual(report["total_eligible"], 93)

    def test_repeated_cohort_labels_are_fatal(self):
        with self.assertRaises(pe.EligibilityError):
            pe.build_report([self.cohort("anp", 2), self.cohort("anp", 2, first=50)])

    def test_design_ids_shared_between_cohorts_are_surfaced(self):
        a = self.cohort("anp", 2)
        b = self.cohort("noanp", 2)
        b["mapping"] = a["mapping"]  # same IDs in both cohorts
        report = pe.build_report([a, b])
        self.assertEqual(len(report["design_ids_shared_between_cohorts"]), 2)


class MappingLoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def write(self, rows, fields=None):
        path = self.tmp / "m.csv"
        fields = fields or list(pe.MAPPING_FIELDS)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_duplicate_ids_are_fatal(self):
        path = self.write(
            [
                {"id": "d1", "file_name": "a.cif", "refold_cif": "a", "refold_cif_present": "True"},
                {"id": "d1", "file_name": "b.cif", "refold_cif": "b", "refold_cif_present": "True"},
            ]
        )
        with self.assertRaises(pe.EligibilityError):
            pe.load_mapping(path)

    def test_missing_required_column_is_fatal(self):
        path = self.write([{"id": "d1"}], fields=["id"])
        with self.assertRaises(pe.EligibilityError):
            pe.load_mapping(path)

    def test_header_only_mapping_is_fatal(self):
        with self.assertRaises(pe.EligibilityError):
            pe.load_mapping(self.write([]))


class CohortArgumentTests(unittest.TestCase):
    def test_label_and_two_paths_are_split_from_the_right(self):
        parsed = pe.parse_cohort_argument("anp:/a/audit.json:/a/map.csv")
        self.assertEqual(parsed["label"], "anp")
        self.assertEqual(parsed["audit_path"], Path("/a/audit.json"))
        self.assertEqual(parsed["mapping_path"], Path("/a/map.csv"))

    def test_malformed_arguments_are_refused(self):
        import argparse

        for bad in ("anp", "anp:only_one", ":/a/x.json:/a/y.csv"):
            with self.assertRaises(argparse.ArgumentTypeError):
                pe.parse_cohort_argument(bad)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.refold = self.tmp / "refold"
        self.refold.mkdir()
        self.paths = {}
        for label, count in (("anp", 2), ("noanp", 3)):
            rows = []
            for i in range(count):
                name = f"{label}_{i}.cif"
                (self.refold / name).write_text("data_x\n")
                rows.append(
                    {
                        "id": f"{label}_{i}",
                        "file_name": name,
                        "designed_sequence": "A" * (8 + i),
                        "sequence_length": str(8 + i),
                        "final_rank": str(i + 1),
                        "refold_cif": str(self.refold / name),
                        "refold_cif_present": "True",
                        "refold_cif_sha256": "",
                    }
                )
            audit_path = self.tmp / f"{label}_audit.json"
            audit_path.write_text(json.dumps(audit(count)))
            map_path = self.tmp / f"{label}_map.csv"
            with map_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.paths[label] = (audit_path, map_path)
        self.thresholds = self.tmp / "thresholds.json"
        self.thresholds.write_text(json.dumps(THRESHOLDS))

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO / "python" / "peptide_eligibility.py"), *args],
            capture_output=True,
            text=True,
        )

    def cohort_args(self):
        out = []
        for label, (audit_path, map_path) in self.paths.items():
            out += ["--cohort", f"{label}:{audit_path}:{map_path}"]
        return out

    def test_cli_reports_equivalence_and_writes_the_candidate_table(self):
        candidates = self.tmp / "candidates.csv"
        result = self.run_cli(
            *self.cohort_args(),
            "--thresholds", f"anp:{self.thresholds}",
            "--thresholds", f"noanp:{self.thresholds}",
            "--from-inverse-folded", "anp:true",
            "--from-inverse-folded", "noanp:true",
            "--candidates-out", str(candidates),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["filter_comparison"]["equivalent"])
        self.assertEqual(report["total_eligible"], 5)
        with candidates.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertEqual({r["cohort"] for r in rows}, {"anp", "noanp"})

    def test_cli_exits_one_when_definitions_are_unverified(self):
        # No thresholds supplied: equivalence cannot be confirmed.
        result = self.run_cli(*self.cohort_args())
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["filter_comparison"]["equivalent"])

    def test_cli_requires_two_cohorts(self):
        label, (audit_path, map_path) = next(iter(self.paths.items()))
        result = self.run_cli("--cohort", f"{label}:{audit_path}:{map_path}")
        self.assertNotEqual(result.returncode, 0)

    def test_cli_rejects_a_record_that_is_not_an_audit(self):
        bogus = self.tmp / "bogus.json"
        bogus.write_text(json.dumps({"hello": "world"}))
        _, map_path = self.paths["anp"]
        result = self.run_cli(
            "--cohort", f"anp:{bogus}:{map_path}",
            "--cohort", f"noanp:{self.paths['noanp'][0]}:{self.paths['noanp'][1]}",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("enabled_hard_filters", result.stderr)


if __name__ == "__main__":
    unittest.main()
