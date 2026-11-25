#!/usr/bin/env python3
"""Checks for the BoltzGen filtering audit.

The fixtures are tiny hand-written metrics tables whose correct answers can be
read off by eye, so a failure indicates a defect in the audit rather than a
disagreement between two copies of the same logic. Nothing here parses a
molecular structure: the "CIF" files are empty placeholders, because the audit
only establishes a filename's presence and digest.

The scientifically important cases are those where a count could silently go
wrong: an unparseable Boolean cell, a recorded conjunction that disagrees with
its parts, the order-dependence of BoltzGen's `num_filters_passed`, and a
budgeted selection containing designs that failed a hard filter.
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

import audit_filtering as af  # noqa: E402

FILTER_COLUMNS = ["pass_has_x_filter", "pass_filter_rmsd_filter", "pass_CYS_fraction_filter"]
FIELDS = ["id", "final_rank", "designed_sequence", "file_name"] + FILTER_COLUMNS + [
    "pass_filters",
    "num_filters_passed",
]


def row(design_id, sequence, verdicts, pass_filters, num_passed, file_name=None):
    """Build one metrics row; `verdicts` are the three per-filter cells."""
    record = {
        "id": design_id,
        "final_rank": "1",
        "designed_sequence": sequence,
        "file_name": file_name if file_name is not None else f"{design_id}.cif",
        "pass_filters": pass_filters,
        "num_filters_passed": num_passed,
    }
    record.update(dict(zip(FILTER_COLUMNS, verdicts)))
    return record


def write_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TruthParsingTests(unittest.TestCase):
    """An unrecognised cell must not be silently read as a failure."""

    def test_recognised_spellings(self):
        for token in ("True", "true", "TRUE", "1"):
            self.assertIs(af.parse_truth(token), True, token)
        for token in ("False", "false", "0"):
            self.assertIs(af.parse_truth(token), False, token)

    def test_unparseable_is_none_not_false(self):
        for token in ("", "nan", "NA", None, "maybe"):
            self.assertIsNone(af.parse_truth(token), repr(token))


class FilterColumnDiscoveryTests(unittest.TestCase):
    def test_order_is_preserved_and_only_filter_columns_match(self):
        fields = ["id", "pass_b_filter", "pass_filters", "pass_a_filter", "passing"]
        self.assertEqual(
            af.filter_columns(fields), ["pass_b_filter", "pass_a_filter"]
        )

    def test_pass_filters_itself_is_not_a_per_filter_column(self):
        self.assertEqual(af.filter_columns(["pass_filters"]), [])


class ConjunctionVerificationTests(unittest.TestCase):
    """`pass_filters` must equal the AND of its parts, or the audit must refuse."""

    def test_agreement_counts_only_all_true_designs(self):
        rows = [
            row("d1", "AAA", ("True", "True", "True"), "True", "3"),
            row("d2", "AAB", ("True", "False", "True"), "False", "1"),
            row("d3", "AAC", ("False", "True", "True"), "False", "0"),
        ]
        result = af.verify_conjunction(rows, FILTER_COLUMNS)
        self.assertEqual(result["recomputed_pass_all"], 1)
        self.assertEqual(result["disagreement_count"], 0)

    def test_disagreement_is_reported_with_the_offending_id(self):
        rows = [row("d1", "AAA", ("True", "False", "True"), "True", "1")]
        result = af.verify_conjunction(rows, FILTER_COLUMNS)
        self.assertEqual(result["disagreement_count"], 1)
        self.assertEqual(result["disagreements"][0]["id"], "d1")

    def test_unparseable_cell_blocks_a_pass_rather_than_counting_one(self):
        rows = [row("d1", "AAA", ("True", "", "True"), "", "0")]
        result = af.verify_conjunction(rows, FILTER_COLUMNS)
        self.assertEqual(result["recomputed_pass_all"], 0)
        self.assertEqual(result["recomputed_unparseable"], 1)
        # A blank recorded cell parses to None, which matches the recomputed
        # None, so this is an absence of a verdict and not a contradiction.
        self.assertEqual(result["disagreement_count"], 0)


class FiltersPassedCountTests(unittest.TestCase):
    """BoltzGen's `num_filters_passed` counts a leading run, not a total."""

    def test_leading_run_differs_from_the_true_count(self):
        # Fails the first filter, passes the other two: true count 2, run 0.
        rows = [row("d1", "AAA", ("False", "True", "True"), "False", "0")]
        result = af.recount_filters_passed(rows, FILTER_COLUMNS)
        self.assertEqual(result["true_count_distribution"], {2: 1})
        self.assertEqual(result["prefix_run_distribution"], {0: 1})
        self.assertEqual(result["designs_where_the_two_differ"], 1)

    def test_a_design_passing_everything_agrees_under_both_definitions(self):
        rows = [row("d1", "AAA", ("True", "True", "True"), "True", "3")]
        result = af.recount_filters_passed(rows, FILTER_COLUMNS)
        self.assertEqual(result["true_count_distribution"], {3: 1})
        self.assertEqual(result["prefix_run_distribution"], {3: 1})
        self.assertEqual(result["designs_where_the_two_differ"], 0)

    def test_recorded_value_is_compared_against_the_prefix_model(self):
        # Recorded 2 while the leading run is 1: the recorded column does not
        # follow the prefix model, which is itself worth knowing.
        rows = [row("d1", "AAA", ("True", "False", "True"), "False", "2")]
        result = af.recount_filters_passed(rows, FILTER_COLUMNS)
        self.assertEqual(result["recorded_rows_compared"], 1)
        self.assertEqual(result["recorded_disagrees_with_prefix_model"], 1)


class RefoldPathTests(unittest.TestCase):
    def test_basename_is_resolved_under_the_refold_subdirectory(self):
        path = af.refold_path(Path("/designs"), "design_7.cif")
        self.assertEqual(path, Path("/designs/refold_cif/design_7.cif"))

    def test_a_path_escaping_the_design_directory_is_refused(self):
        for name in ("../design.cif", "sub/design.cif", "/abs/design.cif"):
            with self.assertRaises(af.AuditError):
                af.refold_path(Path("/designs"), name)

    def test_a_non_cif_name_is_refused(self):
        with self.assertRaises(af.AuditError):
            af.refold_path(Path("/designs"), "design.pdb")


class MetricsReadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_duplicate_ids_are_fatal(self):
        path = self.tmp / "m.csv"
        write_csv(
            path,
            [
                row("d1", "AAA", ("True",) * 3, "True", "3"),
                row("d1", "AAB", ("True",) * 3, "True", "3"),
            ],
        )
        with self.assertRaises(af.AuditError):
            af.read_metrics(path)

    def test_missing_id_column_is_fatal(self):
        path = self.tmp / "m.csv"
        write_csv(path, [{"x": "1"}], fields=["x"])
        with self.assertRaises(af.AuditError):
            af.read_metrics(path)

    def test_header_only_file_is_fatal(self):
        path = self.tmp / "m.csv"
        write_csv(path, [])
        with self.assertRaises(af.AuditError):
            af.read_metrics(path)


class EndToEndAuditTests(unittest.TestCase):
    """A whole small campaign, with the pass count known by inspection."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run = self.tmp / "run"
        self.design = self.run / af.DEFAULT_DESIGN_SUBDIR
        self.refold = self.design / af.REFOLD_SUBDIR
        self.refold.mkdir(parents=True)
        # Four designs; exactly two pass all three filters, and one of those
        # two is missing its refolded structure.
        self.rows = [
            row("d1", "AAAA", ("True", "True", "True"), "True", "3"),
            row("d2", "AAAAAA", ("True", "True", "True"), "True", "3"),
            row("d3", "AAA", ("True", "False", "True"), "False", "1"),
            row("d4", "AA", ("False", "True", "True"), "False", "0"),
        ]
        write_csv(self.run / af.RANKED_SUBDIR / af.METRICS_BASENAME, self.rows)
        (self.refold / "d1.cif").write_text("")
        # d2.cif deliberately absent.

    def audit(self, **kwargs):
        return af.build_audit(
            self.run, None, None, kwargs.get("selected"), kwargs.get("hashes", False)
        )

    def test_pass_count_is_the_conjunction_not_the_per_filter_maximum(self):
        audit, records = self.audit()
        self.assertEqual(audit["metrics_rows"], 4)
        self.assertEqual(audit["pass_all_hard_filters_recorded"]["passed"], 2)
        self.assertEqual(audit["pass_all_hard_filters_recomputed"], 2)
        # The loosest single filter passes all four, which is exactly the
        # figure that must not be mistaken for the overall pass count.
        self.assertEqual(
            audit["per_filter_counts"]["pass_CYS_fraction_filter"]["passed"], 4
        )
        self.assertEqual(len(records), 2)
        self.assertEqual([r["id"] for r in records], ["d1", "d2"])

    def test_a_missing_refolded_structure_is_recorded_not_raised(self):
        audit, records = self.audit()
        self.assertEqual(
            audit["refolded_structures_for_passing_designs"],
            {"present": 1, "missing": 1},
        )
        by_id = {r["id"]: r for r in records}
        self.assertEqual(by_id["d1"]["refold_cif_present"], "True")
        self.assertEqual(by_id["d2"]["refold_cif_present"], "False")

    def test_enabled_filters_are_reported_in_recorded_order(self):
        audit, _ = self.audit()
        self.assertEqual(audit["enabled_hard_filters"], FILTER_COLUMNS)
        self.assertEqual(audit["enabled_hard_filter_count"], 3)

    def test_hashes_are_recorded_only_for_present_structures(self):
        _, records = self.audit(hashes=True)
        by_id = {r["id"]: r for r in records}
        empty_sha = (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        )
        self.assertEqual(by_id["d1"]["refold_cif_sha256"], empty_sha)
        self.assertNotIn("refold_cif_sha256", by_id["d2"])

    def test_budgeted_selection_may_contain_failing_designs(self):
        # A budget of three over four designs, two of which pass: the third
        # slot is necessarily filled by a design that failed a hard filter.
        selected = self.run / af.RANKED_SUBDIR / "final_designs_metrics_3.csv"
        write_csv(selected, [self.rows[0], self.rows[1], self.rows[2]])
        audit, _ = self.audit(selected=selected)
        summary = audit["budgeted_selection"]
        self.assertEqual(summary["selected_rows"], 3)
        self.assertEqual(summary["selected_recorded_pass"], 2)
        self.assertEqual(summary["selected_recorded_fail"], 1)
        self.assertEqual(summary["selected_but_failing_ids"], ["d3"])

    def test_sequence_length_distributions_separate_all_from_passing(self):
        audit, _ = self.audit()
        self.assertEqual(
            audit["sequence_length_distribution_all"], {2: 1, 3: 1, 4: 1, 6: 1}
        )
        self.assertEqual(audit["sequence_length_distribution_passing"], {4: 1, 6: 1})

    def test_recorded_conjunction_disagreement_aborts_the_audit(self):
        bad = [row("d1", "AAAA", ("True", "False", "True"), "True", "3")]
        write_csv(self.run / af.RANKED_SUBDIR / af.METRICS_BASENAME, bad)
        with self.assertRaises(af.AuditError) as caught:
            self.audit()
        self.assertIn("disagrees", str(caught.exception))

    def test_metrics_without_filter_columns_is_refused(self):
        plain = self.run / "plain.csv"
        write_csv(plain, [{"id": "d1", "pass_filters": "True"}],
                  fields=["id", "pass_filters"])
        with self.assertRaises(af.AuditError):
            af.build_audit(self.run, None, plain, None, False)


class UpstreamCountTests(unittest.TestCase):
    """Duplicate-sequence removal and the single-file read must be visible."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.design = self.tmp / "designs"
        self.design.mkdir()

    def test_duplicate_sequences_are_counted(self):
        path = self.design / "aggregate_metrics_0.csv"
        write_csv(
            path,
            [
                {"id": "a", "designed_sequence": "AAA"},
                {"id": "b", "designed_sequence": "AAA"},
                {"id": "c", "designed_sequence": "BBB"},
            ],
            fields=["id", "designed_sequence"],
        )
        summary = af.count_upstream_designs(self.design)
        self.assertEqual(summary["designs_before_deduplication"], 3)
        self.assertEqual(summary["distinct_designed_sequences"], 2)
        self.assertEqual(summary["duplicate_sequences_removed"], 1)

    def test_extra_aggregate_files_are_listed_because_only_one_is_read(self):
        for name in ("aggregate_metrics_0.csv", "aggregate_metrics_1.csv"):
            write_csv(
                self.design / name,
                [{"id": "a", "designed_sequence": "AAA"}],
                fields=["id", "designed_sequence"],
            )
        summary = af.count_upstream_designs(self.design)
        self.assertEqual(len(summary["aggregate_metrics_files_found"]), 2)
        self.assertTrue(summary["used_file"].endswith("aggregate_metrics_0.csv"))

    def test_absent_upstream_file_is_reported_as_unknown(self):
        summary = af.count_upstream_designs(self.design)
        self.assertIsNone(summary["designs_before_deduplication"])
        self.assertEqual(summary["aggregate_metrics_files_found"], [])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run = self.tmp / "run"
        refold = self.run / af.DEFAULT_DESIGN_SUBDIR / af.REFOLD_SUBDIR
        refold.mkdir(parents=True)
        write_csv(
            self.run / af.RANKED_SUBDIR / af.METRICS_BASENAME,
            [
                row("d1", "AAAA", ("True", "True", "True"), "True", "3"),
                row("d2", "AAA", ("True", "False", "True"), "False", "1"),
            ],
        )
        (refold / "d1.cif").write_text("")

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(REPO / "python" / "audit_filtering.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_cli_reports_the_pass_count_and_writes_both_records(self):
        json_out = self.tmp / "audit.json"
        mapping_out = self.tmp / "mapping.csv"
        result = self.run_cli(
            str(self.run), "--json-out", str(json_out), "--mapping-out", str(mapping_out)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        audit = json.loads(result.stdout)
        self.assertEqual(audit["pass_all_hard_filters_recorded"]["passed"], 1)
        self.assertTrue(audit["read_only"])
        self.assertEqual(json.loads(json_out.read_text())["metrics_rows"], 2)
        with mapping_out.open(newline="") as handle:
            mapped = list(csv.DictReader(handle))
        self.assertEqual([r["id"] for r in mapped], ["d1"])

    def test_cli_refuses_to_overwrite_an_existing_record(self):
        json_out = self.tmp / "audit.json"
        json_out.write_text("keep me")
        result = self.run_cli(str(self.run), "--json-out", str(json_out))
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json_out.read_text(), "keep me")

    def test_cli_exits_non_zero_on_a_missing_metrics_csv(self):
        result = self.run_cli(str(self.tmp / "absent"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("audit failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
