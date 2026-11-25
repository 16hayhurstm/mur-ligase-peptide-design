#!/usr/bin/env python3
"""Audit a BoltzGen filtering result without modifying it (standard library only).

Purpose
-------
Establish, from the recorded Boolean filter columns rather than from
output-directory file counts, exactly how many designs of a BoltzGen campaign
passed every enabled hard filter, and map each passing design to its refolded
complex CIF.

This module is strictly read-only. It opens files for reading, and writes only
the audit record and mapping table it is explicitly asked to write. It never
touches the campaign's own outputs.

Why the Boolean columns and not the file counts
-----------------------------------------------
In BoltzGen 0.3.2 the "hard filters" do not remove designs. `Filter.filter_df`
adds one `pass_<feature>_filter` column per enabled filter plus the conjunction
`pass_filters`, prints a recommendation if too few designs pass, and then
returns. Ranking and the lazy-greedy diversity selection run over *all*
surviving rows, so the `final_<budget>_designs/` directory contains exactly
`budget` structures whether or not `budget` designs passed. A file count there
therefore measures the budget, not the filtering outcome.

Inputs
------
`all_designs_metrics.csv`, written by `Filter.write_outdir` into
`<outdir>/final_ranked_designs/`. One row per design after duplicate-sequence
removal. Required columns: `id`; the audit additionally uses `pass_filters`,
any `pass_*_filter` columns, `file_name`, `designed_sequence`, `final_rank` and
`num_filters_passed` when present.

Optionally `final_designs_metrics_<budget>.csv` from the same directory, to
establish how many of the budgeted selection actually passed the filters, and
the upstream `aggregate_metrics_*.csv` in the design directory, to recover the
pre-deduplication design total.

Outputs
-------
A JSON audit record and, optionally, a per-design CSV mapping table. Counts are
reported with their missing-value accounting; nothing is silently dropped.

Units, indexing and conventions
-------------------------------
Counts are dimensionless design counts. Design identity is the `id` column,
used as a string throughout; rows are never joined by position. Truth values
are parsed from the exact strings pandas writes for Python/NumPy booleans
("True"/"False"), case-insensitively, plus "1"/"0"; anything else, including an
empty cell or "nan", is recorded as unparseable rather than coerced to False.

Assumptions
-----------
- A `pass_*_filter` column corresponds to one enabled hard filter, and
  `pass_filters` is their conjunction. The audit verifies the second of these
  rather than assuming it.
- The refolded complex for a design is
  `<design_dir>/refold_cif/<file_name>`, which is the file
  `Filter.write_outdir` copies into the selected-design directories and the
  file `build_visualisation.py` treats as the post-refold structure.

Limitations
-----------
This audit establishes what the filtering recorded. It does not revalidate any
metric, recompute any RMSD, inspect any structure's chemistry, or establish
that the thresholds used were scientifically appropriate. A design passing
every filter is not a validated binder.

Failure conditions
------------------
Raises `AuditError` on a missing or unreadable metrics CSV, a missing `id`
column, duplicate design IDs, an unsafe `file_name`, or a `pass_filters` column
that disagrees with the conjunction of the per-filter columns. Disagreement is
an error and not a warning, because every downstream count depends on it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

METRICS_BASENAME = "all_designs_metrics.csv"
RANKED_SUBDIR = "final_ranked_designs"
DEFAULT_DESIGN_SUBDIR = "intermediate_designs_inverse_folded"
REFOLD_SUBDIR = "refold_cif"

# Accepted spellings of a recorded Boolean. Anything else is "unparseable",
# which is reported rather than treated as a failure to pass.
_TRUE_TOKENS = frozenset({"true", "1"})
_FALSE_TOKENS = frozenset({"false", "0"})

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")


class AuditError(Exception):
    """A condition that makes the audited counts untrustworthy."""


def parse_truth(value: Optional[str]) -> Optional[bool]:
    """Parse a recorded Boolean, returning None when the cell is not a Boolean.

    None is a distinct third outcome, not a synonym for False: a design with an
    unparseable filter cell has no recorded verdict, and conflating that with
    failure would quietly change the pass count.
    """
    token = str(value).strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return None


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metrics(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a metrics CSV, returning its field names and rows.

    Validates the properties every later count depends on: the file parses, it
    is nonempty, it carries an `id` column, and the IDs are unique. Duplicate
    IDs are fatal because the design-to-structure mapping would be ambiguous.
    """
    if not path.is_file():
        raise AuditError(f"Metrics CSV not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise AuditError(f"Metrics CSV contains no data rows: {path}")
    if "id" not in fields:
        raise AuditError(f"Metrics CSV has no 'id' column: {path}")
    ids = [row["id"] for row in rows]
    duplicates = sorted({x for x, n in Counter(ids).items() if n > 1})
    if duplicates:
        raise AuditError(
            f"Duplicate design IDs in {path}: " + ", ".join(duplicates[:10])
        )
    return fields, rows


def filter_columns(fields: Sequence[str]) -> List[str]:
    """Return the per-filter Boolean columns, in their recorded order.

    The recorded order matters: BoltzGen's `num_filters_passed` is accumulated
    inside the filter loop and is therefore order-dependent (see
    `recount_filters_passed`).
    """
    return [f for f in fields if f.startswith("pass_") and f.endswith("_filter")]


def count_column(rows: Iterable[Dict[str, str]], column: str) -> Dict[str, int]:
    """Count True, False and unparseable cells in one Boolean column."""
    counts = Counter(parse_truth(row.get(column)) for row in rows)
    return {
        "passed": counts[True],
        "failed": counts[False],
        "unparseable": counts[None],
    }


def verify_conjunction(
    rows: Sequence[Dict[str, str]], columns: Sequence[str]
) -> Dict[str, object]:
    """Recompute `pass_filters` as the conjunction of the per-filter columns.

    This is an independent check of the figure the audit reports, not a restatement
    of it: the conjunction is recomputed here from the per-filter columns and
    compared against what BoltzGen recorded. A design is counted as passing only
    when every per-filter cell parses as True; an unparseable cell makes the
    recomputed verdict None, which can never equal a recorded True or False.

    Returns the recomputed pass count, the recorded pass count, and the IDs
    where the two disagree.
    """
    recomputed_pass = 0
    recomputed_unknown = 0
    disagreements: List[Dict[str, str]] = []
    for row in rows:
        verdicts = [parse_truth(row.get(c)) for c in columns]
        if None in verdicts:
            recomputed = None
            recomputed_unknown += 1
        else:
            recomputed = all(verdicts)
            if recomputed:
                recomputed_pass += 1
        recorded = parse_truth(row.get("pass_filters"))
        if recorded != recomputed:
            disagreements.append(
                {
                    "id": row["id"],
                    "recorded_pass_filters": str(row.get("pass_filters")),
                    "recomputed": str(recomputed),
                }
            )
    return {
        "recomputed_pass_all": recomputed_pass,
        "recomputed_unparseable": recomputed_unknown,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[:20],
    }


def recount_filters_passed(
    rows: Sequence[Dict[str, str]], columns: Sequence[str]
) -> Dict[str, object]:
    """Compare a true per-design filter-pass count with BoltzGen's recorded one.

    BoltzGen 0.3.2 accumulates `num_filters_passed` inside its filter loop as

        num_filters_passed += df[filter_cols].all(axis=1)

    where `filter_cols` grows by one column per iteration. The sum is therefore
    the number of leading prefixes of the filter list that are entirely true,
    i.e. the length of the design's leading run of passed filters, not the
    number of filters it passed. A design failing the first filter scores 0
    however many later filters it passes, and the value depends on the order in
    which the filters were configured.

    This matters because `num_filters_passed` is the primary key of the ranking
    in `Filter.sort_df`. Designs are ordered by the tuple
    (num_filters_passed, metric), so an order-dependent quantity partly
    determines which designs reach the budgeted selection.

    Returns the distribution of both quantities and how many designs differ,
    counting only designs whose per-filter cells all parse.
    """
    true_counts: Counter = Counter()
    prefix_counts: Counter = Counter()
    recorded_mismatch = 0
    recorded_comparable = 0
    differ = 0
    for row in rows:
        verdicts = [parse_truth(row.get(c)) for c in columns]
        if None in verdicts:
            continue
        true_count = sum(1 for v in verdicts if v)
        prefix_run = 0
        for verdict in verdicts:
            if not verdict:
                break
            prefix_run += 1
        true_counts[true_count] += 1
        prefix_counts[prefix_run] += 1
        if true_count != prefix_run:
            differ += 1
        recorded = row.get("num_filters_passed")
        if recorded not in (None, ""):
            try:
                recorded_value = int(float(recorded))
            except (TypeError, ValueError):
                continue
            recorded_comparable += 1
            if recorded_value != prefix_run:
                recorded_mismatch += 1
    return {
        "true_count_distribution": dict(sorted(true_counts.items())),
        "prefix_run_distribution": dict(sorted(prefix_counts.items())),
        "designs_where_the_two_differ": differ,
        "recorded_rows_compared": recorded_comparable,
        "recorded_disagrees_with_prefix_model": recorded_mismatch,
    }


def refold_path(design_dir: Path, file_name: str) -> Path:
    """Return the refolded-complex CIF path for one design's recorded filename.

    Rejects any `file_name` that is not a plain basename, so a crafted or
    corrupted metrics row cannot direct the audit outside the design directory.
    """
    if Path(file_name).name != file_name or not file_name.endswith(".cif"):
        raise AuditError(f"Unsafe or unexpected file_name: {file_name!r}")
    return design_dir / REFOLD_SUBDIR / file_name


def map_passing_designs(
    rows: Sequence[Dict[str, str]],
    design_dir: Path,
    with_hashes: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Map every design that passed all filters to its refolded complex CIF.

    Returns one record per passing design and a summary of how many refolded
    structures were present. A missing structure is recorded against its design
    ID and counted; it does not abort the audit, because the complete list of
    missing designs is more useful than the first one.
    """
    records: List[Dict[str, str]] = []
    present = 0
    missing = 0
    for row in rows:
        if parse_truth(row.get("pass_filters")) is not True:
            continue
        design_id = row["id"]
        if not _SAFE_ID.fullmatch(design_id) or design_id in (".", ".."):
            raise AuditError(f"Unsafe design id: {design_id!r}")
        file_name = row.get("file_name") or f"{design_id}.cif"
        path = refold_path(design_dir, file_name)
        exists = path.is_file()
        present += int(exists)
        missing += int(not exists)
        record = {
            "id": design_id,
            "file_name": file_name,
            "final_rank": row.get("final_rank", ""),
            "designed_sequence": row.get("designed_sequence", ""),
            "sequence_length": str(len(row.get("designed_sequence", "") or "")),
            "refold_cif": str(path),
            "refold_cif_present": str(exists),
        }
        if with_hashes and exists:
            record["refold_cif_sha256"] = sha256_of(path)
        records.append(record)
    return records, {"present": present, "missing": missing}


def audit_selected_set(
    selected_path: Path, passing_ids: Sequence[str]
) -> Dict[str, object]:
    """Report how many designs in a budgeted selection actually passed the filters.

    This is the figure that distinguishes "the budget was 46" from "46 designs
    passed". Because BoltzGen's filters do not exclude designs, a budgeted
    selection may contain designs that failed one or more hard filters.
    """
    _, rows = read_metrics(selected_path)
    passing = set(passing_ids)
    verdicts = Counter(parse_truth(row.get("pass_filters")) for row in rows)
    in_passing_set = sum(1 for row in rows if row["id"] in passing)
    failing_ids = [
        row["id"] for row in rows if parse_truth(row.get("pass_filters")) is False
    ]
    return {
        "path": str(selected_path),
        "sha256": sha256_of(selected_path),
        "selected_rows": len(rows),
        "selected_recorded_pass": verdicts[True],
        "selected_recorded_fail": verdicts[False],
        "selected_recorded_unparseable": verdicts[None],
        "selected_also_in_passing_set": in_passing_set,
        "selected_but_failing_ids": failing_ids,
    }


def count_upstream_designs(design_dir: Path) -> Dict[str, object]:
    """Recover the pre-deduplication design total from the upstream metrics CSV.

    BoltzGen reads `sorted(design_dir.glob("aggregate_metrics_*.csv"))[0]` and
    ignores any further matches, so the number of matching files is reported
    here: more than one means part of the campaign was never filtered.
    """
    candidates = sorted(design_dir.glob("aggregate_metrics_*.csv"))
    summary: Dict[str, object] = {
        "aggregate_metrics_files_found": [str(p) for p in candidates],
        "note": (
            "BoltzGen 0.3.2 reads only the first of these, sorted by name; "
            "any others were silently ignored."
        ),
    }
    if not candidates:
        summary["designs_before_deduplication"] = None
        return summary
    used = candidates[0]
    with used.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sequences = [r.get("designed_sequence", "") for r in rows]
    summary["used_file"] = str(used)
    summary["used_file_sha256"] = sha256_of(used)
    summary["designs_before_deduplication"] = len(rows)
    summary["distinct_designed_sequences"] = len(set(sequences))
    summary["duplicate_sequences_removed"] = len(rows) - len(set(sequences))
    return summary


def build_audit(
    run_dir: Path,
    design_dir: Optional[Path],
    metrics_path: Optional[Path],
    selected_path: Optional[Path],
    with_hashes: bool,
) -> Tuple[Dict[str, object], List[Dict[str, str]]]:
    """Assemble the complete audit record for one campaign."""
    run_dir = run_dir.resolve()
    ranked_dir = run_dir if run_dir.name == RANKED_SUBDIR else run_dir / RANKED_SUBDIR
    metrics = (metrics_path or ranked_dir / METRICS_BASENAME).resolve()
    designs = (design_dir or run_dir / DEFAULT_DESIGN_SUBDIR).resolve()

    fields, rows = read_metrics(metrics)
    columns = filter_columns(fields)
    if not columns:
        raise AuditError(
            f"No 'pass_*_filter' columns in {metrics}; the enabled hard filters "
            "cannot be established from this file."
        )
    if "pass_filters" not in fields:
        raise AuditError(f"No 'pass_filters' column in {metrics}")

    conjunction = verify_conjunction(rows, columns)
    if conjunction["disagreement_count"]:
        raise AuditError(
            f"'pass_filters' disagrees with the conjunction of {len(columns)} "
            f"per-filter columns for {conjunction['disagreement_count']} designs "
            f"in {metrics}. The pass count cannot be trusted. First few: "
            f"{conjunction['disagreements']}"
        )

    recorded = count_column(rows, "pass_filters")
    records, structures = map_passing_designs(rows, designs, with_hashes)

    sequences = [r.get("designed_sequence", "") for r in rows]
    lengths = Counter(len(s) for s in sequences if s)
    passing_lengths = Counter(int(r["sequence_length"]) for r in records if r["sequence_length"] != "0")

    audit: Dict[str, object] = {
        "audit_tool": "python/audit_filtering.py",
        "read_only": True,
        "run_dir": str(run_dir),
        "ranked_dir": str(ranked_dir),
        "design_dir": str(designs),
        "metrics_csv": str(metrics),
        "metrics_csv_sha256": sha256_of(metrics),
        "metrics_rows": len(rows),
        "enabled_hard_filters": columns,
        "enabled_hard_filter_count": len(columns),
        "per_filter_counts": {c: count_column(rows, c) for c in columns},
        "pass_all_hard_filters_recorded": recorded,
        "pass_all_hard_filters_recomputed": conjunction["recomputed_pass_all"],
        "num_filters_passed_audit": recount_filters_passed(rows, columns),
        "refolded_structures_for_passing_designs": structures,
        "sequence_length_distribution_all": dict(sorted(lengths.items())),
        "sequence_length_distribution_passing": dict(sorted(passing_lengths.items())),
        "upstream": count_upstream_designs(designs),
        "interpretation": (
            "pass_all_hard_filters_recorded['passed'] is the number of designs "
            "passing every enabled hard filter. BoltzGen 0.3.2 does not remove "
            "failing designs, so the size of any final_<budget>_designs "
            "directory reflects the budget and not this number. Passing every "
            "filter is not experimental validation of binding."
        ),
    }
    if selected_path is not None:
        audit["budgeted_selection"] = audit_selected_set(
            selected_path.resolve(), [r["id"] for r in records]
        )
    return audit, records


def write_mapping_csv(path: Path, records: Sequence[Dict[str, str]]) -> None:
    """Write the passing-design to refolded-structure mapping table."""
    fields = [
        "id",
        "file_name",
        "final_rank",
        "designed_sequence",
        "sequence_length",
        "refold_cif",
        "refold_cif_present",
    ]
    if records and "refold_cif_sha256" in records[0]:
        fields.append("refold_cif_sha256")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Campaign output directory containing final_ranked_designs/, "
        "or that directory itself",
    )
    parser.add_argument(
        "--design-dir",
        type=Path,
        help=f"Design directory holding {REFOLD_SUBDIR}/ and "
        f"aggregate_metrics_*.csv (default: <run_dir>/{DEFAULT_DESIGN_SUBDIR})",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help=f"Explicit metrics CSV (default: <ranked_dir>/{METRICS_BASENAME})",
    )
    parser.add_argument(
        "--selected",
        type=Path,
        help="A final_designs_metrics_<budget>.csv, to report how many "
        "budget-selected designs actually passed the filters",
    )
    parser.add_argument(
        "--hash-structures",
        action="store_true",
        help="Record a SHA-256 for each passing design's refolded CIF",
    )
    parser.add_argument(
        "--json-out", type=Path, help="Write the audit record to this new file"
    )
    parser.add_argument(
        "--mapping-out",
        type=Path,
        help="Write the passing-design to refolded-CIF mapping to this new CSV",
    )
    args = parser.parse_args(argv)

    for destination in (args.json_out, args.mapping_out):
        if destination is not None and destination.exists():
            parser.error(f"Refusing to overwrite an existing file: {destination}")

    try:
        audit, records = build_audit(
            args.run_dir,
            args.design_dir,
            args.metrics,
            args.selected,
            args.hash_structures,
        )
    except AuditError as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(audit, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(audit, indent=2) + "\n")
        print(f"audit record: {args.json_out}", file=sys.stderr)
    if args.mapping_out is not None:
        args.mapping_out.parent.mkdir(parents=True, exist_ok=True)
        write_mapping_csv(args.mapping_out, records)
        print(f"mapping table: {args.mapping_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
