#!/usr/bin/env python3
"""Verify two campaigns' filtering are comparable and build the eligible set.

Purpose
-------
Two BoltzGen campaigns are to be compared, so the designs drawn from them must
have been judged by the same rules. This module answers three questions, using
the records written by `audit_filtering.py`:

1. **Are the hard-filter definitions equivalent?** Same filters, in the same
   order, with the same thresholds and the same `from_inverse_folded` setting.
   Anything less and a pass in one campaign does not mean a pass in the other.
2. **Which designs are eligible?** A design is eligible when it passed every
   enabled hard filter *and* exactly one usable refolded complex CIF exists
   for it. Both conditions are required; neither is assumed.
3. **What is the combined candidate set?** One table, carrying the cohort
   label, the stable design ID and the refolded CIF path, from which a pilot
   can be drawn.

Why filter order matters
------------------------
BoltzGen 0.3.2 accumulates `num_filters_passed` inside its filter loop, so the
value is the length of a design's leading run of passed filters and depends on
the configured order. It does not affect `pass_filters`, which is the full
conjunction, but it does affect the ranking and therefore which designs were
historically selected. Two campaigns whose filter *sets* agree but whose
*orders* differ have comparable pass counts and incomparable rankings, so the
order is reported separately from the set.

Units and conventions
---------------------
Counts are dimensionless design counts. Design identity is the `id` string.
Cohort labels are caller-supplied and carried through unchanged. Nothing is
joined by row order.

Assumptions
-----------
- The audit records and mapping CSVs were produced by `audit_filtering.py`
  against the campaigns' own `all_designs_metrics.csv`.
- Thresholds are supplied by the caller from each campaign's saved filtering
  configuration, because they are not recorded in the metrics CSV.

Limitations
-----------
Equivalent filter definitions make two campaigns comparable; they do not make
either campaign correct. Eligibility is a bookkeeping property: an eligible
design is one that can be prepared, not one that binds. This module reads no
structure beyond checking that the recorded path exists and is non-empty;
`peptide_geometry.py` performs the structural validation.

Failure conditions
------------------
Raises `EligibilityError` on an unreadable record, a mapping CSV whose design
IDs are duplicated or absent from the audit, or a cohort label reused for two
campaigns. Filter-definition differences are *reported*, not raised, because
the whole point is to describe them so a common protocol can be agreed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MAPPING_FIELDS = ("id", "file_name", "refold_cif", "refold_cif_present")


class EligibilityError(ValueError):
    """A condition that makes the eligible set untrustworthy."""


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise EligibilityError(f"Record not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise EligibilityError(f"Could not parse {path}: {error}") from error


def load_mapping(path: Path) -> List[Dict[str, str]]:
    """Read a passing-design mapping CSV and check its IDs are unique."""
    if not path.is_file():
        raise EligibilityError(f"Mapping CSV not found: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise EligibilityError(f"Mapping CSV has no rows: {path}")
    for field in MAPPING_FIELDS:
        if field not in rows[0]:
            raise EligibilityError(f"Mapping CSV {path} has no {field!r} column")
    duplicates = sorted(
        {x for x, n in Counter(r["id"] for r in rows).items() if n > 1}
    )
    if duplicates:
        raise EligibilityError(
            f"Duplicate design IDs in {path}: " + ", ".join(duplicates[:10])
        )
    return rows


def compare_filter_definitions(
    cohorts: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Compare the enabled hard filters across cohorts.

    Reports set equality and order equality separately, and compares the
    caller-supplied thresholds and `from_inverse_folded` values. The verdict is
    `equivalent` only when every one of these agrees.
    """
    ordered = {c["label"]: list(c["enabled_hard_filters"]) for c in cohorts}
    as_sets = {label: set(values) for label, values in ordered.items()}
    reference_label = cohorts[0]["label"]
    reference_set = as_sets[reference_label]
    reference_order = ordered[reference_label]

    same_set = all(values == reference_set for values in as_sets.values())
    same_order = all(values == reference_order for values in ordered.values())

    only_in: Dict[str, List[str]] = {}
    for label, values in as_sets.items():
        extra = sorted(values - reference_set)
        missing = sorted(reference_set - values)
        if extra or missing:
            only_in[label] = {"extra": extra, "absent": missing}

    thresholds = {c["label"]: c.get("thresholds") for c in cohorts}
    stated = [t for t in thresholds.values() if t is not None]
    thresholds_supplied = len(stated) == len(cohorts)
    same_thresholds = thresholds_supplied and all(t == stated[0] for t in stated)

    inverse_folded = {c["label"]: c.get("from_inverse_folded") for c in cohorts}
    declared = [v for v in inverse_folded.values() if v is not None]
    inverse_supplied = len(declared) == len(cohorts)
    same_inverse = inverse_supplied and all(v == declared[0] for v in declared)

    equivalent = bool(same_set and same_order and same_thresholds and same_inverse)
    reasons: List[str] = []
    if not same_set:
        reasons.append("the enabled hard-filter sets differ")
    if same_set and not same_order:
        reasons.append(
            "the filter sets agree but their order differs, so pass counts are "
            "comparable while num_filters_passed and the historical rankings "
            "are not"
        )
    if not thresholds_supplied:
        reasons.append(
            "thresholds were not supplied for every cohort, so threshold "
            "equality is unverified; read them from each saved filtering config"
        )
    elif not same_thresholds:
        reasons.append("the thresholds differ")
    if not inverse_supplied:
        reasons.append("from_inverse_folded was not supplied for every cohort")
    elif not same_inverse:
        reasons.append(
            "from_inverse_folded differs, so filter_rmsd aliases a different "
            "metric in each campaign and the RMSD filters are not the same test"
        )

    return {
        "equivalent": equivalent,
        "same_filter_set": same_set,
        "same_filter_order": same_order,
        "thresholds_supplied": thresholds_supplied,
        "same_thresholds": bool(same_thresholds),
        "from_inverse_folded_supplied": inverse_supplied,
        "same_from_inverse_folded": bool(same_inverse),
        "enabled_hard_filters_by_cohort": ordered,
        "filter_set_differences": only_in,
        "thresholds_by_cohort": thresholds,
        "from_inverse_folded_by_cohort": inverse_folded,
        "reasons": reasons,
    }


def eligibility_for_cohort(
    label: str, audit: Dict, mapping: Sequence[Dict[str, str]]
) -> Dict[str, object]:
    """Classify every passing design of one cohort as eligible or not.

    A design is eligible when its recorded refolded CIF exists and is
    non-empty. Both failure modes are recorded against the design ID, because
    a cohort comparison needs the excluded designs as much as the included
    ones.
    """
    recorded_pass = (
        audit.get("pass_all_hard_filters_recorded", {}).get("passed")
    )
    eligible: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    for row in mapping:
        path = Path(row["refold_cif"])
        present = str(row.get("refold_cif_present", "")).strip().lower() == "true"
        reason = ""
        if not present:
            reason = "refolded CIF absent at the recorded path"
        elif not path.is_file():
            reason = "refolded CIF no longer present at the recorded path"
        elif path.stat().st_size == 0:
            reason = "refolded CIF is empty"
        record = {
            "cohort": label,
            "id": row["id"],
            "file_name": row.get("file_name", ""),
            "designed_sequence": row.get("designed_sequence", ""),
            "sequence_length": row.get("sequence_length", ""),
            "final_rank": row.get("final_rank", ""),
            "refold_cif": row["refold_cif"],
            "refold_cif_sha256": row.get("refold_cif_sha256", ""),
        }
        if reason:
            excluded.append({**record, "exclusion_reason": reason})
        else:
            eligible.append(record)

    summary = {
        "cohort": label,
        "metrics_rows": audit.get("metrics_rows"),
        "passed_all_hard_filters": recorded_pass,
        "mapping_rows": len(mapping),
        "eligible": len(eligible),
        "excluded": len(excluded),
        "exclusion_reasons": dict(
            Counter(e["exclusion_reason"] for e in excluded)
        ),
    }
    if recorded_pass is not None and recorded_pass != len(mapping):
        summary["warning"] = (
            f"the audit recorded {recorded_pass} passing designs but the "
            f"mapping CSV has {len(mapping)} rows; they must be from the same run"
        )
    return {"summary": summary, "eligible": eligible, "excluded": excluded}


def build_report(
    cohorts: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Assemble the cross-cohort comparison and the combined eligible set."""
    labels = [c["label"] for c in cohorts]
    duplicate_labels = sorted({x for x, n in Counter(labels).items() if n > 1})
    if duplicate_labels:
        raise EligibilityError(
            "Cohort labels must be distinct; repeated: "
            + ", ".join(duplicate_labels)
        )

    per_cohort = []
    combined: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    for cohort in cohorts:
        result = eligibility_for_cohort(
            str(cohort["label"]), cohort["audit"], cohort["mapping"]
        )
        per_cohort.append(result["summary"])
        combined.extend(result["eligible"])
        excluded.extend(result["excluded"])

    shared = [
        design_id
        for design_id, count in Counter(r["id"] for r in combined).items()
        if count > 1
    ]
    lengths = Counter(
        int(r["sequence_length"]) for r in combined if r["sequence_length"].isdigit()
    )
    lengths_by_cohort = {
        str(c["label"]): dict(
            sorted(
                Counter(
                    int(r["sequence_length"])
                    for r in combined
                    if r["cohort"] == str(c["label"]) and r["sequence_length"].isdigit()
                ).items()
            )
        )
        for c in cohorts
    }

    return {
        "schema": "mur-ligase-peptide-eligibility/1",
        "filter_comparison": compare_filter_definitions(cohorts),
        "per_cohort": per_cohort,
        "total_eligible": len(combined),
        "total_excluded": len(excluded),
        "eligible_by_cohort": {s["cohort"]: s["eligible"] for s in per_cohort},
        "design_ids_shared_between_cohorts": sorted(shared),
        "sequence_length_distribution": dict(sorted(lengths.items())),
        "sequence_length_distribution_by_cohort": lengths_by_cohort,
        "excluded_designs": excluded,
        "interpretation": (
            "Eligibility means a design passed every enabled hard filter and "
            "has one usable refolded CIF. It is not evidence of binding, and "
            "it does not establish that the thresholds were appropriate. "
            "Structural validation is performed separately by "
            "peptide_geometry.py."
        ),
    }


def write_candidates_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fields = [
        "cohort",
        "id",
        "file_name",
        "designed_sequence",
        "sequence_length",
        "final_rank",
        "refold_cif",
        "refold_cif_sha256",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_cohort_argument(value: str) -> Dict[str, object]:
    """Parse ``LABEL:AUDIT_JSON:MAPPING_CSV``.

    Split from the right so a label may contain a colon-free path separator
    without ambiguity; the last two fields are always the two file paths.
    """
    parts = value.rsplit(":", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"Expected LABEL:AUDIT_JSON:MAPPING_CSV, got {value!r}"
        )
    label, audit_path, mapping_path = parts
    return {
        "label": label,
        "audit_path": Path(audit_path),
        "mapping_path": Path(mapping_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        type=parse_cohort_argument,
        metavar="LABEL:AUDIT_JSON:MAPPING_CSV",
        help="One per campaign; give the flag at least twice to compare",
    )
    parser.add_argument(
        "--thresholds",
        action="append",
        default=[],
        metavar="LABEL:THRESHOLDS_JSON",
        help="Per-cohort filtering thresholds read from its saved config",
    )
    parser.add_argument(
        "--from-inverse-folded",
        action="append",
        default=[],
        metavar="LABEL:true|false",
        help="Per-cohort from_inverse_folded value from its saved config",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--candidates-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if len(args.cohort) < 2:
        parser.error("--cohort must be given at least twice to compare campaigns")
    for destination in (args.json_out, args.candidates_out):
        if destination is not None and destination.exists():
            parser.error(f"Refusing to overwrite an existing file: {destination}")

    thresholds: Dict[str, object] = {}
    for item in args.thresholds:
        label, _, path = item.rpartition(":")
        if not label:
            parser.error(f"Expected LABEL:THRESHOLDS_JSON, got {item!r}")
        thresholds[label] = load_json(Path(path))

    inverse: Dict[str, bool] = {}
    for item in args.from_inverse_folded:
        label, _, value = item.rpartition(":")
        token = value.strip().lower()
        if not label or token not in ("true", "false"):
            parser.error(f"Expected LABEL:true|false, got {item!r}")
        inverse[label] = token == "true"

    try:
        cohorts = []
        for entry in args.cohort:
            audit = load_json(entry["audit_path"])
            if "enabled_hard_filters" not in audit:
                raise EligibilityError(
                    f"{entry['audit_path']} has no 'enabled_hard_filters'; it "
                    "does not look like an audit_filtering.py record"
                )
            mapping = load_mapping(entry["mapping_path"])
            cohorts.append(
                {
                    "label": entry["label"],
                    "audit": audit,
                    "mapping": mapping,
                    "enabled_hard_filters": audit["enabled_hard_filters"],
                    "thresholds": thresholds.get(entry["label"]),
                    "from_inverse_folded": inverse.get(entry["label"]),
                }
            )
        report = build_report(cohorts)
    except EligibilityError as error:
        print(f"eligibility check failed: {error}", file=sys.stderr)
        return 2

    candidates = [
        row
        for cohort in cohorts
        for row in eligibility_for_cohort(
            str(cohort["label"]), cohort["audit"], cohort["mapping"]
        )["eligible"]
    ]

    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.json_out}", file=sys.stderr)
    if args.candidates_out is not None:
        args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
        write_candidates_csv(args.candidates_out, candidates)
        print(f"candidates: {args.candidates_out}", file=sys.stderr)

    return 0 if report["filter_comparison"]["equivalent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
