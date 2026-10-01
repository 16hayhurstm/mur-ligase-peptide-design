# Reproducible visualization and selection

Run from the repository root with Python 3.8+ on Epicac. The builder uses only
Python's standard library. ChimeraX is needed only on the viewing computer.

```bash
# Inspect all 20 trial designs (default cap: 30; order: final_rank ascending)
bash scripts/04_visualize_designs.sh runs/murc_fixed_20_20260930 visualizations/trial-v2
# Only recorded passes; zero passes yields summary files and NO viewer
bash scripts/04_visualize_designs.sh RUN DEST --select passed --limit 30
# Exact IDs (not abbreviated numbers)
bash scripts/04_visualize_designs.sh RUN DEST --select ids --ids murc_myles_20260930_fixed_13
# All custom predicates must match; missing/nonfinite values fail
bash scripts/04_visualize_designs.sh RUN DEST --select custom --criteria examples/visualization_criteria.json
# Rank by a different numeric metric
bash scripts/04_visualize_designs.sh RUN DEST --select passed --sort-by design_to_target_iptm --descending
# Check selection and selected-pair availability without writing a bundle
bash scripts/04_visualize_designs.sh RUN DEST --select passed --dry-run
# Explicitly use another filtering table
bash scripts/04_visualize_designs.sh RUN DEST --metrics RUN/output/refiltered/final_ranked_designs/all_designs_metrics.csv
# Optional generation integration (arguments after --visualize go to builder)
bash scripts/03_boltzgen_design.sh SPEC.yaml NEW_OUTPUT 100 --visualize --select passed --limit 30
```

`RUN` accepts a run folder or the actual BoltzGen output folder. `DEST` must not
already exist. `--limit 0` explicitly removes the cap; avoid it for large runs.
`all` means all metrics rows are eligible, not that the display cap is removed.
Selection uses CSV `id` and exact `file_name` (or `id.cif` when absent). IDs must
be unique. Structures absent from the metrics table are not candidates; summary
counts describe metrics rows, not the originally requested number of designs.
Unselected missing structures do not block a shortlist. Missing selected pairs
are errors. No ranked-file prefix stripping and no silent fallbacks are used.

Custom JSON is a nonempty list of objects containing `column`, `op`, `value`.
Operators: `<`, `<=`, `>`, `>=`, `==`, `!=`. Values: finite numbers or booleans;
booleans permit equality/inequality only. Unknown columns/operators are errors.
The example reproduces existing criteria; it is NOT a validated hit-selection
rule. No arbitrary Python expressions are executed. Custom selection does not
implicitly require `pass_filters`; include that predicate when desired.

Ordering is by the specified numeric column, with missing values last and ID
as a deterministic tie-breaker. Sequence/pose diversity selection is not yet
implemented. A high rank or filter pass is not evidence of experimental binding.

## Portable folder

The bundle contains `summary.json` (criteria, counts, source CSV hash),
`selected_metrics.csv`, `manifest.csv` (sources and structure SHA-256 hashes),
`pairs.json`, selected structure pairs, and `compare_refolds.py` when nonempty.
Download the WHOLE folder. Open the viewer with File > Open in an empty ChimeraX
session. It refuses a nonempty session instead of closing existing work.

Before = inverse-folded complex immediately preceding refolding; after = its
`refold_cif` counterpart. These are not the raw diffusion-stage backbones.
MurC is detected as the sole protein chain with >=200 CA-bearing residues and
the peptide as the sole chain with 2–100. This heuristic is specific to this
project, not a sequence-based identity check; ambiguous structures stop loading.
Each full complex is moved using receptor alignment to the first selected
pre-refold receptor. Cyan ghost = before; magenta sticks/cartoon = after.
The common grey receptor is a reference, not every design's actual receptor.
Use the matching receptor when evaluating clashes. Whole-receptor alignment
can leave local domain differences. Pocket highlighting and geometric
contact/clash evaluation are not implemented.

Expand groups and toggle one parent group at a time. Labels report the saved
`pass_filters` value even when custom selection admits failures. All SELECTED
pairs load into memory; the cap keeps this practical. On-demand loading is
not yet implemented. No BoltzGen structures or scores are modified.

## Validation status (2026-10-01)

The original viewer plus grouping fixes was used by Myles in ChimeraX 1.12.
The refactored builder is tested using the uploaded real 20-row metrics table
and synthetic file-pair fixtures. Tests exercise selection/copying, not molecular
validity. The refactored viewer requires a Mac ChimeraX smoke test; no ChimeraX
runtime or actual run CIFs were included in the review archive.
