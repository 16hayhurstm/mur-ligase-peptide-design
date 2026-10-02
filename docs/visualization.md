# Reproducible visualisation and selection

Create a portable ChimeraX bundle to compare peptide complexes before and after refolding. Run from the repository root with Python 3.8+; the builder uses only the standard library. We use Epicac, but it is not required. ChimeraX is needed only for viewing.

## Usage

Replace `RUN` with a run folder or BoltzGen output folder, and `DEST` with a new destination folder. `DEST` must not already exist.

```bash
# Default: all metrics rows eligible, ordered by final_rank; maximum 30 designs
bash scripts/04_visualize_designs.sh RUN DEST

# Example: inspect the initial 20-design MurC trial
bash scripts/04_visualize_designs.sh runs/murc_fixed_20_20260930 visualizations/trial-v2

# Select recorded filter passes
bash scripts/04_visualize_designs.sh RUN DEST --select passed --limit 30

# Select an exact design ID
bash scripts/04_visualize_designs.sh RUN DEST --select ids --ids murc_myles_20260930_fixed_13

# Require every custom criterion to pass
bash scripts/04_visualize_designs.sh RUN DEST --select custom --criteria examples/visualization_criteria.json

# Sort passing designs by another numeric metric
bash scripts/04_visualize_designs.sh RUN DEST --select passed --sort-by design_to_target_iptm --descending

# Check selection and structure-pair availability without writing files
bash scripts/04_visualize_designs.sh RUN DEST --select passed --dry-run

# Use an alternative metrics table
bash scripts/04_visualize_designs.sh RUN DEST --metrics RUN/output/refiltered/final_ranked_designs/all_designs_metrics.csv

# Generate designs, then create a bundle
bash scripts/03_boltzgen_design.sh SPEC.yaml NEW_OUTPUT 100 --visualize --select passed --limit 30
```

Arguments after `--visualize` go to the builder. `--limit 0` removes the cap; use cautiously because all selected pairs load into memory. Selecting `all` does not remove the cap. An empty selection produces summary files without a viewer.

## Selection rules

Only metrics CSV rows are candidates, so summary counts may differ from the number of designs originally requested. IDs must be unique. Structures are matched using the exact `file_name`, or `id.cif` if absent. Ranked prefixes are not stripped, and there are no silent fallbacks. Missing selected pairs cause errors; missing unselected structures do not.

Sorting uses the specified numeric column, with missing values last and ID breaking ties. Sequence and pose diversity selection are not implemented.

Custom criteria are a nonempty JSON list of objects containing `column`, `op` and `value`. Every criterion must pass.

- Operators: `<`, `<=`, `>`, `>=`, `==`, `!=`.
- Values: finite numbers or booleans; booleans support only `==` and `!=`.
- Missing or nonfinite metric values fail.
- Unknown columns or operators cause errors.
- Include a `pass_filters` predicate if custom selection should also require recorded filter passes.
- Arbitrary Python expressions are not executed.

The example JSON reproduces existing criteria; it is not a validated hit-selection rule. A high rank or filter pass does not demonstrate experimental binding.

## Viewing the bundle

Download the **whole destination folder**. It contains:

- `summary.json`: criteria, counts and source CSV hash.
- `selected_metrics.csv`: selected metrics.
- `manifest.csv`: source paths and structure SHA-256 hashes.
- `pairs.json`: structure-pair information.
- Selected structure pairs and `compare_refolds.py` for nonempty selections.

Open `compare_refolds.py` through **File → Open** in an empty ChimeraX session. The viewer refuses a nonempty session to protect existing work. Expand model groups and toggle one parent group at a time.

| Display | Meaning |
|---|---|
| Cyan ghost | Inverse-folded complex immediately before refolding |
| Magenta sticks/cartoon | Corresponding `refold_cif` structure |
| Grey receptor | First selected pre-refold receptor, used as a common reference |

“Before” does not mean the raw diffusion backbone. Each full complex is moved by aligning its receptor to the reference. Use each design’s matching receptor to assess clashes: the grey reference does not represent every receptor, and whole-receptor alignment can leave local domain differences.

MurC is identified as the sole protein chain with at least 200 CA-bearing residues, and the peptide as the sole chain with 2–100. This project-specific heuristic does not verify sequence identity; ambiguous structures stop loading.

Labels show recorded `pass_filters` values, including failures admitted by custom selection. Pocket highlighting, geometric contact/clash analysis and on-demand loading are not implemented. No BoltzGen structures or scores are modified.

