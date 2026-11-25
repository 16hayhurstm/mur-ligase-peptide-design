# Repository code organisation: an evidence-based review

Drafted 25 November 2026 as a **proposal**. No Python file has been deleted,
moved or renamed, and none should be until this is agreed and executed as a
separate change after the validation branch is merged.

## How each module was classified

Three kinds of evidence, all gathered mechanically from the current tree:

1. **Import relationships** — every `import` statement in `python/` and
   `tests/`, so a module's reverse-dependency count is counted, not guessed.
2. **Test coverage** — whether any test file imports the module or invokes
   its script path.
3. **Documented command usage** — whether any file in `docs/`, `README.md` or
   `scripts/` names the module.

A module absent from all three is a candidate for review. A module absent
from documentation but central to the import graph is a supporting module,
not dead code — the two are distinguished below rather than conflated.

## 1. Core modules for the next peptide docking phase

| Module | Role | Evidence |
| --- | --- | --- |
| `dock.py` | the docking campaign driver: box record, receptor manifest gate, per-seed poses and scores, run record | documented in `README.md`, `docs/docking_box.md`, `docs/receptor_preparation.md`, `docs/uma_control.md`; tested in `test_docking_box.py` |
| `prepare_receptor.py` | builds the receptor, with the `--condition` the standard model is named by | documented in `README.md`, `docs/receptor_preparation.md`, `docs/uma_control_results.md`; 48 tests in `test_receptor_preparation.py` |
| `receptor_provenance.py` | manifests, atom correspondence, pocket census, the box/receptor gate | imported by **10** modules; tested directly in five test files |
| `box_geometry.py` | box construction, the box record, hashing, ChimeraX BILD | imported by **13** modules — the most depended-on module here; tested in `test_docking_box.py` |
| `map_pocket.py` | pocket selection and the box record it writes | `README.md`, `docs/docking_box.md`; real CLI exercised in `test_docking_box.py` |
| `box_to_bild.py` | box visualisation | `docs/docking_box.md`; tested |
| `scramble.py` | composition-matched sequence controls for the peptide cohorts | `README.md`; **no test** (see §4) |
| `scripts/06_prepare_ligands.sh` | peptide ligand construction from sequence (PyMOL + OpenBabel) | the only ligand-building entry point for peptides |

**The gap in this list is peptide ligand preparation.** `prepare_ligand.py`
is UMA-specific (CCD-driven, single component, chosen formal charges);
`scripts/06_prepare_ligands.sh` builds peptides from sequence with no
verification record, no per-design failure table and no atom map. The next
phase needs a peptide equivalent of the UMA preparation-and-verification
pattern, and that is new work, not a reorganisation. See
`project_private/next_task_peptide_docking.md` §1–§2.

## 2. Modules needed primarily for the UMA validation

Retained for reproducibility of the control, not expected to be used by the
peptide phase in their current form:

| Module | Role | Evidence |
| --- | --- | --- |
| `prepare_ligand.py` | UMA ligand build / check / atom-map stages, driven by the component dictionary | `docs/uma_control.md`; exercised by `test_uma_control.py`, `test_manifest_provenance.py`, `test_ligand_verification.py` |
| `ligand_reference.py` | the immutable deposited reference pose | `docs/uma_control.md`; tested |
| `analyse_redocking.py` | per-seed recovery analysis against a reference pose | `docs/uma_control.md`, `docs/uma_control_results.md`; imported by `local_minimisation.py` and `pose_to_pdb.py`; tested in four test files |
| `verify_ligand_rdkit.py` | RDKit stereochemistry and representation verification | `docs/uma_control.md`; `test_ligand_verification.py` |
| `independent_rmsd.py` | the independent RMSD comparison against RDKit `CalcRMS` | `docs/uma_control.md`; `test_manifest_provenance.py` |
| `oxygen_pair_audit.py` | the audited equivalent-oxygen mapping convention | `docs/uma_control.md`; `test_manifest_provenance.py` |
| `local_minimisation.py` | the crystallographic-pose minimisation diagnostic | **undocumented**; imported by `rescore_poses.py`; 563 lines of tests in `test_local_minimisation.py` |
| `pose_to_pdb.py` | pose conversion for viewing, with a conversion record | **undocumented**; `test_pose_to_pdb.py` |
| `rescore_poses.py` | pose rescoring under two conventions | **undocumented**; `test_rescore_poses.py` |

Three of these — `local_minimisation.py`, `pose_to_pdb.py`,
`rescore_poses.py` — are **well tested and entirely undocumented**. They
produced results cited in the audit and the handover. That is a documentation
gap, not redundancy, and it is the one finding here I would act on before
structural cleanup: a short `docs/uma_diagnostics.md` describing what each
answers and what it cannot.

`analyse_redocking.py` sits in this section but is **not UMA-only in
principle**: it measures a pose against a reference pose and would serve any
redocking control. Its UMA specificity is in its callers, not its logic.

## 3. Supporting modules imported by other scripts

Not commands, correctly absent from the command documentation, and must not
be read as unused:

| Module | Imported by | Note |
| --- | --- | --- |
| `box_geometry.py` | 13 modules | also holds `file_sha256`, used everywhere for provenance |
| `receptor_provenance.py` | 10 modules | manifest reading/writing for every stage |
| `chemical_component.py` | 8 modules | CCD reading, ionisable sites |
| `chemical_graph.py` | 8 modules | resonance normalisation, automorphisms, in-place RMSD |
| `ligand_atom_map.py` | 5 modules | Meeko index map, serial-to-name correspondence |

`box_geometry.py` carrying the project's hashing helper is a small
cohesion wrinkle: provenance hashing is used by modules that have nothing to
do with box geometry. Worth noting; **not** worth a rename on this branch,
since 13 modules import it.

## 4. Potentially redundant, unused, or unverified

Reviewed rather than condemned — each needs a decision, and none is a
deletion I would make without one.

| Module | Finding | Proposal |
| --- | --- | --- |
| `occupancy.py` | in `README.md`; **no test**; uses chain-length heuristics and has the known gaps recorded in `CLAUDE.md` (no stable per-design IDs, silent omission of unanalysable complexes) | keep; do **not** use for the campaign comparison without the ID and failure-record fixes |
| `build_visualisation.py` | the only module behind `scripts/04_visualise_designs.sh`; tested by `test_visualisation.py` (12 standard-library tests on non-molecular fixtures) | keep as-is; BoltzGen-stage tool, unaffected by the docking phase |
| `inspect_structure.py` | in `README.md`; **no test**; its only dependency, Biopython, is absent from both documented environments' package tables and from the local venv | decide: either add Biopython to the environment documentation, or reimplement the few lines with Gemmi, which every other module already uses. Currently it cannot run anywhere documented |
| `make_spec.py` | in `README.md` and `scripts/03_boltzgen_design.sh`; **no test**; writes an absolute source path | keep; generation-stage tool. The absolute path is already flagged in `CLAUDE.md` as a machine-transfer hazard |
| `occupancy.py`, `scramble.py`, `make_spec.py`, `inspect_structure.py` | the four modules **no test touches at all** | `scramble.py` is 50 lines and seeded, so it is cheap to test and will be used by the peptide phase — worth a small test. The other three are generation/analysis-stage utilities whose risk profile is lower but non-zero |
| `python/chimerax/compare_refolds.py` | imports `chimerax.*`, so it runs only inside ChimeraX; untestable here by design | keep; note in documentation that it is a ChimeraX-internal script, not a CLI |
| `python/__pycache__`, `tests/__pycache__` | present on disk, correctly ignored by `.gitignore` | nothing to do |

**Nothing in `python/` is dead code.** Every module is reached by a test, a
documented command, or an import from another module. The two real problems
this review found are the three undocumented diagnostics (§2) and
`inspect_structure.py`'s undeclared Biopython dependency (above).

## 5. Retained solely for reproducibility

These exist because results were obtained with them, and must not be removed
or "tidied" even if the peptide phase never calls them:

- `independent_rmsd.py` and `oxygen_pair_audit.py` — the audited mapping
  convention under which the RMSD implementation was validated. The
  validation is only interpretable alongside the code that performed it.
- `local_minimisation.py` and `rescore_poses.py` — the diagnostics behind the
  minimisation and rescoring observations in the audit and the handover.
- `verify_ligand_rdkit.py` — the stereochemistry verification that gates
  `analyse_redocking.py`'s `interpretable` flag; the gate is meaningless
  without it.
- `examples/box_murc_uma_chainE_{24,30,33}A.json` — the 24 Å record is the
  one the existing receptors' pocket selection was taken from, so it is
  provenance, not a superseded draft.
- `examples/uma_pose_comparison.cxc` — the viewing script for the recorded
  pose comparison.
- The `apo` receptor and all its records and outputs — now the archived
  comparison condition, per `docs/receptor_preparation.md`.

## 6. Proposed order, after the merge

1. Document the three diagnostics (`local_minimisation.py`,
   `pose_to_pdb.py`, `rescore_poses.py`). No code change.
2. Resolve `inspect_structure.py`'s Biopython dependency, one way or the
   other.
3. Add a small test for `scramble.py` before the peptide phase uses it.
4. Only then consider structure: if the `python/` directory is to be split,
   the natural seam is the one §1–§3 already describe — a supporting layer
   (`box_geometry`, `receptor_provenance`, `chemical_*`, `ligand_atom_map`),
   a docking layer, and a UMA-validation layer. That is a large,
   import-breaking change for presentational benefit, and it should be
   weighed against leaving a flat, working `python/` alone.
