# The docking box

How the AutoDock Vina search box is constructed, recorded, transferred to
docking, and checked by eye. Geometry only: nothing here establishes that a
box is the right search region, and nothing here is a validated setting.

## What a box is, and what it is not

The box is an axis-aligned cuboid given by a centre and three dimensions, in
Ångströms, in the coordinate frame of the source structure as read. Vina's
search space "restricts where the movable atoms … should lie": in this pipeline
the receptor is rigid, with no flexible side chains configured, so the movable
atoms are the peptide's and nothing else. Receptor atoms outside the box are
not removed from the receptor.

A box is not a pocket definition, and a box that covers a region is not
evidence that the region is the correct one. Vina publishes a direction — "as
small as possible, but not smaller" — and a conditional size guideline: avoid
search spaces larger than 30 × 30 × 30 Å *unless* exhaustiveness is increased.
See <https://vina.scripps.edu/manual/> and
<https://autodock-vina.readthedocs.io/en/latest/faq.html>.

## Two boxes, deliberately distinguished

`map_pocket.py` reports both, and they are not the same box.

**Legacy printed lines** (`box centre`, `box size`) appear on every run,
unchanged. They centre on the **ligand atom mean** and give a single cubic
`max(extent) + 10`, formatted to the nearest integer. Both choices have
consequences: atom-mean centring displaces the box towards the denser end of
the ligand, so clearance lost on one face is gained on the opposite one, and
nearest-integer formatting can remove up to 0.5 Å from the nominal padding.
They are kept so previously used boxes stay reconstructable.

**The recorded box** is written only when asked for, with `--box-record` plus
exactly one of `--box-size` or `--box-padding`. It centres on the **midpoint of
the selection's coordinate bounds** by default, carries three independent
dimensions, and is written as JSON that `dock.py` reads directly.

There is no default size or padding. Box dimensions are a scientific choice and
have to be stated.

## Writing a box record

```bash
# Explicit dimensions: one value for a cube, or three for independent axes.
python python/map_pocket.py data/1P3D.cif UMA 5.0 \
    --protein-chain A --ligand-chain E \
    --box-size 24 --box-record boxes/murc_uma_chainE_24A.json

# Clearance per face instead: dimensions become extent + 2 x padding per axis,
# rounded up to 0.1 A so the stated padding is a guaranteed minimum.
python python/map_pocket.py data/1P3D.cif UMA 5.0 \
    --protein-chain A --ligand-chain E \
    --box-padding 5 --box-record boxes/murc_uma_chainE_pad5.json
```

Options: `--box-centre {bounds-midpoint,ligand-atom-mean}` (the second only to
reconstruct a historical box), `--overwrite-box-record` to replace an existing
file. A record is never overwritten silently, because a stale box reused
without noticing is the failure the record exists to prevent.

**Padding is a guarantee, so it constrains the centring.** `--box-padding`
promises at least that clearance on all six faces, which only
`bounds-midpoint` centring can deliver: any other centre is displaced from the
selection's bounds and takes from one face exactly what it adds to the
opposite one. `--box-centre ligand-atom-mean` is therefore accepted with
`--box-size` only, where the dimensions are stated outright and no clearance
is promised, and refused with `--box-padding`.

The record carries the centre, the three dimensions, the units, the centring
and sizing mode with the padding or requested size, the ligand components and
both their canonical and author chain identifiers, the author protein chain the
pocket was mapped on, the cutoff and the pocket residue list, and the source
structure's path, SHA-256 and coordinate frame, plus the commit and whether the
working tree was dirty.

Three provisional examples are kept in `examples/`, sharing one centre:
`box_murc_uma_chainE_24A.json`, `..._30A.json`, `..._33A.json`. They are
geometry comparisons for the MurC/1P3D UMA pocket, **not** validated
production settings.

## Docking from a record

```bash
python python/dock.py murc.pdbqt pep dock_run \
    --box-record boxes/murc_uma_chainE_24A.json \
    --seeds 1,42,137 --exhaustiveness 32 --cpu 8
```

Nothing is transcribed. Before docking anything, `dock.py` refuses to proceed
when:

- the box centre lies outside the receptor's own coordinate range — what a
  mistyped centre or a box from a different protein copy looks like. 1P3D's two
  UMA sites are 98.9 Å apart, so this is not hypothetical;
- the receptor does not contain the author protein chain the box was mapped on;
- the ligand directory already holds `*_docked.pdbqt` files from an earlier run;
- the output directory is the ligand directory or inside it;
- the output directory already exists with anything in it. There is no
  override: results from separate runs are never merged, because a rerun into
  a populated directory leaves the previous run's poses beside the new tables
  and rewrites the tables around them.

`--check-only` runs those checks and stops.

### What the receptor checks are, and are not

They are **preliminary sanity checks**. They confirm that the box centre falls
within the receptor's coordinate range and that the recorded author chain is
present — enough to catch a box paired with the wrong protein copy.

They do **not** verify how the receptor was prepared, which structure it came
from, or that its coordinates are consistent with the box's recorded source
structure and frame. A receptor prepared from a different structure of the
same protein would pass both checks. **Verified receptor-preparation
provenance and coordinate consistency with the box's source structure remain
prerequisites for real docking**, and belong to the receptor-preparation
stage; they are not implemented yet. Passing these checks is not evidence that
the receptor is the right one.

### Outputs

All under the output directory:

| Path | Contents |
| --- | --- |
| `box_record.json` | the record used, copied verbatim |
| `run_record.json` | command, input SHA-256 hashes, seeds, settings, box, Vina version, status |
| `poses/seed_<N>/<ligand>_docked.pdbqt` | poses for **every** seed |
| `per_seed_scores.csv` | one row per ligand, seed and pose, appended as produced |
| `dock_results.csv` | one row per ligand, rewritten after each ligand |
| `failures.csv` | one row per ligand and seed that did not complete |

Nothing is held only in memory. The run record is written with
`"status": "running"` before docking starts and updated at the end, pose rows
are flushed as they are produced, and the summary and failure tables are
rewritten after every ligand, so an interrupted run leaves complete
bookkeeping for everything that finished and a run record that still says it
never completed.

**Each result is validated before it is recorded as a success**: the score
list must be non-empty, as long as requested and entirely finite, and the pose
file must be new to that call and non-empty. A ligand and seed failing any of
those is recorded in `failures.csv` with the reason, not scored. Validation
sits inside the per-seed error handling, so one bad ligand does not end the
campaign.

**Incomplete campaigns exit non-zero.** `dock_results.csv` marks each ligand
`complete`, `partial` or `failed`, and names the seeds that completed and the
seeds that failed. A partial ligand's mean rests on fewer seeds than the
others and is not comparable with them. For a ligand with one completed seed
the spread columns are left **blank** with `spread_estimable = no`, rather
than reporting a standard deviation of 0.00 — over one observation that
describes the sample size and reads as the opposite of what it means.

### Legacy docking mode

`python python/dock.py RECEPTOR.pdbqt LIGAND_DIR CX,CY,CZ SIZE [nseed] [cpu]`
still works and prints a warning. It forces a cube, saves poses for the first
seed only, writes into the ligand directory, and validates nothing about the
centre. It exists for reproducing earlier runs.

## Visual check

```bash
python python/box_to_bild.py examples/box_murc_uma_chainE_24A.json box24.bild \
    --chimerax-script box24.cxc --structure data/1P3D.cif
chimerax box24.cxc
```

This writes a ChimeraX BILD wireframe of the box with a marker at its centre,
and a `.cxc` that opens the structure, the box, the recorded pocket residues
and the recorded ligand together. Confirm by eye that the box sits on the
intended protein copy and not on a second copy elsewhere in the asymmetric
unit, that it covers the pocket mouth, and which pocket residues fall at or
outside a face.

Paths written into the `.cxc` are absolute and double-quoted, so the script
runs from any working directory and tolerates spaces in a path.

The box is drawn in the recorded structure's frame, so it is only meaningful
against that structure or a receptor prepared from it without transformation.
BoltzGen design and refolded outputs are in a different frame and must be
superposed first.

## Checks

`python -m unittest discover -s tests -p 'test_*.py'` covers box geometry,
invalid inputs, the centring/sizing combinations, record transfer, the
ChimeraX path quoting, and `dock.py`'s result validation, per-seed outputs,
partial-result marking, exit status and incremental persistence.

Two kinds of test are deliberately distinguished:

- the `map_pocket.py` box-record tests run the **real** CLI against a small
  asymmetric mmCIF fixture, and skip if Gemmi is unavailable;
- the docking tests run against a **stub** Vina injected on `PYTHONPATH`. They
  check how seeds, poses, validation and failures are organised and say
  nothing about AutoDock Vina's own behaviour.

Anything requiring real Vina — including whether receptor atoms outside the
box contribute to the scoring grid — has to run on the machine that has it.
