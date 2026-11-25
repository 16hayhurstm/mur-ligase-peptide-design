# Receptor preparation

How a crystal structure becomes an AutoDock Vina receptor, what is checked,
what is recorded, and what remains unresolved. Nothing here validates a
receptor biologically: passing every check means the file is the structure it
claims to be, not that it is a good model of the protein.

## Commands

```bash
# Initial condition: ANP absent, manganese absent, UMA absent, waters absent.
python python/prepare_receptor.py data/1P3D.cif receptors/murc_apo \
    --protein-author-chain A --condition apo \
    --box-record examples/box_murc_uma_chainE_24A.json

# The paired comparison's second receptor, same box, ANP retained.
python python/prepare_receptor.py data/1P3D.cif receptors/murc_anp \
    --protein-author-chain A --condition anp-present \
    --box-record examples/box_murc_uma_chainE_24A.json
```

The box record supplies the protein copy and the pocket residue list, so a
receptor and its search box cannot come from different selections.
`--pocket-residues 25,27,...` works without a box. `--skip-meeko` stops after
the cleaned PDB and its checks, and writes no manifest, because no receptor
was produced.

Then dock against the manifest:

```bash
python python/dock.py receptors/murc_apo_rigid.pdbqt pep dock_pilot \
    --box-record examples/box_murc_uma_chainE_24A.json \
    --receptor-manifest receptors/murc_apo.manifest.json \
    --seeds 1,42,137 --exhaustiveness 32
```

## Receptor conditions

| Condition | ANP | Manganese | UMA | Waters | Status |
| --- | --- | --- | --- | --- | --- |
| `apo` | absent | absent | absent | absent | archived comparison condition |
| `anp-present` | **retained** | absent | absent | absent | **standard docking model** |

**Which condition to use, decided 25 November 2026.** `anp-present` is the
standard docking model for the peptide cohorts; `apo` is retained as an
archived comparison condition, unmodified and available. The basis is the
cognate-ligand control: in the 30 Å box, with ten seeds, pose recovery
succeeded in the ANP-containing receptor and failed entirely in the ANP-free
one — see [uma_control_results.md](uma_control_results.md) §4 and §10.

This is a **modelling decision, not a validation**. It does not establish
that peptide affinities, rankings or binding have been validated, and it does
not validate ANP's chemistry, which remains Meeko-perceived (below) and is
now a dependency of the standard model.

**`--condition` is required and has no default**, and nothing in this
repository hard-codes a receptor. A standard condition agreed by decision
must still be typed into every command and must appear in every manifest and
run record, so that a receptor prepared or docked under the other condition
is visible as such instead of being assumed away.

**UMA is removed in every condition.** It is the reaction product whose
position defines the pocket; leaving it in the receptor would block the site
being docked into. It is preserved separately as the reference pose for the
planned UMA redocking control.

**Manganese omission is a documented approximation.** It followed a reported
compatibility problem at the generation stage, not a finding that the metals
are unimportant. Metal-free modelling does not represent the catalytic
environment of a Mur ligase, and this is recorded in every manifest as an
unresolved decision rather than settled by the script's default.

**Removing ligands and metals does not produce an apo structure.** The
coordinates remain those of the complex. An "apo receptor" here means "the
complex with components deleted", which is not the same thing.

### What was actually removed from 1P3D chain A

Observed, and recorded identically in both prepared receptors' manifests:

| Component | Count removed | Why |
| --- | --- | --- |
| UMA | 1 | the pocket-defining reaction product |
| **Mn²⁺** | **2** | documented approximation, not a validated equivalence |
| **HOH** | **397** | crystallographic waters removed |
| ANP | 1, in `apo` only | ATP analogue absent in that condition |

Chain B was discarded in a separate, separately labelled step; it is not a
component removal. Residue transformations: MSE→MET on 14 residues; **KCX→LYS
on none** — 1P3D chain A contains no carbamylated lysine, and the recorded
count is 0. The KCX path exists and was not exercised here.

**Measured, not assumed:** of these removals, the waters are the largest
chemical change to the UMA site. Minimum distances from the 49 heavy atoms of
deposited UMA (author A:602) to the other components of chain A are 2.54 Å for
water (**20** water oxygens within 3.5 Å), 3.96 Å for ANP (none within 3.5 Å)
and 4.15 Å for Mn²⁺ (none within 3.5 Å).

So **manganese does not coordinate UMA in 1P3D**: the two ions coordinate
ANP's phosphate oxygens and protein side chains. A metal-bridged carboxylate
described in the literature concerns UNAM, the *substrate*, in a different
structure, and must not be transferred to UMA, the product. Manganese
omission still matters — through ANP's charge and the pocket's electrostatics
— but not through a lost UMA contact.

## What is checked, and in what order

1. **Explicit protein copy.** `--protein-author-chain` is required. There is
   no "largest chain" default: 1P3D holds two MurC copies whose pockets are
   98.9 Å apart, and the previous default picked one by a 3-residue margin in
   residue count. The author-to-canonical chain and residue mapping is
   recorded, because PDB and PDBQT carry only author numbering and the
   canonical mmCIF identifiers would otherwise be lost.
2. **Pocket census before preparation.** Every requested residue must be
   present on the selected chain, and none may be missing heavy atoms, judged
   against a table of expected heavy-atom counts. An incomplete pocket side
   chain **stops the run**: under this project's rules missing pocket atoms
   need an agreed repair method or must block downstream use. A pocket residue
   carrying an insertion code also stops the run, since a residue-number
   selection cannot represent it unambiguously.
3. **Component removal, attributed.** Each removal is recorded by component
   with its reason. This replaces Gemmi's `remove_ligands_and_waters()`,
   which on 1P3D silently strips UMA, ANP and all four manganese ions in one
   call — three different scientific decisions taken by one line. Discarding
   the other protein copy is recorded separately, as a selection rather than a
   component removal.
4. **Residue transformations, recorded as chemistry.** MSE → MET renames SE to
   SD and sets its element to sulphur; KCX → LYS deletes the carbamyl atoms.
   Each occurrence is listed with its effect. The 1P3D chain A case is 14
   MSE residues, one of them pocket residue 31.
5. **Coordinate preservation, per atom.** Every retained heavy atom is paired
   between the source and the cleaned PDB, and again between the cleaned PDB
   and the receptor PDBQT, on full identity: author chain, residue number,
   insertion code, residue name, atom name and altloc. The **maximum**
   per-atom deviation is reported, never a mean, and the pocket subset is
   reported separately so a local change cannot be averaged away. Agreed
   transformations are followed (MSE SE → MET SD pairs as one atom); anything
   else counts as a missing atom.
6. **Subprocess and output checks.** Meeko's exit status is checked, its full
   output recorded, and the PDBQT must exist, be non-empty, parse, and
   correspond atom-for-atom with the cleaned PDB. A pocket heavy atom lost at
   this step stops the run.
7. **Failure leaves nothing usable.** On any failure the outputs are renamed
   with a `.FAILED` suffix and **no manifest is written**. A manifest's
   existence is the signal that every check passed.

### Coordinate tolerance

PDB and PDBQT store three decimal places, so one coordinate written twice can
differ by up to 0.001 Å per axis and 0.001 × √3 ≈ 0.0017 Å in distance. The
tolerance is **0.002 Å**: large enough that rounding never trips it, far
smaller than any real movement. On 1P3D chain A the observed maximum is
0.0000 Å at both stages, for all 3547 heavy atoms and all 234 pocket heavy
atoms.

This checks that coordinates were *not transformed*. It is not a structural
similarity measure.

## The manifest

`OUT_PREFIX.manifest.json` records the receptor condition; the chain and
residue mappings and the pocket selection with the box record and its hash;
the source structure's path, SHA-256 and coordinate frame; the output paths
and their hashes; every residue transformation, removed component and retained
component; every check and its findings; the exact Meeko command, exit status
and output; the tool versions available; and the unresolved decisions.

`dock.py --receptor-manifest` compares the manifest with the box record and
refuses to dock when the source structure hash, the author protein chain, the
pocket cutoff or the pocket residue list disagree, or when the receptor file
being docked is not the one the manifest describes. **Without a manifest,
recorded-box docking is refused** unless `--allow-unverified-receptor` is
passed, which is recorded in the run record. A `.pdbqt` carries no source
structure, no hash and no account of what was removed from it, so provenance
cannot be recovered from the file itself.

Provenance agreement is a prerequisite, not a validation. It establishes that
the box and the receptor refer to the same protein copy of the same structure.
It says nothing about atom typing, protonation, charges, or whether the pocket
is the right one.

## Retaining ANP: not implemented blind

The `anp-present` condition keeps ANP through the Gemmi cleaning stage and
then **requires** the ANP atoms to be present in Meeko's PDBQT. If they are
not, the run fails. The records are never appended by hand: a PDBQT line
without an AutoDock atom type and a partial charge is not a usable receptor
atom, and a receptor silently missing its cofactor would dock and return
scores.

Meeko does support non-standard residues, through templates.

### The installed flags

Confirmed from `mk_prepare_receptor.py --help` in the `boltz` environment on
Epicac, **Meeko 0.7.1** (`/home/amp/miniconda3/envs/boltz/bin/python`):

| Flag | Observed behaviour |
| --- | --- |
| `-a`, `--allow_bad_res` | deletes residues with missing atoms |
| `-d`, `--delete_residues` | deletes an explicit residue selection |
| `-n`, `--set_template` | assigns residue templates |
| `--add_templates` | accepts JSON, or `resname:file.sdf` |
| `--charge_model` | defaults to `gasteiger` |
| `--read_pdb` | reads a PDB without ProDy |

Two corrections to an earlier draft of this document, which had read Meeko's
`develop` documentation rather than the installed version:

- `-a/--allow_bad_res` has **not** been replaced by `-d/--delete_bad_res` in
  0.7.1. `-a` exists and deletes residues with missing atoms; `-d` is
  `--delete_residues`, a different flag taking an explicit selection. They are
  not interchangeable and `-d` is never substituted for `-a` here.
- `--add_templates` takes JSON *or* an SDF named per residue
  (`resname:file.sdf`), not JSON alone.

`prepare_receptor.py` therefore calls:

```
mk_prepare_receptor.py --read_pdb <cleaned.pdb> -o <prefix> -p --default_altloc A
```

`--read_pdb` is used because this script always hands Meeko a PDB it has just
written, so the ProDy-backed reader is unnecessary. **`-a/--allow_bad_res` is
deliberately not passed**: automatic bad-residue deletion stays disabled for
the first real preparation attempt, so Meeko reports what it cannot match
instead of discarding it, and an incomplete pocket residue blocks the receptor
as this project's rules require. Either flag can still be passed through
`--meeko-argument`, and the manifest records the exact command.

### What Meeko produced for ANP, observed

The `anp-present` receptor exists and was used for one docking campaign, so
what Meeko actually did to ANP is recorded here rather than left implied.

Read from the receptor PDBQT, and consistent with the manifest's captured
subprocess record:

| Quantity | Value |
| --- | --- |
| ANP records in the receptor PDBQT | 36, all at author A:603 |
| heavy atoms | **31** |
| retained polar hydrogens (`HD`) | **5** |
| summed partial charge | **−4.0000** |
| AutoDock types | OA 12, C 5, A 5, NA 4, HD 5, P 3, N 2 |
| receptor heavy atoms | 3578, i.e. 3547 + 31, agreeing with the manifest |

ANP matched **no packaged Meeko template**. The captured stderr reads

```
RuntimeWarning: Input residues {'A:603': 'ANP'} not in residue_templates
```

and Meeko then built a chemical template of its own from a hydrogen-free PDB.
Its bond orders, protonation and −4 formal charge are therefore **perceived
by the tool, not evidenced**. The hydrogen pattern is consistent with the
neutral CCD entry with the four phosphate acids deprotonated — the same
convention applied to the UMA ligand, which makes the two consistent rather
than independently corroborated.

> **This is a plausible working representation of ANP. It is not an
> independently validated ANP parameterisation.** Nothing here establishes
> that the atom types, the partial charges or the protonation are right for
> this cofactor at this site.

Two further consequences, recorded so they are not rediscovered:

- the retained ANP carries its full −4 inside the search box while the two
  Mn²⁺ ions that are inner-sphere coordinated to its phosphates in the
  deposited structure have been deleted, so the local charge balance of the
  model is not that of the crystal;
- the Meeko **version is not captured** in either manifest: the recorded
  entry is the first line of `mk_prepare_receptor.py --help`, which is a
  usage line. "Meeko 0.7.1" comes from [environment.md](environment.md), not
  from a record. Both existing receptors were prepared by a tool of
  unrecorded version; this cannot be fixed retrospectively.

Anything obtained with this receptor is conditional on the above. See
[uma_control_results.md](uma_control_results.md) for the campaign that used
it and for how its result is scoped.

**Template support alone does not establish that ANP preparation works.**
That a flag exists for assigning templates says nothing about whether a
template matches ANP, whether its atom types and partial charges are sensible
for a phosphorylated nucleotide analogue, or whether its protonation is right
at the catalytic site. None of that has been tested.

### Exact integration checks to run on Epicac

Run in the `boltz` environment, from the repository root. These are checks,
not a campaign; none of them docks anything.

The flag inspection (step 1 of an earlier draft) is **done**: its findings are
the table above. What remains needs the structure and the tools together.

```bash
conda activate boltz

# 1. The source structure must be the one the example box records were made
#    from. Expect:
#    dba58d35a1eeedf1440abe536c930422cedadd5763d799bfc996761710a05689
./scripts/01_fetch_structure.sh 1P3D data
sha256sum data/1P3D.cif
python -c "import json; print(json.load(
    open('examples/box_murc_uma_chainE_24A.json'))['source_structure']['sha256'])"

# 2. The test suite, with the real Gemmi of this environment.
python -m unittest discover -s tests -p 'test_*.py'

# 3. The apo receptor: the first real preparation attempt, with automatic
#    bad-residue deletion disabled. Expect max deviation 0.0000 A at both
#    stages, 32 pocket residues and 234 pocket heavy atoms.
python python/prepare_receptor.py data/1P3D.cif receptors/murc_apo \
    --protein-author-chain A --condition apo \
    --box-record examples/box_murc_uma_chainE_24A.json

# 4. What Meeko could not match, and whether any of it lines the pocket.
#    If this is non-empty, decide deliberately: repair, or accept and record.
#    Do NOT reach for -a/--allow_bad_res to make it pass.
python -c "import json; m=json.load(open('receptors/murc_apo.manifest.json')); \
print(m['commands'][0]['template_matching_failures']); \
print(m['checks']['pocket_in_receptor_pdbqt']['residues_absent'])"

# 5. The box/receptor provenance gate, without docking anything.
python python/dock.py receptors/murc_apo_rigid.pdbqt pep /tmp/gate_check \
    --box-record examples/box_murc_uma_chainE_24A.json \
    --receptor-manifest receptors/murc_apo.manifest.json --check-only

# 6. ANP retention, first attempt with no template. EXPECTED to fail at the
#    ANP typing check; Meeko's own diagnostic is the useful output.
python python/prepare_receptor.py data/1P3D.cif receptors/murc_anp_try1 \
    --protein-author-chain A --condition anp-present \
    --box-record examples/box_murc_uma_chainE_24A.json

# 7. Second attempt, naming a template for ANP at author A:603. The template
#    name must be one the installed Meeko knows; check its packaged template
#    list before assuming "ANP" is accepted.
python python/prepare_receptor.py data/1P3D.cif receptors/murc_anp_try2 \
    --protein-author-chain A --condition anp-present \
    --box-record examples/box_murc_uma_chainE_24A.json \
    --meeko-argument=--set_template --meeko-argument=A:603=ANP

# 8. If no packaged template matches, --add_templates takes JSON or
#    resname:file.sdf. Report what step 7 said before writing one: a template
#    defines the chemistry and is a scientific decision, not configuration.
```

What step 5 must establish before an ANP-present receptor is used: that all 31
ANP heavy atoms appear in the PDBQT with their coordinates unchanged (the
script checks this), **and** that their atom types and partial charges are
chemically sensible for a phosphorylated nucleotide analogue — which the
script cannot check and a person must inspect. Record the template used and
the charge model in the manifest's Meeko command, which happens automatically.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Three kinds of check, deliberately distinguished:

- **pure**: atom parsing, correspondence, pocket census, manifest handling and
  box/receptor agreement. Nothing installed is needed.
- **Gemmi-backed**: the cleaning stage, run against a small real mmCIF fixture
  holding two protein copies, an MSE, a UMA, an ANP, a manganese and a water.
  Skipped if Gemmi is absent.
- **stub-Meeko**: a fake `mk_prepare_receptor.py` on `PATH` exercises the
  subprocess, output, correspondence and failure-marking checks. The stub does
  **no chemistry**: no typing, no protonation, no charges. These tests say
  nothing about whether real Meeko produces a correct receptor.

Checks that require the real tools, and therefore Epicac: Meeko's flag set and
version, which residues it drops from this structure, ANP template matching
and typing, partial charges, protonation states, and anything about Vina.

## Unresolved, and recorded in every manifest

- Manganese omission is an approximation whose effect on pocket
  electrostatics has not been assessed.
- Deleting components does not yield an experimentally apo structure.
- MSE → MET replaces selenium with sulphur without re-optimising geometry;
  pocket residue 31 is affected and its geometry has not been inspected.
- Protonation states, tautomers and partial charges come from Meeko's defaults
  and have not been reviewed for this pocket.
- Heavy-atom completeness is judged against a small internal table, not a
  chemical component dictionary.
- For `anp-present`: ANP's atom types, charges and protonation are not
  validated for this project.
