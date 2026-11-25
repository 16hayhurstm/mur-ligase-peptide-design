# UMA redocking control: the paired receptor-condition result

Procedure, criterion and vocabulary: [uma_control.md](uma_control.md).
Receptor construction: [receptor_preparation.md](receptor_preparation.md).

This document records **what was run and what was found** in the paired
comparison of two receptor conditions, together with the provenance that
supports it and the provenance that does not. It changes no threshold, no
measure and no historical result.

## 1. Status of each class of claim

The campaigns themselves ran on Epicac. This document was assembled on the
Mac, where the structure, the component dictionary, both prepared receptors
and their manifests are present but **no run directory and no analysis record
is**. Claims are therefore labelled, and the labels are not decoration:

| Label | Meaning |
| --- | --- |
| **verified here** | recomputed or re-hashed on the Mac from the file named, which is present locally |
| **recorded on Epicac** | written by the pipeline into a record on Epicac. **The file has not been transferred and has not been read on the Mac.** Taken from the operator's report of completed Epicac terminal output |
| **unresolved** | a decision or an investigation, not a result |

Every number in §4 is **recorded on Epicac**. None of the ten-seed campaign
figures has been independently read here. Before this comparison is cited
anywhere beyond the project, §9's checklist should be run on Epicac and its
output stored with the records.

## 2. The two conditions

Both receptors come from the same deposited coordinates of *Haemophilus
influenzae* MurC, PDB 1P3D, protein author chain A (canonical chain A), and
differ **only** in whether ANP is retained. The protein portion of the two
PDBQT files is bit-identical.

| | Condition A | Condition B |
| --- | --- | --- |
| name recorded in the manifest | `apo` | `anp-present` |
| receptor PDBQT | `receptors/murc_noanp_nomn_try1.pdbqt` | `receptors/murc_anp_nomn_try1.pdbqt` |
| ANP | removed | **retained**, author A:603 |
| UMA, Mn²⁺, waters | removed | removed |
| receptor heavy atoms | 3547 | 3578 (= 3547 + 31 ANP) |

"apo" is a **naming convention for an ANP-free receptor**, not an
experimentally apo structure: the coordinates remain those of the ternary
complex.

## 3. Settings, identical in both conditions

| Setting | Value |
| --- | --- |
| ligand | prepared UMA in the **−2** modelled protonation state (see §7) |
| box | 30 × 30 × 30 Å, centre [25.4385, −6.2665, 48.0135] Å |
| engine | AutoDock Vina, `sf_name="vina"`, exhaustiveness 32 |
| seeds | 11, 23, 37, 59, 71, 83, 101, 149, 211, 307 — the same ten in both |
| retained poses | 9 per seed requested (`--n-poses 9`; Vina treats this as a maximum) |
| measure | symmetry-aware heavy-atom RMSD against deposited UMA (author A:602), **no superposition of any kind** |
| criterion | RMSD ≤ 2.0 Å, predefined |

## 4. Results, as recorded on Epicac

| Metric | Condition A, protein only | Condition B, protein + ANP |
| --- | --- | --- |
| completed seeds | 10 / 10 | 10 / 10 |
| seeds with **any** retained pose ≤ 2 Å | **0 / 10** | **9 / 10** |
| seeds with the **rank-1** pose ≤ 2 Å | **0 / 10** | **7 / 10** |
| best RMSD over all retained poses | 2.609 Å | **0.677 Å** |

### How this reads under the predefined vocabulary

The vocabulary was fixed before these runs ([uma_control.md](uma_control.md),
"How to read the result") and is applied here unchanged:

- **Condition A is `no recovery`**: no retained pose in any seed meets the
  criterion. Pooled with the earlier three-seed campaign (seeds 1, 42, 137),
  that is **0 of 13 independent searches**, and the earlier best pose of
  2.240 Å — which also failed the criterion — **does not replicate**: the
  ten-seed best is worse, at 2.609 Å.
- **Condition B is a `mixed outcome`**, not consistent recovery. Consistent
  recovery requires the rank-1 pose to meet the criterion in *every* seed;
  it does so in 7. In 2 further seeds the search reached a pose within 2 Å
  but the scoring did not rank it first, and in 1 seed no retained pose met
  the criterion at all.

### What the difference does and does not mean

Retaining ANP **substantially improved recovery of the crystallographic UMA
pose** under these settings: from nothing in 13 searches to a near-native
rank-1 pose in most of 10. That is a real and useful result about the model.

It is **not**:

- validation of the native state. Both receptors still lack the two Mn²⁺
  ions and all 397 waters, and 20 of those water oxygens lie within 3.5 Å of
  deposited UMA (**verified here**, §6);
- a replicate comparison. The two conditions differ in the contents of the
  search box, so the difference is an effect of the **receptor condition**;
  nothing here estimates run-to-run variability within a condition;
- evidence that ANP's chemical representation is correct. It is
  Meeko-perceived (§5), so condition B's result is conditional on it;
- any statement about peptide binding, peptide ranking, enzyme inhibition or
  antibacterial activity. UMA is a 49-heavy-atom reaction product; the
  candidates are 8–16 residue peptides, and their scores are not comparable.

A plausible mechanism — ANP walling off part of the box and removing the
decoy placements that previously outscored the native site — is **a
hypothesis, not a finding**. Nothing here localises the improvement to a
cause, and pose-by-pose inspection has not been done.

## 5. ANP's representation in condition B

**Verified here** by reading `murc_anp_nomn_try1.pdbqt` directly, and
cross-checked against the manifest:

| Quantity | Value |
| --- | --- |
| ANP records in the receptor PDBQT | 36, all at author A:603 |
| heavy atoms | **31** |
| retained polar hydrogens (`HD`) | **5** |
| summed partial charge | **−4.0000** |
| AutoDock types | OA 12, C 5, A 5, NA 4, HD 5, P 3, N 2 |
| heavy-atom bookkeeping | 3578 − 3547 = 31, agreeing with the manifest |

How this arose, **verified here** from the manifest's captured subprocess
record: `mk_prepare_receptor.py` (Meeko 0.7.1, per
[environment.md](environment.md)) emitted

```
RuntimeWarning: Input residues {'A:603': 'ANP'} not in residue_templates
```

and built a chemical template of its own. ANP matched **no packaged Meeko
template**. Its bond orders, protonation and −4 formal charge are therefore
**perceived by the tool from a hydrogen-free PDB, not evidenced**.

> This is a plausible working representation of ANP. It is **not an
> independently validated ANP parameterisation.** The charge, the atom types
> and the four deprotonated phosphate acids follow the same convention
> applied to the UMA ligand, which makes them consistent — not correct, and
> not independent evidence, since both follow one convention.

The charge also sits in a receptor from which the neutralising metals have
been deleted: in deposited 1P3D the two Mn²⁺ ions are inner-sphere
coordinated to ANP's phosphates, so condition B carries ANP's full −4 inside
the box with +4 absent.

**Outstanding, and gating any stronger claim about condition B:** template
matching, atom typing, partial charges and protonation for ANP, as set out in
[receptor_preparation.md](receptor_preparation.md). Until those are resolved,
condition B's result is reported as obtained under an unvalidated cofactor
representation.

## 6. Components removed from the selected complex

**Verified here** from both manifests, which agree:

| Removed from chain A | Count | Recorded reason |
| --- | --- | --- |
| UMA | 1 | the pocket-defining reaction product; the site must be free to dock into |
| **Mn²⁺** | **2** | manganese omitted as a documented approximation, not a validated biological equivalence |
| **HOH** | **397** | crystallographic waters removed |
| ANP | 1 (condition A only) | ATP analogue absent in the initial condition |

Chain B was discarded as a separate, explicitly labelled step, not as a
component removal.

### What neighbours the deposited ligand, measured rather than assumed

Recomputed here with Gemmi from `data/1P3D.cif`, heavy atoms of chain A,
minimum distance to the 49 heavy atoms of UMA A:602:

| Component | Atoms | Min. distance to UMA | Atoms ≤ 3.5 Å |
| --- | --- | --- | --- |
| HOH | 397 | **2.54 Å** | **20** |
| ANP | 31 | 3.96 Å | 0 |
| Mn²⁺ | 2 | 4.15 Å | 0 |

Two consequences for how the removals are described:

1. **Manganese does not coordinate UMA in 1P3D.** Neither ion comes within
   3.5 Å of any UMA atom. The two metals coordinate ANP's phosphate oxygens
   and protein side chains. A metal-to-carboxylate interaction described in
   the literature concerns **UNAM, the substrate**, in a different structure,
   and must not be transferred to UMA, the product, here. Manganese omission
   remains a documented approximation and an open decision — but the reason
   to care about it is its effect on ANP and on pocket electrostatics, not a
   lost contact with UMA.
2. **Water removal is the largest chemical change to the UMA site itself**,
   and larger than either cofactor deletion. 20 water oxygens contact the
   ligand directly. This is an interpretative limitation of both conditions
   equally, so it does not confound the A-versus-B comparison, and it is not
   a reason to add fixed waters: Vina's hydrated-docking protocol is
   calibrated for the AutoDock4 scoring function, not for `vina`.

## 7. Protonation: the campaigns used the sensitivity state

[uma_control.md](uma_control.md) designates **−3 as the primary modelled
state** (α-phosphate OH, β-phosphate OH and the L-alanine carboxylic acid
deprotonated at pH 7.4) and **−2 as a planned sensitivity condition** in
which `O2B`, the uncertain second diphosphate ionisation, retains its proton.
The repository's own naming agrees: the −2 plan is `sensitivity_minus2`.

**Both campaigns in §4 were run on the −2 ligand.** That is recorded here as
an inconsistency between the designation and the runs, and is **not corrected
retrospectively**: the runs are what they are, and the −2 state is a
deliberately modelled state with its own validated preparation record.

What is affected, and what is not:

- The two states differ in hydrogen-bond donor/acceptor typing and in the
  torsional count (`TORSDOF` 19 at −3, 20 at −2). They do **not** differ in
  partial-charge effect on the score: the `vina` scoring function ignores
  user-supplied partial charges, which only `ad4` uses.
- The state **does** determine the RMSD measure's symmetry. At −3 the
  resonance-normalised graph has 8 automorphisms (O1A/O2A, O1B/O2B, O19/O20
  interchangeable); at −2 it has 4, because the protonated phosphate's
  O1B/O2B pair is correctly **not** treated as equivalent. Both counts were
  recomputed here. This is why the protonation record must travel with the
  analysis, and why the measure is not state-agnostic.
- **Unresolved:** whether to re-designate −2 as primary with its reasoning
  recorded, or to repeat the paired comparison at −3. Either is defensible;
  silently reporting the sensitivity state as the primary one is not.

## 8. The crystallographically observed alternative UMA conformation

**Unresolved investigation. Not an explanation of anything in §4.**

1P3D contains two crystallographically independent UMA molecules, and Mol
*et al.* report them in **different conformations**, with the muramic acid
carboxylate of one rotated roughly 180° from the other, suggested there as a
possible early product-release step.

Quantified here, by best-fit superposition of the two deposited copies
(A:602 against B:1602, 49 atoms — **the protein is not what was superposed in
this measurement, and this measurement is not the docking measure**):

| Fragment | n | mean deviation | max |
| --- | --- | --- | --- |
| uridine | 17 | 0.27 Å | 0.43 Å |
| diphosphate | 8 | 0.32 Å | 0.51 Å |
| muramyl ring | 13 | 0.33 Å | 0.48 Å |
| **lactyl + alanine** | 11 | **1.85 Å** | **4.66 Å** |

Fragments are the CCD atom-name groups used by
[`../examples/uma_pose_comparison.cxc`](../examples/uma_pose_comparison.cxc),
with the fourth row taking every remaining atom, so the four rows partition
all 49. Overall best-fit RMSD **1.141 Å**; the largest displacements are C20
4.66 Å, O18 4.46 Å, C18 2.48 Å and C19 2.38 Å — the lactyl group — with the
alanine carboxylate moving about 1 Å, so the alanine does **not** quite
superimpose.

What this does and does not license:

- It establishes that **the experimental reference is not a point**: the same
  ligand is observed in two conformations differing by more than half the
  2.0 Å criterion in exactly the fragment that moves.
- It is **not** an explanation for displaced docking poses. No pose in either
  campaign has been shown to resemble copy B, because nobody has measured it.
  Attributing a failed pose to "the other conformer" without that measurement
  would be a post-hoc rescue of a failed endpoint.
- The **predefined reference stays primary**, and the 2.0 Å criterion is
  unchanged. Relaxing either because of this observation would be
  retrospective threshold movement.
- The investigation that would settle it, and is **not started**: superpose
  the two **protein** chains — never the ligands — to bring copy B into chain
  A's frame, then measure every retained pose of both campaigns against copy
  B in that frame, reporting it as a second reference alongside the first and
  never instead of it.

## 9. Provenance

### Confirmed here, by re-hashing or recomputation

| Item | Result |
| --- | --- |
| source structure `data/1P3D.cif` | sha256 `dba58d35…05689`, equal to the hash recorded in all three box records and in both receptor manifests |
| `murc_noanp_nomn_try1.pdbqt` | sha256 `76601616…60d13`, equal to its manifest's `receptor_pdbqt_sha256` |
| `murc_anp_nomn_try1.pdbqt` | sha256 `2229becd…ca3f2`, equal to its manifest's `receptor_pdbqt_sha256` |
| prepared PDBs of both conditions | hashes equal to their manifests' `prepared_pdb_sha256` |
| pocket integrity, both conditions | 32 / 32 residues and 234 pocket heavy atoms found before cleaning, after cleaning and in the PDBQT; max coordinate deviation 0.000 Å at both stages against a 0.002 Å tolerance |
| residue transformations, both conditions | MSE→MET **14** residues (A:15, 31, 110, 115, 135, 138, 187, 194, 199, 209, 228, 236, 371, 400); **KCX→LYS count 0** |
| box geometry | all three example records share centre [25.4385, −6.2665, 48.0135] Å, `centre_mode` `bounds-midpoint`, `sizing.mode` `explicit-size`, pocket cutoff 5.0 Å, 32 pocket residues |
| ANP census in condition B | §5 |
| component removals | §6 |
| UMA neighbour distances | §6 |
| automorphism counts at −2 and −3 | §7 |
| CCD `UMA.cif` | sha256 `1efc8757…c32eaf`, the dictionary the pose-conversion records name |

### Flagged, not silently corrected

1. **The Meeko version is not actually recorded.** Both manifests' Meeko
   entry reads `"usage: mk_prepare_receptor.py [-h] [--read_pdb PDB_FILENAME]"`:
   `receptor_provenance.tool_version()` takes the first line of the tool's
   output and is called with `--help`. Gemmi 0.6.5 and Python 3.12.13 are
   captured correctly. "Meeko 0.7.1" comes from
   [environment.md](environment.md), not from any record. Both existing
   receptors were therefore prepared by a tool of **unrecorded version**.
   This cannot be fixed retrospectively and is stated as a provenance
   limitation; fixing the capture for future preparations is a separate,
   small change and is **not** made here.
2. **The receptors' pocket selection is tied to the 24 Å box record**
   (`examples/box_murc_uma_chainE_24A.json`, sha256 `1ae0014c…e055f`), while
   docking used the 30 Å box. Nothing is scientifically wrong: the pocket is
   defined by the 5.0 Å cutoff and is identical in both records, and the
   box/receptor gate deliberately compares source structure, chain, cutoff
   and residue list rather than box size. It is named because a reader
   checking provenance will otherwise look for the 30 Å record and find a
   mismatch.
3. **KCX→LYS was not applied to this receptor.** The code path exists and is
   correctly documented, but 1P3D chain A contains no carbamylated lysine and
   the recorded count is 0. Only the 14 MSE substitutions were applied.
4. **The box records were written by an unidentified working tree.** All
   three record `commit 47723c0…` with `working_tree_dirty: true`, so the
   commit does not identify the code that produced the centre.
5. **The ANP-present campaign's own records have not been read here.** No
   run directory, analysis record, per-pose table or ligand record for either
   ten-seed campaign is present on this machine.

### Evidence files supporting §4

**Every path in this subsection is an Epicac path, established from completed
Epicac terminal output. None of these files exists on the Mac and none has
been read here.** They are named so the record can be reproduced and audited
on the machine that holds them, not because their contents have been
inspected.

| Condition | Run directory (Epicac) | Analysis record (Epicac) |
| --- | --- | --- |
| A, protein only, `apo` | `runs/uma_minus2_30A_seeds10/` | `analysis/uma_minus2_30A_seeds10_validated.analysis.json` |
| B, protein + ANP, `anp-present` | `runs/uma_minus2_anp_30A_seeds10_20261124/` | `analysis/uma_minus2_anp_30A_seeds10_validated.analysis.json` |

Each run directory is expected to hold `run_record.json`, the per-seed
`seed_*` pose files, `dock_results.csv` and `failures.csv`; each analysis
record is expected to sit beside its `*_per_pose.csv` written from the same
prefix. **"Expected" means what the pipeline writes, not what has been
confirmed in these two directories** — §9's checklist is what confirms it.

Shared inputs to both campaigns:

| Input | Where it was inspected |
| --- | --- |
| `data/1P3D.cif` | **Mac**, re-hashed (§9) |
| `examples/box_murc_uma_chainE_{24,30}A.json` | **Mac**, tracked in the repository, re-hashed (§9) |
| both receptor PDBQTs and their manifests | **Mac**, as read-only copies under `project_private/epicac_receptor_validation/receptors/`, re-hashed against the manifests (§9) |
| CCD `UMA.cif` | **Mac**, `project_private/ccd_downloads/`, re-hashed (§9) |
| `ligands/uma_minus2.pdbqt` with its `*.ligand.json`, `*.atom_map.json`, `*.preparation.json` and `uma_minus2_verified.verification.json` | **Epicac only.** The −3 state's PDBQT and verification record were transferred and inspected on the Mac; the −2 state's were not |
| `references/uma_chainE.reference.json` | **Epicac only** |
| `analysis/uma_minus2.independent_rmsd.audited.json` | **Epicac only** |

### Read-only checklist to complete the record on Epicac

Writes nothing. Run from the repository root in the `boltz` environment, and
store the output alongside the analysis records.

```bash
# 1. Which code was executing.
git rev-parse HEAD
git status --short

# 2. Inputs, including the box record the receptors were selected against.
sha256sum data/1P3D.cif \
          examples/box_murc_uma_chainE_24A.json \
          examples/box_murc_uma_chainE_30A.json \
          receptors/murc_noanp_nomn_try1.pdbqt \
          receptors/murc_anp_nomn_try1.pdbqt \
          ligands/uma_minus2.pdbqt

# 3. For each campaign: the receptor actually docked against, the seeds, the
#    box, the exhaustiveness and whether provenance was verified.
for RUN in runs/uma_minus2_30A_seeds10 \
           runs/uma_minus2_anp_30A_seeds10_20261124; do
  python - "$RUN" <<'PY'
import json, sys
record = json.load(open(f"{sys.argv[1]}/run_record.json"))
for key in ("receptor", "receptor_condition", "receptor_provenance_verified",
            "seeds", "exhaustiveness", "cpu", "box_centre_angstrom",
            "box_size_angstrom", "status", "n_failures"):
    print(f"{key:32} {record.get(key)}")
PY
  grep -c '^MODEL' "$RUN"/seed_*  # retained modes per seed
done

# 4. Both analyses: the condition, the prerequisites and the linkage.
for A in analysis/uma_minus2_30A_seeds10_validated \
         analysis/uma_minus2_anp_30A_seeds10_validated; do
  python - "$A.analysis.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print(sys.argv[1])
print("  interpretable          ", record["interpretable"])
print("  receptor_condition     ",
      record["docking_context"].get("receptor_condition"))
print("  provenance verified    ",
      record["docking_context"].get("receptor_provenance_verified"))
print("  pose linkage           ",
      record["measure"]["atom_correspondence"].get("pose_linkage", {})
            .get("established"))
for limit in record["interpretation_limits"]:
    print("  limit:", limit)
PY
done
```

Step 4 is also the check that the condition-specific interpretation reaches
the records: an `anp-present` analysis must **not** print that the receptor
lacks ANP. Analyses written before that fix carry the old unconditional
sentence; re-running `analyse_redocking.py` on the existing run directories
regenerates the record and changes no pose, no score and no RMSD.

## 10. Decision taken: the standard docking receptor

**Agreed project-level modelling decision, 25 November 2026.** The
**ANP-containing MurC receptor** (`anp-present`) is the **standard docking
model** for both BoltzGen peptide cohorts. The ANP-free receptor (`apo`)
becomes an **archived comparison condition**: it is kept, its records and
outputs are kept unmodified, and it remains available as a deliberate
second condition.

Basis: it is the only one of the two in which the cognate-ligand pose
recovery of §4 worked at all, so it is the condition for which we have any
evidence that the setup can reproduce a known answer in this pocket.

What this decision is **not**:

- not evidence that peptide binding affinities have been validated, or
  measured, or that any design binds anything;
- not a validation of ANP's chemical representation, which remains
  Meeko-perceived and unvalidated (§5), and is now a dependency of the
  standard model rather than of a side condition — which raises, not lowers,
  the value of completing the ANP chemical checks;
- not a claim that the standard receptor is the native state. Manganese and
  all 397 waters are still absent (§6);
- not a statement about peptide poses, peptide rankings, enzyme inhibition or
  antibacterial activity.

**It is not hard-coded anywhere, deliberately.** `prepare_receptor.py`
requires `--condition` explicitly, with no default and no fallback;
`dock.py` takes the receptor and its manifest as arguments and records the
condition it found. A standard condition chosen by project decision must
still be named in every command and appear in every run record, so that a
campaign run against the other condition is visible as such rather than
silently mislabelled.

**Consequence for the next phase:** both design campaigns are docked in this
one receptor condition, which keeps the generation-condition factor separate
from the receptor-condition factor. A receptor-condition crossover, if run,
is a separate stratified sub-study against the archived `apo` receptor.

## 11. Summary of what is established

- Under these settings, in the ANP-free receptor, Vina does **not** recover
  the crystallographic UMA pose: 0 of 13 searches, best 2.609 Å over ten
  seeds.
- Retaining ANP changes that substantially: rank-1 recovery in 7 of 10 seeds
  and a best pose of 0.677 Å, a **mixed outcome** under the predefined
  definition and the better of the two conditions by a wide margin.
- Both results are **pose-recovery results for one cognate ligand in a
  simplified receptor**, conditional on a Meeko-perceived ANP representation
  in condition B and obtained in the sensitivity protonation state.
- They say nothing about peptide poses, peptide rankings, binding
  affinities, enzyme inhibition or antibacterial activity.
