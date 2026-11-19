# The UMA redocking control

Redocking the cognate ligand asks one question: can this box, this receptor and
these search settings put UMA where the crystal has it? It tests the docking
setup, not the biology, and either answer is useful. A failure would be strong
evidence against using the setup for peptides.

Nothing in this document has been docked yet.

## What UMA is

`UMA` is **uridine-5'-diphosphate-N-acetylmuramoyl-L-alanine**: the MurC
reaction product, including the L-alanine that MurC adds. Not UDP-MurNAc
alone. From the Chemical Component Dictionary:

| Property | Value |
| --- | --- |
| Formula | C23 H36 N4 O20 P2 |
| Heavy atoms | 49 |
| Heavy-atom bonds | 51 (43 single, 8 double) |
| Hydrogens in the neutral form | 36 |
| Dictionary formal charge | 0 |
| Declared stereocentres | 13 |
| InChIKey | NTMMCWJNQNKACG-KBKUWGQMSA-N |

Deposited 1P3D canonical chain E (author A:602) carries all 49 heavy atoms,
with names and elements identical to the dictionary. Checked, not assumed.

## The criterion

**Symmetry-aware heavy-atom RMSD ≤ 2.0 Å to deposited UMA chain E, in the
receptor coordinate frame, with no independent ligand superposition.**

- **No superposition.** A pose is compared where the search put it. Best-fit
  superposition would answer "is this the right shape?" rather than "is this
  in the right place", and a pose 10 Å out of the pocket must score 10 Å.
- **Symmetry-aware.** Three groups in UMA have interchangeable terminal
  oxygens: both phosphates (`O1A`/`O2A`, `O1B`/`O2B`) and the alanine
  carboxylate (`O19`/`O20`). Their separations in the deposited pose are
  2.61, 2.61 and 2.19 Å. A physically identical pose with those labels
  swapped scores **0.867 Å** by name-ordered RMSD and **0.000 Å** with the
  symmetry correction: 43% of the 2.0 Å budget, spent on nothing.
- **Heavy atoms only.** Hydrogen positions are placed by preparation software
  and are not experimental.

Resonance normalisation is what creates that symmetry: a connection table
records one phosphate oxygen as double-bonded and the other as single-bonded,
so without it the group is frozen. With it, UMA has 8 automorphisms; without
it, 1.

**Equivalence is restricted to deprotonated groups.** A group is only treated
as resonance-equivalent when *none* of its terminal oxygens bears a hydrogen at
the modelled state. A neutral carboxylic acid's two oxygens are not
equivalent — one holds the proton, the other is the carbonyl — and treating
them as interchangeable would let a wrong pose pass. The symmetry therefore
depends on the protonation state, and `analyse_redocking.py` takes it from the
ligand manifest via `--ligand-manifest` rather than assuming one. Measured on
real UMA chemistry:

| Modelled state | Automorphisms | Orbits |
| --- | --- | --- |
| neutral dictionary form | 1 | none |
| −3 (`O2A`, `O2B`, `O20` deprotonated) | 8 | all three oxygen pairs |
| partial (`O2A` only) | 2 | `O1A`/`O2A` only |

Without a ligand manifest nothing is normalised, and the reported RMSD values
are **upper bounds** — the conservative direction.

**Stereochemistry is not verified by this measure and cannot be.** The
automorphism search works on a graph, which holds no 3D information. It
permits only mappings between constitutionally equivalent atoms, and for UMA no
permitted mapping touches a carbon stereocentre. But if a preparation step
inverted a centre, both structures would still share one graph and the RMSD
would report the geometric consequence without identifying it as a
stereochemical error; where an inverted centre is graph-symmetric with another,
an automorphism could even understate the deviation. That blind spot is fixed
by a test so it stays deliberate. The measure **assumes both structures are
the same stereoisomer**, which is why stage 3 checks CIP codes.

**This RMSD implementation is not independently validated.** It is custom
code, and its tests were written against the same understanding that produced
it, so they show self-consistency rather than correctness. Before any value
from it is relied on, compare it against an established toolkit on Epicac.
RDKit's `rdMolAlign.CalcRMS` is the right comparator because it computes a
symmetry-corrected RMSD **without alignment**, which is the same quantity:

```bash
conda activate boltz
python - <<'PY_INNER'
import sys; sys.path.insert(0, "python")
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import chemical_component, chemical_graph

# Same two structures, read independently by each implementation.
reference = Chem.MolFromMolFile("references/uma_reference_withH.mol",
                                removeHs=True, sanitize=True)
pose = Chem.MolFromMolFile("analysis/uma_pose1.mol", removeHs=True,
                           sanitize=True)
print("RDKit CalcRMS (no alignment, symmetry-aware):",
      rdMolAlign.CalcRMS(pose, reference))

component = chemical_component.read_component("<CCD>/UMA.cif")
still = sorted(set(component.hydrogens_on) - {"O2A", "O2B", "O20"})
graph, _ = chemical_graph.normalise_resonance(component.graph,
                                              hydrogen_bearing=still)
# ... supply the same coordinates as dicts and compare the two numbers.
PY_INNER
```

They should agree to within floating-point noise on the same pair of
structures. If they disagree, **this implementation is wrong until shown
otherwise** and the toolkit value is the one to trust. Run the comparison on
at least one near-reference pose and one distant pose, and on a pose with the
phosphate oxygens relabelled, which is where the symmetry handling is actually
exercised.

**Consistent recovery** means the top-ranked pose meets the criterion in **all
three seeds**. Anything else is a mixed outcome, reported as such with the
continuous RMSD values. Two seeds at 2.1 Å is not the same situation as one
seed at 9 Å, and a pass/fail would lose that.

Top-ranked and closest-to-reference poses are reported **separately** for each
seed, with scores and ranks:

- top-ranked asks whether the scoring function *prefers* a near-reference pose;
- closest-to-reference asks whether the search *reached* one at all.

A good closest with a poor top-ranked is a scoring problem; a poor closest is a
sampling or search-space problem.

## Protonation: provisional, and the evidence is thin

The dictionary entry is the **neutral** molecule, formal charge 0, every acid
protonated. That is the authority for connectivity and stereochemistry and the
wrong thing to dock at physiological pH. The modelled state is a separate
decision, and it is recorded as **provisional**.

### What is actually known

**We have not located a measured pKa for UMA.** A search found none, and the numbers an
earlier draft of this document gave — "O2A pKa ≈ 1–2", "O2B pKa ≈ 6.5",
"O20 pKa ≈ 3.5" — were **functional-group estimates presented as if they were
UMA values. They are withdrawn.**

What the evidence does consist of, separated by kind:

| Kind | Source | What it covers |
| --- | --- | --- |
| **Measured, close analogue** | Jancan & Macnaughtan, *Anal. Chim. Acta* **749**, 63–69 (2012), doi:10.1016/j.aca.2012.08.052 | ³¹P NMR determination of the **second diphosphate** and **uridine imide** pKa of UDP, UDP-GlcNAc and UDP-S-GlcNAc. The paper's numeric values have not been read and are not reproduced here. |
| **Software prediction, this compound** | HMDB0011720 and PAMDB000485, for UDP-*N*-acetylmuramate | A predicted strongest-acidic pKa near 1.7 (ChemAxon/ALOGPS). A prediction, not a measurement, and it refers only to the most acidic site. |
| **Functional-group expectation** | General organic chemistry | A phosphate diester's first ionisation is strongly acidic; a second diphosphate ionisation sits near neutrality; an amide-linked carboxylic acid is acidic well below pH 7; alcohols, amides and the uracil imide are not ionised near pH 7. |

Absence of a located measurement is not proof that none exists; it means the modelling decision rests on what is listed above. Note what is *absent* from all three: UMA differs from UDP-GlcNAc by the
muramyl lactyl ether and the L-alanine amide, and the alanine carboxylate has
no analogue in the measured set at all.

### The state being modelled, and its status

| Site | Group | pH 7.4 | Basis |
| --- | --- | --- | --- |
| `O2A` | α-phosphate OH | deprotonated | functional-group expectation; strongly acidic |
| `O2B` | β-phosphate OH | deprotonated | **the uncertain one**; analogue data only |
| `O20` | L-Ala carboxylic acid | deprotonated | functional-group expectation |
| `O2'`, `O3B`, `O4'`, `O6'` | hydroxyls | retained | alcohols |
| `N2'`, `N4` | amide N–H | retained | amides are not acidic |
| `N3` | uracil imide | retained | analogue data puts this well above pH 7 |

**Modelled formal charge −3, provisional**, derived as 0 − 3 acidic protons.

The choice between −3 and −2 rests entirely on `O2B`, the second diphosphate
ionisation, for which only analogue measurements exist. If that pKa is near
7.4 the compound is a mixture at the modelled pH and neither integer state is
right.

**−2 is therefore a planned sensitivity condition, not a contingency.** It is
run and reported whatever the −3 result looks like, because robustness to the
protonation assumption cannot be judged from the −3 run alone: a clean −3
recovery that collapsed at −2 would mean the control was resting on an
unverified assumption, and that is exactly what a sensitivity check exists to
reveal. Making the −2 run conditional on a marginal −3 result would guarantee
it was only ever done when it was least informative.

Reading the Jancan & Macnaughtan values remains worth doing and would narrow
the question, but it does not replace the sensitivity run.

The same treatment is what Meeko applied to ANP in the receptor — neutral
dictionary entry, all four phosphate acids deprotonated, total −4. That
consistency is a reason to prefer this state over the neutral form. It is
**not** evidence that the state is correct, and it is not independent, since
both choices follow the same convention.

### Stereochemistry: how it is carried, and how it is checked

The molfile written by stage 2 carries **no parity flags and no wedge bonds**:
every atom and bond stereo field is zero. Stereochemistry is held **implicitly
in the 3D coordinates**, together with the explicit connection table. That is
sufficient — a tetrahedral centre's configuration is determined by its
substituents' positions — but it means nothing in the file *asserts* a
configuration, so nothing can contradict the coordinates, and equally nothing
checks them.

Verification is therefore an explicit step in stage 3: RDKit's
`AssignStereochemistryFrom3D` derives CIP codes from the coordinates, and those
are compared against the dictionary's declared configurations.

The 13 declared stereocentres are **not** all of one kind:

- **11 carbon centres** (`C1B`, `C2B`, `C3B`, `C4B`, `C1'`, `C2'`, `C3'`,
  `C4'`, `C5'`, `C18`, `C21`) — the muramyl sugar, the ribose and the alanine
  α-carbon. These are genuine, configurationally stable stereocentres. Every
  one must match the dictionary. A mismatch means the wrong stereoisomer and
  blocks use.
- **2 phosphorus labels** (`PA`, `PB`) — the dictionary declares `R` for both.
  In the **neutral** form each phosphorus does carry four different
  substituents, so a configuration can be written. In the **modelled −3
  state** the two terminal oxygens of each phosphate are equivalent by
  resonance, so neither phosphorus is a stereocentre at all, and the label has
  no referent. Phosphate stereochemistry of this kind is in any case not
  configurationally stable in solution.

**This is treated deliberately, not ignored**: a CIP mismatch at `PA` or `PB`
is expected and is not a failure, while a mismatch at any of the 11 carbons
is. The same reasoning licenses the resonance normalisation used by the RMSD —
the oxygens being exchanged belong to groups that are not stereocentres in the
modelled state — so the two treatments are consistent rather than
independently convenient.

## The staged procedure

Every command below is the real CLI, taken from `--help`. Stages 1, 2 and 3
need only Gemmi and run anywhere; stage 4 needs RDKit; stage 5 needs Meeko;
stage 6 needs neither. Only stage 7 docks, and it is not part of this
milestone.

The component dictionary is fetched rather than transferred:

```bash
conda activate boltz
cd ~/murc-docking-validation
./scripts/02_fetch_component.sh UMA components
# Expect sha256 1efc8757b1ffa8a82d060f6e68cf92de9e0ed345f022af4eb6566a22e1c32eaf
```

If that hash differs, the CCD entry has been revised since this procedure was
written. Stop and decide deliberately: a revision can change atom names or
bond orders, and every downstream record pins the hash it used.

### Stage 1 — reference extraction (Gemmi)

```bash
python python/ligand_reference.py \
    data/1P3D.cif UMA references/uma_chainE \
    --ligand-canonical-chain E \
    --component-dictionary components/UMA.cif \
    --box-record examples/box_murc_uma_chainE_24A.json
```

Writes `references/uma_chainE_reference.pdb` and
`references/uma_chainE.reference.json`. Expect 49 heavy atoms, an atom set
identical to the dictionary, 13 declared stereocentres and a maximum deviation
from the deposited coordinates of 0.0000 Å. Both outputs are refused if they
already exist: **the reference is written once and never replaced.** If a
reference already exists from an earlier transfer, keep it and skip this stage.

### Stage 2 — modelled ligand construction (Gemmi)

```bash
python python/prepare_ligand.py build \
    references/uma_chainE.reference.json ligands/uma \
    --component-dictionary components/UMA.cif \
    --ph 7.4 \
    --deprotonate O2A,O2B,O20 \
    --protonation-justification "Provisional -3 at pH 7.4: the two \
diphosphate hydroxyls O2A and O2B and the L-alanine carboxylic acid O20 are \
taken as deprotonated; the O2', O3B, O4', O6' hydroxyls and the N2', N3, N4 \
amide/uracil N-H are retained. No measured UMA pKa has been located; the \
basis is functional-group expectation plus measured analogues (UDP, \
UDP-GlcNAc). The O2B ionisation is the uncertain one and -2 is run as a \
planned sensitivity condition. See docs/uma_control.md."
```

Writes `ligands/uma_ligand.mol` (49 atoms, 51 bonds, `M  CHG` with three −1
entries, no hydrogens) and `ligands/uma.ligand.json`.

Then the same for the sensitivity state, to a separate prefix:

```bash
python python/prepare_ligand.py build \
    references/uma_chainE.reference.json ligands/uma_minus2 \
    --component-dictionary components/UMA.cif \
    --ph 7.4 \
    --deprotonate O2A,O20 \
    --protonation-justification "Planned sensitivity condition at -2: O2B \
retained as the protonated phosphate hydroxyl. Run and reported regardless of \
the -3 outcome, because robustness to the protonation assumption cannot be \
judged from the -3 run alone."
```

### Stage 3 — stereochemistry plan (Gemmi)

```bash
python python/prepare_ligand.py stereo-plan \
    ligands/uma.ligand.json \
    --out ligands/uma.stereo_plan.json \
    --state sensitivity_minus2=O2A,O20
```

Fixes the expectation **before** any result is seen, computing for each state
which declared centres are stereogenic. Expected output:

| State | Charge | Must match | Not stereogenic |
| --- | --- | --- | --- |
| `modelled` | −3 | **11 carbons**: `C1B`, `C2B`, `C3B`, `C4B`, `C1'`, `C2'`, `C3'`, `C4'`, `C5'`, `C18`, `C21` | `PA`, `PB` — terminal oxygens equivalent |
| `sensitivity_minus2` | −2 | the same 11 carbons **plus `PB`** | `PA` only |

`PB` moving between columns is the point of planning per state: at −2 it keeps
its hydroxyl, so its four substituents are distinct and the connectivity rule
makes it formally stereogenic. The plan marks it `fatal_if_mismatched: false`,
because phosphorus configuration of that kind is not configurationally stable
in solution — a mismatch there is **investigated**, not treated as the wrong
stereoisomer. A mismatch at any of the 11 carbons is fatal.

### Stage 4 — RDKit checks and hydrogen addition (RDKit)

```bash
python - <<'PY_INNER'
import json
from rdkit import Chem

plan = json.load(open("ligands/uma.stereo_plan.json"))
state = plan["states"]["modelled"]

mol = Chem.MolFromMolFile("ligands/uma_ligand.mol", removeHs=False,
                          sanitize=True)
assert mol is not None, "RDKit could not read the molfile"
charge = Chem.GetFormalCharge(mol)
print("formal charge:", charge, "expected", state["formal_charge"])
assert charge == state["formal_charge"], (
    "the M CHG block was not applied; this is the neutral molecule")
print("heavy atoms  :", mol.GetNumHeavyAtoms(), "expected 49")

Chem.AssignStereochemistryFrom3D(mol)
assigned = {}
for atom in mol.GetAtoms():
    if atom.HasProp("_CIPCode"):
        name = atom.GetPropsAsDict().get("molFileAlias") or atom.GetIdx()
        assigned[atom.GetIdx()] = (atom.GetSymbol(), atom.GetProp("_CIPCode"))

# The molfile atom order is the dictionary's heavy-atom order, so index i
# corresponds to the i-th name in the plan's dictionary ordering.
names = [entry["atom"] for entry in state["must_match"]]
print("\nmust-match centres:", names)
print("assigned CIP codes by index:", assigned)
print("\nCompare each must-match centre's code with declared_configuration")
print("in the plan. Record the comparison; do not summarise it as a pass.")

withH = Chem.AddHs(mol, addCoords=True)
Chem.MolToMolFile(withH, "ligands/uma_withH.mol")
print("\natoms with hydrogens:", withH.GetNumAtoms(),
      "( added:", withH.GetNumAtoms() - mol.GetNumAtoms(), ")")
PY_INNER
```

Write the comparison to `ligands/uma.stereo_verification.json` — the file
`analyse_redocking.py --stereo-verification` later expects — recording, for
every must-match centre, its declared and assigned configuration and whether
they agree. **Atom-index to name mapping is the one thing to get right here**:
the molfile's atom order is the dictionary's heavy-atom order, so confirm the
correspondence explicitly rather than assuming RDKit preserved names.

### Stage 5 — Meeko preparation (Meeko)

```bash
mk_prepare_ligand.py -i ligands/uma_withH.mol -o ligands/uma.pdbqt
echo "exit status: $?"
```

### Stage 6 — preparation verification (nothing extra)

```bash
python python/prepare_ligand.py check \
    ligands/uma.ligand.json ligands/uma.pdbqt \
    --charge-model gasteiger \
    --preparation-command "mk_prepare_ligand.py -i ligands/uma_withH.mol -o ligands/uma.pdbqt"
```

Refuses to pass if a heavy atom moved beyond 0.002 Å from the reference, if a
heavy atom is missing or unexpected, if an atom name is **repeated** — a
duplicate record would otherwise collapse into a name set that still matched —
or if the partial charges do not sum to the recorded formal charge. Records the
hydrogens added, the AutoDock type census, the charge sum, `TORSDOF` and the
`BRANCH` count.

### Stage 7 — the independent RMSD comparison (RDKit)

This is a **prerequisite for interpreting the control**, not an optional
cross-check, and it must use the **same protonation state and the same
chemically allowed atom equivalences** as this project's implementation.
Otherwise a disagreement would only show that two different quantities were
computed.

```bash
python - <<'PY_INNER'
import json, sys
sys.path.insert(0, "python")
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import chemical_component, chemical_graph

manifest = json.load(open("ligands/uma.ligand.json"))
deprotonated = set(manifest["protonation_state"]["deprotonated_atoms"])

# Same protonation state on both sides: RDKit perceives equivalence from the
# molecule it is given, so it must be given the charged molecule, not the
# neutral dictionary form.
component = chemical_component.read_component(manifest["dictionary"]["path"])
still = sorted(set(component.hydrogens_on) - deprotonated)
graph, resonance = chemical_graph.normalise_resonance(
    component.graph, hydrogen_bearing=still)
permutations = chemical_graph.automorphisms(graph)
print("this project: automorphisms", len(permutations))
print("orbits:", chemical_graph.symmetry_classes(graph, permutations))
print("resonance groups:", resonance)

# RDKit side: CalcRMS is symmetry-corrected and applies NO alignment, which
# is the same quantity this project computes.
reference = Chem.MolFromMolFile("ligands/uma_ligand.mol", removeHs=True,
                                sanitize=True)
for label, pose_file in (("near", "analysis/pose_near.mol"),
                         ("distant", "analysis/pose_distant.mol"),
                         ("oxygens relabelled", "analysis/pose_swapped.mol")):
    pose = Chem.MolFromMolFile(pose_file, removeHs=True, sanitize=True)
    print(f"{label:20s} RDKit CalcRMS:", rdMolAlign.CalcRMS(pose, reference))
PY_INNER
```

Compare each value with the RMSD this project reports for the same pose, which
appears in `<out>_per_pose.csv` alongside
`atoms_exchanged_by_the_minimising_mapping` — the mapping that achieved the
minimum, recorded by atom name precisely so a disagreement can be diagnosed.

**If the two disagree, investigate both implementations and both mappings.**
Neither is presumed right: RDKit's substructure-match symmetry perception and
this project's resonance-normalised graph automorphisms can legitimately
differ in what they consider equivalent, and the first thing to compare is the
set of exchanged atoms, not the numbers. Record the outcome in
`analysis/uma.independent_rmsd.json`, the file
`analyse_redocking.py --independent-rmsd` expects. The "oxygens relabelled"
case is the one that actually exercises the symmetry handling; agreement on
the near and distant cases alone proves little.

## Stage 8: full docking searches (a later milestone)

**Not part of the current milestone.** Stages 1-7 are the transfer target:
real ligand preparation and the independent analytical checks. Nothing below
runs until the stereochemistry comparison and the independent RMSD comparison
are both recorded.

**These are full global searches, not refinements of the crystallographic
pose.** Vina's `dock()` randomises position, orientation and torsions within
the box; the input coordinates serve only to let stage 4 verify that
preparation did not move the molecule. `optimize()` and `--local_only` perform
a local minimisation from the input pose and would make the control
meaningless — neither is used by `dock.py` and neither must be added.

`--cpu 2` is deliberate for this initial pilot: epicac is shared, and Vina
otherwise starts one thread per exhaustiveness pass, which has previously taken
32 of 48 cores. Raise it only after checking who else is on the machine, and
record the value used — run time is not comparable across different thread
counts.

```bash
conda activate boltz
cd ~/murc-docking-validation

# One run per provisional box, same receptor, same seeds, same exhaustiveness.
# Each writes a fresh directory; none is reused.
for SIZE in 24 30 33; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python python/dock.py \
      receptors/murc_noanp_nomn_try1.pdbqt \
      ligands/uma_only \
      runs/uma_control_${SIZE}A \
      --box-record examples/box_murc_uma_chainE_${SIZE}A.json \
      --receptor-manifest receptors/murc_noanp_nomn_try1.manifest.json \
      --seeds 1,42,137 --exhaustiveness 32 --cpu 2 --n-poses 9
  echo "exit status for ${SIZE} A: $?"
done
```

`ligands/uma_only` must contain the prepared UMA `.pdbqt` **and nothing else**:
`dock.py` docks every `.pdbqt` in the directory it is given.

Expect fewer than 9 poses sometimes. Vina documents `num_modes` as a maximum,
further limited by `energy_range` and a 1 Å redundancy filter, so a smaller
retained set is a legitimate result and `dock.py` records the returned count.

Then analyse each run:

```bash
for SIZE in 24 30 33; do
  python python/analyse_redocking.py runs/uma_control_${SIZE}A uma \
      --reference references/uma_chainE.reference.json \
      --component-dictionary <CCD>/UMA.cif \
      --ligand-manifest ligands/uma.ligand.json \
      --stereo-verification ligands/uma.stereo_verification.json \
      --independent-rmsd analysis/uma.independent_rmsd.json \
      --out analysis/uma_control_${SIZE}A
done
```

## How to read the result

The analysis prints, per seed, the top-ranked and closest-to-reference poses
with scores, ranks and continuous RMSD values, then one of four outcomes:

| Outcome | Meaning |
| --- | --- |
| *consistent recovery* | the top-ranked pose meets the criterion in every seed |
| *mixed outcome* | it meets the criterion in some seeds and not others |
| *recovery among retained poses, but failure to rank it first in that run* | a retained pose meets the criterion while the top-ranked one does not. The search reached a near-reference pose; the scoring did not put it first. This is **not** a sampling failure, and it is **not** a recovery of the top-ranked pose — it is precisely and only what it says |
| *no recovery* | no retained pose in any seed meets the criterion |

**The analysis refuses to be interpreted until its prerequisites are met.**
Without `--stereo-verification` and `--independent-rmsd` it still computes and
writes the numbers, but marks `interpretable: false` and prints a banner
naming what is outstanding. An unverified stereoisomer or an unvalidated RMSD
implementation makes a pass and a failure equally uninformative, so the
numbers are not evidence either way until both are supplied.

Interpretation limits, which the analysis record also carries:

- The initial control receptor has **neither ANP nor manganese**. A recovery is
  **recovery in the simplified receptor, not validation against the native
  state.** In deposited 1P3D, UMA, ANP and two Mn²⁺ **coexist**: they occupy
  adjacent sub-sites of one composite site. Their proximity (UMA and ANP come
  within 3.96 Å) is **not** evidence of competition or steric conflict —
  nothing is competing in a structure where all of them are bound at once. It
  means the simplified receptor is missing neighbours that the deposited pose
  was formed against.
- The ANP-present receptor is therefore a **second condition, not a
  replicate**. ANP occupies part of the search box, which changes the search
  problem; that is a different experiment with a different expected answer,
  and it requires the cofactor's chemical state to have been reviewed first.
- Recovering the deposited pose does not establish that peptide poses or
  rankings are reliable. It removes one failure mode.
- **UMA scores are not affinity thresholds for peptides.** UMA is a 49-heavy-atom
  reaction product; the candidates are 8–16 residue peptides. Their scores are
  not comparable, and a docking score is not a measured affinity in either case.
- The reference is one crystallographic conformer, and the pocket side chains
  are those adopted with UMA bound — so this is a self-consistent geometric
  test, not an independent prediction.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

`tests/test_uma_control.py` covers the molecular graph, resonance
normalisation, automorphism enumeration, in-place RMSD, pose-file parsing, the
recovery logic, dictionary reading, reference extraction and both ligand
stages. There is **no stub for a chemistry toolkit**: hydrogen placement, atom
typing and partial charges are not exercised or imitated anywhere, and the
`check` stage is tested against a hand-written PDBQT fixture, which proves the
verification logic and nothing about the preparation.
