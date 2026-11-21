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
stage 6 needs neither; stage 7 needs RDKit again. **Stages 1-7 are the
current milestone.** Docking is stage 8 and is not part of it.

### Getting the code onto the validation machine

The validation checkout is a **detached worktree**, not a branch-tracking
clone, so `git pull` is the wrong command: there is no upstream branch to
merge into and the worktree is deliberately pinned to one commit. Fetch, then
switch to the exact commit being validated:

```bash
cd <validation worktree>
git fetch origin
git switch --detach <commit>
git log --oneline -1          # confirm it is the commit you intended
git status --short            # expect clean; generated outputs are ignored
```

Pinning to an explicit commit is the point: every manifest and record written
below carries input hashes, and the code revision that produced them has to be
identifiable too. Note the commit hash alongside the results.

### Output directories

The scripts do create their own output directories, but creating them up front
keeps the paths below valid whatever order they are run in, and makes the
layout explicit:

```bash
conda activate boltz
mkdir -p components references ligands analysis runs
```

The component dictionary is fetched rather than transferred:

```bash
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

### Stage 4 — RDKit verification and hydrogen addition (RDKit)

This is the only step that perceives chemistry rather than reading it, and it
is a single script rather than a snippet to adapt:

```bash
python python/verify_ligand_rdkit.py \
    ligands/uma.ligand.json \
    ligands/uma.stereo_plan.json \
    modelled \
    --out-prefix ligands/uma_verified
echo "verification exit status: $?"
```

The −2 sensitivity state needs its **own** plan, because a plan records the
SHA-256 of the manifest it was derived from and the script refuses a plan that
was built from a different one. Generate it, then verify:

```bash
python python/prepare_ligand.py stereo-plan \
    ligands/uma_minus2.ligand.json \
    --out ligands/uma_minus2.stereo_plan.json \
    --state comparison_minus3=O2A,O2B,O20

python python/verify_ligand_rdkit.py \
    ligands/uma_minus2.ligand.json \
    ligands/uma_minus2.stereo_plan.json \
    modelled \
    --out-prefix ligands/uma_minus2_verified
echo "verification exit status: $?"
```

In the −2 manifest's own plan, the state called `modelled` **is** the −2
state; the extra `--state` adds −3 alongside it for comparison. The script
checks that the named plan state's charge equals the manifest's modelled
charge and refuses the pair otherwise, so a state and a molfile that describe
different protonations cannot be verified against each other by accident.

Comparing the two plans' `modelled` states is what shows `PB` moving between
`not_stereogenic_at_this_state` at −3 and `must_match` at −2.

#### What it checks, and in what order

1. **Input identity** — the molfile's SHA-256 against the manifest, the
   dictionary's against the manifest, and the plan's recorded manifest hash
   against the manifest being used. A revised CCD entry or an edited molfile
   stops the run.
2. **The molfile's own bond orders**, read from its bond block with no
   toolkit and no aromaticity model, against the dictionary's `value_order`
   bond for bond. This asks only "is the file we wrote the file we meant",
   and a disagreement here means it is not.
3. **Formal charge** must equal the manifest's modelled charge. **A charge of
   0 where −3 was intended means the `M  CHG` block was not applied and the
   molecule is the neutral form** — the most likely silent failure here.
4. **Heavy-atom count** against the reference, and the molfile must carry no
   hydrogens yet.
5. **Index-to-name correspondence, derived from the coordinates**, then
   validated against elements and connectivity, and required to be a
   bijection. The molfile has no atom names, so the mapping is matched by
   position against the reference and *then* checked; it is never assumed
   from file order, and the plan's stereocentre list is never used as an
   ordering.
6. **Connectivity after identical sanitisation.** The expected molecule is
   built from the same connection table and the same modelled formal charges,
   given the same coordinates, and sanitised the same way, so both sides
   carry whatever aromaticity RDKit assigns. See below.
7. **Stereochemistry from 3D** — `AssignStereochemistryFrom3D` followed by
   `rdCIPLabeler.AssignCIPLabels`, the accurate labeller, since the
   comparison is against the dictionary's CIP configurations.
8. **Hydrogen addition** with `AddHs(addCoords=True)`: no embedding, no
   conformer generation, no minimisation. Charge, stereochemistry and every
   heavy-atom coordinate are then rechecked, and any change fails the run.

#### Kekulé input against an aromatising toolkit

The first real run failed here. What has been established is narrow and
worth stating precisely: **the raw molfile bond table matches the CCD**. That
is a statement about the connection table only. Stereochemistry and hydrogen
addition were never reached by that run and remain unverified.

```
order differences:
  C2-N1: dictionary SING, molecule AROMATIC
  C2-N3: dictionary SING, molecule AROMATIC
  C4-C5: dictionary SING, molecule AROMATIC
  C4-N3: dictionary SING, molecule AROMATIC
  C5-C6: dictionary DOUB, molecule AROMATIC
```

UMA's uracil ring is declared in the dictionary in **Kekulé** form — the
entry flags no aromatic atoms at all, and gives `C5=C6` double with the rest
of the ring single. The molfile's bond table reproduces that exactly: checked
bond for bond, all 51 bonds agree in both membership and order, which is now
step 2 above. RDKit then **sanitises** the molecule on reading, its
aromaticity model perceives the ring, and the ring bonds come back as
`AROMATIC`. Comparing a perceived molecule against an unperceived connection
table reports differences that are purely representational.

What this does **not** establish: that the stereochemistry is right, that
hydrogen addition behaves, or that the ligand is fit to dock. The first run
stopped at step 2's successor and never reached steps 7 or 8. Those checks
exist, have never been executed against this ligand, and are the point of
rerunning.

The fix is to compare like with like. The expected molecule is now
constructed through RDKit from the dictionary's connectivity and the modelled
charges, given the same coordinates, and sanitised identically, so both sides
share one aromaticity model. A genuine bond-order difference still shows,
because it would survive identical perception — a regression test and a
negative test cover both halves.

Two things deliberately **not** done, because either would have made the
symptom disappear while removing the check: aromatic differences are not
ignored, and sanitisation is not relaxed or disabled. The raw Kekulé
comparison against the dictionary also still runs, on the file itself, where
no perception is involved.

One observation left open: the first run flagged **five** of the six ring
bonds, with `C6-N1` agreeing. A ring is aromatised as a whole, so five of six
is peculiar and is worth understanding rather than normalising away. It does
not affect the fix — identical perception on both sides handles it either
way — but the Epicac check below prints RDKit's perceived orders for the ring
so the asymmetry can be looked at directly.

#### A changed R/S label is not assumed to be an inversion

Because step 4 verifies that every heavy atom sits where the deposited
reference has it, a label difference found at step 6 **cannot be a physical
inversion** — the geometry is the deposited geometry. The record says so
explicitly and directs the investigation to CIP priority instead: the
dictionary's labels describe the *neutral* component, and deprotonating a
nearby acid can reorder priorities at a centre without moving anything. This
matters most at `C21` and `C18`, which flank the alanine carboxylate.

Differences are reported, never discarded, and never silently recast.

#### Phosphorus is reported separately

Carbon and phosphorus centres appear in separate lists in both the console
output and the record, following the state's plan:

- at **−3**, `PA` and `PB` are not stereogenic, and a label there is expected
  to be absent or meaningless;
- at **−2**, `PB` is formally stereogenic and appears under `must_match` with
  `fatal_if_mismatched: false`, so a difference is **investigated** rather
  than treated as the wrong stereoisomer.

A mismatch at any of the 11 carbons is fatal and fails the run.

#### Outputs

`ligands/uma_verified_withH.mol` and `ligands/uma_verified.verification.json`.

**A failed check writes the record with `result: "failed"`, does not write the
hydrogenated molfile, and exits non-zero.** If an output from an earlier
attempt exists it is renamed with a `.FAILED` suffix. There is no path by
which a failure leaves something that looks usable, and the downstream
analysis validates the record's `result` and its recorded input hashes rather
than accepting any JSON file at that path.

The record carries the RDKit version, every input hash, the output hash, the
derived correspondence, all three check groups and the hydrogen count.

#### Do the existing inputs need rebuilding?

**No.** The molfiles already on Epicac are correct: their bond blocks are the
dictionary's Kekulé form bond for bond, and the fix is entirely in the
comparison. Rerun the verification against the **existing**
`ligands/uma_ligand.mol`, `ligands/uma.ligand.json` and
`ligands/uma.stereo_plan.json`. Preserve the failed verification records; they
are the evidence of what the first run found.

One cosmetic defect was fixed in the generator: line 2 of the molfile had free
text where V2000 keeps the dimensional code in columns 21-22, so RDKit warned
`molecule is tagged as 2D, but at least one Z coordinate is not zero`. It
corrected itself and read the coordinates properly, so this changed no result.
Molfiles written from now on carry `3D` there, with a zeroed timestamp so the
file stays reproducible from the same inputs. Rebuilding an existing ligand
only to pick that up would change its hash and so require regenerating its
stereo plan as well, for no gain — do it only when a ligand is being rebuilt
anyway.

#### If the ring asymmetry needs looking at

Read-only, and prints what RDKit actually perceives:

```bash
python - <<'PY_INNER'
import sys; sys.path.insert(0, "python")
from rdkit import Chem
mol = Chem.MolFromMolFile("ligands/uma_ligand.mol", removeHs=False,
                          sanitize=True)
ring = {"N1", "C2", "N3", "C4", "C5", "C6"}
names = [line.split()[1] for line in open("ligands/uma_ligand.mol")
         if False]            # molfile carries no names; use the CCD order
import chemical_component
component = chemical_component.read_component("components/UMA.cif")
order = list(component.heavy_atom_names)
for bond in mol.GetBonds():
    first = order[bond.GetBeginAtomIdx()]
    second = order[bond.GetEndAtomIdx()]
    if first in ring and second in ring:
        print(f"{first:>3}-{second:<3} perceived {bond.GetBondType()!s:<9} "
              f"aromatic={bond.GetIsAromatic()} "
              f"ringAtomsAromatic="
              f"{mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetIsAromatic()}"
              f"/{mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetIsAromatic()}")
PY_INNER
```

### Stage 5 — Meeko preparation (Meeko)

```bash
mk_prepare_ligand.py -i ligands/uma_verified_withH.mol -o ligands/uma.pdbqt
echo "exit status: $?"
```

### Stage 5b — the atom map (nothing extra)

Meeko writes the **element** in a PDBQT's atom-name column, so a prepared UMA
has 49 heavy atoms sharing five names (`C`, `H`, `N`, `O`, `P`). No name-keyed
correspondence is possible against it, and `check` refuses such a file rather
than collapsing the duplicates — a collapsed duplicate would leave a name set
that still matched the reference while a record was silently discarded.

The route through is Meeko's own index map, which is why stage 5 passes
`--add_index_map`:

```bash
python python/prepare_ligand.py atom-map \
    ligands/uma.ligand.json \
    ligands/uma_verified_v2.verification.json \
    ligands/uma_minus3_try1.pdbqt \
    --out ligands/uma.atom_map.json \
    --preparation-command "mk_prepare_ligand.py -i ligands/uma_verified_v2_withH.mol -o ligands/uma_minus3_try1.pdbqt --charge_model gasteiger --add_index_map"
```

This composes `PDBQT serial -> input molfile atom index -> reference atom
name` and validates every link. The original PDBQT is read only.

**Index-map semantics, established from the real output rather than assumed.**
For the UMA preparation, `REMARK INDEX MAP` pairs read **(input atom index,
PDBQT serial)**, both **1-based**, with the serial column being the atoms in
file order. Confirmed two ways: every element agrees, and the coordinates
agree exactly — serial 1 is input atom 22, which is `PB`, at
`23.570 −3.765 50.318`, the deposited `PB` position.

The direction is still re-derived on every run, by asking which column is
exactly the PDBQT's serials. If both columns are permutations of `1..n`, the
two readings are compared: identical readings mean the direction is immaterial
and are accepted; differing readings are refused rather than guessed.

What the stage requires before writing a mapping:

- the verification record says `passed`, and its ligand-manifest hash matches;
- the hydrogenated molfile's hash matches what was verified;
- that molfile's heavy atoms match the reference by element **and** coordinate,
  so its positional order is checked rather than trusted;
- PDBQT serials are unique and each appears once in the map;
- a **bijection over all 49 heavy atoms** — no reference atom unreached, no two
  serials reaching the same name;
- elements and coordinates agree for every mapped pair, within 0.002 Å.

**Merged hydrogens are expected, not failures.** Meeko merges non-polar
hydrogens into their parent heavy atom, so the input's hydrogens outnumber the
PDBQT's. The real case: 82 input atoms to 56 PDBQT atoms, with 7 polar
hydrogens mapped and **26 merged**. Those are counted and listed in the record.

The mapping is specific to one PDBQT and one input molfile, both hashed in the
record, and must not be reused across a re-preparation.

### Stage 6 — preparation verification (nothing extra)

```bash
python python/prepare_ligand.py check \
    ligands/uma.ligand.json ligands/uma_minus3_try1.pdbqt \
    --out ligands/uma_minus3.preparation.json \
    --atom-map ligands/uma.atom_map.json \
    --charge-model gasteiger \
    --preparation-command "mk_prepare_ligand.py -i ligands/uma_verified_v2_withH.mol -o ligands/uma_minus3_try1.pdbqt --charge_model gasteiger --add_index_map"

python python/prepare_ligand.py check \
    ligands/uma_minus2.ligand.json ligands/uma_minus2_try1.pdbqt \
    --out ligands/uma_minus2.preparation.json \
    --atom-map ligands/uma_minus2.atom_map.json \
    --charge-model gasteiger \
    --preparation-command "mk_prepare_ligand.py -i ligands/uma_minus2_verified_withH.mol -o ligands/uma_minus2_try1.pdbqt --charge_model gasteiger --add_index_map"
```

`--preparation-command` is recorded verbatim, so it must be the **exact**
command that produced the PDBQT being checked, with its original `-i` and `-o`
arguments. An abbreviated command is incomplete provenance: it does not
identify which input molfile was converted or which output was written, and
the record is then not reproducible. Substitute the real −2 filenames if they
differ from those above.

`--out` is required and the ligand manifest is **left unmodified**. An earlier
version wrote the preparation results back into the manifest, which changed the
manifest's SHA-256 after the stereochemistry verification and the atom map had
already recorded that digest as the ligand's identity; every downstream gate
keyed to it then failed for a bookkeeping reason rather than a scientific one.
The evidence now lives in its own `mur-ligase-ligand-preparation/1` record,
tied by digest to both the manifest and the exact prepared PDBQT it describes.

A manifest that an earlier `check` already modified in place stays usable. The
downstream gates restore that one known top-level `preparation` block to the
shape it held before that stage and accept the manifest only when the restored
form reproduces the recorded digest **exactly**; the basis is recorded as
`legacy_compatibility_used`, so a compatibility match is visible as one. No
historical hash is rewritten, and a change to the component, the protonation
state, the reference or the dictionary cannot be restored into agreement.

A failed check still writes its record, with `result: "failed"` and the
reasons, so the diagnostic detail is kept; downstream readers gate on
`result`.

With `--atom-map`, atoms are keyed by the validated mapping. **Without it the
name-based check stays strict** and refuses a repeated atom name, which is the
right behaviour for a ligand whose names survived preparation and the right
refusal for one whose names did not.

Either way it refuses to pass if a heavy atom moved beyond 0.002 Å from the
reference, if a heavy atom is missing or unexpected, or if the partial charges
do not sum to the recorded formal charge. Records the
hydrogens added, the AutoDock type census, the charge sum, `TORSDOF` and the
`BRANCH` count.

### Stage 7 — the independent RMSD comparison (RDKit)

This is a **prerequisite for interpreting the control**, not an optional
cross-check. `chemical_graph.symmetry_aware_rmsd` is a custom implementation
whose own unit tests are not independent of it, so before the control's
numbers mean anything it has to agree with an established toolkit on
structures whose answer is fixed by something outside both implementations.

`python/independent_rmsd.py` runs the comparison and writes the record
`analyse_redocking.py --independent-rmsd` validates. Run it once per modelled
state, against that state's own manifest and verification record:

```bash
# The -3 state.
python python/independent_rmsd.py \
    ligands/uma.ligand.json ligands/uma_verified_v2.verification.json \
    --out analysis/uma_minus3.independent_rmsd.json

# The -2 sensitivity state.
python python/independent_rmsd.py \
    ligands/uma_minus2.ligand.json \
    ligands/uma_minus2_verified.verification.json \
    --out analysis/uma_minus2.independent_rmsd.json
```

Each run reads the **verified hydrogenated molfile** the verification record
names, checks its digest, and refuses to proceed on an unverified molecule or
a manifest the verification did not describe.

#### Two separate validations

The real Epicac runs at `52dcfef` showed these are different questions and
must not share a gate. The **−3** state agreed on every number while failing
the equivalence-class comparison. The **−2** state agreed on the classes while
automatic CalcRMS symmetrisation accepted the PB protonated/unprotonated
oxygen swap, scoring **0 Å where this project's measure gives 0.526446 Å**.
Both failed records are preserved as the evidence of that.

**1. The allowed mappings.** `python/oxygen_pair_audit.py` audits every
explicitly named terminal-oxygen pair against the dictionary's connectivity
and hydrogen attachment at the modelled state. A pair may be exchanged only
when both atoms are oxygen, both are terminal, both are bonded to the same
centre, and **neither bears a hydrogen at that state**. Every candidate is
recorded with each condition evaluated and the verdict in words, so the
convention is reviewable pair by pair. Bond order is recorded but is *not* a
criterion: a delocalised group is conventionally drawn with one double and one
single bond.

The module parses the dictionary itself with Gemmi and takes **nothing** from
`chemical_graph`, its orbits or its automorphisms. The full audited mapping
set is then compared with the set this project's implementation minimises
over — the sets themselves, not merely the orbits they induce, since different
sets can induce the same orbits. A mismatch blocks validation.

**2. The numerical calculation.** CalcRMS is given that audited mapping set
through its own `map` argument, with
`symmetrizeConjugatedTerminalGroups=False`. Both sides then minimise over one
shared, reviewed convention and the only thing left to disagree about is the
arithmetic. No superposition on either side.

**Retained as diagnostics, binding on nothing:** the automatic-symmetrisation
results, including the −2 PB finding, and RDKit's canonical ranks and
self-matches. Automatic symmetrisation implements a more permissive convention
than the one being validated, which is precisely what the −2 run demonstrated,
so judging agreement under it would test the wrong rule. **Ordinary canonical
ranks and self-matches must not be presented as the mappings symmetrised
CalcRMS uses** — they are RDKit's perception of equivalent atoms, and CalcRMS
is a separate routine with its own options.

#### What this establishes, and what it does not

That the numerical implementation agrees with an established toolkit **under
an explicitly audited chemical mapping convention**. It is **not independent
experimental validation of the chemistry**: the convention is a recorded
judgement about the dictionary, and agreement under it says the two
implementations compute the same quantity over the same permitted mappings,
nothing more. The record carries this as `validation_status`.

Still independent of this project: the audited convention comes from a module
that never consults `chemical_graph`; a rigid translation of length *d* has
the geometric expectation *d*, belonging to neither implementation; and this
project's automorphism list is never handed to the toolkit — the maps CalcRMS
receives come from the audit.

#### The cases

Generated deterministically from the verified coordinates, with no randomness
and no embedding:

| Case | Expected | Why |
| --- | --- | --- |
| `unchanged` | exactly 0 | the coordinates as verified |
| `translation_small` | exactly 0.25 Å | rigid translation; geometric expectation |
| `translation_large` | exactly 5.0 Å | as above, well beyond any pose criterion |
| `swap_<centre>` | exactly 0 | two terminal oxygens exchanged at a centre whose oxygens are **all deprotonated**, so they are delocalised and interchangeable: the same structure |
| `negative_swap_<centre>` | non-zero | a **protonated** and an **unprotonated** terminal oxygen exchanged at the same centre |

The negative case is the one that matters. In the −2 state PB retains a
proton on one terminal oxygen, so exchanging it with PB's unprotonated
terminal oxygen describes a **different structure**: the two differ in bond
length and hydrogen-bonding role, and a measure that returned zero there
would let a wrong pose pass. Both implementations must return a non-zero
value, and the record fails if either returns zero. Agreement on the
unchanged and translated cases alone proves very little.

#### Reading the result

The record states its own verdict. `result: "passed"` requires every case to
agree within the tolerance (10⁻⁶ Å, which leaves room only for
floating-point summation order), this project's values to meet the geometric
expectations, the negative case to be distinguished by both sides, and the
**audited mapping set to be the same set** the implementation minimises over.
Anything else is `"failed"` with the reasons listed, and
`analyse_redocking.py` then reports the prerequisite unsatisfied. A diagnostic
disagreement never fails the record.

**If the two disagree, investigate both implementations and both mappings.**
Neither is presumed right, and the first thing to compare is the
exchanged-atom sets the record carries, not the numbers:
`allowed_mappings.audited_exchanged_atom_sets` against
`allowed_mappings.implementation_exchanged_atom_sets`, with
`only_in_the_audited_set` and `only_in_the_implementation_set` naming the
difference. **Do not reconcile a mismatch by changing the equivalence rule or
widening the tolerance** — a disagreement is a finding about which convention
each side implements.

#### What this stage does not establish

The cases are **synthetic coordinate transformations of one deposited
structure**. They validate software. They are not docking results, not poses,
and not evidence about binding; the record labels itself
`content_class: "synthetic software validation..."` so it cannot be quoted as
a docking outcome. Agreement does not prove either implementation correct for
every molecule, and no stereochemistry is tested, since a swap of equivalent
oxygens is stereochemically inert.

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
      --atom-map ligands/uma.atom_map.json \
      --stereo-verification ligands/uma_verified_v2.verification.json \
      --independent-rmsd analysis/uma_minus3.independent_rmsd.json \
      --out analysis/uma_control_${SIZE}A
done
```

Both prerequisites are now **validated, not merely counted**. The
stereochemistry record must state a passing result, report no fatal mismatch
and describe the manifest being analysed. The independent-RMSD record must
state a passing result, report agreement on every case and the negative case
as distinguished, confirm the equivalences were verified against the toolkit's
own perception, and declare `superposition: "none"` — a fitted RMSD is a
different quantity. A record that merely exists satisfies neither.

### Pose provenance: by campaign record, not by file hash

A docked pose file is Vina's **output**. It has its own hash and does not
carry the prepared ligand's, so the analysis must not require the atom map to
hash-match the pose files. It does not: `--atom-map` is read without naming a
file it should describe, whereas the preparation check reads it while naming
the PDBQT being examined, which *is* the prepared input.

What establishes the link is the docking run's own `run_record.json`, which
lists every ligand input with its SHA-256. If the hash the atom map was built
against appears there, the poses in that directory were docked from the ligand
the map describes, and the analysis record says so. If it does not, the poses
may still be sound but nothing ties them to the verified input, and the
analysis prints

```
pose provenance  : NOT established - the atom map was built against a prepared
ligand whose hash is not among this run's recorded ligand inputs
```

rather than assuming either way.

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

287 tests, 5 of which require RDKit and skip without it. What they do and
do not establish, kept apart deliberately:

| Suite | Needs | Covers |
| --- | --- | --- |
| `tests/test_uma_control.py` | nothing / Gemmi | the molecular graph, resonance normalisation restricted to deprotonated groups, automorphism enumeration, in-place RMSD, pose-file parsing, the recovery logic, dictionary reading, reference extraction, both ligand stages and the stereo plan |
| `tests/test_ligand_verification.py` | nothing / Gemmi | the index-to-name correspondence derived from coordinates and every way it can be ambiguous, the raw molfile bond-order reader, bond-order vocabulary normalisation, element and connectivity validation, the per-centre stereochemistry comparison including the carbon/phosphorus separation, the heavy-atom check after hydrogen addition, the failure record, the downstream validation of a verification record, and the five input-identity guards run end to end |
| `tests/test_ligand_verification.py::RdkitRepresentationTests` | **RDKit** | the Kekulé/aromatic regression, that the previous comparison would have disagreed, that a genuinely changed bond is still rejected, and that modelled charges reach the expected molecule in both the charged and the neutral form. **Skipped without RDKit, so never executed on the Mac** |
| `tests/test_ligand_atom_map.py` | nothing | the Meeko index map: direction derivation including the ambiguous and immaterial cases, reordered atoms, missing and duplicate entries, a map from the wrong input, merged hydrogens, the record's tie to its PDBQT by hash, and the campaign-provenance linkage for docked poses |
| `tests/test_ligand_verification.py::ChargeConsistencyTests` | nothing / Gemmi | that a negative charge is only accepted on an atom bearing an acidic proton, reported with the atom's name and bonds rather than RDKit's atom index |
| `tests/test_ligand_verification.py::RdkitAbsenceTests` | RDKit **absent** | the diagnostic given when RDKit is missing. Skips wherever RDKit is installed |

**Expected skips depend on the machine, and one skip is always expected.**
`RdkitRepresentationTests` needs RDKit; `RdkitAbsenceTests` needs its
absence. Exactly one of the two groups runs anywhere:

| Machine | Ran | Skipped | Which |
| --- | --- | --- | --- |
| Mac (no RDKit) | 287 | 5 | the five `RdkitRepresentationTests` |
| Epicac (`boltz`, RDKit present) | 287 | **1** | `RdkitAbsenceTests` only |

An Epicac run reporting more than one skip means something else is being
skipped and is worth reading: before this revision, five `InputGuardTests`
skipped there because their fixtures lived under a git-excluded, Mac-only
path, which is now fixed by generating the fixtures in the test.

**The RDKit integration is only exercised where RDKit is installed.**
`MolFromMolFile`, `AssignStereochemistryFrom3D`,
`rdCIPLabeler.AssignCIPLabels`, `AddHs(addCoords=True)` and `MolToMolFile`
are reached only through the thin adapter in `verify_ligand_rdkit.py`. The
four `RdkitRepresentationTests` do exercise the expected-molecule
construction and the representation comparison, but **they skip on the
development machine and were written without ever being run**, so on Epicac
they are themselves a first execution rather than a passing guarantee. There
is deliberately **no RDKit stub** — stubbing stereochemistry perception would
assert chemistry that nothing in the suite can check, and could hide an
interface error instead of revealing one.

What *is* exercised locally on that path: the module imports, the argument
handling, all four input-identity guards, and the failure route end to end.
The guards deliberately run **before** RDKit is required, so a plan built from
another manifest, a state whose charge contradicts the manifest, an unknown
state name and an edited molfile each fail fast with a clear diagnostic
whether or not a toolkit is present — and are tested. Running the script
without RDKit then produces a `result: "failed"` record naming the missing
dependency, writes no hydrogenated molfile, and exits non-zero, so the gating
behaviour is tested even though the chemistry is not.

The first real execution of the RDKit path will therefore be on Epicac. Treat
its first run as an integration check in its own right: read the console
output and the record before trusting either.

**`index_to_name_correspondence.equals_file_order` is diagnostic, not a pass
requirement.** The correspondence that matters is the one the script derives
from coordinates and then validates against elements and connectivity, and a
validated bijective mapping is correct whether or not it happens to coincide
with file order. A `false` value simply records that the toolkit presented the
atoms in a different order from the molfile, which is useful to know and
changes nothing about the result. What must hold is that the mapping is
bijective and validated, which the script enforces by failing otherwise;
`equals_file_order` is reported so the observation is not lost, not as a
condition to meet.

Checks that need the real tools, and are therefore still pending: RDKit
reading and sanitisation of the molfile, CIP assignment from 3D, hydrogen
placement, Meeko atom typing and partial charges, and the independent RMSD
comparison.
