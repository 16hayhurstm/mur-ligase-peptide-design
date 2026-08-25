# murligase-peptide-docking

A workflow for designing peptides against bacterial Mur ligase binding sites with BoltzGen, and scoring them with AutoDock Vina.

Developed against *Haemophilus influenzae* MurC (PDB 1P3D) and *Pseudomonas aeruginosa* MurF (4CVM), but the scripts take structures, ligands and residue lists as arguments and are not specific to those targets.

## What this does

Given a crystal structure with a bound ligand, the pipeline defines the binding pocket from that ligand, generates peptides against it, filters them on where they actually sit, and docks them.

| step | script | purpose |
|---|---|---|
| 1 | `scripts/01_fetch_structure.sh` | download a structure from RCSB |
| 2 | `python/map_pocket.py` | pocket residues and docking box from a bound ligand |
| 3 | `python/make_spec.py` | write a BoltzGen design spec |
| 4 | `scripts/03_boltzgen_design.sh` | run the design job |
| 5 | `python/occupancy.py` | rank designs by how much sits in the target pocket |
| 6 | `python/prepare_receptor.py` | strip, repair and convert the receptor |
| 7 | `scripts/06_prepare_ligands.sh` | build peptides from sequence, convert to PDBQT |
| 8 | `python/scramble.py` | composition-matched controls |
| 9 | `python/dock.py` | multi-seed docking with reproducibility reporting |

## Quickstart

    ./scripts/01_fetch_structure.sh 1P3D data
    python python/map_pocket.py data/1P3D.cif UMA 5.0

That prints the pocket residues and a box. Feed the residues to `make_spec.py` to design against them, or go straight to receptor preparation and docking:

    python python/prepare_receptor.py data/1P3D.cif murc A "25,27,28,29,30,31,32,48,49,50,51,70,84,85,86,87,88,91,107,152,173,174,175,177,178,198,346,348,376,377,380,459"
    ./scripts/06_prepare_ligands.sh pep TTDPGFGT $(python python/scramble.py TTDPGFGT 5)
    python python/dock.py murc.pdbqt pep 25.69,-5.73,48.55 24

See `docs/environment.md` for the conda environments and system tools required.

## Read this before trusting any output

The scripts run. Whether their numbers mean anything is a separate question, and on these targets the answer was largely no. Findings from developing this workflow:

**AutoDock Vina failed its redocking control.** Given MurC's own substrate and the correct pocket, it placed it 5.67 Å from the crystallographic position; the accepted threshold is under 2 Å.

**Vina scored that substrate better in the wrong pocket.** UMA docked into MurC's ATP site scored −10.17, against −9.26 in its own substrate site.

**Cross-pocket comparisons are dominated by enclosure, not sequence.** Designed peptides preferred MurC's ATP pocket by 1.38 kcal/mol; scrambles of those peptides preferred it by 1.59. The effect carried no sequence information.

**A validated inhibitor ranked last against its own scrambles.** MurFp1 (IC50 250 µM, Paradis-Bleau et al. 2008) docked into *P. aeruginosa* MurF scored below all five composition-matched shuffles of itself.

**Seed variance is comparable to the signal.** One peptide against one receptor, varying only the random seed, spanned 0.91 kcal/mol; the range across ten different peptides was 1.29. `dock.py` reports this and warns when a ranking is not readable.

**12-mer peptides are outside Vina's range.** They carry 42–43 torsional degrees of freedom against a practical limit near 32.

Composition-matched scrambles were the single most useful control. Two apparent results dissolved on contact with them.

## Sources

Peptide inhibitors with measured IC50 values against *P. aeruginosa* Mur ligases:

- Paradis-Bleau et al., *Peptides* **27** (2006) 1693–1700 — MurD
- Paradis-Bleau et al., *BMC Biochemistry* **9**:33 (2008) — MurF
- Paradis-Bleau et al., *Biochem. J.* **421** (2009) 263–272 — MurE
