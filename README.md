# murligase-peptide-design

A workflow for designing peptides against a specified site using [BoltzGen](https://github.com/HannesStark/boltzgen) and scoring them with [AutoDock Vina](https://vina.scripps.edu/). Developed against *Haemophilus influenzae* MurC (PDB 1P3D) and *Pseudomonas aeruginosa* MurF (4CVM), but the scripts take structures, ligands and residue lists as arguments and are not specific to those targets.

## How it works
Download the structure of whatever you're targeting from the PDB:
```bash
# Generic usage
./scripts/01_fetch_structure.sh <PDB_ID> <output_directory>

# Example used for MurC (1P3D)
./scripts/01_fetch_structure.sh 1P3D data
```

Inspect the structure to identify its protein chains, ligands and ions:

```bash
# Generic usage
python python/inspect_structure.py <path_to_structure.cif>

# Example used for MurC (1P3D)
python python/inspect_structure.py data/1P3D.cif
```

This prints the canonical, author and entity IDs, component codes and descriptions. Use it to identify the target protein and ligand code, such as `UMA`. Components sharing an entity ID are copies of the same molecular entity.

Use the **canonical IDs** to select the protein and ligand copy for pocket mapping. For 1P3D, protein A and UMA E share author ID A. Author IDs help match components. Confirm in a structure viewer when working with unfamiliar structures.

Identify pocket residues by measuring their minimum atom-to-atom distance from the selected ligand. Supply the structure, ligand code, distance cutoff in Å, and canonical protein and ligand IDs:

```bash
# Generic usage
python python/map_pocket.py <path_to_structure.cif> <ligand_code> <distance_cutoff> \
  --protein-chain <canonical_protein_id> \
  --ligand-chain <canonical_ligand_id>

# MurC (1P3D): residues on protein A within 5 Å of UMA E
python python/map_pocket.py data/1P3D.cif UMA 5.0 \
  --protein-chain A \
  --ligand-chain E
```

Both chain flags are required. This prints the pocket residues, estimated docking-box coordinates, selected component identifiers and recognised modified residues requiring attention during receptor preparation. Selecting a ligand here defines the pocket; it does not retain that ligand during design.

Then create the [BoltzGen design specification](https://github.com/HannesStark/boltzgen/tree/main#how-to-make-a-design-specification-yaml) (`.yaml`) using `make_spec.py`.

Use the chain identifier and residue list labelled **"BoltzGen residues, canonical chain ..."** in the output from `map_pocket.py`. These use the canonical mmCIF identifiers (`label_asym_id` and `label_seq_id`), which may differ from the author identifiers. `make_spec.py` writes the supplied identifiers without converting or validating them.

**Current limitation:** the generated specification includes only the selected target chain from the structure. Additional ions are not explicitly retained. For example, the command below targets the UMA pocket but does not retain manganese ions.

```bash
# Generic usage: CHAIN and RESIDUES must use canonical identifiers.
python python/make_spec.py <path_to_structure.cif> <canonical_protein_id> <residue_list> <length_AA> <output_directory>.yaml

# MurC (1P3D): target the UMA pocket with peptides of 8–16 amino acids.
python python/make_spec.py data/1P3D.cif A \
  25,27,28,29,30,31,32,48,49,50,51,70,84,85,86,87,88,91,107,152,173,174,175,177,178,198,346,348,376,377,380,459 \
  8..16 murc.yaml
```
Inspect the generated check structure to confirm that the intended pocket residues are marked. Successful parsing alone does not confirm that the correct binding site was selected.

Check the design specification is as intended by running:

```bash
# Generic usage
boltzgen check <path_to_design specification>

# Example used for murc.yaml
boltzgen check murc.yaml
```
Then visulise the resulting '.cif' file using (https://molstar.org/viewer/) which should show the binding residues a different color. Alternatively programmes such as ChimeraX can be used but require knowlege of the programme to visualise the specified binding residues. Then, design some peptides against it;

```bash
# Generic usage
bash scripts/03_boltzgen_design.sh <path_to_design_specification> <output_directory> <number_of_designs>

# Example used for murc.yaml to produce 200 peptides (in practice Boltzgen recomends genrating between 10,000-60,000)
bash scripts/03_boltzgen_design.sh murc.yaml out_murc 200
```

Timing not: roughly an hour for 100 designs on an A6000. Run it under tmux. - Myles' run took 5.5hr for 200, 16.5hrs for 500

Optionally compare the designs before and after refolding in ChimeraX. Use `scripts/04_visualise_designs.sh` to prepare a viewing folder containing up to 20 paired designs:

- Visualisation file still needs sorting

```bash
bash scripts/04_visualise_designs.sh out_murc visualisations/murc --select all --limit 20
```

Use a new destination folder for each set of results.

Download the entire `visualisations/murc` folder to your laptop. In a fresh ChimeraX session, open `compare_refolds.py` using **File > Open**.

Use the **Model Panel** to display one design group at a time:

- **Cyan:** peptide before refolding.
- **Magenta:** peptide after refolding.
- **Grey:** common MurC reference, taken from the first selected design before refolding.

The common reference provides a visual comparison; assess clashes using each design’s matching receptor.

See [docs/visualisation.md](docs/visualisation.md) for selection options and further viewing instructions.











----------- Myles Checked up to here


Now, find out where the designs actually sit. BoltzGen ranks its own output on a refolded complex that doesn't preserve the designed placement, so rank on measured pocket contact instead:

    python python/occupancy.py out_murc/intermediate_designs 25,27,28,29,...

Prepare the receptor. Crystal structures need cleaning before docking — waters and ligands stripped, modified residues repaired, one chain kept:

    python python/prepare_receptor.py data/1P3D.cif murc A 25,27,28,29,...

Pass the pocket residues and it will warn you if any of them get dropped.

Build the ligands and their controls. Peptides come from sequence via PyMOL, and scrambles give you a composition-matched baseline:

    ./scripts/06_prepare_ligands.sh pep TTDPGFGT $(python python/scramble.py TTDPGFGT 5)

Finally, dock across several seeds. One seed isn't a measurement: the search is stochastic enough that the same peptide can move by nearly a kcal/mol between runs:

    python python/dock.py murc.pdbqt pep 25.69,-5.73,48.55 24

It reports the seed spread next to every mean, and warns when the variation between runs is comparable to the difference between ligands.

## Quickstart

    ./scripts/01_fetch_structure.sh 1P3D data
    python python/map_pocket.py data/1P3D.cif UMA 5.0

That prints the pocket residues and a box. Feed the residues to `make_spec.py` to design against them, or go straight to receptor preparation and docking:

    python python/prepare_receptor.py data/1P3D.cif murc A "25,27,28,29,30,31,32,48,49,50,51,70,84,85,86,87,88,91,107,152,173,174,175,177,178,198,346,348,376,377,380,459"
    ./scripts/06_prepare_ligands.sh pep TTDPGFGT $(python python/scramble.py TTDPGFGT 5)
    python python/dock.py murc.pdbqt pep 25.69,-5.73,48.55 24

See `docs/environment.md` for the conda environments and system tools required.


## Sources

Peptide inhibitors with measured IC50 values against *P. aeruginosa* Mur ligases:

- Paradis-Bleau et al., *Peptides* **27** (2006) 1693–1700 — MurD
- Paradis-Bleau et al., *BMC Biochemistry* **9**:33 (2008) — MurF
- Paradis-Bleau et al., *Biochem. J.* **421** (2009) 263–272 — MurE