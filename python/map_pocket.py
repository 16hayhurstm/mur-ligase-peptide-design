"""Map protein residues near explicitly selected ligand components.

Usage:
    python python/map_pocket.py data/1P3D.cif UMA 5.0 --protein-chain A --ligand-chain E

Both selection flags use canonical mmCIF IDs (label_asym_id), as reported
by inspect_structure.py. Multiple ligand IDs and names are comma-separated.
LIGAND accepts component names or integer author residue numbers.

Uses the first model and all stored atoms, including hydrogens if present.
A residue is selected when its minimum atom distance is strictly below the
cutoff. No symmetry copies are generated. The docking box is an estimate.

Use the canonical residue list (label_seq_id) for BoltzGen.
Ligand selection maps the pocket; it does not retain ligands in a design YAML.
"""
import argparse
import sys
import gemmi
import numpy as np

# Recognised protein residue names, including two common modifications.
AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
         "SER THR TRP TYR VAL MSE KCX".split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure_path")
    parser.add_argument("ligand", help="Component names or author residue numbers")
    parser.add_argument("cutoff", nargs="?", type=float, default=5.0)
    parser.add_argument("--protein-chain", required=True,
                        help="Canonical protein chain ID")
    parser.add_argument("--ligand-chain", required=True,
                        help="Comma-separated canonical ligand IDs")
    args = parser.parse_args()

    path = args.structure_path
    cutoff = args.cutoff
    if not np.isfinite(cutoff) or cutoff <= 0:
        parser.error("The cutoff must be positive and finite.")

    spec = [value.strip() for value in args.ligand.split(",")]
    ligand_ids = {value.strip() for value in args.ligand_chain.split(",")}
    if "" in spec or "" in ligand_ids:
        parser.error("Selections must not contain empty entries.")
    if args.protein_chain in ligand_ids:
        parser.error("Protein and ligand canonical IDs must be different.")

    # Keep support for substrates represented by several separate residues.
    lig_names = {value for value in spec if not value.isdigit()}
    lig_nums = {int(value) for value in spec if value.isdigit()}

    try:
        st = gemmi.read_structure(path)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if len(st) == 0:
        parser.error("The structure contains no models.")
    model = st[0]
    if len(st) > 1:
        print("NOTE: using only the first structural model.")

    # Select canonical instances, independently of author-chain grouping.
    # Do not generate missing canonical IDs: they must come from the CIF.
    protein_residues = []
    protein_author_ids = set()
    ligand_residues = []
    found_ligand_ids = set()
    matched_names, matched_nums = set(), set()   

    for chain in model:
        for residue in chain:
            if residue.subchain == args.protein_chain:
                if residue.label_seq is None:
                    parser.error("The selected protein contains non-polymer residues.")
                protein_residues.append(residue)
                protein_author_ids.add(chain.name)

            if residue.subchain in ligand_ids:
                matches_name = residue.name in lig_names
                matches_number = residue.seqid.num in lig_nums
                if not (matches_name or matches_number):
                    parser.error(
                        f"Ligand ID {residue.subchain} contains "
                        f"{residue.name}{residue.seqid}, which does not match "
                        f"the requested ligand selection."
                    )
                ligand_residues.append((chain.name, residue))
                found_ligand_ids.add(residue.subchain)
                if matches_name:
                    matched_names.add(residue.name)
                if matches_number:
                    matched_nums.add(residue.seqid.num)   
    if not protein_residues or not any(r.name in AA for r in protein_residues):
        parser.error("The selected canonical ID has no recognised protein residues.")
    if len(protein_author_ids) != 1:
        parser.error("The protein selection maps to multiple author chains.")
    protein_author_id = next(iter(protein_author_ids))
    missing_ids = ligand_ids - found_ligand_ids
    if missing_ids:
        parser.error(f"Ligand IDs not found: {sorted(missing_ids)}")
    if matched_names != lig_names or matched_nums != lig_nums:
        parser.error("Not all requested ligand names or numbers were found.")

    prot_nums = sorted(r.seqid.num for r in protein_residues if r.name in AA)
    lig_xyz, lig_res = [], []
    for author_id, residue in ligand_residues:
        lig_xyz.extend([[a.pos.x, a.pos.y, a.pos.z] for a in residue])
        lig_res.append(
            f"{residue.name} (canonical {residue.subchain}, "
            f"author {author_id}:{residue.seqid})"
        )
    if not lig_xyz:
        parser.error("The selected ligand components contain no atoms.")
    lig = np.array(lig_xyz)
    if not np.isfinite(lig).all():
        parser.error("Ligand coordinates contain non-finite values.")

    hits = []

    for r in protein_residues:
        if r.name not in AA:
            continue

        xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r])
        if xyz.size == 0 or not np.isfinite(xyz).all():
            parser.error(f"Missing or invalid coordinates for residue {r.seqid}.")
        d = float(
            np.linalg.norm(xyz[:, None] - lig[None, :], axis=-1).min()
        )

        if d < cutoff:
            author_residue_id = str(r.seqid)
            canonical_residue_index = r.label_seq
            canonical_chain_id = r.subchain

            if canonical_residue_index is None or canonical_residue_index < 1:
                raise ValueError(
                    f"Missing or invalid canonical index for "
                    f"{protein_author_id}:{author_residue_id} ({r.name})"
                )

            if not canonical_chain_id:
                raise ValueError(
                    f"Missing canonical chain ID for "
                    f"{protein_author_id}:{author_residue_id} ({r.name})"
                )

            hits.append({
                "author_residue_id": author_residue_id,
                "canonical_residue_index": canonical_residue_index,
                "canonical_chain_id": canonical_chain_id,
                "residue_name": r.name,
                "minimum_distance": d,
            })

    # Sort once, after collecting all pocket residues.
    hits.sort(
        key=lambda hit: (
            hit["canonical_chain_id"],
            hit["canonical_residue_index"],
        )
    )

    centre = lig.mean(0)
    extent = lig.max(0) - lig.min(0)

    print(f"structure : {path}  ({st.resolution} A)")
    print(
        f"protein   : canonical {args.protein_chain}, author {protein_author_id}, {len(prot_nums)} residues "
        f"({prot_nums[0]}-{prot_nums[-1]})"
    )
    print(f"ligand    : {','.join(lig_res)}  ({len(lig)} atoms)")
    print(f"cutoff    : {cutoff} A")
    print(f"\npocket: {len(hits)} residues")

    print("  Author chain:residue   Canonical chain:index   Residue   Distance")
    for hit in hits:
        author_identifier = f"{protein_author_id}:{hit['author_residue_id']}"
        canonical_identifier = (
            f"{hit['canonical_chain_id']}:{hit['canonical_residue_index']}"
        )
        print(
            f"  {author_identifier:>20}   "
            f"{canonical_identifier:>21}   "
            f"{hit['residue_name']:>7}   "
            f"{hit['minimum_distance']:.2f} A"
        )

    # Retain author identifiers for comparison with the original structure.
    author_residue_ids = [
        hit["author_residue_id"] for hit in hits
    ]
    print(
        f"\nauthor residues, chain {protein_author_id}: "
        f"{','.join(author_residue_ids)}"
    )

    # Report BoltzGen indices separately for each canonical chain.
    canonical_chain_ids = sorted({
        hit["canonical_chain_id"] for hit in hits
    })

    for canonical_chain_id in canonical_chain_ids:
        canonical_residue_indices = [
            str(hit["canonical_residue_index"])
            for hit in hits
            if hit["canonical_chain_id"] == canonical_chain_id
        ]
        print(
            f"BoltzGen residues, canonical chain {canonical_chain_id}: "
            f"{','.join(canonical_residue_indices)}"
        )

    if not hits:
        print("\nWARNING: no pocket residues found; no BoltzGen list generated.")

    print(f"\nbox centre: {[round(float(x), 2) for x in centre]}")
    print(
        f"\nIMPORTANT pass author chain {protein_author_id} "
        "to prepare_receptor.py."
    )
    print("          Keep the same protein copy during receptor preparation.")
    print(
        f"box size  : {extent.max() + 10:.0f} A "
        f"(ligand extent {[round(float(x), 1) for x in extent]})"
    )

    modified_residues = [
        f"{protein_author_id}:{hit['author_residue_id']} {hit['residue_name']}"
        for hit in hits
        if hit["residue_name"] in ("MSE", "KCX")
    ]
    if modified_residues:
        print(
            f"\nNOTE modified residues in pocket "
            f"(author identifiers): {modified_residues}"
        )
        print("     check their handling during receptor preparation - see docs")


if __name__ == "__main__":
    main()
     