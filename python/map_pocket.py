"""Define a binding pocket from a bound ligand.

Reports the protein residues within a distance cutoff of a named ligand,
and a docking box that encloses it.

usage:   python map_pocket.py STRUCTURE.cif LIGAND [cutoff]
example: python map_pocket.py data/1P3D.cif UMA 5.0
         python map_pocket.py data/4CVM.cif 1452,1453,1454,1455 5.0

The output reports both author and canonical residue identifiers.
For BoltzGen, use the chain and residue list labelled "BoltzGen residues":
these use label_asym_id and label_seq_id from the mmCIF file.
Author identifiers are retained for comparison with the original structure.
Missing canonical identifiers in selected pocket residues raise an error.        

LIGAND may be several comma-separated residue names, for structures that
model a substrate as separate components, e.g. 4CVM stores
UDP-MurNAc-Ala-Glu as UDP,MUB,ALA,FGA.


"""
import sys
import gemmi
import numpy as np

AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
         "SER THR TRP TYR VAL MSE KCX".split())


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    path = sys.argv[1]
    spec = sys.argv[2].split(",")
    # entries may be residue names (UMA) or residue numbers (1452).
    # numbers are needed where a substrate component shares a name with a
    # standard amino acid - 4CVM models an ALA at 1454 as part of its
    # substrate, and selecting by name would pool every alanine in the chain
    lig_names = {x for x in spec if not x.isdigit()}
    lig_nums = {int(x) for x in spec if x.isdigit()}
    cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0

    st = gemmi.read_structure(path)
    st.setup_entities()
    model = st[0]

    prot = max(model, key=lambda c: sum(1 for r in c if r.name in AA))
    prot_nums = sorted(r.seqid.num for r in prot if r.name in AA)

    # only the copy in the protein chain being analysed; structures with
    # several chains hold one ligand each, and pooling them would give a
    # box spanning the whole crystal
    lig_xyz, lig_res = [], []
    for ch in model:
        if ch.name != prot.name:
            continue
        for r in ch:
            if r.name in lig_names or r.seqid.num in lig_nums:
                lig_xyz += [[a.pos.x, a.pos.y, a.pos.z] for a in r]
                lig_res.append(f"{r.name}{r.seqid.num}")
    if not lig_xyz:
        sys.exit(f"error: no residue named {sorted(lig_names)} in {path}")
    lig = np.array(lig_xyz)

    hits = []

    for r in prot:
        if r.name not in AA:
            continue
        if r.name in lig_names or r.seqid.num in lig_nums:
            continue

        xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r])
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
                    f"{prot.name}:{author_residue_id} ({r.name})"
                )

            if not canonical_chain_id:
                raise ValueError(
                    f"Missing canonical chain ID for "
                    f"{prot.name}:{author_residue_id} ({r.name})"
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
        f"protein   : author chain {prot.name}, {len(prot_nums)} residues "
        f"({prot_nums[0]}-{prot_nums[-1]})"
    )
    print(f"ligand    : {','.join(lig_res)}  ({len(lig)} atoms)")
    print(f"cutoff    : {cutoff} A")
    print(f"\npocket: {len(hits)} residues")

    print("  Author chain:residue   Canonical chain:index   Residue   Distance")
    for hit in hits:
        author_identifier = f"{prot.name}:{hit['author_residue_id']}"
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
        f"\nauthor residues, chain {prot.name}: "
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
        f"\nIMPORTANT pass author chain {prot.name} "
        "to prepare_receptor.py."
    )
    print("          structures with several chains hold one ligand each,")
    print("          and the two scripts can otherwise pick different chains,")
    print("          leaving the docking box outside the receptor.")
    print(
        f"box size  : {extent.max() + 10:.0f} A "
        f"(ligand extent {[round(float(x), 1) for x in extent]})"
    )

    modified_residues = [
        f"{prot.name}:{hit['author_residue_id']} {hit['residue_name']}"
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