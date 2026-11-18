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

Docking box
-----------
Two box reports exist, and they are not the same box.

*Legacy*: the `box centre` and `box size` lines are printed unchanged for
every run. They use the ligand atom mean as the centre and a single cubic
`max(extent) + 10` size formatted to the nearest integer. They are kept so
that previously used boxes remain reconstructable, and they are human-readable
output only.

*Recorded box*: requested with `--box-record`, together with exactly one of
`--box-size` (explicit dimensions) or `--box-padding` (clearance per face).
It centres on the midpoint of the selection's coordinate bounds by default,
carries three independent dimensions, and is written as a machine-readable
record that `dock.py` reads directly, so no centre or dimension is retyped.
Neither sizing mode has a default: the box dimensions are a scientific choice
and must be stated. `--box-padding` is a guarantee on all six faces, so it is
accepted only with bounds-midpoint centring; the historical atom-mean centring
is accepted with `--box-size` only.

Example:
    python python/map_pocket.py data/1P3D.cif UMA 5.0 \
        --protein-chain A --ligand-chain E \
        --box-size 24 --box-record boxes/murc_uma_chainE_24A.json
"""
import argparse
import sys
from pathlib import Path

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry

# Recognised protein residue names, including two common modifications.
AA = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
         "SER THR TRP TYR VAL MSE KCX".split())


def parse_lengths(parser: argparse.ArgumentParser, text: str | None,
                  flag: str) -> np.ndarray | None:
    """Parse one or three comma-separated lengths in A into a (3,) array.

    One value is applied to all three axes; three give independent axes. The
    values are only checked for syntax and finiteness here; whether they make
    a usable box is decided by `box_geometry.validated_dimensions`.
    """
    if text is None:
        return None
    fields = [field.strip() for field in text.split(",")]
    if len(fields) not in (1, 3):
        parser.error(f"{flag} takes one value or three comma-separated values.")
    try:
        values = [float(field) for field in fields]
    except ValueError:
        parser.error(f"{flag} values must be numbers in Angstroms, got {text!r}.")
    if not all(np.isfinite(value) for value in values):
        parser.error(f"{flag} values must be finite, got {text!r}.")
    return np.array(values * 3 if len(values) == 1 else values, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure_path")
    parser.add_argument("ligand", help="Component names or author residue numbers")
    parser.add_argument("cutoff", nargs="?", type=float, default=5.0)
    parser.add_argument("--protein-chain", required=True,
                        help="Canonical protein chain ID")
    parser.add_argument("--ligand-chain", required=True,
                        help="Comma-separated canonical ligand IDs")
    parser.add_argument("--box-record", metavar="PATH",
                        help="Write a machine-readable box record to PATH")
    parser.add_argument("--box-centre", default="bounds-midpoint",
                        choices=list(box_geometry.CENTRE_MODES),
                        help="Centring mode for the recorded box "
                             "(default: bounds-midpoint). ligand-atom-mean "
                             "reconstructs a historical box and is accepted "
                             "with --box-size only, never with --box-padding")
    sizing = parser.add_mutually_exclusive_group()
    sizing.add_argument("--box-size", metavar="S|SX,SY,SZ",
                        help="Explicit recorded-box dimensions in A")
    sizing.add_argument("--box-padding", metavar="P|PX,PY,PZ",
                        help="Clearance per face in A; dimensions become "
                             "extent + 2 x padding, rounded up to 0.1 A")
    parser.add_argument("--overwrite-box-record", action="store_true",
                        help="Replace an existing box record at that path")
    args = parser.parse_args()

    if (args.box_size or args.box_padding) and not args.box_record:
        parser.error("--box-size and --box-padding require --box-record.")
    if args.box_record and not (args.box_size or args.box_padding):
        parser.error(
            "--box-record requires exactly one of --box-size or --box-padding; "
            "the box dimensions are a scientific choice and have no default."
        )
    box_size = parse_lengths(parser, args.box_size, "--box-size")
    box_padding = parse_lengths(parser, args.box_padding, "--box-padding")

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

    if args.box_record:
        write_recorded_box(
            parser=parser,
            args=args,
            ligand_xyz=lig,
            ligand_descriptions=lig_res,
            ligand_author_ids=sorted({author for author, _ in ligand_residues}),
            ligand_author_residue_ids=[
                f"{author}:{residue.seqid.num}{str(residue.seqid.icode).strip()}"
                for author, residue in ligand_residues
            ],
            protein_author_id=protein_author_id,
            hits=hits,
            box_size=box_size,
            box_padding=box_padding,
        )


def write_recorded_box(*, parser, args, ligand_xyz, ligand_descriptions,
                       ligand_author_ids, ligand_author_residue_ids,
                       protein_author_id, hits, box_size, box_padding):
    """Build and write the machine-readable box record, then report it.

    The record is derived from the same ligand atom selection as the legacy
    lines, but centres on the bounds midpoint by default and carries three
    independent dimensions. Face clearances against the selection are printed
    so the clearance actually delivered on each face is visible, rather than
    being inferred from a nominal padding figure.
    """
    try:
        centre, size, sizing = recorded_box_geometry(
            ligand_xyz, args.box_centre, box_size, box_padding)
    except box_geometry.BoxError as error:
        parser.error(str(error))
    record = assemble_record(args, centre, size, sizing, ligand_xyz,
                             ligand_descriptions, ligand_author_ids,
                             ligand_author_residue_ids, protein_author_id, hits)
    try:
        destination = box_geometry.write_record(
            args.box_record, record, overwrite=args.overwrite_box_record)
    except box_geometry.BoxError as error:
        parser.error(str(error))

    low, high = box_geometry.face_clearances(centre, size, ligand_xyz)
    print(f"\nrecorded box : {destination}")
    print(f"  centring   : {args.box_centre}")
    print(f"  centre     : {[round(float(value), 3) for value in centre]} A")
    print(f"  size x,y,z : {[round(float(value), 2) for value in size]} A")
    print(f"  ligand clearance, low faces : "
          f"{[round(float(value), 2) for value in low]} A")
    print(f"  ligand clearance, high faces: "
          f"{[round(float(value), 2) for value in high]} A")
    print(f"  minimum face clearance      : {min(low.min(), high.min()):.2f} A")
    print("  pass this record to dock.py --box-record; do not retype the numbers.")


def recorded_box_geometry(ligand_xyz, centre_mode, box_size, box_padding):
    """Return the recorded box's centre, dimensions and sizing provenance.

    The centring and sizing combination is validated first: a per-face padding
    is a guarantee on all six faces, which only bounds-midpoint centring can
    deliver, so atom-mean centring is accepted with explicit dimensions only.
    """
    box_geometry.validated_centre_and_sizing(
        centre_mode,
        "explicit-size" if box_size is not None else "padding-per-face")
    centre = box_geometry.centre_from_points(ligand_xyz, centre_mode)
    extent = [float(value) for value in box_geometry.extent_from_points(ligand_xyz)]
    if box_size is not None:
        size = box_geometry.validated_dimensions(box_size)
        sizing = {
            "mode": "explicit-size",
            "requested_angstrom": [float(value) for value in box_size],
            "ligand_extent_angstrom": [
                float(value) for value in box_geometry.extent_from_points(ligand_xyz)
            ],
        }
    else:
        size = box_geometry.dimensions_from_padding(ligand_xyz, box_padding)
        sizing = {
            "mode": "padding-per-face",
            "padding_per_face_angstrom": [float(value) for value in box_padding],
            "ligand_extent_angstrom": [
                float(value) for value in box_geometry.extent_from_points(ligand_xyz)
            ],
            "rounding": "dimensions rounded up to 0.1 A so padding is a minimum",
        }
    sizing["ligand_extent_angstrom"] = extent
    return centre, size, sizing


def assemble_record(args, centre, size, sizing, ligand_xyz, ligand_descriptions,
                    ligand_author_ids, ligand_author_residue_ids,
                    protein_author_id, hits):
    """Assemble the box record's selection and provenance fields.

    The selection block exists so that a box can be traced to the protein copy
    it belongs to: canonical and author identifiers for both the ligand and the
    protein, the components matched, the cutoff and the pocket residue list.
    """
    selection = {
        "description": (
            f"{','.join(sorted({d.split(' ')[0] for d in ligand_descriptions}))} "
            f"in canonical chain(s) {args.ligand_chain}, "
            f"author chain(s) {','.join(ligand_author_ids)}"
        ),
        "ligand_argument": args.ligand,
        "ligand_canonical_chain_ids": sorted(
            value.strip() for value in args.ligand_chain.split(",")
        ),
        "ligand_author_chain_ids": ligand_author_ids,
        "ligand_residues": ligand_descriptions,
        "ligand_author_residue_ids": ligand_author_residue_ids,
        "ligand_atom_count": int(len(ligand_xyz)),
        "protein_canonical_chain_id": args.protein_chain,
        "protein_author_chain_id": protein_author_id,
        "pocket_cutoff_angstrom": float(args.cutoff),
        "pocket_residue_count": len(hits),
        "pocket_author_residue_ids": [hit["author_residue_id"] for hit in hits],
        "pocket_canonical_residue_indices": [
            hit["canonical_residue_index"] for hit in hits
        ],
        "pocket_modified_residues": [
            f"{protein_author_id}:{hit['author_residue_id']} {hit['residue_name']}"
            for hit in hits if hit["residue_name"] in ("MSE", "KCX")
        ],
    }

    repository_root = Path(__file__).resolve().parents[1]
    return box_geometry.build_record(
        centre=centre,
        size=size,
        centre_mode=args.box_centre,
        sizing=sizing,
        selection=selection,
        source_structure_path=args.structure_path,
        software={
            "script": "python/map_pocket.py",
            "git": box_geometry.git_commit(repository_root),
            "gemmi": gemmi.__version__,
            "numpy": np.__version__,
        },
        notes=[
            "The box is a search region for AutoDock Vina, not a pocket "
            "definition, and covering a region is not evidence that the "
            "region is the correct one.",
            "Prepare the receptor from the recorded author protein chain and "
            "the recorded source structure; the box carries no transformation.",
            "Pocket residues are reported for reference and do not enter the "
            "box arithmetic.",
        ],
    )


if __name__ == "__main__":
    main()
     