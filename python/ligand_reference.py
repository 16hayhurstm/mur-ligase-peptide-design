"""Preserve a deposited ligand as an immutable reference pose.

usage:
    python python/ligand_reference.py data/1P3D.cif UMA references/uma_chainE \
        --ligand-canonical-chain E \
        --component-dictionary project_private/ccd_downloads/UMA.cif

Why this is a separate stage
----------------------------
The reference pose and the docked ligand must never be the same file. The
reference is the experimental answer the redocking control is tested against;
the docked ligand has been through preparation, which adds hydrogens, may
alter protonation and will be re-placed by the search. A reference that has
been through preparation is no longer a reference, and comparing a pose with
its own preparation product would make the control vacuous.

This writes two things and never writes anything else:

- a reference PDB holding the selected ligand's heavy atoms exactly as
  deposited, with no hydrogens added, no transformation and no minimisation;
- a reference record naming the source structure and its SHA-256, the
  canonical and author selection, the atom mapping, the component dictionary
  used and its SHA-256, and the identity check between them.

Both are refused if they already exist. The source structure is only read.

What the reference is not
-------------------------
It is not evidence that the deposited pose is the only or the best pose, and
it carries the crystal's own limitations: it is one conformer, at one
temperature, in a lattice, with the protein side chains that accompanied it.
Recovering it is a test of the docking setup, not of biology.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry
import chemical_component
import chemical_graph
import receptor_provenance as provenance

REFERENCE_SCHEMA = "mur-ligase-ligand-reference/1"


class ReferenceError(RuntimeError):
    """Raised when a reference cannot be extracted or verified."""


def extract_ligand(structure_path: Path, component: str,
                   canonical_chain: str) -> tuple[gemmi.Structure, list[gemmi.Residue], str]:
    """Return the selected ligand residues and their author chain.

    Selection is by canonical mmCIF chain (`label_asym_id`) and component
    name, matching how `map_pocket.py` selects the pocket-defining ligand, so
    the reference and the box describe the same copy. A selection matching
    more than one residue, or none, raises.
    """
    structure = gemmi.read_structure(str(structure_path))
    structure.setup_entities()
    if len(structure) == 0:
        raise ReferenceError(f"{structure_path} contains no models")
    if len(structure) > 1:
        print(f"NOTE {structure_path} holds {len(structure)} models; using the first.")
    found: list[tuple[str, gemmi.Residue]] = []
    for chain in structure[0]:
        for residue in chain:
            if residue.subchain == canonical_chain and residue.name == component:
                found.append((chain.name, residue))
    if not found:
        raise ReferenceError(
            f"no {component} residue in canonical chain {canonical_chain!r} of "
            f"{structure_path}"
        )
    author_chains = {chain for chain, _ in found}
    if len(found) != 1 or len(author_chains) != 1:
        raise ReferenceError(
            f"the selection matches {len(found)} residues in author chains "
            f"{sorted(author_chains)}; a reference pose must be one copy."
        )
    return structure, [found[0][1]], found[0][0]


def heavy_atoms_of(residue: gemmi.Residue) -> dict[str, tuple[float, float, float]]:
    """Return the residue's non-hydrogen atoms as name -> coordinates."""
    atoms: dict[str, tuple[float, float, float]] = {}
    for atom in residue:
        if atom.element == gemmi.Element("H"):
            continue
        if atom.name in atoms:
            raise ReferenceError(
                f"duplicated atom name {atom.name} in {residue.name}; the "
                f"reference atom mapping would be ambiguous"
            )
        atoms[atom.name] = (atom.pos.x, atom.pos.y, atom.pos.z)
    if not atoms:
        raise ReferenceError(f"{residue.name} has no heavy atoms")
    return atoms


def write_reference_pdb(path: Path, structure: gemmi.Structure,
                        residue: gemmi.Residue, author_chain: str) -> None:
    """Write the ligand alone to a PDB, coordinates untouched."""
    out = gemmi.Structure()
    out.spacegroup_hm = structure.spacegroup_hm
    out.cell = structure.cell
    model = gemmi.Model("1")
    chain = gemmi.Chain(author_chain)
    kept = gemmi.Residue()
    kept.name = residue.name
    kept.seqid = residue.seqid
    kept.het_flag = "H"
    for atom in residue:
        if atom.element == gemmi.Element("H"):
            continue
        kept.add_atom(atom)
    chain.add_residue(kept)
    model.add_chain(chain)
    out.add_model(model)
    out.setup_entities()
    out.write_pdb(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structure", type=Path, help="Source mmCIF")
    parser.add_argument("component", help="Component name, e.g. UMA")
    parser.add_argument("out_prefix", type=Path, help="Output prefix")
    parser.add_argument("--ligand-canonical-chain", required=True,
                        help="Canonical mmCIF chain of the copy to preserve")
    parser.add_argument("--component-dictionary", type=Path,
                        help="CCD entry for the component, for the identity check")
    parser.add_argument("--box-record", type=Path,
                        help="Box record to cross-check the source structure against")
    args = parser.parse_args()

    try:
        build_reference(args)
    except (ReferenceError, chemical_component.ComponentError,
            chemical_graph.CorrespondenceError, box_geometry.BoxError,
            provenance.ReceptorError) as error:
        sys.exit(f"error: {error}")


def build_reference(args: argparse.Namespace) -> None:
    if not args.structure.is_file():
        raise ReferenceError(f"structure not found: {args.structure}")
    pdb_path = args.out_prefix.with_name(args.out_prefix.name + "_reference.pdb")
    record_path = args.out_prefix.with_name(
        args.out_prefix.name + ".reference.json")
    for path in (pdb_path, record_path):
        if path.exists():
            raise ReferenceError(
                f"{path} already exists. A reference pose is written once and "
                f"never replaced; choose a fresh prefix."
            )

    structure, residues, author_chain = extract_ligand(
        args.structure, args.component, args.ligand_canonical_chain)
    residue = residues[0]
    atoms = heavy_atoms_of(residue)
    elements = {atom.name: atom.element.name.upper() for atom in residue
                if atom.element != gemmi.Element("H")}

    identity: dict[str, object] = {"checked": False}
    symmetry: dict[str, object] = {}
    if args.component_dictionary:
        component = chemical_component.read_component(args.component_dictionary)
        identity = chemical_component.compare_with_coordinates(component, elements)
        identity["checked"] = True
        identity["dictionary_path"] = str(args.component_dictionary)
        identity["dictionary_sha256"] = box_geometry.file_sha256(
            args.component_dictionary)
        identity["dictionary_formal_charge"] = component.dictionary_formal_charge
        identity["dictionary_name"] = component.name
        identity["dictionary_formula"] = component.formula
        identity["n_declared_stereocentres"] = len(component.stereocentres)
        if not identity["identical_atom_set"]:
            raise ReferenceError(
                f"the deposited {args.component} does not match its dictionary "
                f"entry: missing {identity['missing_from_observed']}, "
                f"unexpected {identity['not_in_dictionary']}, element "
                f"mismatches {identity['element_mismatches']}. A reference "
                f"whose chemical identity is uncertain is not usable."
            )
        # The reference knows the dictionary's neutral form, not the state the
        # ligand will be docked in, so no group is treated as
        # resonance-equivalent here. The symmetry that applies to an RMSD
        # depends on the modelled protonation and is computed from the ligand
        # manifest at analysis time.
        graph, resonance = chemical_graph.normalise_resonance(
            component.graph, hydrogen_bearing=None)
        permutations = chemical_graph.automorphisms(graph)
        symmetry = {
            "basis": ("dictionary connectivity only; no protonation state is "
                      "assumed, so no resonance equivalence is applied"),
            "resonance_normalisation": resonance,
            "n_automorphisms": len(permutations),
            "orbits": chemical_graph.symmetry_classes(graph, permutations),
            "note": ("the symmetry used for pose comparison is computed from "
                     "the modelled protonation state in the ligand manifest, "
                     "not from this record"),
        }

    box_agreement: dict[str, object] = {"checked": False}
    if args.box_record:
        record = box_geometry.read_record(args.box_record)
        source = record.get("source_structure", {}) or {}
        structure_hash = box_geometry.file_sha256(args.structure)
        box_agreement = {
            "checked": True,
            "box_record": str(args.box_record.resolve()),
            "box_record_sha256": box_geometry.file_sha256(args.box_record),
            "same_source_structure": source.get("sha256") == structure_hash,
            "box_ligand_selection": (record.get("selection", {})
                                     .get("ligand_canonical_chain_ids")),
        }
        if not box_agreement["same_source_structure"]:
            raise ReferenceError(
                f"the box record was built from a different structure "
                f"(sha256 {str(source.get('sha256'))[:12]}...) than "
                f"{args.structure} ({structure_hash[:12]}...)"
            )

    write_reference_pdb(pdb_path, structure, residue, author_chain)
    written = provenance.parse_atom_records(pdb_path)
    if len(written) != len(atoms):
        raise ReferenceError(
            f"wrote {len(written)} atoms but selected {len(atoms)}")
    comparison = provenance.compare_atom_sets(
        {key: record for key, record in written.items()},
        {key: record for key, record in written.items()})
    # Compare the written file back against the source coordinates directly.
    worst = 0.0
    for key, record in written.items():
        source_position = atoms.get(key.atom_name)
        if source_position is None:
            raise ReferenceError(f"{key.atom_name} is not in the source selection")
        worst = max(worst, provenance.distance(record.position, source_position))
    if worst > provenance.COORDINATE_TOLERANCE_ANGSTROM:
        raise ReferenceError(
            f"the written reference deviates from the deposited coordinates by "
            f"{worst:.4f} A, beyond the "
            f"{provenance.COORDINATE_TOLERANCE_ANGSTROM} A rounding tolerance"
        )

    reference_record = {
        "schema": REFERENCE_SCHEMA,
        "component": args.component,
        "role": ("immutable reference pose for redocking comparison; never "
                 "the file that is docked"),
        "source_structure": {
            "path": str(args.structure),
            "sha256": box_geometry.file_sha256(args.structure),
            "model": 1,
            "coordinate_frame": ("source structure as read; no transformation, "
                                 "recentring, minimisation or hydrogen addition"),
        },
        "selection": {
            "component": args.component,
            "ligand_canonical_chain_id": args.ligand_canonical_chain,
            "ligand_author_chain_id": author_chain,
            "author_residue_id": f"{residue.seqid.num}"
                                 f"{str(residue.seqid.icode).strip()}",
            "n_heavy_atoms": len(atoms),
        },
        "atom_mapping": {name: {"element": elements[name],
                                "coordinates_angstrom": list(position)}
                         for name, position in sorted(atoms.items())},
        "chemical_identity": identity,
        "symmetry": symmetry,
        "box_agreement": box_agreement,
        "outputs": {
            "reference_pdb": str(pdb_path),
            "reference_pdb_sha256": box_geometry.file_sha256(pdb_path),
            "max_deviation_from_deposited_angstrom": round(worst, 6),
            "tolerance_angstrom": provenance.COORDINATE_TOLERANCE_ANGSTROM,
        },
        "limitations": [
            "one crystallographic conformer, in a lattice, at the deposited "
            "temperature; not the only possible bound pose",
            "the pocket side-chain conformations in the source structure are "
            "those adopted with this ligand bound, so recovering the pose is "
            "a test of the docking setup rather than an independent prediction",
            "hydrogens are absent by design; comparisons are heavy-atom only",
        ],
    }
    provenance.write_manifest(record_path, reference_record)

    print(f"source structure : {args.structure}")
    print(f"component        : {args.component} "
          f"(canonical {args.ligand_canonical_chain}, author {author_chain}:"
          f"{residue.seqid.num})")
    print(f"heavy atoms      : {len(atoms)}")
    if identity.get("checked"):
        print(f"dictionary       : {identity['dictionary_name']}")
        print(f"  formula        : {identity['dictionary_formula']}, "
              f"dictionary charge {identity['dictionary_formal_charge']}")
        print(f"  identity       : atom set identical to the dictionary")
        print(f"  stereocentres  : {identity['n_declared_stereocentres']} declared")
    if symmetry:
        print(f"symmetry         : {symmetry['n_automorphisms']} automorphisms, "
              f"orbits {symmetry['orbits']}")
    print(f"max deviation    : {worst:.4f} A from the deposited coordinates")
    print(f"wrote            : {pdb_path}")
    print(f"record           : {record_path}")
    print("\nThis file is the reference. Do not dock it, and do not let ligand "
          "preparation\noverwrite it; prepare the docking ligand separately.")


if __name__ == "__main__":
    main()
