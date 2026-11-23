"""Convert a single-pose ligand PDBQT to PDB for viewing, without re-perceiving.

A PDBQT carries AutoDock atom types in its element column and, as Meeko writes
it, repeated element-like strings in its name column. Opening one directly in a
viewer therefore shows a molecule whose atoms cannot be identified and whose
bonds the viewer has guessed from distances.

This script restores identity from evidence instead. Atom names come from the
validated atom map, keyed on PDBQT serial; connectivity comes from the chemical
component dictionary as `CONECT` records. Coordinates are copied unchanged.

**Why not a chemistry toolkit.** OpenBabel, PyMOL and similar converters
perceive bonds and elements from geometry, may add or reposition hydrogens, and
can emit coordinates that differ from the input. Any of those would break the
one thing a pose comparison depends on: that the atom at a given position is
the atom we think it is, where the docking put it. Nothing here perceives
anything.

What this does
--------------
- copies every mapped heavy atom's coordinates verbatim, to PDB's three
  decimals, which is the precision the PDBQT already carried;
- names each atom from the atom map, so names are the reference names;
- writes the element from the component dictionary, not from the AutoDock type;
- writes `CONECT` records from the dictionary's heavy-atom bonds, so the viewer
  draws the experimentally established connectivity rather than guessing it;
- checks that every heavy atom of the reference is present, and that no
  coordinate moved.

What it does not do
-------------------
No hydrogens are written: the poses are compared on heavy atoms, and the
retained polar hydrogens carry no mapping to reference names. No bond orders
are written, because PDB `CONECT` has no field for them; the dictionary remains
the authority on bond order, and the stereochemistry recorded by the
verification stage is unaffected by this conversion, which moves nothing.

Every structure written by this script is in the receptor's own coordinate
frame. No alignment, fitting or superposition is applied anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_redocking
import box_geometry
import chemical_component
import chemical_graph
import ligand_atom_map

# PDB and PDBQT both write coordinates to three decimals, so a faithful copy
# must agree exactly. The tolerance exists only to absorb float formatting.
COORDINATE_TOLERANCE_ANGSTROM = 1e-6


class ConversionError(RuntimeError):
    """The pose could not be converted without losing identity or geometry."""


def pdb_atom_line(serial: int, name: str, element: str, residue: str,
                  chain: str, residue_number: int,
                  position: tuple[float, float, float]) -> str:
    """Return one PDB ATOM record.

    Atom names are padded as the PDB specification expects for names of fewer
    than four characters: a leading space, so the element letter sits in
    column 14. Primed CCD names such as `O3'` are written as they stand.
    """
    if len(name) >= 4:
        padded = name[:4]
    else:
        padded = f" {name:<3s}"
    return (f"HETATM{serial:5d} {padded}{residue:>4s} {chain}"
            f"{residue_number:4d}    "
            f"{position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
            f"  1.00  0.00          {element:>2s}")


def conect_records(graph: chemical_graph.MolecularGraph,
                   serial_of_name: dict[str, int]) -> list[str]:
    """Return CONECT records for the dictionary's heavy-atom bonds.

    One record per bond, which is valid PDB and avoids the four-partner
    continuation rules entirely. Bond order is not representable here.
    """
    lines = []
    for pair in sorted(graph.bonds,
                       key=lambda bond: sorted(graph.names[i] for i in bond)):
        first, second = sorted(pair)
        name_a, name_b = graph.names[first], graph.names[second]
        if name_a not in serial_of_name or name_b not in serial_of_name:
            continue
        lines.append(f"CONECT{serial_of_name[name_a]:5d}"
                     f"{serial_of_name[name_b]:5d}")
    return lines


def convert(pose_path: Path, atom_map_path: Path, dictionary_path: Path,
            destination: Path, residue: str, chain: str,
            residue_number: int, overwrite: bool = False) -> dict[str, object]:
    """Write the pose as a named, connected PDB and return a record of it."""
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise ConversionError(
            f"{destination} already exists; pass --overwrite only if you mean "
            f"to replace it")

    atom_map_record = ligand_atom_map.read_atom_map(atom_map_path)
    names_by_serial = ligand_atom_map.name_by_serial(atom_map_record)
    component = chemical_component.read_component(dictionary_path)

    poses = analyse_redocking.parse_pose_file(pose_path, names_by_serial)
    if len(poses) != 1:
        raise ConversionError(
            f"{pose_path} holds {len(poses)} poses; extract one first so the "
            f"conversion cannot silently take the wrong model")
    coordinates = poses[0]["coordinates"]

    expected = set(component.heavy_atom_names)
    missing = sorted(expected - set(coordinates))
    unexpected = sorted(set(coordinates) - expected)
    if missing or unexpected:
        raise ConversionError(
            f"the pose does not carry the component's heavy atoms; missing "
            f"{missing}, unexpected {unexpected}")

    # Written in the dictionary's atom order, so every converted pose has the
    # same atom order and a viewer's atom indices mean the same thing in each.
    ordered = list(component.heavy_atom_names)
    serial_of_name = {name: index for index, name in enumerate(ordered, start=1)}
    lines = [
        f"REMARK   converted from {Path(pose_path).name} by pose_to_pdb.py",
        "REMARK   coordinates copied unchanged; no alignment or superposition",
        "REMARK   names from the validated atom map, bonds from the CCD",
    ]
    for name in ordered:
        lines.append(pdb_atom_line(
            serial_of_name[name], name, component.elements.get(name, ""),
            residue, chain, residue_number, coordinates[name]))
    lines.extend(conect_records(component.graph, serial_of_name))
    lines.append("END")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")

    # Read the file back and confirm nothing moved. Checking the written bytes
    # rather than the in-memory values is the point: a formatting error that
    # truncated or shifted a column would otherwise pass unnoticed.
    written = {}
    for line in destination.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            written[line[12:16].strip()] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]))
    worst_name, worst = None, 0.0
    for name, original in coordinates.items():
        if name not in written:
            raise ConversionError(f"{name} is absent from the written PDB")
        deviation = max(abs(a - b) for a, b in zip(original, written[name]))
        if deviation > worst:
            worst_name, worst = name, deviation
    if worst > COORDINATE_TOLERANCE_ANGSTROM:
        raise ConversionError(
            f"conversion moved {worst_name} by {worst:.6f} A; coordinates must "
            f"be copied, not recomputed")

    return {
        "schema": "mur-ligase-pose-conversion/1",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_pose": str(Path(pose_path).resolve()),
        "source_pose_sha256": box_geometry.file_sha256(pose_path),
        "atom_map": str(Path(atom_map_path).resolve()),
        "atom_map_sha256": box_geometry.file_sha256(atom_map_path),
        "component_dictionary": str(Path(dictionary_path).resolve()),
        "component_dictionary_sha256": box_geometry.file_sha256(dictionary_path),
        "output_pdb": str(destination.resolve()),
        "output_pdb_sha256": box_geometry.file_sha256(destination),
        "component": component.identifier,
        "n_heavy_atoms": len(ordered),
        "n_conect_records": len(component.graph.bonds),
        "max_coordinate_deviation_angstrom": round(worst, 9),
        "coordinate_frame": (
            "the receptor's own frame, as the pose file carried it; no "
            "alignment, fitting or superposition applied"),
        "hydrogens": "not written; heavy-atom comparison only",
        "perception": "none; names from the atom map, bonds from the dictionary",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Convert a single-pose ligand PDBQT to a named, connected "
                     "PDB for viewing, copying coordinates unchanged."))
    parser.add_argument("pose", type=Path, help="single-pose ligand PDBQT")
    parser.add_argument("output", type=Path, help="PDB file to write")
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--component-dictionary", type=Path, required=True)
    parser.add_argument("--residue", default="UMA",
                        help="residue name for the written records")
    parser.add_argument("--chain", default="L",
                        help="chain identifier; distinct from the receptor's")
    parser.add_argument("--residue-number", type=int, default=1)
    parser.add_argument("--record", type=Path, default=None,
                        help="write the conversion record as JSON")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        record = convert(args.pose, args.atom_map, args.component_dictionary,
                         args.output, args.residue, args.chain,
                         args.residue_number, args.overwrite)
    except (ConversionError, chemical_graph.CorrespondenceError,
            ligand_atom_map.AtomMapError) as error:
        sys.exit(f"error: {error}")

    if args.record is not None:
        Path(args.record).write_text(json.dumps(record, indent=2) + "\n")
    print(f"converted      : {record['source_pose']}")
    print(f"wrote          : {record['output_pdb']}")
    print(f"heavy atoms    : {record['n_heavy_atoms']}")
    print(f"CONECT records : {record['n_conect_records']}")
    print(f"max deviation  : {record['max_coordinate_deviation_angstrom']} A")
    print(f"frame          : {record['coordinate_frame']}")


if __name__ == "__main__":
    main()
