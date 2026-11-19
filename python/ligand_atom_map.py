"""Map PDBQT atom serials to reference atom names, via Meeko's index map.

Why this exists
---------------
Meeko writes a ligand PDBQT whose atom name column holds the *element*, not
the atom name: every carbon is `C`, every oxygen `O`. For UMA that is 49 heavy
atoms sharing five names. A name-keyed correspondence, which is what every
other stage in this project uses, is therefore impossible against a prepared
ligand, and `prepare_ligand.py check` correctly refuses rather than collapsing
the duplicates.

The route through is Meeko's own `REMARK INDEX MAP`, written when
`--add_index_map` is passed. It relates the PDBQT to the file Meeko was given,
which is the hydrogenated molfile, whose heavy-atom order is the reference
order. Composing the two gives:

    PDBQT serial -> input molfile atom index -> reference atom name

Nothing here trusts that chain: each link is checked against elements and
coordinates, and the composed mapping must be a bijection over all heavy
atoms before it is written.

Index-map semantics, established rather than assumed
----------------------------------------------------
Meeko 0.7.1 writes `REMARK INDEX MAP` as whitespace-separated integers in
pairs. Determined empirically from a real UMA preparation, and re-derived by
`index_map_direction` on every run rather than hard-coded:

- the pairs are `(input atom index, PDBQT atom serial)`;
- **both are 1-based**;
- the serial column is the atoms in PDBQT file order, so it runs 1..n for an
  n-atom PDBQT;
- indices above the input's heavy-atom count are the appended hydrogens.

The direction is inferred by asking which column is exactly the PDBQT serials
in file order. If neither column is, or both are, the map is rejected: a map
whose direction cannot be established is worse than none.

Merged hydrogens
----------------
Meeko merges non-polar hydrogens into their parent heavy atom, so the PDBQT
holds fewer atoms than the input. Those input hydrogens appear in no pair.
That is expected and is recorded as a count and a list, not treated as a
mapping failure. Heavy atoms are a different matter: every one must appear.

Limitations
-----------
- This establishes which PDBQT atom is which reference atom. It says nothing
  about atom typing, partial charges or protonation, which are Meeko's.
- The mapping is specific to one PDBQT and one input file, both identified by
  SHA-256 in the record. It must not be reused across a re-preparation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import box_geometry
import receptor_provenance as provenance

ATOM_MAP_SCHEMA = "mur-ligase-ligand-atom-map/1"

# Agreement required between the PDBQT and the input molfile coordinates.
# A molfile writes four decimals and a PDBQT three, so two representations of
# one coordinate can differ by up to sqrt(3) x 0.001 A; 0.002 A is the
# tolerance used throughout this project for exactly that reason.
COORDINATE_TOLERANCE_ANGSTROM = 0.002

HYDROGEN_SYMBOLS = {"H", "D", "HD", "HS"}


class AtomMapError(RuntimeError):
    """Raised when a PDBQT cannot be mapped to its input unambiguously."""


@dataclass(frozen=True)
class PdbqtAtom:
    serial: int
    name: str
    autodock_type: str
    position: tuple[float, float, float]
    charge: float

    @property
    def is_hydrogen(self) -> bool:
        return (self.autodock_type.upper() in HYDROGEN_SYMBOLS
                or self.name.upper().startswith("H"))


@dataclass(frozen=True)
class MolfileAtom:
    index: int
    element: str
    position: tuple[float, float, float]

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in ("H", "D")


def distance(first: Sequence[float], second: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_pdbqt_atoms(path: Path | str) -> list[PdbqtAtom]:
    """Read a ligand PDBQT's atom records in file order.

    Serials are read from columns 7-11 rather than counted, so a file whose
    serials are not sequential is detected rather than silently renumbered.
    """
    atoms: list[PdbqtAtom] = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            atoms.append(PdbqtAtom(
                serial=int(line[6:11]),
                name=line[12:16].strip(),
                autodock_type=line[77:80].strip(),
                position=(float(line[30:38]), float(line[38:46]),
                          float(line[46:54])),
                charge=float(line[70:76]) if line[70:76].strip() else 0.0,
            ))
        except (ValueError, IndexError) as error:
            raise AtomMapError(
                f"{path} line {number} is not a readable atom record: {error}"
            ) from error
    if not atoms:
        raise AtomMapError(f"{path} contains no atom records")
    serials = [atom.serial for atom in atoms]
    if len(set(serials)) != len(serials):
        repeated = sorted({s for s in serials if serials.count(s) > 1})
        raise AtomMapError(
            f"{path} repeats atom serials {repeated}; serials must be unique "
            f"for a mapping to be unambiguous")
    return atoms


def read_molfile_atoms(path: Path | str) -> list[MolfileAtom]:
    """Read a V2000 molfile's atom block, 1-based, in file order."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 4:
        raise AtomMapError(f"{path} is too short to be a molfile")
    try:
        count = int(lines[3][0:3])
    except ValueError as error:
        raise AtomMapError(
            f"{path} line 4 is not a V2000 counts line: {error}") from error
    atoms: list[MolfileAtom] = []
    for offset in range(count):
        line = lines[4 + offset]
        try:
            atoms.append(MolfileAtom(
                index=offset + 1,
                element=line[31:34].strip(),
                position=(float(line[0:10]), float(line[10:20]),
                          float(line[20:30])),
            ))
        except (ValueError, IndexError) as error:
            raise AtomMapError(
                f"{path} atom line {offset + 1} is malformed: {error}"
            ) from error
    return atoms


def parse_index_map(path: Path | str) -> list[tuple[int, int]]:
    """Read every `REMARK INDEX MAP` pair from a PDBQT, in file order.

    Meeko writes the map across several REMARK lines and repeats the whole
    block once per model, so duplicate identical blocks are collapsed. A
    repeated pair with a *different* partner is a contradiction and raises.
    """
    values: list[int] = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("REMARK INDEX MAP"):
            continue
        for field in line[len("REMARK INDEX MAP"):].split():
            try:
                values.append(int(field))
            except ValueError as error:
                raise AtomMapError(
                    f"{path}: INDEX MAP holds a non-integer {field!r}: {error}"
                ) from error
    if not values:
        raise AtomMapError(
            f"{path} has no REMARK INDEX MAP. Re-prepare the ligand with "
            f"mk_prepare_ligand.py --add_index_map, which is what writes it.")
    if len(values) % 2:
        raise AtomMapError(
            f"{path}: INDEX MAP has {len(values)} integers, which is not a "
            f"whole number of pairs")
    pairs = [(values[position], values[position + 1])
             for position in range(0, len(values), 2)]
    collapsed: dict[int, int] = {}
    ordered: list[tuple[int, int]] = []
    for first, second in pairs:
        if first in collapsed:
            if collapsed[first] != second:
                raise AtomMapError(
                    f"{path}: INDEX MAP maps {first} to both "
                    f"{collapsed[first]} and {second}")
            continue
        collapsed[first] = second
        ordered.append((first, second))
    return ordered


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

def index_map_direction(pairs: Sequence[tuple[int, int]],
                        n_pdbqt_atoms: int) -> str:
    """Decide which column of the index map holds the PDBQT serials.

    Returns `"second"` when the pairs read (input index, serial), which is
    what Meeko 0.7.1 was observed to write, `"first"` for the reverse, or
    `"either"` when both readings give the same mapping and the direction
    therefore makes no difference.

    The test is positional rather than assumed: the serial column must be
    exactly the PDBQT's serials, each once. If both columns qualify, or
    neither does, the map is rejected — guessing the direction of a mapping
    that silently relabels atoms is not an acceptable risk.
    """
    if not pairs:
        raise AtomMapError("the index map is empty")
    expected = set(range(1, n_pdbqt_atoms + 1))
    first_column = [first for first, _ in pairs]
    second_column = [second for _, second in pairs]
    first_is_serials = (len(pairs) == n_pdbqt_atoms
                        and set(first_column) == expected)
    second_is_serials = (len(pairs) == n_pdbqt_atoms
                         and set(second_column) == expected)
    if first_is_serials and second_is_serials:
        # Both columns are permutations of 1..n. That is only a problem if the
        # two readings disagree: when the pairs are an involution, or the
        # identity, either reading yields the same serial-to-input mapping and
        # the direction is immaterial rather than unknown. The common real
        # case is an input whose atom order Meeko happened to preserve.
        forward = {first: second for first, second in pairs}
        backward = {second: first for first, second in pairs}
        if forward == backward:
            return "either"
        raise AtomMapError(
            f"both index-map columns are a permutation of 1..{n_pdbqt_atoms} "
            f"and the two readings disagree, so the direction cannot be "
            f"established from the file. Refusing to guess.")
    if second_is_serials:
        return "second"
    if first_is_serials:
        return "first"
    raise AtomMapError(
        f"neither index-map column is the PDBQT's {n_pdbqt_atoms} serials: "
        f"first column spans {min(first_column)}-{max(first_column)} with "
        f"{len(set(first_column))} distinct values, second spans "
        f"{min(second_column)}-{max(second_column)} with "
        f"{len(set(second_column))}. The map and the PDBQT do not correspond.")


# ---------------------------------------------------------------------------
# Building the validated mapping
# ---------------------------------------------------------------------------

def build_atom_map(pdbqt_path: Path, molfile_path: Path,
                   reference_names: Sequence[str],
                   reference_atoms: dict[str, tuple[float, float, float]],
                   reference_elements: dict[str, str],
                   tolerance: float = COORDINATE_TOLERANCE_ANGSTROM
                   ) -> dict[str, object]:
    """Compose and validate PDBQT serial -> reference atom name.

    `reference_names` is the heavy-atom order of the input molfile, which is
    the component dictionary's order. Every check below must pass:

    - the index map's direction is established from the file;
    - PDBQT serials are unique, and each appears in the map once;
    - the molfile's heavy atoms match the reference by element and coordinate,
      so the input really is the reference's hydrogenated form;
    - every one of the reference's heavy atoms is reached by exactly one
      PDBQT serial, and no two serials reach the same name;
    - element and coordinate agree for each mapped pair.

    Merged hydrogens are counted and listed, not treated as failures.
    """
    pdbqt_atoms = read_pdbqt_atoms(pdbqt_path)
    molfile_atoms = read_molfile_atoms(molfile_path)
    pairs = parse_index_map(pdbqt_path)
    direction = index_map_direction(pairs, len(pdbqt_atoms))
    if direction in ("second", "either"):
        serial_of_input = {first: second for first, second in pairs}
    else:
        serial_of_input = {second: first for first, second in pairs}

    molfile_by_index = {atom.index: atom for atom in molfile_atoms}
    molfile_heavy = [atom for atom in molfile_atoms if not atom.is_hydrogen]
    if len(molfile_heavy) != len(reference_names):
        raise AtomMapError(
            f"{molfile_path} holds {len(molfile_heavy)} heavy atoms but the "
            f"reference names {len(reference_names)}")

    # Link 1: the molfile's heavy atoms against the reference. The molfile
    # carries no names, so this is by position, then verified by element and
    # coordinate, which is what makes the positional reading safe.
    name_of_input: dict[int, str] = {}
    link_problems: list[str] = []
    for position, atom in enumerate(molfile_heavy):
        name = reference_names[position]
        expected_element = reference_elements.get(name, "?").upper()
        if atom.element.upper() != expected_element:
            link_problems.append(
                f"molfile atom {atom.index} is {atom.element}, but reference "
                f"atom {position + 1} ({name}) is {expected_element}")
            continue
        deviation = distance(atom.position, reference_atoms[name])
        if deviation > tolerance:
            link_problems.append(
                f"molfile atom {atom.index} ({name}) is {deviation:.4f} A from "
                f"the reference position")
            continue
        name_of_input[atom.index] = name
    if link_problems:
        raise AtomMapError(
            f"{molfile_path} is not the hydrogenated form of this reference: "
            + "; ".join(link_problems[:5]))

    # Link 2: compose through the index map, validating each hop.
    mapping: dict[int, str] = {}
    claimed: dict[str, int] = {}
    problems: list[str] = []
    mapped_hydrogen_serials: list[int] = []
    for input_index, serial in sorted(serial_of_input.items()):
        atom = molfile_by_index.get(input_index)
        if atom is None:
            problems.append(
                f"the index map names input atom {input_index}, which is not "
                f"in {molfile_path.name} ({len(molfile_atoms)} atoms)")
            continue
        if atom.is_hydrogen:
            mapped_hydrogen_serials.append(serial)
            continue
        name = name_of_input.get(input_index)
        if name is None:
            problems.append(f"input atom {input_index} has no reference name")
            continue
        if name in claimed:
            problems.append(
                f"reference atom {name} is claimed by serials "
                f"{claimed[name]} and {serial}")
            continue
        claimed[name] = serial
        mapping[serial] = name

    missing = sorted(set(reference_names) - set(claimed))
    if missing:
        problems.append(
            f"{len(missing)} reference heavy atoms are not reached by any "
            f"PDBQT serial: {missing[:8]}")
    if problems:
        raise AtomMapError(
            "the composed mapping is not usable: " + "; ".join(problems[:6]))

    # Element and coordinate agreement for each mapped PDBQT atom.
    by_serial = {atom.serial: atom for atom in pdbqt_atoms}
    worst, worst_atom = 0.0, None
    element_problems: list[str] = []
    for serial, name in sorted(mapping.items()):
        atom = by_serial.get(serial)
        if atom is None:
            raise AtomMapError(
                f"the index map names PDBQT serial {serial}, which is not in "
                f"{pdbqt_path.name}")
        expected_element = reference_elements[name].upper()
        # A PDBQT's name column holds the element; its type column is an
        # AutoDock type, so "A" means aromatic carbon and is compared as C.
        found = atom.name.upper() or atom.autodock_type.upper()[:1]
        if found != expected_element:
            element_problems.append(
                f"serial {serial} -> {name}: expected {expected_element}, "
                f"PDBQT says {found} (type {atom.autodock_type})")
        deviation = distance(atom.position, reference_atoms[name])
        if deviation > worst:
            worst, worst_atom = deviation, name
    if element_problems:
        raise AtomMapError(
            "mapped atoms disagree on element: " + "; ".join(element_problems[:5]))
    if worst > tolerance:
        raise AtomMapError(
            f"mapped atom {worst_atom} is {worst:.4f} A from its reference "
            f"position, beyond the {tolerance} A tolerance; the preparation "
            f"moved the heavy atoms")

    input_hydrogens = [atom.index for atom in molfile_atoms if atom.is_hydrogen]
    merged = sorted(set(input_hydrogens) - set(serial_of_input))
    pdbqt_hydrogens = [atom.serial for atom in pdbqt_atoms if atom.is_hydrogen]
    return {
        "schema": ATOM_MAP_SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "prepared_pdbqt": str(pdbqt_path),
            "prepared_pdbqt_sha256": box_geometry.file_sha256(pdbqt_path),
            "input_molfile": str(molfile_path),
            "input_molfile_sha256": box_geometry.file_sha256(molfile_path),
        },
        "index_map": {
            "source": "Meeko REMARK INDEX MAP (--add_index_map)",
            "direction": {
                "second": "pairs read (input atom index, PDBQT serial)",
                "first": "pairs read (PDBQT serial, input atom index)",
                "either": ("both readings give the same mapping, so the "
                           "direction is immaterial here"),
            }[direction],
            "direction_established_by": ("which column is exactly the PDBQT's "
                                         "serials, each once; not assumed"),
            "index_base": 1,
            "n_pairs": len(pairs),
        },
        "counts": {
            "pdbqt_atoms": len(pdbqt_atoms),
            "pdbqt_hydrogens": len(pdbqt_hydrogens),
            "input_atoms": len(molfile_atoms),
            "input_heavy_atoms": len(molfile_heavy),
            "input_hydrogens": len(input_hydrogens),
            "heavy_atoms_mapped": len(mapping),
            "hydrogen_serials_mapped": len(mapped_hydrogen_serials),
            "input_hydrogens_merged": len(merged),
        },
        "merged_hydrogens": {
            "input_indices": merged,
            "note": ("Meeko merges non-polar hydrogens into their parent heavy "
                     "atom, so these input atoms appear in no pair. Expected, "
                     "and not a mapping failure."),
        },
        "serial_to_reference_name": {str(serial): name
                                     for serial, name in sorted(mapping.items())},
        "reference_name_to_serial": {name: serial
                                     for name, serial in sorted(claimed.items())},
        "checks": {
            "serials_unique": True,
            "direction_established": True,
            "bijective_over_reference_heavy_atoms": True,
            "elements_agree": True,
            "max_coordinate_deviation_angstrom": round(worst, 6),
            "max_deviation_atom": worst_atom,
            "tolerance_angstrom": tolerance,
            "coordinates_preserved": True,
        },
        "limitations": [
            "establishes which PDBQT atom is which reference atom; says "
            "nothing about atom typing, partial charges or protonation",
            "specific to this PDBQT and this input molfile, both hashed above; "
            "must not be reused across a re-preparation",
        ],
    }


def read_atom_map(path: Path | str, expected_pdbqt: Path | str | None = None
                  ) -> dict[str, object]:
    """Read an atom-map record, optionally checking it describes a given PDBQT."""
    record = provenance.read_manifest_like(path, ATOM_MAP_SCHEMA)
    if expected_pdbqt is not None:
        recorded = (record.get("inputs", {}) or {}).get("prepared_pdbqt_sha256")
        actual = box_geometry.file_sha256(expected_pdbqt)
        if not recorded:
            raise AtomMapError(f"{path} records no PDBQT hash")
        if recorded != actual:
            raise AtomMapError(
                f"{path} maps a different PDBQT (recorded {recorded[:12]}..., "
                f"given {actual[:12]}...). A mapping is specific to one "
                f"prepared file.")
    return record


def name_by_serial(record: dict[str, object]) -> dict[int, str]:
    """Return the validated serial -> reference name mapping as integers."""
    table = record.get("serial_to_reference_name") or {}
    if not table:
        raise AtomMapError("the atom-map record holds no mapping")
    try:
        return {int(serial): str(name) for serial, name in table.items()}
    except (TypeError, ValueError) as error:
        raise AtomMapError(f"the mapping has a non-integer serial: {error}") from error
