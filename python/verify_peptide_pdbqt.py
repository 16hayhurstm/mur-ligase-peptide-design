"""Verify a Meeko-produced peptide PDBQT against the structure it came from.

Stage 4 of the peptide scoring pilot runs Meeko, which this project cannot
check from the inside: Meeko's documented remit is small organic molecules as
ligands and proteins as *receptors*, and its source carries no peptide or
polymer path on the ligand side. Treating an eight-residue peptide as a Meeko
ligand is therefore off the documented path, and the PDBQT it writes has to be
checked independently.

    python python/verify_peptide_pdbqt.py \
        ligands/spec_0574/spec_0574_bbrigid.pdbqt \
        --source poses/spec_0574/spec_0574_with_oxt.pdb \
        --design-id spec_0574 --expect-backbone-rigid \
        --json-out ligands/spec_0574/spec_0574_pdbqt_check.json

Four questions, and why each one can fail silently
--------------------------------------------------
**Did Meeko move an atom?** Coordinates are compared against the source by
full atom identity, never by file order, and the maximum per-atom deviation is
reported rather than a mean. The whole pilot rests on the ligand carrying
BoltzGen's coordinates; a re-generated conformer would look entirely normal in
a PDBQT.

**Which bonds did Meeko make rotatable?** This is the question `grep TORSDOF`
cannot answer. A PDBQT's torsion tree is written as `BRANCH i j` records
naming the two atom serials of each rotatable bond, so the bonds themselves
can be read out and classified. For a peptide the classification that matters
is backbone against side chain:

- `N`-`CA` within a residue is phi
- `CA`-`C` within a residue is psi
- `C`-`N` between consecutive residues is omega, the amide, which should never
  appear

If `--rigidify_bonds_smarts` fails to match on the installed Meeko build, the
rigidification silently does nothing and the resulting PDBQT looks perfectly
valid while carrying every backbone torsion. `--expect-backbone-rigid` turns
that silent failure into a non-zero exit.

**Did any amide become rotatable?** `--flexible_amides` was not passed, and
Meeko's own help calls rotating them "bad". An omega torsion in the tree means
the molecule was not interpreted as a peptide.

**How were the ionisable groups typed?** The atom-type census is reported, and
the types on named atoms can be inspected, because the arginine guanidinium
and the C-terminal carboxylate are the groups whose typing the recorded
ionisation decision changes, and Vina's scoring function reads those types.

Atom names must survive
-----------------------
Every check here pairs atoms on their PDB names, so Meeko's `--rename_atoms`
must **not** be used: it rewrites each name to the original plus its 1-based
index, which breaks the pairing and makes this verification impossible.
`--add_index_map` provides the index correspondence without touching names.

What this cannot establish
--------------------------
That the partial charges or atom types are *correct*, only what they are. That
the protonation state is right, only that it is the one recorded. Nothing
about binding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import receptor_provenance  # noqa: E402

SCHEMA = "mur-ligase-peptide-pdbqt-check/1"

BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})


class PdbqtVerificationError(ValueError):
    """Raised when a PDBQT cannot be read or verified as specified."""


def atoms_by_serial(path: Path) -> Dict[int, Dict[str, object]]:
    """Read a PDBQT's atoms keyed by serial number.

    The serial is needed because the torsion tree refers to atoms by serial,
    while every other check works on names. PDBQT shares columns 1-54 with
    PDB, and adds the partial charge at 71-76 and the AutoDock atom type at
    78-79.
    """
    if not path.is_file():
        raise PdbqtVerificationError(f"PDBQT not found: {path}")
    atoms: Dict[int, Dict[str, object]] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            serial = int(line[6:11])
            name = line[12:16].strip()
            residue_name = line[17:20].strip()
            residue_seq = int(line[22:26])
        except (ValueError, IndexError) as error:
            raise PdbqtVerificationError(
                f"{path} line {number}: malformed atom record ({error})"
            ) from error
        charge: Optional[float] = None
        if len(line) >= 76:
            try:
                charge = float(line[70:76])
            except ValueError:
                charge = None
        atom_type = line[77:79].strip() if len(line) >= 78 else ""
        if serial in atoms:
            raise PdbqtVerificationError(
                f"{path} line {number}: serial {serial} appears twice, so the "
                "torsion tree cannot be resolved unambiguously."
            )
        atoms[serial] = {
            "serial": serial,
            "atom_name": name,
            "residue_name": residue_name,
            "residue_seq": residue_seq,
            "atom_type": atom_type,
            "partial_charge": charge,
            "label": f"{residue_name}{residue_seq}:{name}",
        }
    if not atoms:
        raise PdbqtVerificationError(f"{path} contains no atom records")
    return atoms


def parse_torsion_tree(path: Path) -> List[Tuple[int, int]]:
    """Return the rotatable bonds a PDBQT declares, as atom-serial pairs.

    Each `BRANCH i j` record names one rotatable bond. This reads the bonds
    themselves rather than the TORSDOF count, because the count says how many
    there are and not which ones, and for a peptide which ones is the whole
    question.
    """
    bonds: List[Tuple[int, int]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith("BRANCH"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise PdbqtVerificationError(
                f"{path} line {number}: malformed BRANCH record {line!r}"
            )
        try:
            bonds.append((int(parts[1]), int(parts[2])))
        except ValueError as error:
            raise PdbqtVerificationError(
                f"{path} line {number}: BRANCH serials are not integers "
                f"({error})"
            ) from error
    return bonds


def torsdof(path: Path) -> Optional[int]:
    """Read the TORSDOF line, or None when the file carries none."""
    for line in path.read_text().splitlines():
        if line.startswith("TORSDOF"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def classify_torsions(
    bonds: Sequence[Tuple[int, int]], atoms: Dict[int, Dict[str, object]]
) -> Dict[str, object]:
    """Classify each rotatable bond as backbone phi/psi, omega, or side chain.

    An unresolvable serial is reported rather than skipped: a torsion whose
    atoms cannot be named cannot be classified, and silently dropping it would
    understate the backbone count.
    """
    phi_psi: List[Dict[str, object]] = []
    omega: List[Dict[str, object]] = []
    side_chain: List[Dict[str, object]] = []
    unresolved: List[str] = []

    for first_serial, second_serial in bonds:
        first = atoms.get(first_serial)
        second = atoms.get(second_serial)
        if first is None or second is None:
            unresolved.append(
                f"BRANCH {first_serial} {second_serial}: serial not in the "
                "atom records"
            )
            continue
        names = {str(first["atom_name"]), str(second["atom_name"])}
        same_residue = first["residue_seq"] == second["residue_seq"]
        entry = {
            "bond": f"{first['label']} - {second['label']}",
            "serials": [first_serial, second_serial],
        }
        if same_residue and names == {"N", "CA"}:
            entry["torsion"] = "phi"
            phi_psi.append(entry)
        elif same_residue and names == {"CA", "C"}:
            entry["torsion"] = "psi"
            phi_psi.append(entry)
        elif not same_residue and names == {"C", "N"}:
            entry["torsion"] = "omega"
            omega.append(entry)
        else:
            side_chain.append(entry)

    return {
        "rotatable_bonds_declared": len(bonds),
        "backbone_phi_psi": phi_psi,
        "backbone_phi_psi_count": len(phi_psi),
        "amide_omega": omega,
        "amide_omega_count": len(omega),
        "side_chain": side_chain,
        "side_chain_count": len(side_chain),
        "unresolved": unresolved,
        "backbone_is_rigid": bool(not phi_psi and not omega and not unresolved),
        "note": (
            "Read from the PDBQT's own BRANCH records. A backbone torsion here "
            "means Vina's local optimisation can reshape the predicted "
            "backbone; an omega torsion means the molecule was not treated as "
            "a peptide."
        ),
    }


def atom_type_census(atoms: Dict[int, Dict[str, object]]) -> Dict[str, int]:
    """Count AutoDock atom types, so the typing is visible rather than assumed."""
    census: Dict[str, int] = {}
    for atom in atoms.values():
        key = str(atom["atom_type"]) or "(blank)"
        census[key] = census.get(key, 0) + 1
    return dict(sorted(census.items()))


def types_for_atom_names(
    atoms: Dict[int, Dict[str, object]], wanted: Sequence[str]
) -> Dict[str, object]:
    """Report the type and charge assigned to each named atom.

    Used for the groups the ionisation decision changes: the arginine
    guanidinium and the C-terminal carboxylate.
    """
    out: Dict[str, object] = {}
    for atom in atoms.values():
        if str(atom["atom_name"]) in wanted:
            out[str(atom["label"])] = {
                "atom_type": atom["atom_type"],
                "partial_charge": atom["partial_charge"],
            }
    return dict(sorted(out.items()))


def compare_against_source(
    pdbqt: Path, source: Path
) -> Dict[str, object]:
    """Confirm the PDBQT's heavy atoms sit where the source structure put them.

    Hydrogens are excluded from the comparison: Meeko merges non-polar
    hydrogens into their heavy atoms by default, so a hydrogen present in the
    source need not appear in the PDBQT, and that is expected rather than a
    discrepancy.
    """
    try:
        source_records = receptor_provenance.parse_atom_records(source)
        pdbqt_records = receptor_provenance.parse_atom_records(pdbqt)
    except receptor_provenance.ReceptorError as error:
        raise PdbqtVerificationError(str(error)) from error

    def heavy(records):
        return {
            key: record
            for key, record in records.items()
            if not key.atom_name.upper().startswith("H")
        }

    source_heavy = heavy(source_records)
    pdbqt_heavy = heavy(pdbqt_records)
    report = receptor_provenance.compare_atom_sets(source_heavy, pdbqt_heavy)
    result = report.as_dict()
    result["source"] = str(source)
    result["source_heavy_atoms"] = len(source_heavy)
    result["pdbqt_heavy_atoms"] = len(pdbqt_heavy)
    result["paired_on"] = (
        "full atom identity (chain, residue number, insertion code, residue "
        "name, atom name, altloc), never file order"
    )
    result["hydrogens"] = (
        "excluded: Meeko merges non-polar hydrogens by default, so their "
        "absence from the PDBQT is expected"
    )
    return result


def verify(
    pdbqt: Path,
    source: Optional[Path],
    design_id: str,
    expect_backbone_rigid: bool = False,
    inspect_atom_names: Sequence[str] = (
        "NE", "CZ", "NH1", "NH2", "OXT", "O",
    ),
) -> Dict[str, object]:
    """Run every PDBQT check and return a single record."""
    atoms = atoms_by_serial(pdbqt)
    bonds = parse_torsion_tree(pdbqt)
    torsions = classify_torsions(bonds, atoms)
    declared = torsdof(pdbqt)

    record: Dict[str, object] = {
        "schema": SCHEMA,
        "design_id": design_id,
        "pdbqt": str(pdbqt),
        "atoms_in_pdbqt": len(atoms),
        "heavy_atoms_in_pdbqt": sum(
            1 for a in atoms.values()
            if not str(a["atom_name"]).upper().startswith("H")
        ),
        "torsdof": {
            "declared": declared,
            "branch_records": len(bonds),
            "agrees_with_branch_count": (
                None if declared is None else declared == len(bonds)
            ),
            "note": (
                "TORSDOF is what Vina reads. A disagreement with the number of "
                "BRANCH records means the file is internally inconsistent."
            ),
        },
        "torsion_tree": torsions,
        "atom_type_census": atom_type_census(atoms),
        "ionisable_group_typing": types_for_atom_names(atoms, inspect_atom_names),
        "expectations": {
            "backbone_rigid_expected": expect_backbone_rigid,
            "rename_atoms_must_not_be_used": (
                "every check here pairs atoms by name; --rename_atoms breaks "
                "that pairing"
            ),
        },
    }

    if source is not None:
        record["coordinate_preservation"] = compare_against_source(pdbqt, source)
    else:
        record["coordinate_preservation"] = {
            "checked": False,
            "reason": (
                "no --source given, so it is not established that Meeko "
                "preserved the predicted coordinates"
            ),
        }

    failures: List[str] = []
    if torsions["amide_omega_count"]:
        failures.append(
            f"{torsions['amide_omega_count']} amide (omega) torsion(s) in the "
            "tree: the molecule was not treated as a peptide"
        )
    if torsions["unresolved"]:
        failures.append(
            f"unresolvable torsion serials: {torsions['unresolved']}"
        )
    if expect_backbone_rigid and torsions["backbone_phi_psi_count"]:
        failures.append(
            f"{torsions['backbone_phi_psi_count']} backbone phi/psi torsion(s) "
            "remain although backbone rigidity was expected: the "
            "--rigidify_bonds_smarts pattern did not match on this Meeko build"
        )
    if record["torsdof"]["agrees_with_branch_count"] is False:
        failures.append(
            f"TORSDOF {declared} disagrees with {len(bonds)} BRANCH records"
        )
    preservation = record["coordinate_preservation"]
    if preservation.get("checked") is not False:
        if not preservation.get("coordinates_preserved"):
            failures.append(
                "heavy-atom coordinates not preserved: maximum deviation "
                f"{preservation.get('max_deviation_angstrom')} A at "
                f"{preservation.get('max_deviation_atom')}"
            )
        if preservation.get("missing_atoms"):
            failures.append(
                f"heavy atoms absent from the PDBQT: "
                f"{preservation['missing_atoms'][:10]}"
            )

    record["verification_failures"] = failures
    record["verified"] = not failures
    record["interpretation"] = (
        "Structural and bookkeeping verification of a PDBQT. It establishes "
        "what Meeko produced, not that the charges or types are correct, and "
        "it is not evidence of binding."
    )
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdbqt", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Structure the ligand was prepared from, for the coordinate "
        "comparison. Omitting it leaves coordinate preservation unestablished.",
    )
    parser.add_argument("--design-id", required=True)
    parser.add_argument(
        "--expect-backbone-rigid",
        action="store_true",
        help="Fail if any backbone phi/psi torsion remains. Use for a PDBQT "
        "prepared with --rigidify_bonds_smarts.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.json_out is not None and args.json_out.exists():
        parser.error(f"Refusing to overwrite an existing file: {args.json_out}")

    try:
        record = verify(
            args.pdbqt,
            args.source,
            args.design_id,
            expect_backbone_rigid=args.expect_backbone_rigid,
        )
    except PdbqtVerificationError as error:
        print(f"PDBQT verification failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(record, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"record: {args.json_out}", file=sys.stderr)
    if record["verification_failures"]:
        print(
            "verification failures: "
            + "; ".join(record["verification_failures"]),
            file=sys.stderr,
        )
    return 0 if record["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
