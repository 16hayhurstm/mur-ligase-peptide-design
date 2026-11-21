"""Build a docking ligand from authoritative chemistry, and check what was made.

Two stages, because they need different things installed:

    # Stage 1, needs only Gemmi: authoritative connectivity, a stated
    # protonation state, and the deposited heavy-atom coordinates.
    python python/prepare_ligand.py build \
        references/uma_chainE.reference.json ligands/uma \
        --component-dictionary project_private/ccd_downloads/UMA.cif \
        --ph 7.4 --deprotonate O2A,O2B,O20 \
        --protonation-justification "..."

    # Stage 2, needs the prepared PDBQT from Meeko: verify it against the
    # reference and record what the preparation actually produced. The
    # evidence goes to its own record; the ligand manifest is not modified.
    python python/prepare_ligand.py check \
        ligands/uma.ligand.json ligands/uma.pdbqt \
        --out ligands/uma.preparation.json

Why a molfile rather than a PDB
-------------------------------
Stage 1 writes a V2000 molfile carrying the dictionary's bond orders and the
chosen formal charges alongside the deposited coordinates. A PDB carries no
bond orders, so any tool reading one has to perceive the chemistry, which is
how the ANP cofactor in the receptor ended up with a chemistry nobody chose.
Writing the connection table explicitly removes that step: the bond orders are
the dictionary's, and the charges are the ones recorded in the manifest.

No hydrogens are added here. Placing hydrogens needs a chemistry toolkit and
is the real-tool step; stage 1 states *which* atoms are to be deprotonated and
what formal charge that implies, so the toolkit has nothing to decide.

The dictionary structure and the modelled state are different things
-------------------------------------------------------------------
The CCD entry for a ligand like UMA is the **neutral** molecule: every
phosphate and carboxylate protonated, formal charge 0. That is the right
reference for connectivity and stereochemistry and the wrong thing to dock at
physiological pH. `--ph`, `--deprotonate` and `--protonation-justification`
are all required, with no defaults, so the modelled state is a recorded
decision rather than an inherited accident.

Limitations
-----------
- Stereochemistry is carried implicitly by the 3D coordinates and the
  connection table. Declared CCD stereo configurations are recorded for
  comparison but are not independently re-derived from the coordinates here.
- Partial charges, atom typing and hydrogen positions come from the
  preparation tool; stage 2 records them and checks the heavy-atom geometry,
  it does not validate the chemistry of the typing.
- Tautomers are not enumerated. The dictionary's tautomer is used.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry
import chemical_component
import ligand_atom_map
import chemical_graph
import receptor_provenance as provenance

LIGAND_SCHEMA = "mur-ligase-ligand/1"
PREPARATION_SCHEMA = "mur-ligase-ligand-preparation/1"

# V2000 bond-order codes for the CCD's value_order strings.
BOND_ORDER_CODES = {"SING": 1, "DOUB": 2, "TRIP": 3, "AROM": 4,
                    chemical_graph.RESONANCE_ORDER: 1}


class LigandError(RuntimeError):
    """Raised when a ligand cannot be built or its preparation not trusted."""


# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------

def molfile_lines(component: chemical_component.ChemicalComponent,
                  coordinates: dict[str, tuple[float, float, float]],
                  charges: dict[str, int], title: str) -> list[str]:
    """Return a V2000 molfile for the heavy atoms, with explicit bond orders.

    Atom order follows the dictionary's heavy-atom order, so the molfile,
    the connection table and the manifest all index atoms the same way.
    Formal charges go in an `M  CHG` block, which takes arbitrary values,
    rather than the legacy per-atom charge column.
    """
    names = list(component.heavy_atom_names)
    missing = [name for name in names if name not in coordinates]
    if missing:
        raise LigandError(f"no coordinates for {missing}")
    index = {name: position for position, name in enumerate(names, start=1)}

    # V2000 header. Line 2 is fixed-width and its columns 21-22 carry the
    # dimensional code: 2 characters of initials, 8 of program name, 10 of
    # timestamp, then "3D". The previous version put free text there, so
    # RDKit read no dimensional code, assumed 2D, and warned that it had found
    # non-zero Z coordinates and was overriding its own assumption. The
    # timestamp is written as zeros deliberately: the molfile's SHA-256 is
    # recorded in the manifest, so the file has to be reproducible from the
    # same inputs.
    lines = [title,
             "  " + "murligpd" + "0" * 10 + "3D",
             "CCD connectivity and modelled formal charges; no hydrogens"]
    bonds = sorted(
        (tuple(sorted((component.graph.names[a], component.graph.names[b]))), order)
        for pair, order in component.graph.bonds.items()
        for a, b in (tuple(pair),)
    )
    lines.append(f"{len(names):>3}{len(bonds):>3}  0  0  0  0  0  0  0  0999 V2000")
    for name in names:
        x, y, z = coordinates[name]
        element = component.elements[name].capitalize()
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {element:<3} 0  0  0  0  0"
                     f"  0  0  0  0  0  0")
    for (first, second), order in bonds:
        code = BOND_ORDER_CODES.get(order)
        if code is None:
            raise LigandError(f"unmapped bond order {order!r} between "
                              f"{first} and {second}")
        lines.append(f"{index[first]:>3}{index[second]:>3}{code:>3}  0  0  0  0")
    charged = {name: value for name, value in charges.items() if value}
    if charged:
        items = sorted(charged.items(), key=lambda item: index[item[0]])
        for start in range(0, len(items), 8):
            chunk = items[start:start + 8]
            entries = "".join(f"{index[name]:>4}{value:>4}" for name, value in chunk)
            lines.append(f"M  CHG{len(chunk):>3}{entries}")
    lines.append("M  END")
    return lines


def build(args: argparse.Namespace) -> None:
    """Stage 1: write the molfile and the ligand manifest."""
    reference = provenance.read_manifest_like(args.reference_record,
                                              expected_schema=None)
    if reference.get("schema") != "mur-ligase-ligand-reference/1":
        raise LigandError(
            f"{args.reference_record} is not a ligand reference record "
            f"(schema {reference.get('schema')!r})"
        )
    component = chemical_component.read_component(args.component_dictionary)
    expected = reference.get("component")
    if expected and expected != component.identifier:
        raise LigandError(
            f"the reference is for {expected} but the dictionary is for "
            f"{component.identifier}"
        )

    mapping = reference["atom_mapping"]
    coordinates = {name: tuple(entry["coordinates_angstrom"])
                   for name, entry in mapping.items()}
    observed_elements = {name: entry["element"] for name, entry in mapping.items()}
    identity = chemical_component.compare_with_coordinates(
        component, observed_elements)
    if not identity["identical_atom_set"]:
        raise LigandError(
            f"the reference atoms do not match {component.identifier}: "
            f"missing {identity['missing_from_observed']}, unexpected "
            f"{identity['not_in_dictionary']}, element mismatches "
            f"{identity['element_mismatches']}"
        )

    deprotonate = [atom.strip() for atom in args.deprotonate.split(",")
                   if atom.strip()] if args.deprotonate else []
    state = chemical_component.validated_protonation(
        component, deprotonate, args.ph, args.protonation_justification)

    molfile_path = args.out_prefix.with_name(args.out_prefix.name + "_ligand.mol")
    manifest_path = args.out_prefix.with_name(args.out_prefix.name + ".ligand.json")
    for path in (molfile_path, manifest_path):
        if path.exists():
            raise LigandError(f"{path} already exists; choose a fresh prefix.")

    charges = {atom: -1 for atom in state.deprotonated}
    lines = molfile_lines(component, coordinates, charges,
                          f"{component.identifier} from "
                          f"{reference['source_structure']['path']}")
    molfile_path.parent.mkdir(parents=True, exist_ok=True)
    molfile_path.write_text("\n".join(lines) + "\n")

    retained_polar = sorted(
        atom for atom in component.hydrogens_on
        if atom not in state.deprotonated
        and component.elements[atom].upper() in ("N", "O", "S")
    )
    # Resonance equivalence is decided at the modelled state: a group is only
    # equivalent once every one of its terminal oxygens has lost its hydrogen.
    still_protonated = sorted(set(component.hydrogens_on) - set(state.deprotonated))
    normalised, resonance = chemical_graph.normalise_resonance(
        component.graph, hydrogen_bearing=still_protonated)
    permutations = chemical_graph.automorphisms(normalised)

    manifest = {
        "schema": LIGAND_SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": "built; not yet prepared by a chemistry toolkit",
        "component": component.identifier,
        "dictionary": {
            "path": str(args.component_dictionary),
            "sha256": box_geometry.file_sha256(args.component_dictionary),
            "name": component.name,
            "formula": component.formula,
            "formal_charge": component.dictionary_formal_charge,
            "n_heavy_atoms": len(component.heavy_atom_names),
            "n_hydrogens_neutral_form": component.n_hydrogens,
            "n_heavy_bonds": len(component.graph.bonds),
            "declared_stereocentres": component.stereocentres,
            "aromatic_atoms": list(component.aromatic_atoms),
            "note": ("the dictionary entry is the neutral component; it is the "
                     "authority for connectivity and stereochemistry, not for "
                     "the modelled protonation state"),
        },
        "reference": {
            "path": str(args.reference_record),
            "sha256": box_geometry.file_sha256(args.reference_record),
            "reference_pdb": reference["outputs"]["reference_pdb"],
            "source_structure": reference["source_structure"],
            "selection": reference["selection"],
        },
        "chemical_identity": identity,
        "protonation_state": state.as_dict(),
        "ionisable_sites_in_neutral_form": [
            {"atom": site.atom, "group": site.group,
             "attached_to": site.attached_to,
             "hydrogens": list(site.hydrogens),
             "deprotonated_here": site.atom in state.deprotonated}
            for site in component.ionisable_sites()
        ],
        "retained_polar_hydrogen_sites": retained_polar,
        "symmetry": {
            "resonance_normalisation": resonance,
            "n_automorphisms": len(permutations),
            "orbits": chemical_graph.symmetry_classes(normalised, permutations),
            "note": ("used only for symmetry-aware RMSD; it does not change "
                     "the molecule"),
        },
        "outputs": {
            "molfile": str(molfile_path),
            "molfile_sha256": box_geometry.file_sha256(molfile_path),
            "n_atoms_written": len(component.heavy_atom_names),
            "hydrogens_written": 0,
        },
        # This manifest never records the preparation. The check stage writes a
        # separate `mur-ligase-ligand-preparation/1` record, so that this
        # file's SHA-256 stays the ligand's identity for the verification, the
        # atom map and the analysis.
        "preparation": {"performed": False,
                        "recorded_separately": (
                            "prepare_ligand.py check --out writes a "
                            "mur-ligase-ligand-preparation/1 record")},
        "unresolved": [
            "hydrogen positions, partial charges and atom types are not set "
            "here; they come from the preparation tool and are recorded by the "
            "check stage",
            "stereochemistry is carried by the 3D coordinates and the "
            "connection table; CCD stereo labels are recorded but not "
            "re-derived from the coordinates",
            "tautomers are not enumerated; the dictionary's tautomer is used",
        ],
    }
    provenance.write_manifest(manifest_path, manifest)

    print(f"component        : {component.identifier}  {component.name}")
    print(f"dictionary       : {args.component_dictionary}")
    print(f"  formula        : {component.formula}, neutral charge "
          f"{component.dictionary_formal_charge}, "
          f"{component.n_hydrogens} hydrogens in the neutral form")
    print(f"  heavy atoms    : {len(component.heavy_atom_names)}, "
          f"{len(component.graph.bonds)} heavy bonds, "
          f"{len(component.stereocentres)} declared stereocentres")
    print(f"protonation      : pH {args.ph}, deprotonating "
          f"{list(state.deprotonated)}")
    print(f"  formal charge  : {state.formal_charge} "
          f"(dictionary {component.dictionary_formal_charge} "
          f"- {len(state.deprotonated)} acidic protons)")
    print(f"  retained polar : {retained_polar}")
    print(f"symmetry         : {len(permutations)} automorphisms, orbits "
          f"{chemical_graph.symmetry_classes(normalised, permutations)}")
    print(f"wrote            : {molfile_path}")
    print(f"manifest         : {manifest_path}")
    print("\nNext, on a machine with a chemistry toolkit: add hydrogens at this "
          "state,\nconvert to PDBQT, then run the check stage. See "
          "docs/uma_control.md.")


# ---------------------------------------------------------------------------
# Stage 2: check a prepared PDBQT
# ---------------------------------------------------------------------------

def parse_pdbqt_ligand(path: Path) -> dict[str, object]:
    """Read a prepared ligand PDBQT: atoms, types, charges, torsions.

    `TORSDOF` and the `BRANCH` records are read as written by the preparation
    tool. The rotatable-bond count is the number of `BRANCH` records, which is
    the tree the docking search will actually sample; `TORSDOF` is the tool's
    own count of torsional degrees of freedom and the two need not agree.
    """
    atoms: dict[str, dict[str, object]] = {}
    branches: list[tuple[int, int]] = []
    torsdof: int | None = None
    duplicated: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            entry = {
                "element_or_type": line[77:80].strip(),
                "charge": float(line[70:76]) if line[70:76].strip() else 0.0,
                "coordinates": (float(line[30:38]), float(line[38:46]),
                                float(line[46:54])),
                "residue_name": line[17:20].strip(),
            }
            if name in atoms:
                duplicated.append(name)
            atoms[name] = entry
        elif line.startswith("BRANCH"):
            fields = line.split()
            if len(fields) >= 3:
                branches.append((int(fields[1]), int(fields[2])))
        elif line.startswith("TORSDOF"):
            torsdof = int(line.split()[1])
    if not atoms:
        raise LigandError(f"{path} contains no atom records")
    if duplicated:
        raise LigandError(
            f"{path} repeats atom names {sorted(set(duplicated))}; the "
            f"correspondence with the reference would be ambiguous"
        )
    return {"atoms": atoms, "n_branch_records": len(branches),
            "torsdof": torsdof}


def read_pdbqt_torsions(path: Path) -> dict[str, object]:
    """Read TORSDOF and the BRANCH count, independently of atom naming.

    Needed on both correspondence paths: the torsion tree is a property of the
    file, not of how its atoms happen to be keyed.
    """
    branches = 0
    torsdof: int | None = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("BRANCH"):
            branches += 1
        elif line.startswith("TORSDOF"):
            fields = line.split()
            if len(fields) >= 2:
                torsdof = int(fields[1])
    return {"torsdof": torsdof, "n_branch_records": branches}


def _atoms_by_atom_map(atom_map_path: Path, pdbqt_path: Path):
    """Re-key a prepared PDBQT's atoms by reference name, via the atom map.

    Returns the same shape `parse_pdbqt_ligand` produces, so the checks that
    follow are identical whichever route supplied the names. Hydrogens keep
    their serials as keys, prefixed, because they have no reference name: they
    are counted and recorded, never compared.
    """
    record = ligand_atom_map.read_atom_map(atom_map_path,
                                           expected_pdbqt=pdbqt_path)
    names = ligand_atom_map.name_by_serial(record)
    atoms: dict[str, dict[str, object]] = {}
    hydrogens: set[str] = set()
    for atom in ligand_atom_map.read_pdbqt_atoms(pdbqt_path):
        key = names.get(atom.serial)
        if key is None:
            if not atom.is_hydrogen:
                raise LigandError(
                    f"PDBQT serial {atom.serial} ({atom.name}) is a heavy atom "
                    f"with no entry in {atom_map_path}; the mapping does not "
                    f"cover this file")
            key = f"H#{atom.serial}"
            hydrogens.add(key)
        atoms[key] = {
            "element_or_type": atom.autodock_type,
            "charge": atom.charge,
            "coordinates": atom.position,
            "residue_name": "",
            "pdbqt_serial": atom.serial,
        }
    return atoms, hydrogens, {
        "basis": "validated Meeko index map, serial to reference name",
        "atom_map_record": str(atom_map_path),
        "atom_map_record_sha256": box_geometry.file_sha256(atom_map_path),
        "n_heavy_atoms_mapped": len(names),
        "index_map_direction": (record.get("index_map", {}) or {}).get("direction"),
        "input_hydrogens_merged": (record.get("counts", {}) or {}).get(
            "input_hydrogens_merged"),
        "note": ("the PDBQT's atom-name column holds the element, so names "
                 "there are not unique and are not used for correspondence"),
    }


def check(args: argparse.Namespace) -> None:
    """Stage 2: verify the prepared PDBQT and record what it contains.

    Writes a `mur-ligase-ligand-preparation/1` record and leaves the ligand
    manifest untouched. An earlier version wrote the evidence back into the
    manifest, which changed the manifest's SHA-256 after the stereochemistry
    verification and the atom map had already recorded that digest as the
    ligand's identity; both gates then rejected the manifest. The record is
    tied to the prepared PDBQT and the manifest by digest instead.
    """
    manifest = provenance.read_manifest_like(args.ligand_manifest,
                                             expected_schema=LIGAND_SCHEMA)
    reference_atoms = {
        name: tuple(entry["coordinates_angstrom"]) for name, entry
        in provenance.read_manifest_like(
            Path(manifest["reference"]["path"]),
            expected_schema=None)["atom_mapping"].items()
    }
    expected_heavy = set(reference_atoms)
    tolerance = args.coordinate_tolerance

    torsions = read_pdbqt_torsions(args.prepared_pdbqt)

    if args.atom_map:
        # Serial-keyed, via the validated mapping. Meeko writes the element in
        # the atom-name column, so a prepared ligand has 49 heavy atoms
        # sharing five names and no name-keyed comparison is possible.
        atoms, hydrogens, correspondence = _atoms_by_atom_map(
            args.atom_map, args.prepared_pdbqt)
    else:
        # Name-keyed, and deliberately strict: a repeated name is refused
        # rather than collapsed. This path is right for a ligand whose names
        # survived preparation, and refuses one whose names did not.
        prepared = parse_pdbqt_ligand(args.prepared_pdbqt)
        atoms = prepared["atoms"]
        hydrogens = {name for name, entry in atoms.items()
                     if str(entry["element_or_type"]).upper() in ("HD", "H")
                     or name.upper().startswith("H")}
        correspondence = {
            "basis": "atom names in the prepared PDBQT",
            "atom_map_record": None,
            "note": ("no --atom-map was supplied, so names were required to "
                     "match the reference exactly"),
        }

    heavy_present = set(atoms) - hydrogens
    missing = sorted(expected_heavy - heavy_present)
    unexpected = sorted(heavy_present - expected_heavy)
    if missing or unexpected:
        raise LigandError(
            f"the prepared ligand's heavy atoms do not correspond to the "
            f"reference: missing {missing}, unexpected {unexpected}. The "
            f"comparison is not made over a subset."
        )

    worst, worst_atom = 0.0, None
    for name in sorted(expected_heavy):
        deviation = provenance.distance(atoms[name]["coordinates"],
                                        reference_atoms[name])
        if deviation > worst:
            worst, worst_atom = deviation, name
    preserved = worst <= tolerance

    prepared_pdbqt_sha256 = box_geometry.file_sha256(args.prepared_pdbqt)
    charge_sum = sum(float(entry["charge"]) for entry in atoms.values())
    expected_charge = manifest["protonation_state"]["modelled_formal_charge"]
    types: dict[str, int] = {}
    for entry in atoms.values():
        key = str(entry["element_or_type"])
        types[key] = types.get(key, 0) + 1

    # The preparation evidence is written to its own record rather than back
    # into the ligand manifest. Rewriting the manifest changed its SHA-256,
    # which is the identity the stereochemistry verification and the atom map
    # record; every downstream gate keyed to that digest then failed for a
    # bookkeeping reason rather than a scientific one.
    preparation = {
        "performed": True,
        "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prepared_pdbqt": str(args.prepared_pdbqt),
        "prepared_pdbqt_sha256": prepared_pdbqt_sha256,
        "n_atoms_total": len(atoms),
        "n_heavy_atoms": len(heavy_present),
        "n_hydrogens": len(hydrogens),
        "hydrogen_atom_names": sorted(hydrogens),
        "autodock_type_census": dict(sorted(types.items())),
        "partial_charge_sum": round(charge_sum, 4),
        "expected_formal_charge": expected_charge,
        "charge_sum_matches_formal_charge": abs(charge_sum - expected_charge) < 0.05,
        "torsdof": torsions["torsdof"],
        "n_branch_records": torsions["n_branch_records"],
        "heavy_atom_coordinates": {
            "max_deviation_from_reference_angstrom": round(worst, 6),
            "max_deviation_atom": worst_atom,
            "tolerance_angstrom": tolerance,
            "preserved": preserved,
        },
        "partial_charge_model": args.charge_model,
        "preparation_command": args.preparation_command,
        "atom_correspondence": correspondence,
        "not_validated_here": [
            "atom typing correctness",
            "partial-charge model appropriateness",
            "hydrogen orientations, which are placed by the tool and are "
            "rotatable for every hydroxyl",
            "stereochemistry re-derivation from the prepared coordinates",
        ],
    }

    # The record is tied to the exact prepared PDBQT it describes and to the
    # manifest it was checked against, both by digest. The manifest itself is
    # not touched, so the digest recorded here is the one verification saw.
    # A failed check still writes its record: the diagnostic detail is the
    # evidence of what the preparation produced. `result` is what downstream
    # readers gate on, so a failed record cannot be mistaken for a passed one.
    failures = []
    if not preserved:
        failures.append(
            f"heavy atoms deviate from the reference by {worst:.4f} A at "
            f"{worst_atom}, beyond the {tolerance} A tolerance")
    if not preparation["charge_sum_matches_formal_charge"]:
        failures.append(
            f"partial charges sum to {charge_sum:+.4f} against a recorded "
            f"formal charge of {expected_charge:+d}")

    record = {
        "schema": PREPARATION_SCHEMA,
        "result": "failed" if failures else "passed",
        "failure_reasons": failures,
        "component": manifest["component"],
        "inputs": {
            "ligand_manifest": str(args.ligand_manifest),
            "ligand_manifest_sha256": box_geometry.file_sha256(
                args.ligand_manifest),
            "prepared_pdbqt": str(args.prepared_pdbqt),
            "prepared_pdbqt_sha256": prepared_pdbqt_sha256,
            "reference_record": manifest["reference"]["path"],
            "atom_map_record": correspondence.get("atom_map_record"),
            "atom_map_record_sha256": correspondence.get(
                "atom_map_record_sha256"),
        },
        "protonation_state": manifest["protonation_state"],
        "preparation": preparation,
        "note": ("the ligand manifest is deliberately left unmodified: its "
                 "SHA-256 is the identity the stereochemistry verification "
                 "and the atom map were recorded against"),
    }
    destination = Path(args.out)
    provenance.write_manifest(destination, record, overwrite=args.overwrite)

    print(f"prepared ligand  : {args.prepared_pdbqt}")
    print(f"  atoms          : {len(atoms)} total, {len(heavy_present)} heavy, "
          f"{len(hydrogens)} hydrogens")
    print(f"  type census    : {dict(sorted(types.items()))}")
    print(f"  charge sum     : {charge_sum:+.4f} against an expected formal "
          f"charge of {expected_charge:+d}")
    print(f"  TORSDOF        : {torsions['torsdof']}, "
          f"{torsions['n_branch_records']} BRANCH records")
    print(f"  max deviation  : {worst:.4f} A from the reference "
          f"(tolerance {tolerance} A) at {worst_atom}")
    print(f"wrote            : {destination}")
    print(f"manifest          : {args.ligand_manifest} left unmodified")

    if not preserved:
        raise LigandError(
            f"the prepared ligand's heavy atoms deviate from the reference by "
            f"{worst:.4f} A at {worst_atom}, beyond the {tolerance} A "
            f"tolerance. Preparation must not move the heavy atoms; a moved "
            f"ligand cannot be compared with the reference pose."
        )
    if not preparation["charge_sum_matches_formal_charge"]:
        raise LigandError(
            f"the partial charges sum to {charge_sum:+.4f} but the recorded "
            f"protonation state implies {expected_charge:+d}. The preparation "
            f"did not apply the protonation state that was decided."
        )


# ---------------------------------------------------------------------------
# Stereochemistry verification plan
# ---------------------------------------------------------------------------

def stereocentre_status(component: chemical_component.ChemicalComponent,
                        deprotonated: Sequence[str]) -> dict[str, object]:
    """Classify every declared stereocentre at one modelled protonation state.

    A declared centre is **not** a stereocentre at a given state when two of
    its substituents are equivalent, which is decided from the symmetry orbits
    computed at that state rather than asserted. This is what distinguishes the
    carbon centres, whose four substituents stay distinct, from the phosphorus
    labels, whose two terminal oxygens become equivalent once deprotonated.

    Returns the centres that must match the dictionary, those expected not to,
    and the orbit evidence for each decision.
    """
    still_protonated = sorted(set(component.hydrogens_on) - set(deprotonated))
    graph, resonance = chemical_graph.normalise_resonance(
        component.graph, hydrogen_bearing=still_protonated)
    permutations = chemical_graph.automorphisms(graph)
    orbit_of: dict[str, int] = {}
    for number, orbit in enumerate(
            chemical_graph.symmetry_classes(graph, permutations)):
        for name in orbit:
            orbit_of[name] = number

    adjacency = graph.adjacency
    index = graph.index
    must_match: list[dict[str, object]] = []
    not_stereogenic: list[dict[str, object]] = []
    for atom in sorted(component.stereocentres):
        if atom not in index:
            continue
        neighbours = [graph.names[n] for n, _ in adjacency[index[atom]]]
        shared = [
            sorted((first, second))
            for position, first in enumerate(neighbours)
            for second in neighbours[position + 1:]
            if first in orbit_of and orbit_of[first] == orbit_of.get(second)
        ]
        entry = {
            "atom": atom,
            "element": component.elements[atom],
            "declared_configuration": component.stereocentres[atom],
            "substituents": sorted(neighbours),
        }
        if shared:
            entry["equivalent_substituents"] = shared[0]
            entry["treatment"] = (
                f"NOT a stereocentre at this state: substituents "
                f"{shared[0]} are equivalent, so the declared "
                f"{component.stereocentres[atom]!r} has no referent. A CIP "
                f"mismatch here is expected and is not a failure.")
            not_stereogenic.append(entry)
        elif component.elements[atom].upper() in ("P", "S"):
            # A phosphate monoester's P formally has four distinct
            # substituents while it keeps its hydroxyl, so the connectivity
            # rule classifies it here. It is still not configurationally
            # stable in solution: the acidic proton exchanges, and the
            # configuration has no fixed meaning at a pH where the group is
            # partly ionised. Flagged for investigation, not as fatal.
            entry["treatment"] = (
                f"formally stereogenic at this state, because its hydroxyl is "
                f"retained and all four substituents are distinct. But "
                f"phosphorus configuration of this kind is not "
                f"configurationally stable in solution, so a CIP mismatch "
                f"here should be INVESTIGATED, not treated as the wrong "
                f"stereoisomer. Compare with the state where this group is "
                f"deprotonated.")
            entry["fatal_if_mismatched"] = False
            must_match.append(entry)
        else:
            entry["treatment"] = (
                f"MUST match the dictionary's {component.stereocentres[atom]!r}; "
                f"all four substituents remain distinct. A mismatch means the "
                f"wrong stereoisomer and blocks use.")
            entry["fatal_if_mismatched"] = True
            must_match.append(entry)
    return {
        "deprotonated_atoms": list(deprotonated),
        "formal_charge": component.dictionary_formal_charge - len(deprotonated),
        "resonance_normalisation": resonance,
        "n_automorphisms": len(permutations),
        "orbits": chemical_graph.symmetry_classes(graph, permutations),
        "n_declared_stereocentres": len(component.stereocentres),
        "must_match": must_match,
        "not_stereogenic_at_this_state": not_stereogenic,
        "n_must_match": len(must_match),
        "n_not_stereogenic": len(not_stereogenic),
    }


def parse_state_argument(text: str) -> tuple[str, list[str]]:
    """Parse a `label=ATOM,ATOM` modelled-state specification."""
    if "=" not in text:
        raise LigandError(
            f"--state takes LABEL=ATOM,ATOM (an empty atom list means the "
            f"neutral form), got {text!r}")
    label, _, atoms = text.partition("=")
    if not label.strip():
        raise LigandError(f"--state needs a label, got {text!r}")
    return label.strip(), [atom.strip() for atom in atoms.split(",") if atom.strip()]


def stereo_plan(args: argparse.Namespace) -> None:
    """Write the stereochemistry verification plan for each modelled state.

    The plan is what the RDKit check on the preparation machine compares
    against, so the expectation is fixed here, from the dictionary, before any
    result is seen. Every declared centre appears in exactly one of the two
    lists for every state.
    """
    manifest = provenance.read_manifest_like(args.ligand_manifest, LIGAND_SCHEMA)
    dictionary_path = Path(args.component_dictionary
                           or manifest["dictionary"]["path"])
    component = chemical_component.read_component(dictionary_path)
    if component.identifier != manifest["component"]:
        raise LigandError(
            f"the dictionary is for {component.identifier} but the manifest is "
            f"for {manifest['component']}")
    recorded_hash = manifest["dictionary"].get("sha256")
    actual_hash = box_geometry.file_sha256(dictionary_path)
    if recorded_hash and recorded_hash != actual_hash:
        raise LigandError(
            f"{dictionary_path} is not the dictionary entry the ligand was "
            f"built from: recorded {recorded_hash[:12]}..., found "
            f"{actual_hash[:12]}.... CCD entries are revised; use the pinned "
            f"copy or rebuild the ligand."
        )

    states: list[tuple[str, list[str]]] = [
        ("modelled", list(manifest["protonation_state"]["deprotonated_atoms"]))
    ]
    for specification in args.state or []:
        states.append(parse_state_argument(specification))

    plan = {
        "schema": "mur-ligase-stereo-plan/1",
        "component": component.identifier,
        "dictionary": {"path": str(dictionary_path), "sha256": actual_hash,
                       "modified_date_note": (
                           "CCD entries are revised; this hash pins the "
                           "revision the expectation was derived from")},
        "ligand_manifest": {"path": str(args.ligand_manifest),
                            "sha256": box_geometry.file_sha256(args.ligand_manifest)},
        "method": ("a declared centre is treated as not stereogenic at a state "
                   "when two of its substituents fall in one symmetry orbit at "
                   "that state; the orbits are computed, not asserted"),
        "verification": ("assign CIP codes from the prepared 3D coordinates "
                         "with RDKit AssignStereochemistryFrom3D and compare "
                         "with declared_configuration for every entry under "
                         "must_match"),
        "states": {label: stereocentre_status(component, atoms)
                   for label, atoms in states},
    }
    destination = Path(args.out)
    provenance.write_manifest(destination, plan, overwrite=args.overwrite)

    print(f"component        : {component.identifier}")
    print(f"dictionary       : {dictionary_path} ({actual_hash[:12]}...)")
    for label, status in plan["states"].items():
        print(f"\nstate {label!r}: charge {status['formal_charge']}, "
              f"deprotonated {status['deprotonated_atoms'] or 'none'}")
        print(f"  declared stereocentres : {status['n_declared_stereocentres']}")
        print(f"  must match the dictionary ({status['n_must_match']}): "
              f"{[entry['atom'] for entry in status['must_match']]}")
        print(f"  not stereogenic here ({status['n_not_stereogenic']}): "
              f"{[entry['atom'] for entry in status['not_stereogenic_at_this_state']]}")
        for entry in status["not_stereogenic_at_this_state"]:
            print(f"    {entry['atom']} ({entry['element']}): equivalent "
                  f"substituents {entry['equivalent_substituents']}")
    print(f"\nwrote            : {destination}")
    print("\nVerify every must_match centre against the prepared coordinates "
          "before\ninterpreting any docking result. See docs/uma_control.md.")


def atom_map(args: argparse.Namespace) -> None:
    """Build and record the validated PDBQT-serial to reference-name mapping.

    Meeko writes the element in a PDBQT's atom-name column, so a prepared
    ligand has 49 heavy atoms sharing five names and no name-keyed
    correspondence is possible against it. This composes Meeko's own
    `REMARK INDEX MAP` with the input molfile's heavy-atom order, validating
    every link, and writes the result for the check and analysis stages.
    """
    manifest = provenance.read_manifest_like(args.ligand_manifest, LIGAND_SCHEMA)
    verification = provenance.read_manifest_like(
        args.verification_record, "mur-ligase-ligand-verification/1")
    if verification.get("result") != "passed":
        raise LigandError(
            f"{args.verification_record} records result "
            f"{verification.get('result')!r}; a mapping must not be built on "
            f"an unverified ligand")
    recorded_manifest = (verification.get("inputs", {}) or {}).get(
        "ligand_manifest_sha256")
    actual_manifest = box_geometry.file_sha256(args.ligand_manifest)
    agreement = provenance.manifest_hash_agreement(args.ligand_manifest,
                                                   recorded_manifest)
    if recorded_manifest and not agreement["agrees"]:
        raise LigandError(
            f"{args.verification_record} verified a different ligand manifest "
            f"(recorded {recorded_manifest[:12]}..., given "
            f"{actual_manifest[:12]}...): {agreement['reason']}")
    if agreement["legacy_compatibility_used"]:
        print(f"note: {args.ligand_manifest} was modified in place by an "
              f"earlier check stage; {agreement['basis']}")

    # The input Meeko was given is the hydrogenated molfile the verification
    # wrote. Its hash is checked, and its heavy atoms are then validated
    # against the reference directly rather than taken on trust.
    molfile = Path(args.input_molfile or
                   (verification.get("outputs", {}) or {}).get(
                       "hydrogenated_molfile", ""))
    if not molfile or not molfile.is_file():
        raise LigandError(
            f"the hydrogenated molfile is missing: {molfile!s}. It is the file "
            f"Meeko was given and the mapping cannot be built without it.")
    recorded_molfile = (verification.get("outputs", {}) or {}).get(
        "hydrogenated_molfile_sha256")
    actual_molfile = box_geometry.file_sha256(molfile)
    if recorded_molfile and recorded_molfile != actual_molfile:
        raise LigandError(
            f"{molfile} has changed since it was verified (recorded "
            f"{recorded_molfile[:12]}..., found {actual_molfile[:12]}...)")

    reference = provenance.read_manifest_like(
        Path(manifest["reference"]["path"]), None)
    reference_atoms = {name: tuple(entry["coordinates_angstrom"])
                       for name, entry in reference["atom_mapping"].items()}
    reference_elements = {name: entry["element"]
                          for name, entry in reference["atom_mapping"].items()}
    component = chemical_component.read_component(
        args.component_dictionary or manifest["dictionary"]["path"])

    record = ligand_atom_map.build_atom_map(
        args.prepared_pdbqt, molfile, component.heavy_atom_names,
        reference_atoms, reference_elements,
        tolerance=args.coordinate_tolerance)
    record["provenance"] = {
        "ligand_manifest": str(args.ligand_manifest),
        "ligand_manifest_sha256": actual_manifest,
        "ligand_manifest_identity": agreement,
        "verification_record": str(args.verification_record),
        "verification_record_sha256": box_geometry.file_sha256(
            args.verification_record),
        "verification_result": verification.get("result"),
        "component_dictionary": str(
            args.component_dictionary or manifest["dictionary"]["path"]),
        "reference_record": manifest["reference"]["path"],
        "modelled_formal_charge":
            manifest["protonation_state"]["modelled_formal_charge"],
        "preparation_command": args.preparation_command,
    }
    destination = Path(args.out)
    provenance.write_manifest(destination, record, overwrite=args.overwrite)

    counts = record["counts"]
    print(f"prepared PDBQT   : {args.prepared_pdbqt}")
    print(f"input molfile    : {molfile}")
    print(f"index map        : {record['index_map']['n_pairs']} pairs, "
          f"{record['index_map']['direction']}")
    print(f"  base           : {record['index_map']['index_base']}, "
          f"direction established from the file, not assumed")
    print(f"atoms            : PDBQT {counts['pdbqt_atoms']} "
          f"({counts['pdbqt_hydrogens']} hydrogens), input "
          f"{counts['input_atoms']} ({counts['input_heavy_atoms']} heavy, "
          f"{counts['input_hydrogens']} hydrogens)")
    print(f"heavy atoms mapped: {counts['heavy_atoms_mapped']} of "
          f"{len(component.heavy_atom_names)}, bijective")
    print(f"hydrogens merged : {counts['input_hydrogens_merged']} input "
          f"hydrogens appear in no pair, as expected")
    print(f"max deviation    : "
          f"{record['checks']['max_coordinate_deviation_angstrom']} A "
          f"(tolerance {record['checks']['tolerance_angstrom']} A)")
    print(f"wrote            : {destination}")
    print("\nPass this record to prepare_ligand.py check --atom-map and to "
          "analyse_redocking.py\n--atom-map; neither can use the PDBQT's "
          "repeated element names.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    builder = subparsers.add_parser("build", help="Write the molfile and manifest")
    builder.add_argument("reference_record", type=Path)
    builder.add_argument("out_prefix", type=Path)
    builder.add_argument("--component-dictionary", type=Path, required=True)
    builder.add_argument("--ph", type=float, required=True,
                         help="pH the protonation state represents")
    builder.add_argument("--deprotonate", required=True,
                         help="Comma-separated oxygen acids to deprotonate; "
                              "pass an empty string only to model the neutral form")
    builder.add_argument("--protonation-justification", required=True,
                         help="Why this state, in words; recorded verbatim")

    checker = subparsers.add_parser("check", help="Verify a prepared PDBQT")
    checker.add_argument("ligand_manifest", type=Path)
    checker.add_argument("prepared_pdbqt", type=Path)
    checker.add_argument("--out", type=Path, required=True,
                         help="Preparation record to write. The ligand "
                              "manifest is left unmodified, because its "
                              "SHA-256 is the identity the verification and "
                              "the atom map were recorded against")
    checker.add_argument("--overwrite", action="store_true")
    checker.add_argument("--coordinate-tolerance", type=float,
                         default=provenance.COORDINATE_TOLERANCE_ANGSTROM)
    checker.add_argument("--charge-model", default="unrecorded",
                         help="Charge model the preparation used, for the record")
    checker.add_argument("--preparation-command", default="unrecorded",
                         help="Exact command used, for the record")
    checker.add_argument("--atom-map", type=Path,
                         help="Validated atom-map record from the atom-map "
                              "stage. Required when preparation did not "
                              "preserve atom names, which Meeko does not")

    mapper = subparsers.add_parser(
        "atom-map",
        help="Build the validated PDBQT-serial to reference-name mapping")
    mapper.add_argument("ligand_manifest", type=Path)
    mapper.add_argument("verification_record", type=Path)
    mapper.add_argument("prepared_pdbqt", type=Path)
    mapper.add_argument("--out", type=Path, required=True)
    mapper.add_argument("--input-molfile", type=Path,
                        help="Defaults to the hydrogenated molfile the "
                             "verification record names")
    mapper.add_argument("--component-dictionary", type=Path)
    mapper.add_argument("--coordinate-tolerance", type=float,
                        default=ligand_atom_map.COORDINATE_TOLERANCE_ANGSTROM)
    mapper.add_argument("--preparation-command", default="unrecorded")
    mapper.add_argument("--overwrite", action="store_true")

    planner = subparsers.add_parser(
        "stereo-plan",
        help="Write the stereochemistry verification plan for each state")
    planner.add_argument("ligand_manifest", type=Path)
    planner.add_argument("--out", type=Path, required=True)
    planner.add_argument("--component-dictionary", type=Path,
                         help="Defaults to the path recorded in the manifest")
    planner.add_argument("--state", action="append", metavar="LABEL=ATOMS",
                         help="An additional modelled state to plan for, e.g. "
                              "sensitivity_minus2=O2A,O20; repeat for several")
    planner.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        if args.stage == "build":
            build(args)
        elif args.stage == "check":
            check(args)
        elif args.stage == "atom-map":
            atom_map(args)
        else:
            stereo_plan(args)
    except (LigandError, chemical_component.ComponentError,
            chemical_graph.CorrespondenceError, provenance.ReceptorError,
            box_geometry.BoxError, ligand_atom_map.AtomMapError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
