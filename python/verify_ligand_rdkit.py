"""Verify a modelled ligand with RDKit, then add hydrogens without moving it.

usage:
    python python/verify_ligand_rdkit.py \
        ligands/uma.ligand.json ligands/uma.stereo_plan.json modelled \
        --out-prefix ligands/uma_verified

This is the one step in the UMA control that needs a chemistry toolkit, and it
is deliberately the only place where chemistry is *perceived* rather than
read: RDKit assigns stereochemistry from the 3D coordinates and places
hydrogens. Nothing here reimplements chemistry.

What it checks, in order, stopping at the first failure
------------------------------------------------------
1. **The molfile is the one the manifest describes** — its SHA-256 matches,
    and it carries the recorded atom count.
2. **Formal charge, heavy-atom count, elements.** The charge must equal the
    manifest's modelled formal charge. A charge of 0 where −3 was intended
    means the `M  CHG` block was not applied and the molecule is the neutral
    form.
3. **Index-to-name correspondence, derived rather than assumed.** The molfile
    carries no atom names, so the mapping from RDKit atom index to reference
    atom name is established **from the coordinates** and then validated
    against elements and connectivity. The stereo plan's list of stereocentre
    names is *not* an atom ordering and is never used as one.
4. **Connectivity** against the component dictionary, through that mapping.
5. **Stereochemistry from 3D**, compared per centre with the dictionary's
    declared configuration, following the state-specific plan: carbon centres
    that must match, and phosphorus centres reported separately.
6. **Hydrogen addition** with `AddHs(addCoords=True)`: no embedding, no
    minimisation, no conformer generation. Charge, stereochemistry and every
    heavy-atom coordinate are rechecked afterwards.

A changed R/S label is not assumed to be an inversion
-----------------------------------------------------
CIP priorities depend on the substituents, so deprotonating an acid can change
a neighbouring centre's R/S label with no change in geometry whatsoever. This
script therefore verifies that every heavy atom sits where the deposited
reference has it, to within the coordinate tolerance, **before** comparing any
label. Once that holds, a label difference cannot be a physical inversion,
because the coordinates are the deposited ones. Differences are reported as
requiring investigation, with that evidence attached, and never silently
recast as inversions or discarded.

Outputs
-------
`<prefix>_withH.mol` and `<prefix>.verification.json`. **A failed check writes
the record with `result: "failed"` and does not write the hydrogenated
molfile**, so a failure cannot leave an output that looks usable. The record
carries input and output hashes and the RDKit version.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry
import chemical_component
import receptor_provenance as provenance

VERIFICATION_SCHEMA = "mur-ligase-ligand-verification/1"

# Coordinate agreement required between the molfile, the reference and the
# hydrogenated molecule. A molfile writes four decimal places, so this is far
# above representation error and far below any real movement.
COORDINATE_TOLERANCE_ANGSTROM = 0.002

# RDKit bond types mapped to the CCD's value_order strings, for comparison.
BOND_TYPE_TO_CCD = {"SINGLE": "SING", "DOUBLE": "DOUB", "TRIPLE": "TRIP",
                    "AROMATIC": "AROM"}


class VerificationError(RuntimeError):
    """Raised when a ligand cannot be verified. Never produces a pass."""


# ---------------------------------------------------------------------------
# A toolkit-independent snapshot, so the checks can be tested without RDKit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoleculeSnapshot:
    """Atoms, coordinates and bonds of a molecule, indexed as the file orders them.

    This is the boundary between the toolkit and the checks. One thin adapter
    builds it from an RDKit molecule; every check below operates on it and is
    testable without a toolkit installed.
    """
    elements: tuple[str, ...]
    positions: tuple[tuple[float, float, float], ...]
    bonds: frozenset[tuple[int, int, str]] = field(default_factory=frozenset)
    n_atoms_total: int = 0
    formal_charge: int = 0
    cip_codes: tuple[tuple[int, str], ...] = ()

    @property
    def n_heavy(self) -> int:
        return sum(1 for element in self.elements if element.upper() not in ("H", "D"))


def distance(first: Sequence[float], second: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5


def correspondence_from_coordinates(
        snapshot: MoleculeSnapshot,
        reference: dict[str, tuple[float, float, float]],
        tolerance: float = COORDINATE_TOLERANCE_ANGSTROM
) -> dict[int, str]:
    """Map each heavy-atom index to a reference atom name, using coordinates.

    The mapping is **derived from the data**, not assumed from file order. Each
    heavy atom is matched to the reference atom within `tolerance` of it, and
    the result must be a bijection: every reference name used exactly once.

    Raises when an atom has no match, more than one match, or when two atoms
    claim the same name. Any of those makes the correspondence ambiguous, and
    an ambiguous correspondence would silently invalidate every comparison
    that follows, including the stereochemistry.
    """
    heavy_indices = [index for index, element in enumerate(snapshot.elements)
                     if element.upper() not in ("H", "D")]
    if len(heavy_indices) != len(reference):
        raise VerificationError(
            f"{len(heavy_indices)} heavy atoms in the molecule against "
            f"{len(reference)} in the reference; the correspondence cannot be "
            f"a bijection"
        )
    mapping: dict[int, str] = {}
    claimed: dict[str, int] = {}
    for index in heavy_indices:
        position = snapshot.positions[index]
        matches = [name for name, reference_position in reference.items()
                   if distance(position, reference_position) <= tolerance]
        if not matches:
            nearest = min(
                ((distance(position, value), name)
                 for name, value in reference.items()), default=(float("inf"), "?"))
            raise VerificationError(
                f"molecule atom {index} ({snapshot.elements[index]}) at "
                f"{tuple(round(value, 3) for value in position)} matches no "
                f"reference atom within {tolerance} A; nearest is "
                f"{nearest[1]} at {nearest[0]:.3f} A. Either the heavy atoms "
                f"moved or this is not the same molecule."
            )
        if len(matches) > 1:
            raise VerificationError(
                f"molecule atom {index} is within {tolerance} A of several "
                f"reference atoms {sorted(matches)}; the reference has "
                f"coincident atoms and the correspondence is ambiguous"
            )
        name = matches[0]
        if name in claimed:
            raise VerificationError(
                f"reference atom {name} is claimed by molecule atoms "
                f"{claimed[name]} and {index}; the correspondence is not a "
                f"bijection"
            )
        claimed[name] = index
        mapping[index] = name
    return mapping


def validate_elements(mapping: dict[int, str], snapshot: MoleculeSnapshot,
                      reference_elements: dict[str, str]) -> list[str]:
    """Return element disagreements under the derived correspondence."""
    problems = []
    for index, name in sorted(mapping.items()):
        expected = reference_elements.get(name, "?").upper()
        found = snapshot.elements[index].upper()
        if expected != found:
            problems.append(f"atom {index} -> {name}: expected {expected}, "
                            f"molecule has {found}")
    return problems


def validate_connectivity(mapping: dict[int, str], snapshot: MoleculeSnapshot,
                          dictionary_bonds: Iterable[tuple[str, str, str]]
                          ) -> dict[str, object]:
    """Compare the molecule's heavy-atom bonds with the dictionary's.

    Bond orders are compared after mapping RDKit's types onto the CCD's
    strings. Order differences are reported separately from missing or extra
    bonds: a wrong order is a different molecule, but it is a different
    failure from a wrong skeleton and the distinction matters when diagnosing.
    """
    expected: dict[frozenset[str], str] = {
        frozenset((first, second)): order
        for first, second, order in dictionary_bonds
    }
    found: dict[frozenset[str], str] = {}
    for first, second, order in snapshot.bonds:
        if first not in mapping or second not in mapping:
            continue          # a bond to a hydrogen
        found[frozenset((mapping[first], mapping[second]))] = order
    missing = sorted("-".join(sorted(pair)) for pair in set(expected) - set(found))
    extra = sorted("-".join(sorted(pair)) for pair in set(found) - set(expected))
    order_differences = sorted(
        f"{'-'.join(sorted(pair))}: dictionary {expected[pair]}, "
        f"molecule {found[pair]}"
        for pair in set(expected) & set(found)
        if BOND_TYPE_TO_CCD.get(found[pair], found[pair]) != expected[pair]
    )
    return {
        "n_dictionary_bonds": len(expected),
        "n_molecule_heavy_bonds": len(found),
        "missing_bonds": missing,
        "unexpected_bonds": extra,
        "bond_order_differences": order_differences,
        "agrees": not (missing or extra or order_differences),
    }


def compare_stereocentres(plan_state: dict[str, object],
                          assigned: dict[str, str],
                          coordinates_match_reference: bool
                          ) -> dict[str, object]:
    """Compare assigned CIP codes with the plan, per centre, keeping the kinds apart.

    `assigned` maps reference atom name to the CIP code RDKit derived from the
    coordinates; a name absent from it means RDKit assigned no code, which is
    itself reportable rather than a pass.

    Nothing is discarded. Every centre in the plan appears in the result, and
    a difference is labelled as requiring investigation, with the reason a
    physical inversion is or is not possible.
    """
    carbon_results: list[dict[str, object]] = []
    phosphorus_results: list[dict[str, object]] = []
    fatal: list[str] = []
    investigate: list[str] = []

    def classify(entry: dict[str, object], must_match: bool) -> dict[str, object]:
        atom = str(entry["atom"])
        declared = entry.get("declared_configuration")
        found = assigned.get(atom)
        agrees = found is not None and found == declared
        result: dict[str, object] = {
            "atom": atom,
            "element": entry.get("element"),
            "declared_configuration": declared,
            "assigned_from_3d": found,
            "expected_to_match": must_match,
            "agrees": agrees,
            "fatal_if_mismatched": bool(entry.get("fatal_if_mismatched", must_match)),
        }
        if agrees:
            result["finding"] = "agrees with the chemical definition"
            return result
        if found is None:
            result["finding"] = (
                "no CIP code was assigned from the 3D coordinates. For a "
                "centre the plan expects to match, this needs investigation: "
                "it may mean the perceived substituents are not all distinct."
                if must_match else
                "no CIP code was assigned, which is consistent with the plan's "
                "classification of this centre as not stereogenic at this state")
            if must_match:
                investigate.append(atom)
                if result["fatal_if_mismatched"]:
                    fatal.append(atom)
            return result
        result["finding"] = (
            f"assigned {found!r} against a declared {declared!r}. "
            + ("Heavy-atom coordinates were verified against the deposited "
               "reference, so this is NOT a physical inversion: the geometry "
               "is the deposited geometry. Investigate CIP priority instead - "
               "changing the protonation of a nearby acid can change priority "
               "order, and the dictionary's label refers to the neutral "
               "component."
               if coordinates_match_reference else
               "Heavy-atom coordinates were NOT verified against the reference "
               "in this run, so a physical inversion cannot be excluded. "
               "Resolve the coordinate check first."))
        investigate.append(atom)
        if result["fatal_if_mismatched"]:
            fatal.append(atom)
        return result

    for entry in plan_state.get("must_match", []):
        result = classify(entry, must_match=True)
        target = (phosphorus_results
                  if str(entry.get("element", "")).upper() in ("P", "S")
                  else carbon_results)
        target.append(result)
    for entry in plan_state.get("not_stereogenic_at_this_state", []):
        result = classify(entry, must_match=False)
        target = (phosphorus_results
                  if str(entry.get("element", "")).upper() in ("P", "S")
                  else carbon_results)
        target.append(result)

    return {
        "centres_expected_to_match": sorted(
            str(entry["atom"]) for entry in plan_state.get("must_match", [])),
        "carbon_and_other_centres": carbon_results,
        "phosphorus_and_sulphur_centres": phosphorus_results,
        "n_agreeing": sum(1 for entry in carbon_results + phosphorus_results
                          if entry["agrees"]),
        "requires_investigation": sorted(set(investigate)),
        "fatal_mismatches": sorted(set(fatal)),
        "passes": not fatal,
        "note": ("phosphorus and sulphur centres are reported separately and "
                 "are not merged into the carbon result; a difference there is "
                 "recorded whether or not it is fatal"),
    }


def compare_heavy_coordinates(before: MoleculeSnapshot, after: MoleculeSnapshot,
                              mapping: dict[int, str],
                              tolerance: float = COORDINATE_TOLERANCE_ANGSTROM
                              ) -> dict[str, object]:
    """Check that hydrogen addition left every heavy atom where it was.

    `AddHs` appends hydrogens and keeps the original atom indices, but that is
    verified here rather than assumed: the heavy atoms are compared by index
    through the same correspondence, and the worst deviation is reported.
    """
    worst, worst_atom = 0.0, None
    for index, name in sorted(mapping.items()):
        if index >= len(after.positions):
            raise VerificationError(
                f"heavy atom {index} ({name}) is absent after hydrogen "
                f"addition; atom order was not preserved")
        deviation = distance(before.positions[index], after.positions[index])
        if deviation > worst:
            worst, worst_atom = deviation, name
    return {
        "n_heavy_atoms_compared": len(mapping),
        "max_deviation_angstrom": round(worst, 6),
        "max_deviation_atom": worst_atom,
        "tolerance_angstrom": tolerance,
        "preserved": worst <= tolerance,
    }


# ---------------------------------------------------------------------------
# The RDKit adapter: the only part that needs a toolkit
# ---------------------------------------------------------------------------

def rdkit_version() -> str:
    try:
        import rdkit
    except ImportError as error:
        raise VerificationError(
            f"RDKit is not available: {error}. This step must run in an "
            f"environment that provides it (the `boltz` environment on Epicac)."
        ) from error
    return rdkit.__version__


def snapshot_from_rdkit(mol) -> MoleculeSnapshot:
    """Build a toolkit-independent snapshot from an RDKit molecule."""
    from rdkit import Chem

    conformer = mol.GetConformer()
    positions = []
    elements = []
    for atom in mol.GetAtoms():
        point = conformer.GetAtomPosition(atom.GetIdx())
        positions.append((point.x, point.y, point.z))
        elements.append(atom.GetSymbol())
    bonds = set()
    for bond in mol.GetBonds():
        bonds.add((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   str(bond.GetBondType())))
    codes = []
    for atom in mol.GetAtoms():
        if atom.HasProp("_CIPCode"):
            codes.append((atom.GetIdx(), atom.GetProp("_CIPCode")))
    return MoleculeSnapshot(
        elements=tuple(elements), positions=tuple(positions),
        bonds=frozenset(bonds), n_atoms_total=mol.GetNumAtoms(),
        formal_charge=Chem.GetFormalCharge(mol), cip_codes=tuple(codes))


def read_and_assign(path: Path) -> tuple[object, MoleculeSnapshot]:
    """Read a molfile and assign stereochemistry from its 3D coordinates.

    Two steps, both RDKit's: `AssignStereochemistryFrom3D` sets the chiral
    tags from the geometry, then `rdCIPLabeler.AssignCIPLabels` computes true
    CIP labels, which the legacy assignment approximates. The accurate
    labeller is used because the comparison is against the dictionary's CIP
    configurations.
    """
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if mol is None:
        raise VerificationError(
            f"RDKit could not read {path}. A sanitisation failure here means "
            f"the connection table and the charges describe a molecule RDKit "
            f"rejects; the molfile, not RDKit, is the thing to inspect."
        )
    if mol.GetNumConformers() == 0:
        raise VerificationError(f"{path} carries no 3D conformer")
    Chem.AssignStereochemistryFrom3D(mol)
    rdCIPLabeler.AssignCIPLabels(mol)
    return mol, snapshot_from_rdkit(mol)


def add_hydrogens(mol):
    """Add hydrogens with coordinates, without embedding or minimising.

    `addCoords=True` places hydrogens geometrically from the existing heavy
    atoms. No conformer is generated and no force field is applied, so the
    heavy atoms cannot move; `compare_heavy_coordinates` verifies that rather
    than trusting it.
    """
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    withH = Chem.AddHs(mol, addCoords=True)
    Chem.AssignStereochemistryFrom3D(withH)
    rdCIPLabeler.AssignCIPLabels(withH)
    return withH, snapshot_from_rdkit(withH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ligand_manifest", type=Path,
                        help="Manifest from prepare_ligand.py build")
    parser.add_argument("stereo_plan", type=Path,
                        help="Plan from prepare_ligand.py stereo-plan")
    parser.add_argument("state",
                        help="Which state in the plan this molfile is, e.g. "
                             "modelled or sensitivity_minus2")
    parser.add_argument("--out-prefix", type=Path, required=True,
                        help="Prefix for <prefix>_withH.mol and "
                             "<prefix>.verification.json")
    parser.add_argument("--component-dictionary", type=Path,
                        help="Defaults to the path recorded in the manifest")
    parser.add_argument("--coordinate-tolerance", type=float,
                        default=COORDINATE_TOLERANCE_ANGSTROM)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    record_path = args.out_prefix.with_name(
        args.out_prefix.name + ".verification.json")
    hydrogenated_path = args.out_prefix.with_name(
        args.out_prefix.name + "_withH.mol")
    record: dict[str, object] = {}
    try:
        record = verify(args, hydrogenated_path)
    except (VerificationError, chemical_component.ComponentError,
            provenance.ReceptorError, box_geometry.BoxError) as error:
        record = failure_record(args, str(error))
        write_record(record_path, record, args.overwrite)
        if hydrogenated_path.exists():
            failed = hydrogenated_path.with_name(hydrogenated_path.name + ".FAILED")
            hydrogenated_path.replace(failed)
            print(f"renamed {hydrogenated_path.name} -> {failed.name}",
                  file=sys.stderr)
        print(f"\nrecord           : {record_path} (result: failed)",
              file=sys.stderr)
        sys.exit(f"error: {error}")
    write_record(record_path, record, args.overwrite)
    report(record, record_path, hydrogenated_path)


def failure_record(args: argparse.Namespace, reason: str) -> dict[str, object]:
    """A record of a failed verification. Never says passed."""
    return {
        "schema": VERIFICATION_SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result": "failed",
        "state": args.state,
        "ligand_manifest": str(args.ligand_manifest),
        "stereo_plan": str(args.stereo_plan),
        "failure_reason": reason,
        "outputs": {"hydrogenated_molfile": None},
        "note": ("this verification failed; no hydrogenated ligand was "
                 "produced and nothing downstream may treat this as a pass"),
    }


def write_record(path: Path, record: dict[str, object], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_manifest(path, record, overwrite=True if overwrite else
                              not path.exists())


def verify(args: argparse.Namespace, hydrogenated_path: Path) -> dict[str, object]:
    """Run every check and return a passing record, or raise.

    The input-identity checks come before RDKit is required, so a mismatched
    plan, a stale molfile or a revised dictionary fails fast with a clear
    message rather than behind a missing-dependency error. It also means those
    guards can be exercised without a toolkit installed.
    """
    manifest = provenance.read_manifest_like(args.ligand_manifest,
                                             "mur-ligase-ligand/1")
    plan = provenance.read_manifest_like(args.stereo_plan,
                                         "mur-ligase-stereo-plan/1")
    if plan["ligand_manifest"]["sha256"] != box_geometry.file_sha256(
            args.ligand_manifest):
        raise VerificationError(
            f"{args.stereo_plan} was built from a different revision of "
            f"{args.ligand_manifest}; regenerate the plan")
    if args.state not in plan["states"]:
        raise VerificationError(
            f"the plan has no state {args.state!r}; it has "
            f"{sorted(plan['states'])}")
    plan_state = plan["states"][args.state]

    expected_charge = manifest["protonation_state"]["modelled_formal_charge"]
    if plan_state["formal_charge"] != expected_charge:
        raise VerificationError(
            f"state {args.state!r} in the plan is charge "
            f"{plan_state['formal_charge']} but the manifest's modelled state "
            f"is {expected_charge}; the plan state and the molfile do not "
            f"describe the same protonation")

    molfile = Path(manifest["outputs"]["molfile"])
    if not molfile.is_file():
        raise VerificationError(f"the manifest's molfile is missing: {molfile}")
    molfile_hash = box_geometry.file_sha256(molfile)
    if molfile_hash != manifest["outputs"]["molfile_sha256"]:
        raise VerificationError(
            f"{molfile} has changed since the manifest was written "
            f"(recorded {manifest['outputs']['molfile_sha256'][:12]}..., found "
            f"{molfile_hash[:12]}...)")

    dictionary_path = Path(args.component_dictionary
                           or manifest["dictionary"]["path"])
    component = chemical_component.read_component(dictionary_path)
    dictionary_hash = box_geometry.file_sha256(dictionary_path)
    if dictionary_hash != manifest["dictionary"]["sha256"]:
        raise VerificationError(
            f"{dictionary_path} is not the dictionary revision the ligand was "
            f"built from; CCD entries are revised")

    reference_record = provenance.read_manifest_like(
        Path(manifest["reference"]["path"]), None)
    reference_atoms = {
        name: tuple(entry["coordinates_angstrom"])
        for name, entry in reference_record["atom_mapping"].items()
    }
    reference_elements = {name: entry["element"]
                          for name, entry in reference_record["atom_mapping"].items()}

    # Every input is now established; from here a toolkit is required.
    version = rdkit_version()
    mol, snapshot = read_and_assign(molfile)

    if snapshot.formal_charge != expected_charge:
        raise VerificationError(
            f"RDKit reads a formal charge of {snapshot.formal_charge} but the "
            f"modelled state is {expected_charge}. A charge of 0 means the "
            f"M  CHG block was not applied and this is the neutral molecule."
        )
    if snapshot.n_heavy != len(reference_atoms):
        raise VerificationError(
            f"{snapshot.n_heavy} heavy atoms against {len(reference_atoms)} in "
            f"the reference")
    if snapshot.n_atoms_total != snapshot.n_heavy:
        raise VerificationError(
            f"the molfile already contains "
            f"{snapshot.n_atoms_total - snapshot.n_heavy} hydrogens; this "
            f"stage expects the heavy-atom molfile")

    mapping = correspondence_from_coordinates(
        snapshot, reference_atoms, args.coordinate_tolerance)
    element_problems = validate_elements(mapping, snapshot, reference_elements)
    if element_problems:
        raise VerificationError(
            "elements disagree under the derived correspondence: "
            + "; ".join(element_problems[:5]))

    dictionary_bonds = [
        (component.graph.names[first], component.graph.names[second], order)
        for pair, order in component.graph.bonds.items()
        for first, second in (tuple(pair),)
    ]
    connectivity = validate_connectivity(mapping, snapshot, dictionary_bonds)
    if not connectivity["agrees"]:
        raise VerificationError(
            f"connectivity disagrees with {component.identifier}: missing "
            f"{connectivity['missing_bonds'][:5]}, unexpected "
            f"{connectivity['unexpected_bonds'][:5]}, order differences "
            f"{connectivity['bond_order_differences'][:5]}")

    assigned = {mapping[index]: code for index, code in snapshot.cip_codes
                if index in mapping}
    stereo = compare_stereocentres(plan_state, assigned,
                                  coordinates_match_reference=True)

    withH, after = add_hydrogens(mol)
    coordinates = compare_heavy_coordinates(
        snapshot, after, mapping, args.coordinate_tolerance)
    if not coordinates["preserved"]:
        raise VerificationError(
            f"hydrogen addition moved heavy atom "
            f"{coordinates['max_deviation_atom']} by "
            f"{coordinates['max_deviation_angstrom']} A")
    if after.formal_charge != expected_charge:
        raise VerificationError(
            f"the formal charge changed to {after.formal_charge} when "
            f"hydrogens were added; expected {expected_charge}")
    assigned_after = {mapping[index]: code for index, code in after.cip_codes
                      if index in mapping}
    stereo_after = compare_stereocentres(plan_state, assigned_after,
                                        coordinates_match_reference=True)
    changed = sorted(
        atom for atom in set(assigned) | set(assigned_after)
        if assigned.get(atom) != assigned_after.get(atom))
    if changed:
        raise VerificationError(
            f"stereochemistry changed when hydrogens were added, at {changed}. "
            f"Hydrogen addition must not alter perceived configuration.")
    if stereo["fatal_mismatches"]:
        raise VerificationError(
            f"stereochemistry does not match the chemical definition at "
            f"{stereo['fatal_mismatches']}, which the plan marks fatal. "
            f"Findings: "
            + "; ".join(entry["finding"]
                        for entry in stereo["carbon_and_other_centres"]
                        if not entry["agrees"])[:600])

    from rdkit import Chem
    hydrogenated_path.parent.mkdir(parents=True, exist_ok=True)
    if hydrogenated_path.exists() and not args.overwrite:
        raise VerificationError(
            f"{hydrogenated_path} already exists; pass --overwrite to replace it")
    Chem.MolToMolFile(withH, str(hydrogenated_path))

    return {
        "schema": VERIFICATION_SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result": "passed",
        "state": args.state,
        "component": component.identifier,
        "tools": {"rdkit": version, "python": sys.version.split()[0]},
        "inputs": {
            "ligand_manifest": str(args.ligand_manifest),
            "ligand_manifest_sha256": box_geometry.file_sha256(args.ligand_manifest),
            "stereo_plan": str(args.stereo_plan),
            "stereo_plan_sha256": box_geometry.file_sha256(args.stereo_plan),
            "molfile": str(molfile),
            "molfile_sha256": molfile_hash,
            "component_dictionary": str(dictionary_path),
            "component_dictionary_sha256": dictionary_hash,
            "reference_record": manifest["reference"]["path"],
            "reference_record_sha256": box_geometry.file_sha256(
                Path(manifest["reference"]["path"])),
        },
        "protonation_state": manifest["protonation_state"],
        "checks": {
            "formal_charge": {"expected": expected_charge,
                              "found": snapshot.formal_charge, "agrees": True},
            "heavy_atom_count": {"expected": len(reference_atoms),
                                 "found": snapshot.n_heavy},
            "index_to_name_correspondence": {
                "method": ("derived from coordinates within "
                           f"{args.coordinate_tolerance} A, then validated "
                           "against elements and connectivity; the plan's "
                           "stereocentre list is not used as an atom ordering"),
                "mapping": {str(index): name
                            for index, name in sorted(mapping.items())},
                "equals_file_order": [mapping[index] for index
                                      in sorted(mapping)] == list(
                                          component.graph.names),
                "bijective": True,
            },
            "elements": {"disagreements": element_problems, "agrees": True},
            "connectivity": connectivity,
            "stereochemistry_before_hydrogens": stereo,
            "stereochemistry_after_hydrogens": stereo_after,
            "stereochemistry_unchanged_by_hydrogen_addition": True,
            "heavy_atom_coordinates_after_hydrogens": coordinates,
            "hydrogen_addition": {
                "method": "RDKit AddHs(addCoords=True)",
                "embedding_or_minimisation": "none",
                "n_hydrogens_added": after.n_atoms_total - snapshot.n_atoms_total,
                "n_atoms_after": after.n_atoms_total,
            },
        },
        "outputs": {
            "hydrogenated_molfile": str(hydrogenated_path),
            "hydrogenated_molfile_sha256": box_geometry.file_sha256(
                hydrogenated_path),
        },
        "limitations": [
            "partial charges and AutoDock atom types are not set here; they "
            "come from Meeko and are recorded by prepare_ligand.py check",
            "hydrogen orientations on rotatable hydroxyls are geometric "
            "placements, not experimental positions",
            "the dictionary's CIP labels describe the neutral component; a "
            "difference at a centre near an ionised group may be a priority "
            "change rather than a geometric difference, and is reported for "
            "investigation rather than resolved here",
        ],
    }


def report(record: dict[str, object], record_path: Path,
           hydrogenated_path: Path) -> None:
    checks = record["checks"]
    print(f"state            : {record['state']}")
    print(f"component        : {record['component']}")
    print(f"rdkit            : {record['tools']['rdkit']}")
    print(f"formal charge    : {checks['formal_charge']['found']} "
          f"(expected {checks['formal_charge']['expected']})")
    print(f"heavy atoms      : {checks['heavy_atom_count']['found']}")
    correspondence = checks["index_to_name_correspondence"]
    print(f"correspondence   : bijective, derived from coordinates; "
          f"equals file order: {correspondence['equals_file_order']}")
    print(f"connectivity     : {checks['connectivity']['n_molecule_heavy_bonds']}"
          f"/{checks['connectivity']['n_dictionary_bonds']} bonds agree")
    stereo = checks["stereochemistry_before_hydrogens"]
    print(f"\nstereochemistry, centres expected to match "
          f"({len(stereo['centres_expected_to_match'])}):")
    for entry in stereo["carbon_and_other_centres"]:
        mark = "ok " if entry["agrees"] else "** "
        print(f"  {mark}{entry['atom']:<6} {entry['element']:<2} declared "
              f"{str(entry['declared_configuration']):<4} assigned "
              f"{str(entry['assigned_from_3d']):<4}")
    print("phosphorus and sulphur centres, reported separately:")
    for entry in stereo["phosphorus_and_sulphur_centres"]:
        mark = "ok " if entry["agrees"] else "-- "
        print(f"  {mark}{entry['atom']:<6} {entry['element']:<2} declared "
              f"{str(entry['declared_configuration']):<4} assigned "
              f"{str(entry['assigned_from_3d']):<4} "
              f"(expected to match: {entry['expected_to_match']})")
    if stereo["requires_investigation"]:
        print(f"\nREQUIRES INVESTIGATION: {stereo['requires_investigation']}")
        for entry in (stereo["carbon_and_other_centres"]
                      + stereo["phosphorus_and_sulphur_centres"]):
            if not entry["agrees"]:
                print(f"  {entry['atom']}: {entry['finding']}")
    hydrogen = checks["hydrogen_addition"]
    coordinates = checks["heavy_atom_coordinates_after_hydrogens"]
    print(f"\nhydrogens added  : {hydrogen['n_hydrogens_added']} "
          f"({hydrogen['method']}, {hydrogen['embedding_or_minimisation']})")
    print(f"heavy atoms moved: {coordinates['max_deviation_angstrom']} A max "
          f"(tolerance {coordinates['tolerance_angstrom']} A)")
    print(f"\nwrote            : {hydrogenated_path}")
    print(f"record           : {record_path} (result: {record['result']})")


if __name__ == "__main__":
    main()
