"""Assess redocked poses against an immutable reference pose.

usage:
    python python/analyse_redocking.py DOCK_RUN_DIR LIGAND_NAME \
        --reference references/uma_chainE.reference.json \
        --component-dictionary project_private/ccd_downloads/UMA.cif \
        --out analysis/uma_recovery

What is measured
----------------
For every retained pose of every seed: the **symmetry-aware heavy-atom RMSD**
to the reference, **in the receptor coordinate frame, with no superposition of
any kind**. A pose is compared where the search put it. Superposing first
would answer a different question and would hide a pose that has the right
shape in the wrong place.

The criterion is RMSD <= 2.0 A. Two poses are reported per seed, because they
answer different questions:

- the **top-ranked** pose, which asks whether the scoring function prefers a
  near-reference pose;
- the **closest-to-reference** retained pose with its rank, which asks whether
  the search reached one at all.

A good closest with a poor top-ranked is a scoring problem. A poor closest is
a sampling or search-space problem. Collapsing the two loses the distinction
that makes the control useful, so they are never merged.

**Consistent recovery** means the top-ranked pose meets the criterion in every
seed. Anything else is a mixed outcome and is reported as such, with the
continuous RMSD values, never reduced to a pass or fail. Continuous values
matter: 2.1 A in two seeds out of three is a different situation from 9 A in
one.

What this cannot establish
--------------------------
Recovering the deposited pose shows that this box, this receptor and these
search settings can place the cognate ligand where the crystal has it. It does
not establish that peptide poses or peptide rankings are reliable, and it is
not an affinity measurement. The receptor used for the initial control has
neither ANP nor manganese, so a recovery is **recovery in the simplified
receptor, not validation against the native state**.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry
import chemical_component
import chemical_graph
import ligand_atom_map
import receptor_provenance as provenance

RECOVERY_CRITERION_ANGSTROM = 2.0
ANALYSIS_SCHEMA = "mur-ligase-redocking-analysis/1"

POSE_FILE_PATTERN = re.compile(r"^seed_(?P<seed>-?\d+)$")


@dataclass
class Pose:
    """One retained pose from one seed."""
    seed: int
    rank: int
    score: float | None
    coordinates: dict[str, tuple[float, float, float]]
    source: str


def parse_pose_file(path: Path,
                   names_by_serial: dict[int, str] | None = None
                   ) -> list[dict[str, object]]:
    """Read a multi-model ligand PDBQT written by Vina.

    Vina writes one `MODEL`/`ENDMDL` block per retained pose, in rank order,
    each carrying a `REMARK VINA RESULT` line whose first number is the score
    in kcal/mol. Models are returned in file order; the rank is the caller's
    business, since rank 1 is simply the first retained pose.
    """
    poses: list[dict[str, object]] = []
    current: dict[str, tuple[float, float, float]] | None = None
    score: float | None = None
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if line.startswith("MODEL"):
            current, score = {}, None
            continue
        if line.startswith("REMARK VINA RESULT"):
            fields = line.split()
            try:
                score = float(fields[3])
            except (IndexError, ValueError):
                score = None
            continue
        if line.startswith(("ATOM", "HETATM")):
            if current is None:
                # A single-pose file with no MODEL records.
                current, score = {}, score
            name = line[12:16].strip()
            element_or_type = line[77:80].strip().upper()
            if element_or_type in ("HD", "H") or name.upper().startswith("H"):
                continue
            if names_by_serial is not None:
                # Docked poses inherit the prepared ligand's atom records, and
                # Meeko writes the element in the name column, so names there
                # are not unique. The validated serial mapping is used instead.
                try:
                    serial = int(line[6:11])
                except ValueError as error:
                    raise chemical_graph.CorrespondenceError(
                        f"{path} line {number}: unreadable atom serial ({error})")
                mapped = names_by_serial.get(serial)
                if mapped is None:
                    raise chemical_graph.CorrespondenceError(
                        f"{path} line {number}: heavy-atom serial {serial} has "
                        f"no entry in the atom map; the map does not cover "
                        f"this pose file")
                name = mapped
            try:
                position = (float(line[30:38]), float(line[38:46]),
                            float(line[46:54]))
            except ValueError as error:
                raise chemical_graph.CorrespondenceError(
                    f"{path} line {number}: malformed coordinate ({error})")
            if name in current:
                raise chemical_graph.CorrespondenceError(
                    f"{path} line {number}: repeated atom name {name} within one pose")
            current[name] = position
            continue
        if line.startswith("ENDMDL"):
            if current:
                poses.append({"coordinates": current, "score": score})
            current, score = None, None
    if current:
        poses.append({"coordinates": current, "score": score})
    if not poses:
        raise chemical_graph.CorrespondenceError(f"{path} contains no poses")
    return poses


def collect_poses(run_directory: Path, ligand: str,
                  names_by_serial: dict[int, str] | None = None) -> list[Pose]:
    """Gather every retained pose of every seed from a docking run directory.

    Expects the layout `dock.py` writes: `poses/seed_<N>/<ligand>_docked.pdbqt`.
    """
    pose_root = run_directory / "poses"
    if not pose_root.is_dir():
        raise chemical_graph.CorrespondenceError(
            f"{pose_root} does not exist; point this at a dock.py output directory")
    poses: list[Pose] = []
    seeds_found: list[int] = []
    for seed_directory in sorted(pose_root.iterdir()):
        match = POSE_FILE_PATTERN.match(seed_directory.name)
        if not seed_directory.is_dir() or not match:
            continue
        seed = int(match.group("seed"))
        pose_file = seed_directory / f"{ligand}_docked.pdbqt"
        if not pose_file.is_file():
            continue
        seeds_found.append(seed)
        for rank, entry in enumerate(
                parse_pose_file(pose_file, names_by_serial), start=1):
            poses.append(Pose(seed=seed, rank=rank,
                              score=entry["score"],
                              coordinates=entry["coordinates"],
                              source=str(pose_file)))
    if not poses:
        raise chemical_graph.CorrespondenceError(
            f"no poses for ligand {ligand!r} under {pose_root}")
    return poses


def analyse(reference_atoms: dict[str, tuple[float, float, float]],
            graph: chemical_graph.MolecularGraph,
            permutations: list[tuple[int, ...]],
            poses: list[Pose],
            criterion: float) -> dict[str, object]:
    """Compute per-pose RMSD and the per-seed and overall summaries."""
    rows: list[dict[str, object]] = []
    for pose in poses:
        rmsd, permutation = chemical_graph.symmetry_aware_rmsd(
            graph, reference_atoms, pose.coordinates, permutations)
        identity = tuple(range(len(graph.names)))
        # The mapping that achieved the minimum is recorded by atom name, not
        # as a flag: if an independent implementation disagrees on an RMSD,
        # the two mappings are the first thing to compare.
        exchanged = " ".join(
            f"{graph.names[source]}->{graph.names[image]}"
            for source, image in enumerate(permutation) if source != image)
        rows.append({
            "seed": pose.seed,
            "rank": pose.rank,
            "score_kcal_per_mol": pose.score,
            "symmetry_aware_rmsd_angstrom": round(rmsd, 3),
            "meets_criterion": "yes" if rmsd <= criterion else "no",
            "permutation_is_identity": permutation == identity,
            "atoms_exchanged_by_the_minimising_mapping": exchanged,
            "pose_file": pose.source,
        })

    per_seed: list[dict[str, object]] = []
    for seed in sorted({pose.seed for pose in poses}):
        seed_rows = [row for row in rows if row["seed"] == seed]
        top = min(seed_rows, key=lambda row: row["rank"])
        closest = min(seed_rows,
                      key=lambda row: row["symmetry_aware_rmsd_angstrom"])
        per_seed.append({
            "seed": seed,
            "n_poses_retained": len(seed_rows),
            "top_ranked": {
                "rank": top["rank"],
                "score_kcal_per_mol": top["score_kcal_per_mol"],
                "rmsd_angstrom": top["symmetry_aware_rmsd_angstrom"],
                "meets_criterion": top["meets_criterion"],
            },
            "closest_to_reference": {
                "rank": closest["rank"],
                "score_kcal_per_mol": closest["score_kcal_per_mol"],
                "rmsd_angstrom": closest["symmetry_aware_rmsd_angstrom"],
                "meets_criterion": closest["meets_criterion"],
            },
        })

    top_values = [entry["top_ranked"]["rmsd_angstrom"] for entry in per_seed]
    closest_values = [entry["closest_to_reference"]["rmsd_angstrom"]
                      for entry in per_seed]
    n_top_pass = sum(1 for entry in per_seed
                     if entry["top_ranked"]["meets_criterion"] == "yes")
    consistent = n_top_pass == len(per_seed) and len(per_seed) > 0
    if consistent:
        outcome = "consistent recovery"
    elif n_top_pass:
        outcome = "mixed outcome"
    elif any(entry["closest_to_reference"]["meets_criterion"] == "yes"
             for entry in per_seed):
        outcome = ("recovery among retained poses, but failure to rank it "
                   "first in that run")
    else:
        outcome = "no recovery"

    return {
        "criterion_angstrom": criterion,
        "n_seeds": len(per_seed),
        "n_poses_total": len(rows),
        "per_pose": rows,
        "per_seed": per_seed,
        "summary": {
            "outcome": outcome,
            "consistent_recovery": consistent,
            "definition": ("consistent recovery means the top-ranked pose meets "
                           "the criterion in every seed"),
            "n_seeds_top_ranked_meets_criterion": n_top_pass,
            "top_ranked_rmsd_range_angstrom": [min(top_values), max(top_values)]
                                              if top_values else None,
            "closest_rmsd_range_angstrom": [min(closest_values), max(closest_values)]
                                           if closest_values else None,
        },
    }


def validated_stereo_verification(path: Path | None,
                                  ligand_manifest: Path | None
                                  ) -> dict[str, object]:
    """Check a stereochemistry verification record, rather than its existence.

    A file being present is not evidence. The record must carry the
    verification schema, state `result: "passed"`, report no fatal mismatch,
    and describe **the ligand manifest actually being analysed**, matched by
    SHA-256. A record from a different ligand, a different protonation state
    or a failed run does not satisfy the prerequisite.
    """
    requirement = ("a verification record from verify_ligand_rdkit.py whose "
                   "result is 'passed', with no fatal stereochemistry "
                   "mismatch, for the same ligand manifest being analysed")
    manifest_identity: dict[str, object] | None = None
    if path is None:
        return {"satisfied": False, "evidence": None, "requirement": requirement,
                "reason": "no record supplied"}
    try:
        record = provenance.read_manifest_like(
            path, "mur-ligase-ligand-verification/1")
    except provenance.ReceptorError as error:
        return {"satisfied": False, "evidence": str(path),
                "requirement": requirement,
                "reason": f"not a usable verification record: {error}"}

    problems: list[str] = []
    if record.get("result") != "passed":
        problems.append(
            f"the record's result is {record.get('result')!r}, not 'passed'"
            + (f": {record.get('failure_reason')}"
               if record.get("failure_reason") else ""))
    fatal = ((record.get("checks", {}) or {})
             .get("stereochemistry_before_hydrogens", {}) or {}).get(
                 "fatal_mismatches") or []
    if fatal:
        problems.append(f"fatal stereochemistry mismatches at {fatal}")
    if ligand_manifest is None:
        problems.append("no --ligand-manifest given, so the record cannot be "
                        "tied to the ligand being analysed")
    else:
        recorded = (record.get("inputs", {}) or {}).get("ligand_manifest_sha256")
        actual = box_geometry.file_sha256(ligand_manifest)
        # A manifest modified in place by a historical check stage is still the
        # manifest that was verified, provided restoring that one known block
        # reproduces the recorded digest exactly. The basis is carried through
        # to the analysis record so a compatibility match is visible as one.
        agreement = provenance.manifest_hash_agreement(ligand_manifest, recorded)
        manifest_identity = agreement
        if not recorded:
            problems.append("the record carries no ligand manifest hash")
        elif not agreement["agrees"]:
            problems.append(
                f"the record verifies a different ligand manifest "
                f"(recorded {recorded[:12]}..., analysing {actual[:12]}...): "
                f"{agreement['reason']}")

    investigation = ((record.get("checks", {}) or {})
                     .get("stereochemistry_before_hydrogens", {}) or {}).get(
                         "requires_investigation") or []
    return {
        "satisfied": not problems,
        "evidence": str(Path(path).resolve()),
        "requirement": requirement,
        "reason": "; ".join(problems) if problems else "record validated",
        "record_result": record.get("result"),
        "record_state": record.get("state"),
        "ligand_manifest_identity": manifest_identity,
        "rdkit_version": (record.get("tools", {}) or {}).get("rdkit"),
        "centres_requiring_investigation": investigation,
        "note": ("centres requiring investigation do not block the "
                 "prerequisite when none is fatal, but they are carried here "
                 "so they are not lost"
                 if investigation else None),
    }


INDEPENDENT_RMSD_SCHEMA = "mur-ligase-independent-rmsd/1"


def validated_independent_rmsd(path: Path | None,
                               ligand_manifest: Path | None
                               ) -> dict[str, object]:
    """Check an independent RMSD comparison record, rather than its existence.

    The prerequisite is that this project's symmetry-aware RMSD has been
    compared against an established toolkit, on the same structures, at the
    same protonation state and under the same chemically allowed atom
    equivalences. A file being present evidences none of that, so the record
    must state a passing result, describe **the ligand manifest being
    analysed** matched by SHA-256, report every case as agreeing within its
    stated tolerance, carry the negative case that would expose a wrongly
    permitted oxygen exchange, and attest that no coordinate fitting was
    applied. A record whose own checks failed, or which compared a different
    ligand or protonation state, does not satisfy the prerequisite.
    """
    requirement = ("this project's symmetry-aware RMSD compared against an "
                   "established toolkit on the same structures, using the "
                   "same protonation state and the same chemically allowed "
                   "atom equivalences, with no coordinate fitting")
    if path is None:
        return {"satisfied": False, "evidence": None, "requirement": requirement,
                "reason": "no record supplied"}
    try:
        record = provenance.read_manifest_like(path, INDEPENDENT_RMSD_SCHEMA)
    except provenance.ReceptorError as error:
        return {"satisfied": False, "evidence": str(path),
                "requirement": requirement,
                "reason": f"not a usable comparison record: {error}"}

    problems: list[str] = []
    if record.get("result") != "passed":
        problems.append(
            f"the record's result is {record.get('result')!r}, not 'passed'"
            + (f": {record.get('failure_reasons')}"
               if record.get("failure_reasons") else ""))

    checks = record.get("checks", {}) or {}
    if not checks.get("all_cases_agree"):
        problems.append(
            f"the record does not report agreement on every case "
            f"(maximum absolute difference "
            f"{checks.get('max_absolute_difference_angstrom')} A against a "
            f"tolerance of {record.get('tolerance_angstrom')} A)")
    if not checks.get("negative_case_distinguished"):
        problems.append(
            "the record does not report the negative case as distinguished, "
            "so a wrongly permitted oxygen exchange would not have been caught")
    if not (checks.get("allowed_mappings_independently_audited")
            or checks.get("equivalences_independently_verified")):
        problems.append(
            "the record does not report the allowed mappings as "
            "independently audited and agreeing with this project's, so the "
            "two sides may have minimised over different conventions, or the "
            "comparison may have been made under this project's assumptions "
            "on both sides")
    if (record.get("measure", {}) or {}).get("superposition") != "none":
        problems.append(
            f"the record's superposition is "
            f"{(record.get('measure', {}) or {}).get('superposition')!r}, not "
            f"'none'; a fitted RMSD is a different quantity")

    manifest_identity: dict[str, object] | None = None
    if ligand_manifest is None:
        problems.append("no --ligand-manifest given, so the record cannot be "
                        "tied to the ligand being analysed")
    else:
        recorded = (record.get("inputs", {}) or {}).get("ligand_manifest_sha256")
        manifest_identity = provenance.manifest_hash_agreement(ligand_manifest,
                                                               recorded)
        if not recorded:
            problems.append("the record carries no ligand manifest hash")
        elif not manifest_identity["agrees"]:
            problems.append(
                f"the record compared a different ligand manifest: "
                f"{manifest_identity['reason']}")

    return {
        "satisfied": not problems,
        "evidence": str(Path(path).resolve()),
        "requirement": requirement,
        "reason": "; ".join(problems) if problems else "record validated",
        "record_result": record.get("result"),
        "record_state": record.get("state"),
        "content_class": record.get("content_class"),
        "ligand_manifest_identity": manifest_identity,
        "max_absolute_difference_angstrom": checks.get(
            "max_absolute_difference_angstrom"),
        "tolerance_angstrom": record.get("tolerance_angstrom"),
        "n_cases": len(record.get("cases", []) or []),
        "toolkit": (record.get("tools", {}) or {}).get("rdkit"),
        "note": ("the cases are synthetic coordinate transformations, which "
                 "validate the RMSD implementation and not any docking "
                 "result"),
    }


def pose_linkage(atom_map_record: dict, run_record: dict,
                 run_record_path: Path) -> dict[str, object]:
    """Link docked poses to the prepared ligand through campaign provenance.

    A docked pose file is Vina's **output**. It has its own hash and does not
    carry the prepared ligand's, so an atom map must not be required to
    hash-match the pose files; `read_atom_map` is therefore called without
    `expected_pdbqt` here, unlike in the preparation check where the PDBQT
    being examined *is* the prepared input.

    What does establish the link is the docking run's own record, which lists
    every ligand input with its SHA-256. If the hash the atom map was built
    against appears there, the poses in that directory were docked from the
    ligand the map describes. If it does not, the poses may still be fine but
    nothing ties them to the verified input, and that is reported rather than
    assumed either way.
    """
    recorded = (atom_map_record.get("inputs", {}) or {}).get(
        "prepared_pdbqt_sha256")
    requirement = ("the atom map's prepared-ligand hash appears among the "
                   "docking run's recorded ligand inputs")
    if not run_record:
        return {"established": False, "requirement": requirement,
                "reason": f"no run record at {run_record_path}, so the "
                          f"campaign's ligand inputs are unknown"}
    if not recorded:
        return {"established": False, "requirement": requirement,
                "reason": "the atom map records no prepared-ligand hash"}
    inputs = run_record.get("ligand_inputs") or []
    hashes = {}
    for entry in inputs:
        if isinstance(entry, dict) and entry.get("sha256"):
            hashes[entry["sha256"]] = entry.get("path")
    if recorded in hashes:
        return {
            "established": True,
            "requirement": requirement,
            "prepared_pdbqt_sha256": recorded,
            "matched_campaign_input": hashes[recorded],
            "note": ("the poses were docked from the ligand this map "
                     "describes; pose files carry their own hashes and are "
                     "not compared against the input's"),
        }
    return {
        "established": False,
        "requirement": requirement,
        "prepared_pdbqt_sha256": recorded,
        "campaign_ligand_inputs": sorted(hashes.values()),
        "reason": ("the atom map was built against a prepared ligand whose "
                   "hash is not among this run's recorded ligand inputs; the "
                   "poses may not come from the ligand the map describes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_directory", type=Path,
                        help="dock.py output directory")
    parser.add_argument("ligand", help="Ligand stem, as in the pose filenames")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Reference record from ligand_reference.py")
    parser.add_argument("--component-dictionary", type=Path, required=True,
                        help="CCD entry supplying connectivity for the symmetry")
    parser.add_argument("--ligand-manifest", type=Path,
                        help="Ligand manifest from prepare_ligand.py. It "
                             "supplies the modelled protonation state, which "
                             "decides which groups are resonance-equivalent. "
                             "Without it no equivalence is applied and the "
                             "RMSD values are upper bounds")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output prefix for the analysis record and table")
    parser.add_argument("--criterion", type=float,
                        default=RECOVERY_CRITERION_ANGSTROM,
                        help=f"RMSD criterion in A (default "
                             f"{RECOVERY_CRITERION_ANGSTROM})")
    parser.add_argument("--atom-map", type=Path,
                        help="Validated atom-map record from "
                             "prepare_ligand.py atom-map. Required when the "
                             "docked poses carry element names rather than "
                             "atom names, which is what Meeko writes")
    parser.add_argument("--stereo-verification", type=Path,
                        help="Record of the stereochemistry check against the "
                             "stereo plan. A prerequisite for interpreting "
                             "this control")
    parser.add_argument("--independent-rmsd", type=Path,
                        help="Record of the independent toolkit RMSD "
                             "comparison. A prerequisite for interpreting "
                             "this control")
    parser.add_argument("--no-resonance-normalisation", action="store_true",
                        help="Treat terminal oxygens as distinguishable; "
                             "reported in the record and not recommended")
    args = parser.parse_args()

    try:
        run(args)
    except (chemical_graph.CorrespondenceError, chemical_component.ComponentError,
            provenance.ReceptorError, box_geometry.BoxError) as error:
        sys.exit(f"error: {error}")


def run(args: argparse.Namespace) -> None:
    if args.criterion <= 0:
        raise chemical_graph.CorrespondenceError(
            f"the criterion must be positive, got {args.criterion}")
    reference_record = provenance.read_manifest_like(args.reference, None)
    if reference_record.get("schema") != "mur-ligase-ligand-reference/1":
        raise chemical_graph.CorrespondenceError(
            f"{args.reference} is not a ligand reference record")
    reference_atoms = {
        name: tuple(entry["coordinates_angstrom"])
        for name, entry in reference_record["atom_mapping"].items()
    }
    component = chemical_component.read_component(args.component_dictionary)
    if component.identifier != reference_record["component"]:
        raise chemical_graph.CorrespondenceError(
            f"the dictionary is for {component.identifier} but the reference "
            f"is for {reference_record['component']}")

    # The symmetry that applies depends on the modelled protonation state, so
    # it is taken from the ligand manifest rather than assumed. Without one,
    # no group is treated as resonance-equivalent: an unjustified equivalence
    # is the error that matters, and the consequence of omitting a justified
    # one is a conservatively larger RMSD.
    still_protonated: list[str] | None = None
    protonation_basis = ("no ligand manifest supplied; no resonance "
                         "equivalence applied, so reported RMSD values are "
                         "upper bounds")
    if args.ligand_manifest:
        ligand_manifest = provenance.read_manifest_like(
            args.ligand_manifest, "mur-ligase-ligand/1")
        if ligand_manifest["component"] != component.identifier:
            raise chemical_graph.CorrespondenceError(
                f"the ligand manifest is for {ligand_manifest['component']} "
                f"but the dictionary is for {component.identifier}")
        deprotonated = set(
            ligand_manifest["protonation_state"]["deprotonated_atoms"])
        still_protonated = sorted(set(component.hydrogens_on) - deprotonated)
        protonation_basis = (
            f"modelled state from {args.ligand_manifest}: pH "
            f"{ligand_manifest['protonation_state']['modelled_ph']}, "
            f"formal charge "
            f"{ligand_manifest['protonation_state']['modelled_formal_charge']}, "
            f"deprotonated {sorted(deprotonated)}")
    if args.no_resonance_normalisation:
        still_protonated = None
        protonation_basis += "; resonance normalisation disabled by request"
    graph, resonance = chemical_graph.normalise_resonance(
        component.graph, hydrogen_bearing=still_protonated)
    permutations = chemical_graph.automorphisms(graph)

    # Read the campaign record first: it is what links docked poses back to
    # the prepared ligand. A pose file is Vina's output and has its own hash;
    # it is not expected to carry the prepared input's hash, and demanding
    # that would be wrong.
    run_record_path = args.run_directory / "run_record.json"
    run_record: dict[str, object] = {}
    if run_record_path.is_file():
        run_record = json.loads(run_record_path.read_text())

    names_by_serial = None
    atom_map_provenance: dict[str, object] = {
        "basis": "atom names in the pose files",
        "atom_map_record": None,
        "note": ("no --atom-map was supplied, so pose atom names were used "
                 "directly and were required to match the reference"),
    }
    if args.atom_map:
        atom_map_record = ligand_atom_map.read_atom_map(args.atom_map)
        names_by_serial = ligand_atom_map.name_by_serial(atom_map_record)
        atom_map_provenance = {
            "basis": "validated Meeko index map, serial to reference name",
            "atom_map_record": str(Path(args.atom_map).resolve()),
            "atom_map_record_sha256": box_geometry.file_sha256(args.atom_map),
            "prepared_pdbqt_sha256": (atom_map_record.get("inputs", {}) or {}).get(
                "prepared_pdbqt_sha256"),
            "n_heavy_atoms_mapped": len(names_by_serial),
            "note": ("pose files inherit the prepared ligand's atom records, "
                     "whose name column holds the element; names there are "
                     "not unique and are not used"),
            "pose_linkage": pose_linkage(atom_map_record, run_record,
                                         run_record_path),
        }
    poses = collect_poses(args.run_directory, args.ligand, names_by_serial)
    analysis = analyse(reference_atoms, graph, permutations, poses,
                       args.criterion)

    docking_context: dict[str, object] = {}
    if run_record:
        docking_context = {
            "box_centre_angstrom": run_record.get("box_centre_angstrom"),
            "box_size_angstrom": run_record.get("box_size_angstrom"),
            "seeds_requested": run_record.get("seeds"),
            "exhaustiveness": run_record.get("exhaustiveness"),
            "receptor": run_record.get("receptor"),
            "receptor_condition": run_record.get("receptor_condition"),
            "receptor_provenance_verified": run_record.get(
                "receptor_provenance_verified"),
        }

    stereo_status = validated_stereo_verification(
        args.stereo_verification, args.ligand_manifest)
    prerequisites = {
        "stereochemistry_verified": stereo_status,
        "independent_rmsd_comparison": validated_independent_rmsd(
            args.independent_rmsd, args.ligand_manifest),
    }
    interpretable = all(entry["satisfied"] for entry in prerequisites.values())

    record = {
        "schema": ANALYSIS_SCHEMA,
        "ligand": args.ligand,
        "component": component.identifier,
        "prerequisites": prerequisites,
        "interpretable": interpretable,
        "interpretability_note": (
            "the numbers below are computed regardless, but the control must "
            "not be interpreted until both prerequisites are satisfied: an "
            "unverified stereoisomer or an unvalidated RMSD implementation "
            "would make a pass or a failure equally uninformative"),
        "reference": {
            "path": str(args.reference),
            "sha256": box_geometry.file_sha256(args.reference),
            "source_structure": reference_record["source_structure"],
            "selection": reference_record["selection"],
        },
        "measure": {
            "name": "symmetry-aware heavy-atom RMSD",
            "superposition": "none; computed in the receptor coordinate frame",
            "hydrogens": "excluded",
            "resonance_normalisation": resonance,
            "resonance_normalisation_applied": not args.no_resonance_normalisation,
            "protonation_basis": protonation_basis,
            "stereochemistry": (
                "not verified by this measure; the RMSD assumes both "
                "structures are the same stereoisomer. CIP codes assigned "
                "from 3D coordinates must be compared at the preparation "
                "stage"),
            "atom_correspondence": atom_map_provenance,
            "independent_validation": (
                "this is a custom implementation whose tests are not "
                "independent of it; compare against an established toolkit "
                "before relying on the values (see docs/uma_control.md)"),
            "n_automorphisms": len(permutations),
            "orbits": chemical_graph.symmetry_classes(graph, permutations),
        },
        "docking_context": docking_context,
        "analysis": analysis,
        "interpretation_limits": [
            "recovery of the deposited pose tests the box, receptor and search "
            "settings; it does not establish that peptide poses or rankings "
            "are reliable",
            "a docking score is not a measured affinity, and these scores are "
            "not thresholds for peptides",
            "the initial control receptor has neither ANP nor manganese, so "
            "this is recovery in the simplified receptor, not validation "
            "against the native state",
            "the reference is one crystallographic conformer, and the pocket "
            "side chains are those adopted with this ligand bound",
        ],
    }

    record_path = args.out.with_name(args.out.name + ".analysis.json")
    table_path = args.out.with_name(args.out.name + "_per_pose.csv")
    provenance.write_manifest(record_path, record)
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis["per_pose"][0]))
        writer.writeheader()
        writer.writerows(analysis["per_pose"])

    print(f"ligand           : {args.ligand} ({component.identifier})")
    print(f"reference        : {args.reference}")
    print(f"measure          : symmetry-aware heavy-atom RMSD, no superposition,"
          f" {len(permutations)} automorphisms")
    print(f"criterion        : RMSD <= {args.criterion} A")
    if docking_context:
        print(f"box              : size {docking_context.get('box_size_angstrom')} A,"
              f" receptor condition {docking_context.get('receptor_condition')!r}")
    linkage = (record["measure"]["atom_correspondence"].get("pose_linkage")
               if isinstance(record["measure"]["atom_correspondence"], dict)
               else None)
    if linkage:
        if linkage.get("established"):
            print(f"pose provenance  : linked to the prepared ligand through "
                  f"the campaign record")
        else:
            print(f"pose provenance  : NOT established - {linkage.get('reason')}")
    print()
    header = (f"{'seed':>6}{'poses':>7}{'top rank':>10}{'top score':>11}"
              f"{'top RMSD':>10}{'crit':>6}{'closest rank':>14}"
              f"{'closest score':>15}{'closest RMSD':>14}{'crit':>6}")
    print(header)
    print("-" * len(header))
    for entry in analysis["per_seed"]:
        top, closest = entry["top_ranked"], entry["closest_to_reference"]
        print(f"{entry['seed']:>6}{entry['n_poses_retained']:>7}"
              f"{top['rank']:>10}"
              f"{_score(top['score_kcal_per_mol']):>11}"
              f"{top['rmsd_angstrom']:>10.3f}{top['meets_criterion']:>6}"
              f"{closest['rank']:>14}"
              f"{_score(closest['score_kcal_per_mol']):>15}"
              f"{closest['rmsd_angstrom']:>14.3f}"
              f"{closest['meets_criterion']:>6}")
    print("-" * len(header))
    summary = analysis["summary"]
    print(f"\noutcome          : {summary['outcome'].upper()}")
    print(f"  top-ranked meets the criterion in "
          f"{summary['n_seeds_top_ranked_meets_criterion']} of "
          f"{analysis['n_seeds']} seeds")
    print(f"  top-ranked RMSD range   : {summary['top_ranked_rmsd_range_angstrom']} A")
    print(f"  closest-pose RMSD range : {summary['closest_rmsd_range_angstrom']} A")
    print(f"\nwrote            : {record_path}")
    print(f"                   {table_path}")
    print()
    if interpretable:
        print("prerequisites    : stereochemistry verified and RMSD "
              "independently compared;")
        print("                   both recorded in the analysis record.")
    else:
        outstanding = [name for name, entry in prerequisites.items()
                       if not entry["satisfied"]]
        print("=" * 70)
        print("DO NOT INTERPRET THIS CONTROL YET. Outstanding prerequisites:")
        for name in outstanding:
            entry = prerequisites[name]
            print(f"  - {name}: {entry['requirement']}")
            if entry.get("reason"):
                print(f"      reason: {entry['reason']}")
        print("The RMSD values above are computed, but an unverified")
        print("stereoisomer or an unvalidated RMSD implementation makes a")
        print("pass and a failure equally uninformative. Supply")
        print("--stereo-verification and --independent-rmsd once done.")
        print("=" * 70)
    print("\nRecovery here is recovery in the simplified receptor (no ANP, no "
          "manganese).\nIt is not validation against the native state, and it "
          "says nothing about\npeptide affinity.")


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
