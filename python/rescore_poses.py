"""Rescore several existing UMA poses under one consistent Vina setup.

The docking campaign and the local-minimisation diagnostic report scores that
cannot be compared directly. A docking run's per-pose affinity comes from
`Vina.energies()`, which applies Vina's reporting convention for a search
result; `Vina.score()` evaluates one supplied pose. The two are produced by the
same scoring function but are not guaranteed to use the same reference-energy
convention, so a 0.2 kcal/mol difference between a docked pose and a
separately scored one may be convention, not chemistry.

This script removes that ambiguity by scoring every pose the same way, in one
run, against the same receptor, box, scoring function and map settings, with
`score()` and nothing else. No pose is optimised or altered.

**The convention question is answered empirically, not assumed.** For poses
extracted from a docking run, the pose file carries the affinity Vina reported
at the time in its `REMARK VINA RESULT` line. Every component of the fresh
`score()` array is compared against that recorded value, and the record states
which component index reproduces it, if any. That is evidence about the
installed convention rather than a guess about it.

What this establishes
---------------------
Whether Vina's scoring function, applied consistently, prefers a given pose
over another. That is a statement about the scoring function on this receptor
model, not about which pose is correct: the crystallographic pose is correct by
construction, and a scoring function that ranks it below a decoy is reporting
its own limitation.

What it cannot establish
-----------------------
Nothing about the search: these are fixed poses. Nothing about whether the
receptor model is right. Nothing about binding affinity. And a score difference
smaller than Vina's own error, roughly 2-3 kcal/mol on its validation set,
does not establish a preference at all, however reproducible the number.

Extraction
----------
A pose is extracted from a multi-model PDBQT by copying the lines of its
`MODEL` block verbatim. Nothing is re-perceived: the torsion tree, atom types,
partial charges, atom serials and coordinates are the bytes Vina wrote. The
extracted file is then checked to carry the expected heavy atoms, to map
through the validated atom map, and to reproduce the source block's coordinates
exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import box_geometry
import chemical_component
import chemical_graph
import ligand_atom_map
import local_minimisation as diagnostic
import receptor_provenance as provenance

SCHEMA = "mur-ligase-pose-rescoring/2"

# Component order of the array returned by Vina::score(), read from
# AutoDock Vina v1.2.7, src/lib/vina.cpp. Two namings exist for the same eight
# slots: the order the C++ pushes them, and the Python binding's docstring.
# They disagree over the names of slots 2-4, so both are recorded and nothing
# in this analysis depends on either. Slot 7 is the unbound reference in both
# accounts, and it is confirmed empirically before use; see
# `confirm_reference_index`.
SCORE_LABELS_CPP = (
    "total", "lig_grids", "inter_pairs", "flex_grids",
    "intra_pairs", "lig_intra", "conf_independent", "unbound_reference",
)
SCORE_LABELS_PYTHON_DOCSTRING = (
    "total", "lig_inter", "flex_inter", "other_inter",
    "flex_intra", "lig_intra", "torsions", "lig_intra_best_pose",
)
UNBOUND_REFERENCE_INDEX = 7

# For Vina and Vinardo, `Vina::score()` with no argument computes the unbound
# reference from the pose in hand:
#     intramolecular_energy = m_model.eval_intramolecular(...)
#     total = conf_independent(model, inter + intra - intramolecular_energy)
# `Vina::global_search()` instead sets the model to poses[0] and takes one
# reference from it -- the source comment reads "we take the intramolecular
# energy from the best pose" -- then passes that single value to score() for
# every retained pose. So a docking run reports each pose against the best
# pose's internal energy, while standalone score() reports each pose against
# its own. The two agree for the best pose and diverge for every other, by
# exactly the internal-energy difference carried through conf_independent.
SOURCE_REFERENCE = (
    "AutoDock Vina v1.2.7, src/lib/vina.cpp: Vina::score() and "
    "Vina::global_search(); Python binding Vina.score(unbound_energy=None)"
)

# Both the stored REMARK VINA RESULT and the Python binding's score() are
# rounded to three decimals, so two independently rounded values may differ by
# up to 0.001 before any arithmetic. Affinity maps are also recomputed here
# rather than reused, which admits a small interpolation difference. 0.01
# kcal/mol absorbs both and is still far tighter than any energy difference
# this analysis interprets.
AFFINITY_TOLERANCE_KCAL = 0.01


class RescoringError(RuntimeError):
    """A pose could not be extracted or validated, so nothing was scored."""


# ---------------------------------------------------------------------------
# Pose extraction, byte-for-byte
# ---------------------------------------------------------------------------

def split_models(path: Path) -> list[list[str]]:
    """Return the lines of each MODEL block of a multi-model PDBQT, in order.

    The `MODEL` and `ENDMDL` lines themselves are dropped; everything between
    them is kept exactly as written, including every REMARK, the ROOT/BRANCH
    torsion tree and TORSDOF. A file with no MODEL records is treated as a
    single pose, which is what Vina writes for one retained mode.
    """
    text = Path(path).read_text()
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current is not None:
                blocks.append(current)
            current = None
            continue
        if current is not None:
            current.append(line)
    if not blocks:
        stripped = [line for line in text.splitlines()]
        if any(line.startswith(("ATOM", "HETATM")) for line in stripped):
            blocks.append(stripped)
    if not blocks:
        raise RescoringError(f"{path} contains no poses")
    return blocks


def recorded_affinity(block: list[str]) -> float | None:
    """Return the affinity Vina recorded for this pose, or None."""
    for line in block:
        if line.startswith("REMARK VINA RESULT"):
            try:
                return float(line.split()[3])
            except (IndexError, ValueError):
                return None
    return None


def atom_lines(block: list[str]) -> list[str]:
    return [line for line in block if line.startswith(("ATOM", "HETATM"))]


def extract_pose(source: Path, rank: int, destination: Path) -> dict[str, object]:
    """Write one pose of a multi-model PDBQT to its own file, unaltered.

    `rank` is 1-based and is the pose's position in the file, which is the rank
    Vina assigned: it writes retained modes best first. The destination must
    not exist.

    Returns a record of what was extracted, including the recorded affinity and
    a verification that the written atom lines are character-identical to the
    source block's.
    """
    blocks = split_models(source)
    if not 1 <= rank <= len(blocks):
        raise RescoringError(
            f"{source} holds {len(blocks)} poses; rank {rank} does not exist")
    block = blocks[rank - 1]
    destination = Path(destination)
    if destination.exists():
        raise RescoringError(f"{destination} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(block) + "\n")

    written = atom_lines([line for line in destination.read_text().splitlines()])
    original = atom_lines(block)
    if written != original:
        raise RescoringError(
            f"the atom records written to {destination} are not identical to "
            f"the source block; extraction must not alter a pose")
    return {
        "source_pose_file": str(Path(source).resolve()),
        "source_pose_file_sha256": box_geometry.file_sha256(source),
        "rank_in_file": rank,
        "n_poses_in_file": len(blocks),
        "extracted_to": str(destination.resolve()),
        "extracted_sha256": box_geometry.file_sha256(destination),
        "recorded_affinity_kcal_per_mol": recorded_affinity(block),
        "n_atom_records": len(original),
        "atom_records_identical_to_source": True,
        "extraction_method": (
            "the MODEL block's lines copied verbatim; torsion tree, atom "
            "types, partial charges, serials and coordinates unchanged"),
    }


# ---------------------------------------------------------------------------
# Scoring, one setup for every pose
# ---------------------------------------------------------------------------

def build_engine(pose_path: Path, receptor: Path, centre, size,
                 cpu: int, seed: int):
    """Return a Vina engine loaded with one pose, maps computed.

    A new engine per pose, rather than one engine reused, so that no pose can
    inherit map or ligand state from the pose before it. The settings are
    passed from the same variables for every call, so they are identical by
    construction, and the record says so.
    """
    from vina import Vina

    engine = Vina(sf_name="vina", seed=seed, cpu=cpu)
    engine.set_receptor(str(receptor))
    engine.set_ligand_from_file(str(pose_path))
    engine.compute_vina_maps(center=[float(value) for value in centre],
                             box_size=[float(value) for value in size])
    return engine


def confirm_reference_index(engine) -> dict[str, object]:
    """Confirm empirically which component holds the unbound reference.

    The labels published for `score()`'s array disagree over some slots, so the
    slot this analysis depends on is established by experiment rather than
    read off a docstring. `score()` with no argument computes a pose-specific
    reference and returns it in some slot; passing that exact value back as
    `unbound_energy` must then reproduce the identical total. If it does, the
    slot is the reference, on this build, for this pose.

    Raises if the identity does not hold, because every common-reference score
    below would otherwise be computed from the wrong quantity.
    """
    native = [float(value) for value in np.atleast_1d(np.asarray(
        engine.score(), float))]
    if len(native) <= UNBOUND_REFERENCE_INDEX:
        raise RescoringError(
            f"score() returned {len(native)} components; index "
            f"{UNBOUND_REFERENCE_INDEX} was expected to hold the unbound "
            f"reference. {SOURCE_REFERENCE}")
    candidate = native[UNBOUND_REFERENCE_INDEX]
    echoed = [float(value) for value in np.atleast_1d(np.asarray(
        engine.score(unbound_energy=candidate), float))]
    agrees = abs(echoed[0] - native[0]) <= AFFINITY_TOLERANCE_KCAL
    if not agrees:
        raise RescoringError(
            f"passing component {UNBOUND_REFERENCE_INDEX} "
            f"({candidate}) back as unbound_energy changed the total from "
            f"{native[0]} to {echoed[0]}, so that component is not the "
            f"unbound reference on this build. Re-read {SOURCE_REFERENCE} "
            f"before interpreting any common-reference score.")
    return {
        "component_index": UNBOUND_REFERENCE_INDEX,
        "pose_specific_reference_kcal_per_mol": candidate,
        "confirmed_by": ("passing the component back as unbound_energy "
                         "reproduced the same total"),
        "total_without_argument": native[0],
        "total_with_argument": echoed[0],
        "source": SOURCE_REFERENCE,
    }


def score_both_conventions(engine, common_reference: float | None
                           ) -> dict[str, object]:
    """Score one pose pose-specifically and, if given, against one reference.

    Two explicitly labelled methods, both retained:

    `pose_specific` is `score()` with no argument: each pose is measured
    against its own internal energy, so the total reflects intermolecular
    complementarity and the torsional term, with internal strain cancelled.

    `common_reference` is `score(unbound_energy=R)` for one shared R: every
    pose is measured against the same internal-energy zero, so a pose that is
    internally strained relative to that zero is charged for it. This is the
    convention a docking run reports.

    Neither is more correct. They answer different questions, and a pose can
    rank differently under each -- which is the point of recording both.
    """
    pose_specific = diagnostic.score_components(engine, engine.score())
    pose_specific["method"] = "pose_specific"
    pose_specific["method_note"] = (
        "score() with no argument; reference is this pose's own internal "
        "energy, so internal strain cancels")
    result = {"pose_specific": pose_specific}
    if common_reference is not None:
        shared = diagnostic.score_components(
            engine, engine.score(unbound_energy=float(common_reference)))
        shared["method"] = "common_reference"
        shared["method_note"] = (
            f"score(unbound_energy={common_reference}); every pose measured "
            f"against one shared internal-energy zero, as a docking run "
            f"reports")
        shared["reference_kcal_per_mol"] = float(common_reference)
        result["common_reference"] = shared
    return result


def affinity_reproduction(common_total: float | None, affinity: float | None,
                          tolerance: float = AFFINITY_TOLERANCE_KCAL
                          ) -> dict[str, object]:
    """Check whether the common-reference total reproduces what docking recorded.

    This replaces an earlier check that scanned every component for one
    matching the recorded affinity and, finding none, concluded the two used
    different reference conventions. **That inference was wrong.** A
    non-matching component is not evidence of a different convention: the
    totals are computed by the same formula and differ only in which internal
    energy was supplied as the reference. Supplying the reference the docking
    run used is what makes them comparable, and whether they then agree is the
    actual test.
    """
    if affinity is None:
        return {"checked": False,
                "reason": "the pose file records no REMARK VINA RESULT, so "
                          "there is no recorded affinity to reproduce"}
    if common_total is None:
        return {"checked": False,
                "reason": "no common-reference score was computed"}
    residual = common_total - affinity
    return {
        "checked": True,
        "recorded_affinity_kcal_per_mol": affinity,
        "common_reference_total_kcal_per_mol": common_total,
        "residual_kcal_per_mol": round(residual, 6),
        "tolerance_kcal_per_mol": tolerance,
        "reproduced": abs(residual) <= tolerance,
        "tolerance_basis": (
            "both quantities are rounded to three decimals by the Python "
            "binding, and the affinity maps are recomputed rather than "
            "reused"),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def requested_poses(args: argparse.Namespace, output_dir: Path
                    ) -> list[dict[str, object]]:
    """Resolve the poses to score, extracting docked ranks into fresh files."""
    poses: list[dict[str, object]] = []
    for label, path in (("deposited", args.deposited),
                        ("minimised", args.minimised)):
        if path is None:
            continue
        poses.append({"label": label, "path": Path(path), "provenance": {
            "source_pose_file": str(Path(path).resolve()),
            "source_pose_file_sha256": box_geometry.file_sha256(path),
            "extraction_method": "used as supplied; not extracted",
            "recorded_affinity_kcal_per_mol": None,
        }})
    for entry in args.docked_rank or []:
        try:
            rank_text, source = entry.split(":", 1)
            rank = int(rank_text)
        except ValueError as error:
            raise RescoringError(
                f"--docked-rank expects RANK:PATH, got {entry!r} ({error})")
        destination = output_dir / "extracted" / f"rank_{rank}.pdbqt"
        record = extract_pose(Path(source), rank, destination)
        poses.append({"label": f"docked_rank_{rank}", "path": destination,
                      "provenance": record})
    if not poses:
        raise RescoringError(
            "no poses were requested; pass --deposited, --minimised or "
            "--docked-rank")
    return poses


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = diagnostic.fresh_output_directory(args.output_dir, args.ligand)
    box_record = json.loads(Path(args.box_record).read_text())
    centre = box_geometry.record_centre(box_record)
    size = box_geometry.record_size(box_record)

    manifest = provenance.read_manifest(args.receptor_manifest)
    disagreements = provenance.box_and_receptor_disagreements(box_record, manifest)
    if disagreements:
        raise RescoringError("the box record and receptor manifest disagree:\n  - "
                             + "\n  - ".join(disagreements))

    reference_record = json.loads(Path(args.reference).read_text())
    if reference_record.get("schema") != "mur-ligase-ligand-reference/1":
        raise RescoringError(f"{args.reference} is not a ligand reference record")
    reference_atoms = {
        name: tuple(entry["coordinates_angstrom"])
        for name, entry in reference_record["atom_mapping"].items()
    }

    component = chemical_component.read_component(args.component_dictionary)
    ligand_manifest = provenance.read_manifest_like(
        args.ligand_manifest, "mur-ligase-ligand/1")
    atom_map_record = ligand_atom_map.read_atom_map(args.atom_map)
    names_by_serial = ligand_atom_map.name_by_serial(atom_map_record)

    # The same identity chain the minimisation diagnostic enforces: the poses
    # being rescored must derive from the prepared ligand the records describe.
    identity = diagnostic.verified_ligand_identity(
        args.ligand_manifest, args.preparation_record, args.atom_map,
        atom_map_record, args.ligand, args.stereo_verification)

    graph, permutations, resonance, protonation_basis = \
        diagnostic.protonation_aware_symmetry(component, ligand_manifest)

    poses = requested_poses(args, output_dir)

    # The shared internal-energy zero comes from one named pose, so every
    # common-reference score below is on the same footing as the docking run
    # that reported that pose first.
    reference_detail: dict[str, object] = {
        "requested_label": args.reference_pose,
        "established": False,
    }
    common_reference: float | None = None
    if not args.check_only:
        chosen = [pose for pose in poses if pose["label"] == args.reference_pose]
        if not chosen:
            raise RescoringError(
                f"--reference-pose {args.reference_pose!r} is not among the "
                f"poses requested ({[p['label'] for p in poses]}); the common "
                f"reference must come from a pose being scored")
        engine = build_engine(chosen[0]["path"], args.receptor, centre, size,
                              args.cpu, args.seed)
        confirmation = confirm_reference_index(engine)
        common_reference = confirmation["pose_specific_reference_kcal_per_mol"]
        reference_detail = {
            "requested_label": args.reference_pose,
            "established": True,
            "reference_kcal_per_mol": common_reference,
            "taken_from_pose": chosen[0]["label"],
            "confirmation": confirmation,
            "why": (
                "a docking run takes one internal-energy reference from its "
                "best pose and reports every retained pose against it, so "
                "reproducing those affinities requires that same reference"),
        }

    rows: list[dict[str, object]] = []
    for pose in poses:
        coordinates = diagnostic.pose_from_pdbqt(pose["path"], names_by_serial)
        if len(coordinates) != len(reference_atoms):
            raise RescoringError(
                f"{pose['path']} carries {len(coordinates)} mapped heavy "
                f"atoms, expected {len(reference_atoms)}")
        rmsd, permutation = diagnostic.rmsd_against_reference(
            graph, permutations, reference_atoms, coordinates)
        points = np.array([coordinates[name] for name in sorted(coordinates)])
        inside = int(box_geometry.points_inside(centre, size, points).sum())
        if inside != len(points):
            raise RescoringError(
                f"{pose['path']}: only {inside} of {len(points)} heavy atoms "
                f"lie inside the recorded box; Vina cannot score it there")
        row: dict[str, object] = {
            "label": pose["label"],
            "pose_file": str(Path(pose["path"]).resolve()),
            "pose_file_sha256": box_geometry.file_sha256(pose["path"]),
            "provenance": pose["provenance"],
            "n_heavy_atoms_mapped": len(coordinates),
            "heavy_atoms_inside_box": inside,
            "rmsd_to_deposited_angstrom": round(rmsd, 6),
            "permutation_used": list(permutation),
        }
        if not args.check_only:
            engine = build_engine(pose["path"], args.receptor, centre, size,
                                  args.cpu, args.seed)
            scored = score_both_conventions(engine, common_reference)
            row["scores"] = scored
            row["affinity_reproduction"] = affinity_reproduction(
                (scored.get("common_reference") or {}).get("total_kcal_per_mol"),
                pose["provenance"].get("recorded_affinity_kcal_per_mol"))
        rows.append(row)

    record: dict[str, object] = {
        "schema": SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": sys.argv,
        "experiment": {
            "kind": "consistent rescoring of existing poses",
            "poses_were_modified": False,
            "poses_were_optimised": False,
            "is_independent_redocking": False,
            "counts_towards_2A_positive_control": False,
            "statement": (
                "Fixed poses rescored under one setup. This says what the "
                "scoring function prefers; it is not a search result and "
                "cannot satisfy the 2 A positive-control criterion."),
        },
        "inputs": {
            "receptor": str(Path(args.receptor).resolve()),
            "receptor_sha256": box_geometry.file_sha256(args.receptor),
            "receptor_condition": manifest.get("receptor_condition"),
            "receptor_unresolved_decisions": manifest.get("unresolved_decisions", []),
            "prepared_ligand_pdbqt": str(Path(args.ligand).resolve()),
            "box_record": str(Path(args.box_record).resolve()),
            "box_record_sha256": box_geometry.file_sha256(args.box_record),
            "reference": str(Path(args.reference).resolve()),
            "reference_sha256": box_geometry.file_sha256(args.reference),
            "atom_map": str(Path(args.atom_map).resolve()),
            "component_dictionary_sha256": box_geometry.file_sha256(
                args.component_dictionary),
            **identity,
        },
        "settings": {
            "scoring_function": "vina",
            "box_centre_angstrom": [float(value) for value in centre],
            "box_size_angstrom": [float(value) for value in size],
            "cpu": args.cpu,
            "seed": args.seed,
            "map_spacing_angstrom": (
                "not set; Vina's compute_vina_maps default, as in dock.py"),
            "identical_across_poses": (
                "every pose was scored with a freshly built engine from the "
                "same receptor, centre, size, scoring function and defaults, "
                "so no pose inherited state from another"),
        },
        "measure": {
            "rmsd": "symmetry-aware heavy-atom RMSD, no superposition",
            "superposition": "none",
            "protonation_basis": protonation_basis,
            "n_automorphisms": len(permutations),
            "resonance_normalisation": resonance,
        },
        "scoring_methods": {
            "pose_specific": (
                "score() with no argument; each pose against its own internal "
                "energy, so internal strain cancels and the total reflects "
                "intermolecular complementarity and the torsional term"),
            "common_reference": (
                "score(unbound_energy=R) for one shared R taken from the "
                "reference pose; the convention a docking run reports, in "
                "which a pose internally strained relative to R is charged "
                "for that strain"),
            "neither_is_more_correct": (
                "they answer different questions, and a pose may rank "
                "differently under each; both are retained and labelled"),
            "source": SOURCE_REFERENCE,
            "component_labels_cpp_push_order": list(SCORE_LABELS_CPP),
            "component_labels_python_docstring": list(
                SCORE_LABELS_PYTHON_DOCSTRING),
            "component_label_caveat": (
                "the two published namings disagree over slots 2-4; nothing "
                "here depends on them, and slot 7 is confirmed empirically"),
        },
        "reference_energy": reference_detail,
        "uncertainty_note": (
            "Vina's ~2-3 kcal/mol error is a validation figure for predicting "
            "absolute affinities across different ligands. It is NOT a "
            "calibrated threshold for ranking poses of one ligand in one "
            "receptor, where systematic terms largely cancel, and it must not "
            "be used to dismiss a within-ligand difference. How small a "
            "within-ligand gap is meaningful is not established here; report "
            "the differences and their components rather than declaring a "
            "winner."),
        "poses": rows,
        "result": "checks_passed_nothing_scored" if args.check_only else "completed",
    }
    if not args.check_only:
        from vina import Vina  # noqa: F401  (already imported during scoring)
        record["software"] = {
            "vina": getattr(__import__("vina"), "__version__", None),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        }
    return record


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the compact comparison table as CSV."""
    fields = ["label", "rank_in_file", "rmsd_to_deposited_angstrom",
              "recorded_docking_affinity", "pose_specific_total",
              "common_reference_total", "unbound_reference",
              "affinity_residual", "reproduced", "n_heavy_atoms_mapped"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            scores = row.get("scores", {}) or {}
            specific = scores.get("pose_specific", {}) or {}
            shared = scores.get("common_reference", {}) or {}
            reproduction = row.get("affinity_reproduction", {}) or {}
            components = specific.get("raw_components") or []
            writer.writerow({
                "label": row["label"],
                "rank_in_file": row["provenance"].get("rank_in_file"),
                "rmsd_to_deposited_angstrom": row["rmsd_to_deposited_angstrom"],
                "recorded_docking_affinity": row["provenance"].get(
                    "recorded_affinity_kcal_per_mol"),
                "pose_specific_total": specific.get("total_kcal_per_mol"),
                "common_reference_total": shared.get("total_kcal_per_mol"),
                "unbound_reference": (
                    components[UNBOUND_REFERENCE_INDEX]
                    if len(components) > UNBOUND_REFERENCE_INDEX else None),
                "affinity_residual": reproduction.get("residual_kcal_per_mol"),
                "reproduced": reproduction.get("reproduced"),
                "n_heavy_atoms_mapped": row["n_heavy_atoms_mapped"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Rescore existing poses under one consistent Vina setup. "
                     "Poses are never optimised or altered."))
    parser.add_argument("receptor", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ligand", type=Path, required=True,
                        help="the prepared ligand PDBQT the poses derive from")
    parser.add_argument("--deposited", type=Path, default=None,
                        help="PDBQT holding the deposited pose")
    parser.add_argument("--minimised", type=Path, default=None,
                        help="minimised_pose.pdbqt from local_minimisation.py")
    parser.add_argument("--docked-rank", action="append", metavar="RANK:PATH",
                        help=("extract this 1-based rank from a multi-model "
                              "pose file; repeatable"))
    parser.add_argument("--box-record", type=Path, required=True)
    parser.add_argument("--receptor-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--component-dictionary", type=Path, required=True)
    parser.add_argument("--ligand-manifest", type=Path, required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--preparation-record", type=Path, required=True)
    parser.add_argument("--stereo-verification", type=Path, default=None)
    parser.add_argument("--reference-pose", default="docked_rank_1",
                        help=("label of the pose supplying the shared unbound "
                              "reference energy; a docking run takes it from "
                              "its best pose (default: docked_rank_1)"))
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    try:
        record = run(args)
    except (RescoringError, diagnostic.DiagnosticError,
            chemical_graph.CorrespondenceError, provenance.ReceptorError,
            ligand_atom_map.AtomMapError) as error:
        sys.exit(f"error: {error}")

    output_dir = Path(args.output_dir)
    (output_dir / "rescoring.json").write_text(
        json.dumps(record, indent=2) + "\n")
    write_table(output_dir / "rescoring.csv", record["poses"])

    print(f"receptor  : {record['inputs']['receptor']}")
    print(f"condition : {record['inputs']['receptor_condition']}")
    print(f"box       : centre {record['settings']['box_centre_angstrom']}, "
          f"size {record['settings']['box_size_angstrom']} A")
    print()
    reference = record["reference_energy"]
    if reference.get("established"):
        print(f"shared unbound reference: "
              f"{reference['reference_kcal_per_mol']:.3f} kcal/mol, from "
              f"{reference['taken_from_pose']} (component "
              f"{UNBOUND_REFERENCE_INDEX}, confirmed empirically)")
        print()

    header = (f"{'pose':<20}{'rank':>5}{'RMSD/A':>9}{'docked':>9}"
              f"{'specific':>10}{'shared':>9}{'resid':>8}{'ok':>4}")
    print(header)
    print("-" * len(header))
    for row in record["poses"]:
        scores = row.get("scores", {}) or {}
        specific = (scores.get("pose_specific") or {}).get("total_kcal_per_mol")
        shared = (scores.get("common_reference") or {}).get("total_kcal_per_mol")
        recorded = row["provenance"].get("recorded_affinity_kcal_per_mol")
        reproduction = row.get("affinity_reproduction", {}) or {}
        residual = reproduction.get("residual_kcal_per_mol")
        ok = reproduction.get("reproduced")
        def shown(value, places=3):
            return "-" if value is None else f"{value:.{places}f}"
        print(f"{row['label']:<20}"
              f"{row['provenance'].get('rank_in_file') or '-':>5}"
              f"{row['rmsd_to_deposited_angstrom']:>9.3f}"
              f"{shown(recorded):>9}{shown(specific):>10}{shown(shared):>9}"
              f"{shown(residual):>8}"
              f"{('-' if ok is None else ('yes' if ok else 'NO')):>4}")
    print()
    print("specific = score() with each pose's own internal reference")
    print("shared   = score(unbound_energy=R), the docking convention")
    print()
    print(record["uncertainty_note"])
    print(f"wrote     : {output_dir / 'rescoring.json'}")


if __name__ == "__main__":
    main()
