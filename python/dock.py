"""Dock ligands with AutoDock Vina across several random seeds.

Recorded-box mode (preferred):
    python python/dock.py RECEPTOR.pdbqt LIGAND_DIR OUTPUT_DIR \
        --box-record boxes/murc_uma_chainE_24A.json [--seeds 1,42,137]

Legacy mode (retained, deprecated):
    python python/dock.py RECEPTOR.pdbqt LIGAND_DIR CX,CY,CZ SIZE [nseed] [cpu]

The two modes differ in more than syntax:

*Recorded-box mode* reads the centre and the three independent dimensions from
the record written by `map_pocket.py --box-record`, so no coordinate is
transcribed by hand. It runs preliminary sanity checks of the box against the
receptor it was given — not a validation of that receptor — and writes poses
and scores for **every** seed into a fresh output directory, required to be
outside the ligand directory, so no docked pose can be rediscovered as a
ligand input on a later run. The output directory must not already contain
anything: results from separate runs are never merged.

Every ligand and seed is validated before being recorded as a result, the run
exits non-zero if any requested docking failed, and partial ligands are marked
as such rather than averaged in silently.

*Legacy mode* reproduces the previous behaviour: a hand-typed centre, a single
cubic size, poses from the first seed only, and results written into the ligand
directory. It is kept so earlier runs remain reproducible. It prints a warning
and should not be used for new work.

Multi-seed is the default, and deliberately so. Vina's search is stochastic, so
a single-seed number is not a measurement; the seed spread is reported beside
every mean so that differences smaller than it are not over-read. The spread
figures previously quoted in this docstring described one historical system and
have been removed: assess the spread of the campaign actually run.

`--cpu` limits threads per docking. Vina otherwise spawns one thread per
exhaustiveness pass, which took 32 of 48 cores on a shared machine.

Docking scores are model-dependent scores. They are not measured binding free
energies and do not establish binding, inhibition or antibacterial activity.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry
import receptor_provenance

DEFAULT_SEEDS = (1, 42, 137)
LEGACY_SEEDS = (1, 42, 137, 2024, 31337)
DEFAULT_EXHAUSTIVENESS = 32
DEFAULT_CPU = 8
DEFAULT_POSES = 9


# ---------------------------------------------------------------------------
# Receptor checks that need no Vina
# ---------------------------------------------------------------------------

RECEPTOR_CHECK_CAVEAT = (
    "NOTE the receptor checks above are preliminary sanity checks only. They\n"
    "     confirm that the box centre falls inside the receptor's coordinate\n"
    "     range and that the recorded author chain is present. They do NOT\n"
    "     verify how the receptor was prepared, which structure it came from,\n"
    "     or that its coordinates are consistent with the box's source\n"
    "     structure and frame. Verified receptor provenance and coordinate\n"
    "     consistency remain prerequisites for real docking and belong to the\n"
    "     receptor-preparation stage."
)


def read_pdbqt_atoms(path: Path) -> tuple[np.ndarray, set[str]]:
    """Return receptor atom coordinates in A and the chain IDs present.

    Parses `ATOM`/`HETATM` records by PDB column positions, which `.pdbqt`
    preserves: x, y, z in columns 31-54 and the chain ID in column 22. This
    reads geometry only: it does not validate atom typing, charges, or how the
    file was produced.
    """
    coordinates: list[list[float]] = []
    chains: set[str] = set()
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            coordinates.append([float(line[30:38]), float(line[38:46]),
                                float(line[46:54])])
        except ValueError as error:
            raise SystemExit(f"error: malformed coordinate line in {path}: {error}")
        chain = line[21:22].strip()
        if chain:
            chains.add(chain)
    if not coordinates:
        raise SystemExit(f"error: no ATOM/HETATM records in {path}")
    array = np.array(coordinates, dtype=float)
    if not np.isfinite(array).all():
        raise SystemExit(f"error: non-finite receptor coordinates in {path}")
    return array, chains


def check_box_against_receptor(record: dict, receptor: Path) -> list[str]:
    """Preliminary sanity checks that the box and the receptor are not mismatched.

    These are cheap, necessary checks, not validation. Two gross failures are
    caught, both of which previously ran to completion and returned scores:

    - a box centre outside the receptor's own coordinate range, which is what a
      mistyped centre or a box taken from a different protein copy looks like.
      1P3D's two UMA sites are ~99 A apart, so this is not hypothetical;
    - a receptor that does not contain the author protein chain the box was
      mapped on.

    What is **not** checked, and must not be inferred from these passing: that
    the receptor was prepared from the recorded source structure, that its
    coordinates are unchanged from that structure's frame, that the chain
    label means the same thing in both, or that preparation preserved the
    pocket. A receptor prepared from a different structure of the same protein
    would pass both checks. Verified receptor provenance and coordinate
    consistency are prerequisites for real docking and belong to the
    receptor-preparation stage; see `docs/docking_box.md`.

    Returns advisory notes for conditions worth reporting but not fatal.
    """
    centre = box_geometry.record_centre(record)
    size = box_geometry.record_size(record)
    atoms, chains = read_pdbqt_atoms(receptor)

    lower, upper = atoms.min(0), atoms.max(0)
    outside = np.maximum(np.maximum(lower - centre, centre - upper), 0.0)
    if (outside > 0).any():
        raise SystemExit(
            f"error: the box centre {np.round(centre, 2).tolist()} lies "
            f"{float(np.linalg.norm(outside)):.1f} A "
            f"outside the coordinate range of {receptor}\n"
            f"       receptor spans {np.round(lower, 1).tolist()} to "
            f"{np.round(upper, 1).tolist()} A\n"
            f"       the box record names author protein chain "
            f"{record.get('selection', {}).get('protein_author_chain_id')!r} of "
            f"{record.get('source_structure', {}).get('path')!r}; prepare the "
            f"receptor from that structure and that chain."
        )

    notes: list[str] = []
    expected_chain = record.get("selection", {}).get("protein_author_chain_id")
    if expected_chain and chains and expected_chain not in chains:
        raise SystemExit(
            f"error: the box was mapped on author protein chain {expected_chain!r}, "
            f"but {receptor} contains chain(s) {sorted(chains)}.\n"
            f"       Prepare the receptor from the recorded chain, or map the box "
            f"on the chain actually prepared."
        )
    if not chains:
        notes.append(
            f"{receptor} carries no chain identifier, so the recorded author chain "
            f"{expected_chain!r} could not be confirmed."
        )

    inside = int(box_geometry.points_inside(centre, size, atoms).sum())
    notes.append(
        f"{inside} of {len(atoms)} receptor atoms lie within the box volume; "
        f"receptor atoms outside the box are not removed from the receptor."
    )
    return notes


def check_receptor_provenance(record: dict, receptor: Path,
                              manifest_path: Path | None,
                              allow_unverified: bool) -> tuple[dict | None, list[str]]:
    """Confirm the box and the receptor describe the same protein copy.

    With a receptor manifest from `prepare_receptor.py`, this compares the
    source structure's SHA-256, the author protein chain, the pocket cutoff and
    residue list, and that the receptor file being docked is the one the
    manifest describes. Any disagreement stops the run.

    Without a manifest the run is refused unless `--allow-unverified-receptor`
    is passed, which is recorded in the run record. The provenance of a
    receptor cannot be recovered from a `.pdbqt`: it carries no source
    structure, no hash and no account of what was removed from it.

    Returns the manifest (or None) and advisory notes.
    """
    if manifest_path is None:
        if not allow_unverified:
            raise SystemExit(
                "error: no --receptor-manifest given, so the receptor's "
                "provenance cannot be verified against the box.\n"
                "       Prepare the receptor with python/prepare_receptor.py, "
                "which writes a manifest, or pass "
                "--allow-unverified-receptor deliberately; that choice is "
                "recorded in the run record."
            )
        return None, [
            "the receptor's provenance was NOT verified: no manifest was "
            "given and --allow-unverified-receptor was passed. Nothing "
            "establishes that this receptor came from the box's source "
            "structure or protein copy."
        ]

    manifest = receptor_provenance.read_manifest(manifest_path)
    problems = receptor_provenance.box_and_receptor_disagreements(
        record, manifest, receptor_sha256=box_geometry.file_sha256(receptor))
    if problems:
        raise SystemExit(
            "error: the box record and the receptor manifest do not agree:\n"
            + "\n".join(f"       - {problem}" for problem in problems)
            + "\n       Map the box and prepare the receptor from the same "
              "structure and the same protein copy."
        )
    condition = manifest.get("receptor_condition")
    unresolved = manifest.get("unresolved_decisions", [])
    return manifest, [
        f"receptor provenance agrees with the box: same source structure, "
        f"author chain and pocket list; receptor condition {condition!r}.",
        f"the manifest records {len(unresolved)} unresolved preparation "
        f"decision(s); agreement of provenance is not validation of the "
        f"preparation.",
    ]


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------

def collect_ligands(ligand_dir: Path) -> list[Path]:
    """Return the ligand `.pdbqt` files to dock, excluding previous outputs.

    A file whose name ends in `_docked` is a previous Vina output, not an
    input; finding one here means an earlier run wrote into this directory, so
    the run is stopped rather than silently redocking poses.
    """
    files = sorted(Path(ligand_dir).glob("*.pdbqt"))
    if not files:
        raise SystemExit(f"error: no .pdbqt files in {ligand_dir}")
    docked = [path.name for path in files if path.stem.endswith("_docked")]
    if docked:
        raise SystemExit(
            f"error: {ligand_dir} contains docked output from an earlier run "
            f"({docked[:3]}{'...' if len(docked) > 3 else ''}).\n"
            f"       Dock from a directory of inputs only, and send outputs to a "
            f"separate directory."
        )
    return files


def prepare_output_dir(output_dir: Path, ligand_dir: Path) -> Path:
    """Create a fresh output directory that cannot be read back as inputs.

    Two requirements, both refusals rather than warnings:

    - the output directory must not be the ligand directory, nor inside it,
      because docked `.pdbqt` files placed beside the inputs are picked up as
      ligands by the next run;
    - it must not already exist with anything in it. Reuse is refused outright
      rather than offered behind a flag: a rerun into a populated directory
      leaves poses from the previous run next to the new tables, and rewrites
      the score and failure tables around them, so what the directory holds no
      longer describes one campaign.
    """
    output_dir = Path(output_dir).resolve()
    ligand_dir = Path(ligand_dir).resolve()
    if output_dir == ligand_dir or ligand_dir in output_dir.parents:
        raise SystemExit(
            f"error: the output directory {output_dir} is the ligand directory or "
            f"inside it. Choose a separate directory so docked poses cannot be "
            f"rediscovered as ligand inputs."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"error: {output_dir} already exists and is not empty. Choose a fresh "
            f"output directory; results from separate runs are not merged."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validated_scores(energies, n_poses: int, ligand: Path, seed: int) -> list[float]:
    """Return finite per-pose scores, or raise so the seed is recorded as failed.

    Vina's bindings return a row per pose with the total score first.

    **A smaller pose set than requested is not a failure.** Vina documents
    `num_modes` as the *maximum* number of binding modes to output, not a
    guarantee: the count is additionally limited by `energy_range` (3 kcal/mol
    by default) and by a redundancy filter that requires reported modes to
    differ by at least 1 A RMSD over movable heavy atoms. Fewer poses
    therefore means the search found fewer distinct modes worth reporting,
    which is legitimate output. Only an *empty* result, an unreadable table or
    a non-finite score is treated as a failure, because a NaN carried into the
    summary would be averaged into a mean and reported as a number.
    """
    try:
        scores = [float(row[0]) for row in energies]
    except (TypeError, IndexError, ValueError) as error:
        raise ValueError(f"unreadable energy table: {error}") from error
    if not scores:
        raise ValueError("Vina returned no poses")
    if not all(np.isfinite(score) for score in scores):
        raise ValueError(f"non-finite scores returned: {scores}")
    return scores


def validated_pose_file(pose_path: Path, existed_before: bool) -> int:
    """Return the written pose file's size, or raise if it is not usable.

    The file must be new to this call and non-empty. A pose file that already
    existed is a stale pose from an earlier run, and an empty one means Vina
    wrote nothing; neither may be recorded as a success.
    """
    if existed_before:
        raise ValueError(f"{pose_path} already existed before this docking")
    if not pose_path.is_file():
        raise ValueError(f"no pose file was written to {pose_path}")
    size = pose_path.stat().st_size
    if size == 0:
        raise ValueError(f"the pose file {pose_path} is empty")
    return size


def dock_one(ligand: Path, receptor: Path, centre, size, seed: int,
             cpu: int, exhaustiveness: int, n_poses: int,
             pose_path: Path | None, overwrite_poses: bool = False):
    """Dock one ligand with one seed, write its poses, and return the scores.

    Returns the per-pose score list in Vina's own order (best first), validated
    finite and complete, having confirmed that a fresh non-empty pose file was
    written. Any of those checks failing raises, so the caller records the
    ligand and seed as a failure instead of a result.

    In recorded-box mode `pose_path` is unique per ligand and seed, so no
    seed's poses are discarded; `pose_path=None` suppresses writing and the
    pose check, which only legacy mode uses.
    """
    from vina import Vina  # imported here so argument checks need no Vina

    engine = Vina(sf_name="vina", seed=seed, cpu=cpu)
    engine.set_receptor(str(receptor))
    engine.set_ligand_from_file(str(ligand))
    engine.compute_vina_maps(center=[float(value) for value in centre],
                             box_size=[float(value) for value in size])
    engine.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    scores = validated_scores(engine.energies(n_poses=n_poses), n_poses,
                              ligand, seed)
    if pose_path is not None:
        existed_before = pose_path.exists()
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        engine.write_poses(str(pose_path), n_poses=n_poses,
                           overwrite=overwrite_poses)
        if not overwrite_poses:
            validated_pose_file(pose_path, existed_before)
    return scores


def vina_version() -> str | None:
    """Return the installed Vina version string, or None if not determinable."""
    try:
        import vina
    except ImportError:
        return None
    for attribute in ("__version__", "VERSION", "version"):
        value = getattr(vina, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def run_campaign(*, receptor: Path, ligands: list[Path], output_dir: Path,
                 centre, size, seeds: list[int], cpu: int,
                 exhaustiveness: int, n_poses: int,
                 record: dict | None, record_path: Path | None,
                 notes: list[str], manifest: dict | None = None,
                 manifest_path: Path | None = None) -> int:
    """Dock every ligand with every seed, persisting results as they arrive.

    Returns the number of requested ligand/seed combinations that failed, so
    the caller can exit non-zero: a campaign that produced some scores and
    some failures is not a successful campaign, and a later reader of the
    tables has no way to know a run was meant to cover more.

    Nothing is held only in memory. The run record is written before docking
    starts, every pose row is appended and flushed as it is produced, and the
    summary and failure tables are rewritten after each ligand, so an
    interrupted run leaves the bookkeeping for everything completed so far.

    Output layout, all inside `output_dir`:
        box_record.json     the box record used, copied verbatim
        run_record.json     command, input hashes, seeds, settings, status
        per_seed_scores.csv one row per ligand, seed and pose, appended live
        dock_results.csv    one row per ligand, with the seed spread
        poses/seed_<N>/<ligand>_docked.pdbqt
        failures.csv        one row per ligand/seed that did not complete
    """
    if record_path is not None:
        shutil.copyfile(record_path, output_dir / "box_record.json")

    run_record_path = output_dir / "run_record.json"
    run_record = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": sys.argv,
        "receptor": str(Path(receptor).resolve()),
        "receptor_sha256": box_geometry.file_sha256(receptor),
        "receptor_manifest": (
            None if manifest_path is None else str(Path(manifest_path).resolve())),
        "receptor_provenance_verified": manifest is not None,
        "receptor_condition": (
            None if manifest is None else manifest.get("receptor_condition")),
        "receptor_unresolved_decisions": (
            [] if manifest is None else manifest.get("unresolved_decisions", [])),
        "receptor_manifest_contents": manifest,
        "ligand_inputs": [
            {"path": str(path.resolve()), "sha256": box_geometry.file_sha256(path)}
            for path in ligands
        ],
        "seeds": list(seeds),
        "exhaustiveness": exhaustiveness,
        "cpu": cpu,
        "n_poses_requested": n_poses,
        "n_poses_note": (
            "n_poses is Vina's maximum, not a guarantee; energy_range and the "
            "1 A redundancy filter can return fewer, which is not a failure"
        ),
        "box_centre_angstrom": [float(value) for value in centre],
        "box_size_angstrom": [float(value) for value in size],
        "box_record": None if record_path is None else str(Path(record_path).resolve()),
        "box_record_sha256": (
            None if record_path is None else box_geometry.file_sha256(record_path)),
        "box_record_contents": record,
        "vina_version": vina_version(),
        "numpy_version": np.__version__,
        "receptor_checks": notes,
        "receptor_check_scope": (
            "preliminary sanity checks only: box centre within the receptor's "
            "coordinate range and the recorded author chain present. The "
            "receptor's preparation provenance and coordinate consistency with "
            "the recorded source structure are NOT verified here."
        ),
        "n_ligands": len(ligands),
        "n_requested_dockings": len(ligands) * len(seeds),
    }
    write_run_record(run_record_path, run_record)

    failures: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    with (output_dir / "per_seed_scores.csv").open("w", newline="") as pose_handle:
        pose_writer = csv.DictWriter(
            pose_handle, fieldnames=["ligand", "seed", "pose_rank",
                                     "n_poses_returned", "n_poses_requested",
                                     "score_kcal_per_mol", "pose_file",
                                     "pose_file_bytes"])
        pose_writer.writeheader()
        pose_handle.flush()

        for ligand in ligands:
            name = ligand.stem
            best_by_seed: dict[int, float] = {}
            for seed in seeds:
                pose_path = (output_dir / "poses" / f"seed_{seed}"
                             / f"{name}_docked.pdbqt")
                try:
                    scores = dock_one(ligand, receptor, centre, size, seed, cpu,
                                      exhaustiveness, n_poses, pose_path)
                    pose_bytes = pose_path.stat().st_size
                except Exception as error:                  # noqa: BLE001
                    # Batch continuation is intentional, but every failure is
                    # recorded per ligand and seed rather than skipped, and the
                    # run exits non-zero at the end.
                    failures.append({"ligand": name, "seed": seed,
                                     "error": f"{type(error).__name__}: {error}"})
                    print(f"  FAILED {name} seed {seed}: {error}",
                          file=sys.stderr, flush=True)
                    continue
                best_by_seed[seed] = scores[0]
                for rank, score in enumerate(scores, start=1):
                    pose_writer.writerow({
                        "ligand": name, "seed": seed, "pose_rank": rank,
                        "n_poses_returned": len(scores),
                        "n_poses_requested": n_poses,
                        "score_kcal_per_mol": round(score, 3),
                        "pose_file": str(pose_path.relative_to(output_dir)),
                        "pose_file_bytes": pose_bytes,
                    })
                pose_handle.flush()

            summary_rows.append(summarise_ligand(name, best_by_seed, seeds))
            print(describe_ligand(summary_rows[-1]), flush=True)
            # Rewritten per ligand so an interruption still leaves the tables
            # describing every ligand that finished.
            write_csv(output_dir / "dock_results.csv", summary_rows)
            write_csv(output_dir / "failures.csv", failures)

    run_record.update({
        "status": "completed" if not failures else "completed with failures",
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_failures": len(failures),
        "n_ligands_with_all_seeds": sum(
            1 for row in summary_rows if row["result"] == "complete"),
    })
    write_run_record(run_record_path, run_record)

    report(summary_rows, failures, output_dir, seeds)
    return len(failures)


def summarise_ligand(name: str, best_by_seed: dict[int, float],
                     requested_seeds: list[int]) -> dict[str, object]:
    """Summarise one ligand's per-seed best scores, marking partial results.

    A ligand missing any requested seed is marked `partial`, and one with no
    seed at all `failed`. The spread statistics are left blank for a single
    completed seed rather than reported as zero: a standard deviation of 0.00
    over one observation describes the sample size, not the reproducibility of
    the score, and reads as the opposite of what it means.
    """
    seeds_done = sorted(best_by_seed)
    values = np.array([best_by_seed[seed] for seed in seeds_done], dtype=float)
    missing = [seed for seed in requested_seeds if seed not in best_by_seed]
    result = ("complete" if not missing
              else "failed" if not seeds_done else "partial")
    estimable = len(seeds_done) > 1
    return {
        "ligand": name,
        "result": result,
        "n_seeds_requested": len(requested_seeds),
        "n_seeds_completed": len(seeds_done),
        "seeds_completed": " ".join(str(seed) for seed in seeds_done),
        "seeds_failed": " ".join(str(seed) for seed in missing),
        "best_score_mean": "" if not seeds_done else round(float(values.mean()), 3),
        "best_score_sd": round(float(values.std(ddof=1)), 3) if estimable else "",
        "best_score_seed_range": round(float(np.ptp(values)), 3) if estimable else "",
        "spread_estimable": "yes" if estimable else "no",
        "best_score_per_seed": " ".join(
            f"{best_by_seed[seed]:.3f}" for seed in seeds_done),
    }


def describe_ligand(row: dict[str, object]) -> str:
    """One console line per ligand, naming partial and failed results."""
    if row["result"] == "failed":
        return (f"{row['ligand']:<24}  FAILED: no seed completed "
                f"(requested {row['n_seeds_requested']})")
    spread = (f"sd {row['best_score_sd']:>5}  seed range "
              f"{row['best_score_seed_range']:>5}"
              if row["spread_estimable"] == "yes"
              else "spread not estimable from one seed")
    marker = "" if row["result"] == "complete" else (
        f"  PARTIAL: {row['n_seeds_completed']}/{row['n_seeds_requested']} seeds, "
        f"failed {row['seeds_failed']}")
    return f"{row['ligand']:<24}{float(row['best_score_mean']):8.2f}  {spread}{marker}"


def write_run_record(path: Path, record: dict[str, object]) -> None:
    """Write the run record, replacing any earlier version of it."""
    Path(path).write_text(json.dumps(record, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows as CSV, including a header-only file when there are none.

    An empty `failures.csv` is written deliberately: its absence would be
    indistinguishable from a run that never checked.
    """
    fieldnames = list(rows[0]) if rows else ["note"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report(summary_rows: list[dict[str, object]],
           failures: list[dict[str, object]], output_dir: Path,
           seeds: list[int]) -> None:
    """Print the campaign summary, the seed-spread caution and any shortfall."""
    complete = [row for row in summary_rows if row["result"] == "complete"]
    partial = [row for row in summary_rows if row["result"] == "partial"]
    failed = [row for row in summary_rows if row["result"] == "failed"]
    print(f"\n{f'ligands with all {len(seeds)} seeds':<26}: {len(complete)}")
    print(f"{'ligands partially docked':<26}: {len(partial)}")
    print(f"{'ligands with no result':<26}: {len(failed)}")
    print(f"{'ligand/seed failures':<26}: {len(failures)}")
    print(f"{'outputs':<26}: {output_dir}")
    if partial or failed:
        print("\nWARNING: this campaign is incomplete. Partial ligands rest on "
              "fewer\nseeds than the others, so their means are not comparable "
              "with the rest.\nSee failures.csv; the exit status is non-zero.")
    scored = [row for row in summary_rows if row["best_score_mean"] != ""]
    if not scored:
        return
    means = np.array([float(row["best_score_mean"]) for row in scored])
    ranges = [float(row["best_score_seed_range"]) for row in scored
              if row["spread_estimable"] == "yes"]
    span = float(means.max() - means.min())
    print(f"\nspread across ligands : {span:.2f} kcal/mol")
    if not ranges:
        print("largest seed range    : not estimable; no ligand completed "
              "more than one seed")
        return
    worst = max(ranges)
    print(f"largest seed range    : {worst:.2f} kcal/mol "
          f"(over {len(ranges)} ligands with more than one seed)")
    if worst > span * 0.5:
        print("\nWARNING: seed variation is comparable to the difference")
        print("between ligands. Treat this as a coarse tier, not a ranking.")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_seeds(text: str) -> list[int]:
    """Parse a comma-separated seed list, preserving order and rejecting repeats."""
    try:
        seeds = [int(field.strip()) for field in text.split(",") if field.strip()]
    except ValueError:
        raise SystemExit(f"error: --seeds must be integers, got {text!r}")
    if not seeds:
        raise SystemExit("error: --seeds must name at least one seed")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"error: --seeds contains repeats: {seeds}")
    return seeds


def main_recorded_box(argv: list[str]) -> None:
    """Recorded-box mode: read the box from its record and dock every seed."""
    parser = argparse.ArgumentParser(
        prog="dock.py", description="Dock ligands against a recorded search box.")
    parser.add_argument("receptor", type=Path, help="Prepared receptor .pdbqt")
    parser.add_argument("ligand_dir", type=Path, help="Directory of ligand .pdbqt")
    parser.add_argument("output_dir", type=Path,
                        help="Fresh output directory, outside the ligand directory")
    parser.add_argument("--box-record", type=Path, required=True,
                        help="Box record from map_pocket.py --box-record")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help=f"Comma-separated seeds (default "
                             f"{','.join(str(s) for s in DEFAULT_SEEDS)})")
    parser.add_argument("--exhaustiveness", type=int, default=DEFAULT_EXHAUSTIVENESS)
    parser.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    parser.add_argument("--n-poses", type=int, default=DEFAULT_POSES,
                        help=f"Maximum poses per docking (default "
                             f"{DEFAULT_POSES}); Vina may return fewer")
    parser.add_argument("--receptor-manifest", type=Path,
                        help="Manifest from prepare_receptor.py; its "
                             "provenance must agree with the box record")
    parser.add_argument("--allow-unverified-receptor", action="store_true",
                        help="Dock without a receptor manifest. The run record "
                             "marks the receptor as unverified")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the preliminary checks, then stop")
    args = parser.parse_args(argv)

    for value, flag in ((args.exhaustiveness, "--exhaustiveness"),
                        (args.cpu, "--cpu"), (args.n_poses, "--n-poses")):
        if value < 1:
            raise SystemExit(f"error: {flag} must be at least 1, got {value}")
    if not args.receptor.is_file():
        raise SystemExit(f"error: receptor not found: {args.receptor}")
    if not args.ligand_dir.is_dir():
        raise SystemExit(f"error: ligand directory not found: {args.ligand_dir}")

    seeds = parse_seeds(args.seeds)
    try:
        record = box_geometry.read_record(args.box_record)
    except box_geometry.BoxError as error:
        raise SystemExit(f"error: {error}")
    centre = box_geometry.record_centre(record)
    size = box_geometry.record_size(record)

    ligands = collect_ligands(args.ligand_dir)
    notes = check_box_against_receptor(record, args.receptor)
    try:
        manifest, provenance_notes = check_receptor_provenance(
            record, args.receptor, args.receptor_manifest,
            args.allow_unverified_receptor)
    except receptor_provenance.ReceptorError as error:
        raise SystemExit(f"error: {error}")
    notes.extend(provenance_notes)
    output_dir = prepare_output_dir(args.output_dir, args.ligand_dir)

    print(f"receptor   : {args.receptor}")
    print(f"box record : {args.box_record}")
    print(f"box        : {box_geometry.describe(record)}")
    print(f"ligands    : {len(ligands)}")
    print(f"seeds      : {seeds}   exhaustiveness {args.exhaustiveness}")
    print(f"outputs    : {output_dir}")
    for note in notes:
        print(f"NOTE {note}")
    print(RECEPTOR_CHECK_CAVEAT)
    # Flushed so this header cannot appear after failure lines on stderr when
    # both streams are redirected to the same file.
    print(flush=True)

    if args.check_only:
        print("--check-only: preliminary checks passed; nothing docked.")
        return

    n_failures = run_campaign(
        receptor=args.receptor, ligands=ligands, output_dir=output_dir,
        centre=centre, size=size, seeds=seeds, cpu=args.cpu,
        exhaustiveness=args.exhaustiveness, n_poses=args.n_poses,
        record=record, record_path=args.box_record, notes=notes,
        manifest=manifest, manifest_path=args.receptor_manifest)
    if n_failures:
        raise SystemExit(
            f"error: {n_failures} of {len(ligands) * len(seeds)} requested "
            f"ligand/seed dockings failed; see {output_dir / 'failures.csv'}")


def main_legacy(argv: list[str]) -> None:
    """Legacy mode: hand-typed cubic box, poses from the first seed only.

    Retained unchanged in behaviour so that earlier runs stay reproducible.
    Results are written into the ligand directory, which means a rerun can
    rediscover them as inputs; recorded-box mode exists to avoid that.
    """
    if len(argv) < 4:
        sys.exit(__doc__)
    receptor, ligand_dir = Path(argv[0]), Path(argv[1])
    try:
        centre = [float(value) for value in argv[2].split(",")]
        size = float(argv[3])
    except ValueError as error:
        sys.exit(f"error: could not parse the legacy centre or size: {error}")
    if len(centre) != 3 or not all(np.isfinite(value) for value in centre):
        sys.exit(f"error: legacy centre must be three finite numbers, got {argv[2]!r}")
    if not np.isfinite(size) or size <= 0:
        sys.exit(f"error: legacy size must be positive and finite, got {argv[3]!r}")
    n_seeds = int(argv[4]) if len(argv) > 4 else 3
    cpu = int(argv[5]) if len(argv) > 5 else DEFAULT_CPU
    if not 1 <= n_seeds <= len(LEGACY_SEEDS):
        sys.exit(f"error: legacy mode has {len(LEGACY_SEEDS)} seeds available")

    print("WARNING: legacy mode. The centre and size are hand-typed and unchecked,")
    print("         the box is forced cubic, poses are saved for the first seed")
    print("         only, and results are written into the ligand directory where")
    print("         a rerun can rediscover them as inputs. Prefer --box-record.")
    print()

    seeds = list(LEGACY_SEEDS[:n_seeds])
    ligands = sorted(ligand_dir.glob("*.pdbqt"))
    if not ligands:
        sys.exit(f"error: no .pdbqt files in {ligand_dir}")

    print(f"receptor : {receptor}")
    print(f"box      : centre {centre}, size {size} A (cubic)")
    print(f"ligands  : {len(ligands)}")
    print(f"seeds    : {seeds}\n")

    rows = []
    for ligand in ligands:
        name = ligand.stem
        values = []
        pose_spread = float("nan")
        for seed in seeds:
            first_seed = seed == seeds[0]
            scores = dock_one(
                ligand, receptor, centre, [size] * 3, seed, cpu,
                DEFAULT_EXHAUSTIVENESS, DEFAULT_POSES,
                ligand_dir / f"{name}_docked.pdbqt" if first_seed else None,
                overwrite_poses=True,
            )
            values.append(scores[0])
            if seed == seeds[0]:
                pose_spread = scores[0] - scores[-1]
        array = np.array(values)
        rows.append({"ligand": name, "mean": round(float(array.mean()), 3),
                     "sd": round(float(array.std()), 3),
                     "seed_range": round(float(np.ptp(array)), 3),
                     "pose_spread": round(float(pose_spread), 3),
                     "seed_scores": " ".join(f"{value:.3f}" for value in values)})
        print(f"{name:<24}{array.mean():8.2f}  sd {array.std():5.2f}  "
              f"seed range {np.ptp(array):5.2f}", flush=True)

    rows.sort(key=lambda row: row["mean"])
    write_csv(ligand_dir / "dock_results.csv", rows)
    print(f"\nwrote {ligand_dir / 'dock_results.csv'}")

    # Legacy mode keeps its own summary: it has no per-seed pose files, no
    # failure table and no partial results, because a failure here raises.
    means = np.array([row["mean"] for row in rows], dtype=float)
    worst = max(float(row["seed_range"]) for row in rows)
    span = float(means.max() - means.min())
    print(f"spread across ligands : {span:.2f} kcal/mol")
    print(f"largest seed range    : {worst:.2f} kcal/mol")
    if len(seeds) == 1:
        print("NOTE one seed only: the seed range above is not an estimate of "
              "reproducibility.")
    if worst > span * 0.5:
        print("\nWARNING: seed variation is comparable to the difference")
        print("between ligands. Treat this as a coarse tier, not a ranking.")


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        sys.exit(__doc__)
    if "--box-record" in argv:
        main_recorded_box(argv)
    else:
        main_legacy(argv)


if __name__ == "__main__":
    main()
