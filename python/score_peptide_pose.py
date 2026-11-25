"""Score a placed peptide with AutoDock Vina, and optionally relax it locally.

This is stage four of the peptide scoring pilot, after Meeko has written the
ligand PDBQT. It evaluates the BoltzGen pose as given, and separately reports
what happens when Vina is allowed to relax it locally. The two results are
never merged: one is a property of the prediction, the other is a property of
the prediction plus Vina's local search.

    python python/score_peptide_pose.py \
        --ligand ligands/spec_0574/spec_0574_ligand.pdbqt \
        --receptor receptors/murc_anp_prepared.pdbqt \
        --box-record examples/box_murc_uma_chainE_30A.json \
        --design-id spec_0574 --out-dir scores/spec_0574 \
        --seed 1 --cpu 2 --local-optimisation

**This stage requires AutoDock Vina and runs only where Vina is installed.**
On a machine without it, every function here raises rather than returning a
degraded result, because a missing scoring function is a blocker and not a
check that can be skipped.

No search is performed
----------------------
`Vina.score()` evaluates the supplied coordinates. `Vina.optimize()` performs a
*local* optimisation from them. Neither explores the box, and this script never
calls `dock()`. That is deliberate: an eight-to-sixteen-residue peptide carries
far more torsional freedom than a drug-like ligand, and an unrestrained global
search over that space is not expected to be reliable. The pose under test is
the one BoltzGen predicted, so searching for a different one would discard the
hypothesis.

What local optimisation moves, and the trap in it
-------------------------------------------------
Vina relaxes the rigid-body placement **and every torsion in the ligand's
torsion tree**. For a peptide that includes the backbone: phi and psi are
ordinary rotatable single bonds, and Meeko makes them torsions by default.
Measured on `spec_0574`, an eight-residue peptide, 15 of its 26 rotatable
bonds are backbone phi/psi and 11 are side-chain.

So an unqualified local optimisation **does not preserve the predicted
backbone**. It is not a side-chain relaxation, and reporting it as one would be
wrong. To hold the backbone rigid, the backbone torsions must be removed at
Meeko time with `--rigidify_bonds_smarts`/`--rigidify_bonds_indices`; see
`project_private/peptide_pilot_spec_0574_2026-11-25.md`. This script records the torsion
count from the PDBQT's own TORSDOF line, and the displacement it measures is
reported per atom and split by backbone and side chain, so a run that moved
the backbone cannot be mistaken for one that did not.

The receptor is rigid throughout, as in the campaign. A clash that the real
complex would relieve by moving a receptor side chain cannot be relieved here.

What a score means
------------------
A Vina score is a model score in kcal/mol, not a measured binding free energy
and not evidence of binding. Vina's own validation error is roughly 2-3
kcal/mol, so a difference smaller than that does not establish a preference
however reproducibly it is reported. Scores for a peptide are additionally
outside the chemical space Vina's scoring function was fitted on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import peptide_geometry  # noqa: E402
import receptor_provenance  # noqa: E402

try:
    from vina import Vina

    VINA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where Vina is absent
    VINA_AVAILABLE = False

SCHEMA = "mur-ligase-peptide-score/1"

BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})


class ScoringError(RuntimeError):
    """Raised when a peptide pose cannot be scored as specified."""


def require_vina() -> None:
    if not VINA_AVAILABLE:
        raise ScoringError(
            "AutoDock Vina's Python bindings are not importable in this "
            "interpreter. This stage runs only where Vina is installed; it is "
            "a blocker, not a skippable check."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(131072), b""):
            digest.update(block)
    return digest.hexdigest()


def load_box_record(path: Path) -> Tuple[List[float], List[float], Dict]:
    """Read and validate a recorded docking box.

    The box is read from the project's recorded JSON rather than passed as
    numbers, so a run cannot silently use a box nobody wrote down. Returns
    `(centre, size, record)`.
    """
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ScoringError(f"cannot read box record {path}: {error}") from error
    for key in ("centre_angstrom", "size_angstrom"):
        if key not in record:
            raise ScoringError(f"box record {path} has no {key!r}")
    centre = [float(v) for v in record["centre_angstrom"]]
    size = [float(v) for v in record["size_angstrom"]]
    if len(centre) != 3 or len(size) != 3:
        raise ScoringError("box centre and size must each have three values")
    if not all(np.isfinite(centre)) or not all(v > 0 for v in size):
        raise ScoringError("box centre must be finite and dimensions positive")
    return centre, size, record


def pdbqt_torsion_count(path: Path) -> Optional[int]:
    """Read the TORSDOF line from a PDBQT, which governs what Vina searches.

    Returns `None` when the file carries no TORSDOF line, rather than
    substituting a count computed some other way: the number that matters is
    the one in the file Vina reads.
    """
    for line in path.read_text().splitlines():
        if line.startswith("TORSDOF"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def pose_positions(path: Path) -> Dict[receptor_provenance.AtomKey, np.ndarray]:
    """Read a pose's atom positions, keyed by full atom identity.

    Atoms are keyed on identity rather than file order, because Vina's written
    pose need not preserve ordering and a positional comparison would then
    report motion that did not happen.
    """
    try:
        records = receptor_provenance.parse_atom_records(path)
    except receptor_provenance.ReceptorError as error:
        raise ScoringError(f"cannot read pose {path}: {error}") from error
    return {
        key: np.asarray(record.position, dtype=float)
        for key, record in records.items()
    }


def displacement_report(before: Path, after: Path) -> Dict[str, object]:
    """Measure how far each atom moved between two poses.

    The split by backbone and side chain is the point: a local optimisation
    that relieved a side-chain clash while leaving the predicted backbone in
    place is a different result from one that reshaped the peptide, and a
    single RMSD cannot tell them apart.

    Hydrogens are excluded, because Vina's scoring function ignores them and
    their positions are idealised rather than predicted.
    """
    first, second = pose_positions(before), pose_positions(after)
    shared = sorted(set(first) & set(second), key=str)
    only_before = sorted(set(first) - set(second), key=str)
    only_after = sorted(set(second) - set(first), key=str)

    backbone: List[float] = []
    side_chain: List[float] = []
    per_atom: List[Dict[str, object]] = []
    for key in shared:
        if key.atom_name.upper().startswith("H"):
            continue
        distance = float(np.linalg.norm(second[key] - first[key]))
        is_backbone = key.atom_name.upper() in BACKBONE_ATOM_NAMES
        (backbone if is_backbone else side_chain).append(distance)
        per_atom.append(
            {
                "atom": str(key),
                "moved_angstrom": round(distance, 4),
                "backbone": is_backbone,
            }
        )
    per_atom.sort(key=lambda entry: entry["moved_angstrom"], reverse=True)

    def statistics(values: List[float]) -> Dict[str, object]:
        if not values:
            return {
                "atoms": 0,
                "rmsd_angstrom": None,
                "mean_angstrom": None,
                "maximum_angstrom": None,
            }
        array = np.asarray(values, dtype=float)
        return {
            "atoms": int(array.size),
            "rmsd_angstrom": round(float(np.sqrt((array ** 2).mean())), 4),
            "mean_angstrom": round(float(array.mean()), 4),
            "maximum_angstrom": round(float(array.max()), 4),
        }

    everything = backbone + side_chain
    return {
        "all_heavy_atoms": statistics(everything),
        "backbone": statistics(backbone),
        "side_chain": statistics(side_chain),
        "largest_movements": per_atom[:15],
        "atoms_only_in_before": [str(k) for k in only_before],
        "atoms_only_in_after": [str(k) for k in only_after],
        "paired_on": "full atom identity, not file order",
        "note": (
            "Hydrogens are excluded: Vina ignores them and their positions "
            "were idealised, not predicted."
        ),
        "interpretation": (
            "A backbone RMSD near zero means the local optimisation relaxed "
            "side chains and placement only. A substantial backbone RMSD "
            "means the predicted backbone was altered, so the relaxed pose is "
            "no longer the pose BoltzGen predicted."
        ),
    }


def pose_box_containment(pose: Path, box_record: Dict) -> Dict[str, object]:
    """Check a pose lies inside the box Vina will build maps for.

    An atom outside the box is not merely untidy: Vina's grid maps are only
    defined inside it, so a score computed with atoms outside is not the score
    of this pose.
    """
    positions = pose_positions(pose)
    heavy = np.asarray(
        [
            value
            for key, value in positions.items()
            if not key.atom_name.upper().startswith("H")
        ],
        dtype=float,
    )
    if heavy.size == 0:
        raise ScoringError(f"{pose} has no heavy atoms to check against the box")
    containment = peptide_geometry.box_containment(heavy, box_record)
    containment["consequence"] = (
        "Vina's affinity maps are defined only inside the box. A score "
        "computed while atoms lie outside it is not a score of this pose."
        if not containment["fully_contained"]
        else "Every heavy atom lies inside the box used to build the maps."
    )
    return containment


def build_record(
    design_id: str,
    ligand: Path,
    receptor: Path,
    box_record_path: Path,
    box_record: Dict,
    centre: Sequence[float],
    size: Sequence[float],
    seed: int,
    cpu: int,
    containment: Dict[str, object],
    torsions: Optional[int],
    initial: Dict[str, object],
    optimised: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Assemble the scoring record, keeping the two results separate."""
    record: Dict[str, object] = {
        "schema": SCHEMA,
        "design_id": design_id,
        "inputs": {
            "ligand_pdbqt": str(ligand),
            "ligand_pdbqt_sha256": file_sha256(ligand),
            "receptor_pdbqt": str(receptor),
            "receptor_pdbqt_sha256": file_sha256(receptor),
            "box_record": str(box_record_path),
            "box_record_sha256": file_sha256(box_record_path),
        },
        "box": {
            "centre_angstrom": list(centre),
            "size_angstrom": list(size),
            "spacing_angstrom": (
                "not set by this script; Vina's compute_vina_maps default, as "
                "in dock.py"
            ),
        },
        "scoring_parameters": {
            "scoring_function": "vina",
            "seed": seed,
            "cpu": cpu,
            "search_performed": False,
            "note": (
                "score() evaluates the supplied pose; optimize() relaxes it "
                "locally. dock() is never called, so no conformational search "
                "over the box is performed."
            ),
        },
        "ligand_torsions": {
            "torsdof_from_pdbqt": torsions,
            "source": "the TORSDOF line of the ligand PDBQT",
            "caveat": (
                "For a peptide, backbone phi and psi are ordinary rotatable "
                "bonds and become torsions unless rigidified at Meeko time. "
                "A high TORSDOF here means local optimisation can reshape the "
                "predicted backbone."
            ),
        },
        "box_containment_before_scoring": containment,
        "initial_pose": initial,
        "after_local_optimisation": optimised,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "interpretation": (
            "Vina scores are model scores in kcal/mol, not measured binding "
            "free energies and not evidence of binding. Vina's validation "
            "error is roughly 2-3 kcal/mol, so smaller differences do not "
            "establish a preference. The receptor was rigid throughout."
        ),
    }
    if VINA_AVAILABLE:  # pragma: no cover - depends on the installation
        try:
            import vina

            record["software"]["vina"] = getattr(vina, "__version__", "unknown")
        except Exception:
            record["software"]["vina"] = "unknown"
    return record


def run(args: argparse.Namespace) -> Dict[str, object]:  # pragma: no cover
    """Score, and optionally relax, one peptide pose. Requires Vina."""
    require_vina()
    import local_minimisation

    for path in (args.ligand, args.receptor, args.box_record):
        if not path.is_file():
            raise ScoringError(f"input not found: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.out_dir / f"{args.design_id}_score.json"
    optimised_path = args.out_dir / f"{args.design_id}_optimised.pdbqt"
    for path in (record_path, optimised_path):
        if path.exists() and not args.overwrite:
            raise ScoringError(
                f"Refusing to overwrite {path}. Choose a fresh output "
                "directory, or pass --overwrite deliberately."
            )

    centre, size, box_record = load_box_record(args.box_record)
    containment = pose_box_containment(args.ligand, box_record)
    if not containment["fully_contained"] and not args.allow_outside_box:
        raise ScoringError(
            "The pose has heavy atoms outside the box, so Vina's maps do not "
            "cover it and the resulting score would not be a score of this "
            "pose. Per-axis overshoot (A): "
            f"{containment['per_axis_overshoot_angstrom']}. Enlarge the box "
            "and record the new one, or pass --allow-outside-box to record "
            "the score with this limitation stated."
        )

    engine = Vina(sf_name="vina", seed=args.seed, cpu=args.cpu)
    engine.set_receptor(str(args.receptor))
    engine.set_ligand_from_file(str(args.ligand))
    engine.compute_vina_maps(center=list(centre), box_size=list(size))

    initial = local_minimisation.score_components(engine, engine.score())
    initial["pose"] = "as supplied; no atom was moved before scoring"

    optimised: Optional[Dict[str, object]] = None
    if args.local_optimisation:
        relaxed_values = (
            engine.optimize(max_steps=args.max_steps)
            if args.max_steps is not None
            else engine.optimize()
        )
        optimised = local_minimisation.score_components(engine, relaxed_values)
        engine.write_pose(str(optimised_path), overwrite=False)
        optimised["pose_file"] = str(optimised_path)
        optimised["pose_file_sha256"] = file_sha256(optimised_path)
        optimised["max_steps"] = args.max_steps
        optimised["displacement"] = displacement_report(
            args.ligand, optimised_path
        )
        optimised["box_containment_after"] = pose_box_containment(
            optimised_path, box_record
        )
        if (
            optimised["displacement"]["all_heavy_atoms"]["maximum_angstrom"]
            == 0.0
        ):
            optimised["warning"] = (
                "No atom moved. Either the pose was already at a local "
                "minimum, or the installed write_pose() wrote the input pose "
                "rather than the optimised one. Check the installation before "
                "interpreting this."
            )

    torsions = pdbqt_torsion_count(args.ligand)
    record = build_record(
        args.design_id,
        args.ligand,
        args.receptor,
        args.box_record,
        box_record,
        centre,
        size,
        args.seed,
        args.cpu,
        containment,
        torsions,
        initial,
        optimised,
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    record["record_path"] = str(record_path)
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--box-record", type=Path, required=True)
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument(
        "--local-optimisation",
        action="store_true",
        help="Also relax the pose locally with Vina.optimize() and report the "
        "result separately from the initial score",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap on local-optimisation steps; Vina's default when omitted",
    )
    parser.add_argument(
        "--allow-outside-box",
        action="store_true",
        help="Record a score although atoms lie outside the box, with that "
        "limitation stated in the record",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        record = run(args)
    except ScoringError as error:
        print(f"scoring failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(record, indent=2))
    print(f"record: {record['record_path']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
