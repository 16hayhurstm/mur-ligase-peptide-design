"""Score and locally minimise the deposited UMA pose in a prepared receptor.

This is a **diagnostic**, not a docking experiment. It starts from the
crystallographic ligand coordinates, asks what Vina's scoring function makes of
them, and then lets Vina relax them locally. It answers one question:

    is the crystallographic pose a local minimum of the model we are scoring
    with, and what does it score there?

**What the two outcomes do and do not license.** A pose that stays put and
scores competitively shows that the experimental mode is a local minimum of
this model. It does **not** show that the receptor model is scientifically
sound, and it does not establish that the recovery failure must be due to
conformational search: minimisation only probes the basin it starts in, so it
cannot see that some other basin has become deeper, or that interactions
removed during preparation would have made this basin the global optimum. A
pose that drifts or scores poorly shows the experimental mode is not a
minimum of this model, but does not identify which preparation choice caused
that — the rigid receptor, the unreviewed protonation assignment, the omitted
cofactors, the omitted waters and the scoring function all remain candidates.

The result narrows the hypothesis space. It does not close it, and no single
cause may be read off it.

**What this can never be.** Local minimisation starts from the answer. It is
not independent redocking and it cannot satisfy the project's <= 2.0 A
positive-control criterion, however small the RMSD it reports. Every record
written here says so in its own fields, so a reader who finds the file without
this docstring cannot mistake it for a recovered control.

Scope and limitations
---------------------
- The receptor is rigid, as in the campaign. Local minimisation relaxes the
  ligand only, so a native pose could be penalised by a receptor side chain
  that the real complex would have moved.
- Vina's scoring function ignores hydrogens and has no electrostatic term.
  Scores are model scores, not binding free energies.
- `score()` and the per-pose affinities from a docking run are produced by the
  same scoring function, but they are not guaranteed to be assembled from the
  same component terms in every Vina release. This script records every
  component it is given, so the comparison can be made on matching terms
  rather than assumed to be like-for-like. See `comparability_note` in the
  record.
- Nothing here validates the omitted cofactor or metal chemistry, or the
  receptor's protonation assignment. Those are separate questions.

Provenance
----------
Nothing is rebuilt. The prepared receptor, prepared ligand PDBQT, box record,
ligand manifest, atom map and reference pose are all read as they are, hashed,
and recorded. The ligand manifest is checked through the same
`manifest_hash_agreement` gate the rest of the pipeline uses, so a manifest
whose identity has drifted is refused here too.

Before touching Vina, the script re-derives the RMSD of the *input* pose
against the deposited reference through the full atom-map and symmetry path. The
prepared PDBQT preserves the deposited heavy-atom coordinates, so that value
must be essentially zero. It is a free end-to-end test of the whole
correspondence chain, and a non-zero value aborts the run rather than being
reported alongside a minimisation result that could not be interpreted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_redocking
import box_geometry
import chemical_component
import chemical_graph
import ligand_atom_map
import receptor_provenance as provenance

SCHEMA = "mur-ligase-local-minimisation/1"

# The prepared PDBQT preserves the deposited heavy-atom coordinates, and PDBQT
# writes three decimals, so the input pose must reproduce the reference to
# within rounding. Anything larger is a broken correspondence chain, not a
# tolerance to be widened.
INPUT_POSE_TOLERANCE_ANGSTROM = 0.01

NOT_A_CONTROL = (
    "Local minimisation starts from the deposited coordinates. It is NOT "
    "independent redocking and cannot satisfy the <= 2.0 A positive-control "
    "criterion, whatever RMSD it reports."
)


class DiagnosticError(RuntimeError):
    """A prerequisite failed, so no diagnostic was run."""


def fresh_output_directory(output_dir: Path, ligand: Path) -> Path:
    """Return a usable fresh output directory, or raise.

    Mirrors `dock.py`'s refusals: results from separate runs are never merged,
    and outputs never land beside the inputs where a later run could read them
    back as ligands.
    """
    output_dir = Path(output_dir).resolve()
    ligand_dir = Path(ligand).resolve().parent
    if output_dir == ligand_dir or ligand_dir in output_dir.parents:
        raise DiagnosticError(
            f"{output_dir} is the ligand directory or inside it; choose an "
            f"output directory outside {ligand_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DiagnosticError(
            f"{output_dir} already exists and is not empty; choose a fresh "
            f"output directory so no existing result is overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def verified_ligand_identity(manifest_path: Path, preparation_path: Path,
                             atom_map_path: Path, atom_map_record: dict,
                             prepared_pdbqt: Path,
                             verification_path: Path | None = None
                             ) -> dict[str, object]:
    """Tie the manifest, the prepared PDBQT and the atom map to one another.

    **Which record names what.** The atom map is built from the prepared PDBQT
    and the hydrogenated molfile, and never sees the ligand manifest, so it
    records `prepared_pdbqt_sha256` and `input_molfile_sha256` and no manifest
    hash. The record that binds all three together is the preparation record
    (`mur-ligase-ligand-preparation/1`), whose `inputs` name the ligand
    manifest, the prepared PDBQT *and* the atom-map record by SHA-256. It is
    therefore the anchor for this check, and is required.

    Checked here, every one by exact digest:

    1. the preparation record passed;
    2. the ligand manifest is the one the preparation recorded, through the
       legacy-compatible `manifest_hash_agreement` gate;
    3. the prepared PDBQT is the one the preparation recorded;
    4. the atom-map *file* is the one the preparation recorded;
    5. the atom map's own recorded PDBQT is the file we are about to score.

    Optionally, when a stereochemistry verification record is supplied, the
    atom map's input molfile is checked against the hydrogenated molfile that
    verification wrote. Note that this is **not** the ligand manifest's
    `outputs.molfile`: the manifest's molfile is the heavy-atom form written
    before hydrogens were added, so comparing it with the atom map's input
    would always mismatch and would be the wrong check.
    """
    actual_pdbqt = provenance.file_digest_of(prepared_pdbqt)
    actual_atom_map = provenance.file_digest_of(atom_map_path)

    preparation = provenance.read_manifest_like(
        preparation_path, "mur-ligase-ligand-preparation/1")
    if preparation.get("result") != "passed":
        raise DiagnosticError(
            f"{preparation_path} records result "
            f"{preparation.get('result')!r}, not 'passed': "
            f"{preparation.get('failure_reasons')}. The prepared ligand has "
            f"not passed its own checks, so there is nothing to score.")
    prepared_inputs = preparation.get("inputs", {}) or {}

    # 2. manifest identity, reported by the shared gate and decided here.
    agreement = provenance.manifest_hash_agreement(
        manifest_path, prepared_inputs.get("ligand_manifest_sha256"))
    if not agreement.get("agrees"):
        raise DiagnosticError(
            f"the ligand manifest {manifest_path} is not the one "
            f"{preparation_path} was built on: {agreement.get('reason')}. "
            f"Recorded {agreement.get('recorded_sha256')}, actual "
            f"{agreement.get('actual_sha256')}. The modelled protonation "
            f"state comes from this manifest, so it must be the prepared one.")

    # 3, 4, 5. the three file identities, each named by a record.
    for label, recorded, actual, path in (
            ("prepared PDBQT", prepared_inputs.get("prepared_pdbqt_sha256"),
             actual_pdbqt, prepared_pdbqt),
            ("atom-map record", prepared_inputs.get("atom_map_record_sha256"),
             actual_atom_map, atom_map_path),
            ("prepared PDBQT, as named by the atom map",
             (atom_map_record.get("inputs", {}) or {}).get(
                 "prepared_pdbqt_sha256"),
             actual_pdbqt, prepared_pdbqt)):
        if not recorded:
            raise DiagnosticError(
                f"no {label} SHA-256 is recorded, so {path} cannot be tied to "
                f"the preparation evidence. This check is not optional: "
                f"without it the pose being scored is not demonstrably the "
                f"prepared ligand.")
        if recorded != actual:
            raise DiagnosticError(
                f"the {label} does not match: recorded {recorded}, actual "
                f"{actual} for {path}.")

    molfile_check: dict[str, object] = {
        "performed": False,
        "reason": "no stereochemistry verification record was supplied",
    }
    if verification_path is not None:
        verification = provenance.read_manifest_like(
            verification_path, "mur-ligase-ligand-verification/1")
        recorded_molfile = (verification.get("outputs", {}) or {}).get(
            "hydrogenated_molfile_sha256")
        atom_map_molfile = (atom_map_record.get("inputs", {}) or {}).get(
            "input_molfile_sha256")
        if not recorded_molfile or not atom_map_molfile:
            raise DiagnosticError(
                f"cannot cross-check the hydrogenated molfile: verification "
                f"records {recorded_molfile!r}, atom map records "
                f"{atom_map_molfile!r}")
        if recorded_molfile != atom_map_molfile:
            raise DiagnosticError(
                f"the atom map was built from a different molfile than "
                f"verification wrote: atom map {atom_map_molfile}, "
                f"verification {recorded_molfile}. The mapping would not "
                f"describe the verified stereochemistry.")
        molfile_check = {
            "performed": True,
            "hydrogenated_molfile_sha256": recorded_molfile,
            "agrees": True,
            "note": ("the atom map's input molfile is the hydrogenated "
                     "molfile verification wrote, not the manifest's "
                     "pre-hydrogen outputs.molfile"),
        }

    return {
        "ligand_manifest_identity": agreement,
        "preparation_record_sha256": provenance.file_digest_of(preparation_path),
        "preparation_result": preparation.get("result"),
        "prepared_pdbqt_sha256": actual_pdbqt,
        "atom_map_record_sha256": actual_atom_map,
        "identity_chain": (
            "preparation record -> ligand manifest, prepared PDBQT and "
            "atom-map record; atom-map record -> prepared PDBQT"),
        "hydrogenated_molfile_cross_check": molfile_check,
    }


def protonation_aware_symmetry(component, ligand_manifest: dict):
    """Return the graph and automorphisms for the modelled protonation state.

    Taken from the manifest, exactly as `analyse_redocking.py` does it: the
    equivalences that apply depend on which oxygens still carry a proton, and
    assuming RDKit-style defaults here would reintroduce the error the
    independent RMSD audit was built to exclude.
    """
    deprotonated = set(ligand_manifest["protonation_state"]["deprotonated_atoms"])
    still_protonated = sorted(set(component.hydrogens_on) - deprotonated)
    graph, resonance = chemical_graph.normalise_resonance(
        component.graph, hydrogen_bearing=still_protonated)
    basis = (
        f"pH {ligand_manifest['protonation_state']['modelled_ph']}, formal "
        f"charge {ligand_manifest['protonation_state']['modelled_formal_charge']}, "
        f"deprotonated {sorted(deprotonated)}, still protonated "
        f"{still_protonated}")
    return graph, chemical_graph.automorphisms(graph), resonance, basis


def pose_from_pdbqt(path: Path, names_by_serial: dict[int, str]
                    ) -> dict[str, tuple[float, float, float]]:
    """Read the single pose in a PDBQT, named through the validated atom map."""
    poses = analyse_redocking.parse_pose_file(path, names_by_serial)
    if len(poses) != 1:
        raise DiagnosticError(
            f"{path} holds {len(poses)} poses; a minimised pose file must "
            f"hold exactly one")
    return poses[0]["coordinates"]


def rmsd_against_reference(graph, permutations, reference, pose):
    """Symmetry-aware heavy-atom RMSD with no superposition."""
    value, permutation = chemical_graph.symmetry_aware_rmsd(
        graph, reference, pose, permutations)
    return float(value), list(permutation)


def score_components(engine, values) -> dict[str, object]:
    """Record Vina's score array without inventing labels for its terms.

    The number and meaning of the components returned by `score()` is a
    property of the installed Vina, not something to hard-code here. The raw
    array is always recorded; labels are attached only when the installed
    docstring supplies them, and the record says which case applied.
    """
    raw = [float(value) for value in np.atleast_1d(np.asarray(values, float))]
    documented = getattr(type(engine).score, "__doc__", None) or ""
    return {
        "total_kcal_per_mol": raw[0] if raw else None,
        "raw_components": raw,
        "n_components": len(raw),
        "component_labels_source": (
            "installed Vina score() docstring, reproduced verbatim below; "
            "labels were NOT assumed by this script"
            if documented.strip() else
            "the installed Vina score() carries no docstring, so components "
            "are recorded positionally and unlabelled"),
        "installed_score_docstring": documented.strip() or None,
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    """Validate every prerequisite, then score and minimise. Returns the record."""
    box_record = json.loads(Path(args.box_record).read_text())
    centre = box_geometry.record_centre(box_record)
    size = box_geometry.record_size(box_record)

    manifest = provenance.read_manifest(args.receptor_manifest)
    disagreements = provenance.box_and_receptor_disagreements(
        box_record, manifest)
    if disagreements:
        raise DiagnosticError(
            "the box record and receptor manifest disagree:\n  - "
            + "\n  - ".join(disagreements))

    reference_record = json.loads(Path(args.reference).read_text())
    if reference_record.get("schema") != "mur-ligase-ligand-reference/1":
        raise DiagnosticError(f"{args.reference} is not a ligand reference record")
    reference_atoms = {
        name: tuple(entry["coordinates_angstrom"])
        for name, entry in reference_record["atom_mapping"].items()
    }

    component = chemical_component.read_component(args.component_dictionary)
    if component.identifier != reference_record["component"]:
        raise DiagnosticError(
            f"the dictionary is for {component.identifier} but the reference "
            f"is for {reference_record['component']}")

    ligand_manifest = provenance.read_manifest_like(
        args.ligand_manifest, "mur-ligase-ligand/1")
    if ligand_manifest["component"] != component.identifier:
        raise DiagnosticError(
            f"the ligand manifest is for {ligand_manifest['component']} but "
            f"the dictionary is for {component.identifier}")

    atom_map_record = ligand_atom_map.read_atom_map(args.atom_map)
    names_by_serial = ligand_atom_map.name_by_serial(atom_map_record)
    identity = verified_ligand_identity(
        args.ligand_manifest, args.preparation_record, args.atom_map,
        atom_map_record, args.ligand, args.stereo_verification)

    graph, permutations, resonance, protonation_basis = \
        protonation_aware_symmetry(component, ligand_manifest)

    # End-to-end check of the correspondence chain before Vina is involved.
    input_pose = pose_from_pdbqt(args.ligand, names_by_serial)
    input_rmsd, input_permutation = rmsd_against_reference(
        graph, permutations, reference_atoms, input_pose)
    if input_rmsd > args.input_tolerance:
        raise DiagnosticError(
            f"the input pose is {input_rmsd:.6f} A from the deposited "
            f"reference, above the {args.input_tolerance} A tolerance. The "
            f"prepared PDBQT is supposed to preserve the deposited "
            f"coordinates, so the atom map, the reference or the prepared "
            f"file disagree. Nothing was scored: a minimisation result would "
            f"not be interpretable against a reference this does not match.")

    coordinates = np.array([input_pose[name] for name in sorted(input_pose)])
    inside = int(box_geometry.points_inside(centre, size, coordinates).sum())
    if inside != len(coordinates):
        raise DiagnosticError(
            f"only {inside} of {len(coordinates)} ligand heavy atoms lie "
            f"inside the recorded box; Vina cannot score a pose that leaves "
            f"the search volume")

    record: dict[str, object] = {
        "schema": SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": sys.argv,
        "experiment": {
            "kind": "local minimisation from the deposited pose",
            "is_independent_redocking": False,
            "counts_towards_2A_positive_control": False,
            "statement": NOT_A_CONTROL,
        },
        "inputs": {
            "receptor": str(Path(args.receptor).resolve()),
            "receptor_sha256": box_geometry.file_sha256(args.receptor),
            "receptor_manifest": str(Path(args.receptor_manifest).resolve()),
            "receptor_manifest_sha256": box_geometry.file_sha256(
                args.receptor_manifest),
            "receptor_condition": manifest.get("receptor_condition"),
            "receptor_unresolved_decisions": manifest.get(
                "unresolved_decisions", []),
            "ligand_pdbqt": str(Path(args.ligand).resolve()),
            "ligand_manifest": str(Path(args.ligand_manifest).resolve()),
            "atom_map": str(Path(args.atom_map).resolve()),
            "atom_map_sha256": box_geometry.file_sha256(args.atom_map),
            "reference": str(Path(args.reference).resolve()),
            "reference_sha256": box_geometry.file_sha256(args.reference),
            "component_dictionary": str(Path(args.component_dictionary).resolve()),
            "component_dictionary_sha256": box_geometry.file_sha256(
                args.component_dictionary),
            "box_record": str(Path(args.box_record).resolve()),
            "box_record_sha256": box_geometry.file_sha256(args.box_record),
            "preparation_record": str(Path(args.preparation_record).resolve()),
            "stereo_verification": (
                None if args.stereo_verification is None
                else str(Path(args.stereo_verification).resolve())),
            "stereo_verification_sha256": (
                None if args.stereo_verification is None
                else box_geometry.file_sha256(args.stereo_verification)),
            "independent_rmsd_record": (
                None if args.independent_rmsd is None
                else str(Path(args.independent_rmsd).resolve())),
            **identity,
        },
        "box": {
            "centre_angstrom": [float(value) for value in centre],
            "size_angstrom": [float(value) for value in size],
            "centre_mode": box_record.get("centre_mode"),
            "ligand_heavy_atoms_inside": inside,
        },
        "measure": {
            "rmsd": "symmetry-aware heavy-atom RMSD, no superposition",
            "superposition": "none",
            "protonation_basis": protonation_basis,
            "n_automorphisms": len(permutations),
            "resonance_normalisation": resonance,
        },
        "input_pose_check": {
            "purpose": (
                "the prepared PDBQT preserves the deposited coordinates, so "
                "this must be ~0; it tests the atom map, reference and "
                "symmetry path end to end before Vina is involved"),
            "rmsd_angstrom": round(input_rmsd, 6),
            "tolerance_angstrom": args.input_tolerance,
            "passed": True,
        },
        "parameters": {
            "scoring_function": "vina",
            "cpu": args.cpu,
            "seed": args.seed,
            "seed_note": (
                "recorded for reproducibility; score() and optimize() are "
                "deterministic local operations and do not use the random seed "
                "the way a Monte Carlo search does"),
            "receptor_flexibility": "rigid, as in the campaign",
            "max_steps": args.max_steps,
            "map_spacing_angstrom": (
                "not set by this script; Vina's compute_vina_maps default "
                "applies, exactly as in dock.py, so the grids match the "
                "baseline campaign's"),
            "matches_campaign_settings": {
                "scoring_function": True,
                "box_centre_and_size": True,
                "map_spacing": True,
                "receptor_file": True,
                "differs_by": ("no Monte Carlo search, no exhaustiveness, no "
                               "pose retention: score() and optimize() are "
                               "local operations on one supplied pose"),
            },
        },
        "comparability_note": (
            "score() totals and a docking run's per-pose affinities come from "
            "the same scoring function, but are not guaranteed to be "
            "assembled from identical component terms in every Vina release. "
            "Compare on matching components using raw_components below, and "
            "state which terms were compared."
        ),
    }

    if args.check_only:
        record["result"] = "checks_passed_nothing_scored"
        print("preliminary checks passed; nothing scored or minimised.")
        return record

    try:
        from vina import Vina  # imported here so the checks need no Vina
    except ImportError as error:
        raise DiagnosticError(
            f"Vina is not importable: {error}. Activate the docking "
            f"environment (conda activate boltz on Epicac), or use "
            f"--check-only to validate the prerequisites without it.") from error

    engine = Vina(sf_name="vina", seed=args.seed, cpu=args.cpu)
    engine.set_receptor(str(args.receptor))
    engine.set_ligand_from_file(str(args.ligand))
    engine.compute_vina_maps(center=[float(value) for value in centre],
                             box_size=[float(value) for value in size])

    before = score_components(engine, engine.score())
    # `max_steps` is accepted by Vina 1.2's optimize(); older bindings take no
    # argument. The fallback is recorded rather than silently substituted,
    # because the step count affects how far the pose is allowed to relax.
    try:
        relaxed = engine.optimize(max_steps=args.max_steps)
        max_steps_used: object = args.max_steps
    except TypeError:
        relaxed = engine.optimize()
        max_steps_used = ("the installed optimize() accepts no max_steps "
                          "argument; its own default was used")
    after = score_components(engine, relaxed)

    minimised_path = Path(args.output_dir) / "minimised_pose.pdbqt"
    if minimised_path.exists():
        raise DiagnosticError(f"{minimised_path} already exists")
    engine.write_pose(str(minimised_path), overwrite=False)
    if not minimised_path.is_file() or minimised_path.stat().st_size == 0:
        raise DiagnosticError(f"no usable minimised pose was written to "
                              f"{minimised_path}")

    minimised_pose = pose_from_pdbqt(minimised_path, names_by_serial)
    minimised_rmsd, minimised_permutation = rmsd_against_reference(
        graph, permutations, reference_atoms, minimised_pose)
    drift, _ = rmsd_against_reference(
        graph, permutations, input_pose, minimised_pose)

    # A changed energy with bit-identical coordinates is impossible. If it
    # happens, `write_pose` wrote the input pose rather than the optimised one,
    # and the RMSD below would read as "the deposited pose is a stable
    # minimum" when nothing had been minimised at all. That false negative is
    # the worst outcome this script could produce, so it is refused.
    energy_changed = (before["total_kcal_per_mol"] is not None
                      and after["total_kcal_per_mol"] is not None
                      and abs(after["total_kcal_per_mol"]
                              - before["total_kcal_per_mol"]) > 1e-9)
    if energy_changed and drift == 0.0:
        raise DiagnosticError(
            f"the score changed from {before['total_kcal_per_mol']} to "
            f"{after['total_kcal_per_mol']} kcal/mol, yet the written pose is "
            f"identical to the input. The installed write_pose() appears not "
            f"to emit the optimised coordinates, so the RMSD would be "
            f"meaningless. Check the installed Vina's score/optimize/write_pose "
            f"behaviour before trusting any result from this script.")

    vina_version = getattr(__import__("vina"), "__version__", None)
    record["software"] = {
        "vina": vina_version,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    record["scores"] = {"before_minimisation": before, "after_minimisation": after}
    record["parameters"]["max_steps_used"] = max_steps_used
    record["geometry"] = {
        "minimised_pose_file": str(minimised_path.resolve()),
        "minimised_pose_sha256": box_geometry.file_sha256(minimised_path),
        "rmsd_to_deposited_angstrom": round(minimised_rmsd, 6),
        "drift_from_input_angstrom": round(drift, 6),
        "permutation_used": minimised_permutation,
        "n_heavy_atoms": len(minimised_pose),
        "interpretation_note": (
            "rmsd_to_deposited is how far local relaxation moved the pose away "
            "from the crystallographic coordinates in the receptor frame. A "
            "small value means the deposited mode is a local minimum of the "
            "docking model; a large value means it is not."
        ),
    }
    record["coordinates"] = {
        "deposited": {name: list(reference_atoms[name]) for name in sorted(reference_atoms)},
        "input_pose": {name: list(input_pose[name]) for name in sorted(input_pose)},
        "minimised_pose": {name: list(minimised_pose[name]) for name in sorted(minimised_pose)},
    }
    record["result"] = "completed"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Score and locally minimise the deposited ligand pose in "
                     "a prepared receptor. A diagnostic, not redocking: it "
                     "cannot satisfy the 2 A positive-control criterion."))
    parser.add_argument("receptor", type=Path)
    parser.add_argument("ligand", type=Path,
                        help="prepared ligand PDBQT holding the deposited pose")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--box-record", type=Path, required=True)
    parser.add_argument("--receptor-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--component-dictionary", type=Path, required=True)
    parser.add_argument("--ligand-manifest", type=Path, required=True)
    parser.add_argument("--atom-map", type=Path, required=True)
    parser.add_argument("--preparation-record", type=Path, required=True,
                        help=("the mur-ligase-ligand-preparation/1 record; it "
                              "is the only record naming the ligand manifest, "
                              "prepared PDBQT and atom map together, so the "
                              "identity chain cannot be checked without it"))
    parser.add_argument("--stereo-verification", type=Path, default=None,
                        help=("optional; enables the hydrogenated-molfile "
                              "cross-check against the atom map"))
    parser.add_argument("--independent-rmsd", type=Path, default=None,
                        help="recorded for provenance; not required")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0 lets Vina choose its own step count")
    parser.add_argument("--input-tolerance", type=float,
                        default=INPUT_POSE_TOLERANCE_ANGSTROM)
    parser.add_argument("--check-only", action="store_true",
                        help="validate every prerequisite and stop")
    args = parser.parse_args()

    for value, flag in ((args.cpu, "--cpu"), (args.seed, "--seed")):
        if value < 0:
            sys.exit(f"error: {flag} must not be negative, got {value}")
    if args.input_tolerance <= 0:
        sys.exit(f"error: --input-tolerance must be positive, got "
                 f"{args.input_tolerance}")

    try:
        output_dir = fresh_output_directory(args.output_dir, args.ligand)
        args.output_dir = output_dir
        record = run_diagnostic(args)
    except (DiagnosticError, chemical_graph.CorrespondenceError,
            provenance.ReceptorError) as error:
        sys.exit(f"error: {error}")

    record_path = Path(args.output_dir) / "local_minimisation.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    print(f"receptor         : {record['inputs']['receptor']}")
    print(f"condition        : {record['inputs']['receptor_condition']}")
    print(f"ligand           : {record['inputs']['ligand_pdbqt']}")
    print(f"box              : centre {record['box']['centre_angstrom']}, "
          f"size {record['box']['size_angstrom']} A")
    print(f"input pose RMSD  : {record['input_pose_check']['rmsd_angstrom']:.6f} A "
          f"(must be ~0; end-to-end correspondence check)")
    if record["result"] == "completed":
        before = record["scores"]["before_minimisation"]["total_kcal_per_mol"]
        after = record["scores"]["after_minimisation"]["total_kcal_per_mol"]
        print(f"score, deposited : {before:.3f} kcal/mol")
        print(f"score, minimised : {after:.3f} kcal/mol")
        print(f"RMSD to deposited: "
              f"{record['geometry']['rmsd_to_deposited_angstrom']:.6f} A")
        print(f"drift from input : "
              f"{record['geometry']['drift_from_input_angstrom']:.6f} A")
        print(f"minimised pose   : {record['geometry']['minimised_pose_file']}")
    print(f"wrote            : {record_path}")
    print()
    print(NOT_A_CONTROL)


if __name__ == "__main__":
    main()
