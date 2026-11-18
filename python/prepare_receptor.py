"""Prepare a crystal structure as an AutoDock Vina receptor, with provenance.

usage:
    python python/prepare_receptor.py STRUCTURE.cif OUT_PREFIX \
        --protein-author-chain A --condition apo \
        --box-record boxes/murc_uma_chainE_24A.json

The box record supplies the protein copy and the pocket residue list, so the
receptor and the search box cannot be defined from different selections. A
pocket list may be given directly with `--pocket-residues` instead, for use
without a box.

What this script does, in order
------------------------------
1. Reads the source structure and leaves it untouched. All output goes to new
    files; an existing output is refused rather than replaced.
2. Requires the protein copy to be named explicitly. There is no "largest
    chain" default: 1P3D holds two MurC copies whose pockets are 98.9 A apart,
    and picking one by residue count is not a scientific choice.
3. Takes a pocket census before preparation: every requested residue must be
    present, and no pocket residue may be missing heavy atoms. An incomplete
    pocket side chain stops the run.
4. Removes components according to the receptor condition (below), recording
    every removed component rather than reporting a net count.
5. Applies the agreed residue transformations, recording each one as a change
    of chemistry rather than a relabelling.
6. Writes the cleaned PDB, then checks atom-by-atom that every retained atom
    survived with its coordinates unchanged, reporting the pocket separately.
7. Runs Meeko's `mk_prepare_receptor.py`, checks its exit status, and checks
    that the PDBQT is new, non-empty, parseable, and in per-atom
    correspondence with the cleaned PDB. A pocket heavy atom lost at this step
    stops the run.
8. Writes the manifest **only** on success. On failure the outputs are renamed
    with a `.FAILED` suffix so nothing is left looking usable.

Receptor conditions
-------------------
`apo` (the initial condition)
    ANP absent, manganese absent, UMA absent, waters absent. This is the
    condition the first docking pilot uses.

`anp-present`
    ANP retained for the later paired comparison; manganese, UMA and waters
    still absent. Retaining ANP needs Meeko to type a non-standard residue,
    which this script does **not** fake: if the prepared PDBQT does not
    contain the ANP atoms, the run fails rather than producing a receptor that
    is silently missing its cofactor. See docs/receptor_preparation.md for the
    integration check this requires.

UMA is removed in every condition. It is the reaction product that defines the
pocket, and leaving it in the receptor would block the site being docked into.
It is preserved separately as a reference pose and, for the planned control,
prepared as a ligand in its own right.

Scientific limitations, which this script records rather than resolves
---------------------------------------------------------------------
- Manganese omission is a documented approximation. It followed a reported
  compatibility problem at the generation stage, not a finding that the metals
  are unimportant, and metal-free modelling does not represent the catalytic
  environment.
- Removing ligands and metals from a crystal structure does not produce an
  experimentally apo structure; the coordinates remain those of the complex.
- MSE to MET replaces selenium with sulphur without re-optimising geometry.
- Nothing here validates atom typing, partial charges or protonation states.
  Those belong to the preparation tools and must be checked where they are
  installed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry
import receptor_provenance as provenance

AMINO_ACIDS = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO "
    "SER THR TRP TYR VAL".split()
)

# Atoms present in KCX but not in LYS; deleted by the agreed transformation.
KCX_EXTRA_ATOMS = {"CX", "OQ1", "OQ2", "OX1", "OX2", "C1", "O1", "O2"}

WATER_NAMES = {"HOH", "DOD", "WAT"}

# Components removed in every condition, with the reason recorded in the
# manifest. UMA defines the pocket being docked into; the metals are omitted
# as a documented approximation, not because they are unimportant.
ALWAYS_REMOVED = {
    "UMA": "the pocket-defining reaction product; the site must be free to dock into",
    "MN": "manganese omitted as a documented approximation, not a validated "
          "biological equivalence",
}

CONDITIONS = {
    "apo": {
        "removed": {**ALWAYS_REMOVED,
                    "ANP": "ATP analogue absent in the initial condition"},
        "retained": [],
        "description": "ANP absent, manganese absent, UMA absent, waters absent",
    },
    "anp-present": {
        "removed": ALWAYS_REMOVED,
        "retained": ["ANP"],
        "description": "ANP retained; manganese absent, UMA absent, waters absent",
    },
}


class PreparationError(RuntimeError):
    """Raised when preparation cannot proceed or its output is not trustworthy."""


# ---------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------

def pocket_from_box_record(path: Path) -> tuple[list[int], str, float, dict]:
    """Return the pocket residues, author chain and cutoff from a box record.

    Taking the selection from the record is what keeps the receptor and the
    box describing the same thing. Residue identifiers carrying insertion
    codes are rejected here rather than silently truncated to a number, since
    a PDB-numbered selection cannot represent them.
    """
    record = box_geometry.read_record(path)
    selection = record.get("selection", {})
    identifiers = [str(value) for value in
                   selection.get("pocket_author_residue_ids", [])]
    if not identifiers:
        raise PreparationError(
            f"{path} records no pocket residues, so there is nothing to check "
            f"the receptor against."
        )
    with_codes = [value for value in identifiers if not value.lstrip("-").isdigit()]
    if with_codes:
        raise PreparationError(
            f"{path} lists pocket residues carrying insertion codes or "
            f"non-numeric identifiers ({with_codes[:5]}). This script selects "
            f"by residue number and cannot represent them; resolve the "
            f"numbering before preparing a receptor."
        )
    chain = selection.get("protein_author_chain_id")
    if not chain:
        raise PreparationError(f"{path} records no author protein chain.")
    return ([int(value) for value in identifiers], chain,
            float(selection.get("pocket_cutoff_angstrom", float("nan"))), record)


def parse_residue_numbers(text: str) -> list[int]:
    """Parse a comma-separated list of author residue numbers."""
    cleaned = text.strip().lstrip("[").rstrip("]").replace(" ", "")
    try:
        return [int(value) for value in cleaned.split(",") if value]
    except ValueError as error:
        raise PreparationError(
            f"--pocket-residues takes author residue numbers, got {text!r} "
            f"({error})"
        ) from error


def chain_mapping(model: gemmi.Model, author_chain: str) -> dict[str, object]:
    """Record the canonical-to-author chain and residue mapping being kept.

    PDB and PDBQT carry only author numbering, so the canonical mmCIF
    identifiers have to be recorded here if they are to survive at all. The
    mapping is stored per residue for the protein chain, which is what lets a
    later stage translate a BoltzGen canonical index back to this receptor.
    """
    canonical_ids, residues = set(), {}
    for chain in model:
        if chain.name != author_chain:
            continue
        for residue in chain:
            if residue.name not in AMINO_ACIDS | {"MSE", "KCX"}:
                continue
            canonical_ids.add(residue.subchain)
            label = (f"{residue.seqid.num}"
                     f"{str(residue.seqid.icode).strip()}")
            residues[label] = {
                "canonical_chain_id": residue.subchain,
                "canonical_residue_index": residue.label_seq,
                "residue_name": residue.name,
            }
    if not residues:
        raise PreparationError(
            f"author chain {author_chain!r} holds no recognised protein residues."
        )
    if len(canonical_ids) != 1:
        raise PreparationError(
            f"author chain {author_chain!r} maps to several canonical chains "
            f"{sorted(canonical_ids)}; the protein copy is ambiguous."
        )
    return {
        "protein_author_chain_id": author_chain,
        "protein_canonical_chain_id": sorted(canonical_ids)[0],
        "n_residues": len(residues),
        "author_to_canonical_residues": residues,
    }


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def strip_components(model: gemmi.Model, author_chain: str,
                     condition: str) -> dict[str, object]:
    """Remove waters, other chains and the condition's components, recording all.

    Components are removed by name so that each removal is attributable, in
    place of Gemmi's `remove_ligands_and_waters()`, which strips every
    heteroatom at once: on 1P3D that silently removes UMA, ANP and all four
    manganese ions together, which is three different scientific decisions
    taken by one call.
    """
    settings = CONDITIONS[condition]
    removed_counts: dict[str, int] = {}
    retained_counts: dict[str, int] = {}
    other_chains: dict[str, dict[str, int]] = {}

    # Discarding the other protein copy is a selection of which copy to dock
    # against, not a removal of components, so it is recorded separately.
    for chain in list(model):
        if chain.name != author_chain:
            counts: dict[str, int] = {}
            for residue in chain:
                counts[residue.name] = counts.get(residue.name, 0) + 1
            other_chains[chain.name] = counts
            model.remove_chain(chain.name)

    for chain in model:
        # Indices are collected first and deleted in reverse, so no index
        # shifts under the loop and no residue reference outlives its slot.
        # Gemmi has no filtered-removal call that takes a predicate.
        drop: list[int] = []
        for index, residue in enumerate(chain):
            if residue.name in WATER_NAMES or residue.name in settings["removed"]:
                removed_counts[residue.name] = removed_counts.get(residue.name, 0) + 1
                drop.append(index)
                continue
            if residue.name not in AMINO_ACIDS | {"MSE", "KCX"}:
                retained_counts[residue.name] = retained_counts.get(residue.name, 0) + 1
        for index in reversed(drop):
            del chain[index]

    unexpected = sorted(set(retained_counts) - set(settings["retained"]))
    if unexpected:
        raise PreparationError(
            f"condition {condition!r} retained unexpected components "
            f"{unexpected}; it expects {settings['retained'] or 'none'}. "
            f"Decide deliberately what belongs in the receptor."
        )
    return {
        "removed": {
            "components_in_selected_chain": removed_counts,
            "reasons": dict(settings["removed"]),
            "waters_removed": removed_counts.get("HOH", 0),
            "other_chains_discarded": other_chains,
            "other_chains_note": (
                "discarding the other protein copy selects which copy is "
                "docked against; it is not a component removal"),
        },
        "retained": {"by_component": retained_counts,
                     "expected": list(settings["retained"])},
    }


def apply_residue_transformations(model: gemmi.Model) -> dict[str, object]:
    """Apply the agreed residue transformations, recording each occurrence.

    MSE becomes MET with SE renamed to SD and its element set to sulphur; KCX
    becomes LYS with the carbamyl atoms deleted. Both are changes of chemistry
    and are recorded as such, with the atoms affected listed.
    """
    applied: dict[str, list[str]] = {"MSE_to_MET": [], "KCX_to_LYS": []}
    deleted_atoms: list[str] = []
    transformations: dict[tuple[str, int, str], tuple[str, str]] = {}

    for chain in model:
        for residue in chain:
            label = (f"{chain.name}:{residue.seqid.num}"
                     f"{str(residue.seqid.icode).strip()}")
            residue_key = (chain.name, residue.seqid.num,
                           str(residue.seqid.icode).strip())
            if residue.name == "MSE":
                residue.name = "MET"
                residue.het_flag = "A"
                applied["MSE_to_MET"].append(label)
                transformations[residue_key] = ("MSE", "MET")
                for atom in residue:
                    if atom.name == "SE":
                        atom.name = "SD"
                        atom.element = gemmi.Element("S")
            elif residue.name == "KCX":
                residue.name = "LYS"
                residue.het_flag = "A"
                applied["KCX_to_LYS"].append(label)
                transformations[residue_key] = ("KCX", "LYS")
                for name in [atom.name for atom in residue]:
                    if name in KCX_EXTRA_ATOMS:
                        deleted_atoms.append(f"{label} {name}")
                        del residue[[atom.name for atom in residue].index(name)]

    return {
        "record": {
            "MSE_to_MET": {
                "count": len(applied["MSE_to_MET"]),
                "residues": applied["MSE_to_MET"],
                "effect": provenance.AGREED_RESIDUE_TRANSFORMATIONS[
                    ("MSE", "MET")][2],
            },
            "KCX_to_LYS": {
                "count": len(applied["KCX_to_LYS"]),
                "residues": applied["KCX_to_LYS"],
                "deleted_atoms": deleted_atoms,
                "effect": provenance.AGREED_RESIDUE_TRANSFORMATIONS[
                    ("KCX", "LYS")][2],
            },
        },
        "transformations": transformations,
    }


# ---------------------------------------------------------------------------
# Meeko
# ---------------------------------------------------------------------------

def run_meeko(pdb_path: Path, prefix: Path,
              extra_arguments: list[str]) -> dict[str, object]:
    """Run Meeko's receptor preparation and return the command record.

    The exit status is checked. Meeko's own output is captured in full and
    recorded, because the useful diagnostics (template matching failures, the
    residues it ignored) appear there and nowhere else.

    Flags used, as confirmed against the installed Meeko 0.7.1 on Epicac:

    `--read_pdb`
        Reads a PDB without ProDy. This script always hands Meeko a PDB it
        has just written, so the ProDy-backed reader (`-i`,
        `--read_with_prody`) is not needed and is not used, removing a
        dependency from the path that matters.
    `-o` / `-p`
        Output basename, and write PDBQT.
    `--default_altloc A`
        Alternate conformations are already removed upstream; this guards
        against any that survive.

    `-a/--allow_bad_res` is **not** passed. In Meeko 0.7.1 it deletes residues
    with missing atoms, which is the opposite of this project's stop rule:
    an incomplete pocket residue must block the receptor, not be quietly
    dropped. It is left disabled for the first real preparation attempt so
    that Meeko reports what it cannot match instead of discarding it.

    `-d/--delete_residues` is a *different* flag: it takes an explicit residue
    selection to delete. It is not a replacement for `-a` and is never
    substituted for it; deleting named residues is a deliberate scientific
    act, not a way to get past a failure.

    Either flag can still be passed explicitly through `--meeko-argument`, and
    the manifest records the exact command, so any such use is visible.
    """
    command = ["mk_prepare_receptor.py", "--read_pdb", str(pdb_path),
               "-o", str(prefix), "-p", "--default_altloc", "A",
               *extra_arguments]
    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=3600)
    except FileNotFoundError as error:
        raise PreparationError(
            f"mk_prepare_receptor.py was not found on PATH: {error}. Activate "
            f"the environment that provides Meeko."
        ) from error
    except subprocess.SubprocessError as error:
        raise PreparationError(f"mk_prepare_receptor.py failed to run: {error}") from error

    output = finished.stdout + finished.stderr
    record = {
        "command": command,
        "returncode": finished.returncode,
        "stdout_tail": finished.stdout[-4000:],
        "stderr_tail": finished.stderr[-4000:],
        "template_matching_failures": [
            line.strip() for line in output.splitlines()
            if "Template matching failed" in line
        ],
    }
    if finished.returncode != 0:
        raise PreparationError(
            f"mk_prepare_receptor.py exited {finished.returncode}.\n"
            f"--- tool output ---\n{output.strip()}"
        )
    return record


def locate_receptor_pdbqt(prefix: Path) -> Path:
    """Return the rigid receptor PDBQT Meeko wrote, or raise.

    Meeko names its output `<prefix>_rigid.pdbqt` when writing rigid and
    flexible parts, and `<prefix>.pdbqt` otherwise; which appears depends on
    the version and the options used, so both are accepted and the one found
    is recorded. Finding neither is a failure, not something to work around.
    """
    candidates = [prefix.with_name(prefix.name + "_rigid.pdbqt"),
                  prefix.with_suffix(".pdbqt")]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise PreparationError(
            f"Meeko reported success but no receptor PDBQT was written; "
            f"looked for {[str(path) for path in candidates]}."
        )
    if len(found) > 1:
        raise PreparationError(
            f"several candidate receptor files exist "
            f"({[str(path) for path in found]}); remove the stale one so the "
            f"receptor being docked is unambiguous."
        )
    return found[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("structure", type=Path, help="Source mmCIF or PDB")
    parser.add_argument("out_prefix", type=Path,
                        help="Output prefix; nothing existing is replaced")
    parser.add_argument("--protein-author-chain", required=True,
                        help="Author chain of the protein copy to prepare")
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS),
                        help="Receptor condition; see the module docstring")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--box-record", type=Path,
                           help="Box record supplying the pocket and chain")
    selection.add_argument("--pocket-residues",
                           help="Comma-separated author residue numbers")
    parser.add_argument("--meeko-argument", action="append", default=[],
                        metavar="ARG",
                        help="Extra argument passed to mk_prepare_receptor.py; "
                             "repeat for several. Recorded in the manifest")
    parser.add_argument("--manifest", type=Path,
                        help="Manifest path (default: OUT_PREFIX.manifest.json)")
    parser.add_argument("--skip-meeko", action="store_true",
                        help="Stop after the cleaned PDB and its checks. Writes "
                             "no manifest: a receptor was not produced")
    args = parser.parse_args()

    try:
        prepare(args)
    except (PreparationError, provenance.ReceptorError,
            box_geometry.BoxError) as error:
        sys.exit(f"error: {error}")


def prepare(args: argparse.Namespace) -> None:
    """Run the preparation, stopping at the first condition that fails."""
    structure_path: Path = args.structure
    prefix: Path = args.out_prefix
    if not structure_path.is_file():
        raise PreparationError(f"structure not found: {structure_path}")

    pdb_path = prefix.with_name(prefix.name + "_prepared.pdb")
    manifest_path = args.manifest or prefix.with_name(
        prefix.name + ".manifest.json")
    for path in (pdb_path, manifest_path):
        if path.exists():
            raise PreparationError(
                f"{path} already exists. Choose a fresh output prefix; source "
                f"structures and earlier receptors are never replaced."
            )

    box_record = None
    if args.box_record:
        pocket_residues, box_chain, cutoff, box_record = pocket_from_box_record(
            args.box_record)
        if box_chain != args.protein_author_chain:
            raise PreparationError(
                f"the box record was mapped on author chain {box_chain!r} but "
                f"--protein-author-chain is {args.protein_author_chain!r}. "
                f"These must be the same protein copy."
            )
    else:
        pocket_residues = parse_residue_numbers(args.pocket_residues)
        cutoff = None

    # ---- read the source, which is never modified on disk ----------------
    structure = gemmi.read_structure(str(structure_path))
    structure.setup_entities()
    if len(structure) == 0:
        raise PreparationError(f"{structure_path} contains no models")
    if len(structure) > 1:
        print(f"NOTE {structure_path} holds {len(structure)} models; using the first.")
    mapping = chain_mapping(structure[0], args.protein_author_chain)

    source_pdb = pdb_path.with_name(pdb_path.name + ".source_subset.pdb")
    source_structure = gemmi.read_structure(str(structure_path))
    source_structure.setup_entities()
    source_structure.remove_hydrogens()
    source_structure.remove_alternative_conformations()
    source_structure.setup_entities()
    source_structure.write_pdb(str(source_pdb))
    source_atoms = provenance.parse_atom_records(source_pdb)

    before = provenance.pocket_census(
        source_atoms, args.protein_author_chain, pocket_residues)
    print(f"source structure : {structure_path}")
    print(f"protein copy     : author {mapping['protein_author_chain_id']}, "
          f"canonical {mapping['protein_canonical_chain_id']}, "
          f"{mapping['n_residues']} residues")
    print(f"condition        : {args.condition} "
          f"({CONDITIONS[args.condition]['description']})")
    print(f"pocket requested : {before['n_requested']} residues, "
          f"{before['total_heavy_atoms']} heavy atoms found")

    if before["residues_absent"]:
        raise PreparationError(
            f"pocket residues absent from the source structure's chain "
            f"{args.protein_author_chain}: {before['residues_absent']}. "
            f"The pocket list and the structure do not agree."
        )
    if before["residues_incomplete"]:
        raise PreparationError(
            "pocket residues are incomplete in the source structure:\n  "
            + "\n  ".join(str(value) for value in before["residues_incomplete"])
            + "\nMissing pocket atoms must be repaired by an agreed, documented "
              "method or block downstream use; they are not dropped silently."
        )
    if before["residues_with_insertion_codes"]:
        raise PreparationError(
            f"pocket residues carry insertion codes "
            f"{before['residues_with_insertion_codes']}, which a residue-number "
            f"selection cannot represent unambiguously."
        )

    # ---- clean ----------------------------------------------------------
    model = structure[0]
    component_record = strip_components(model, args.protein_author_chain,
                                        args.condition)
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    transformation_record = apply_residue_transformations(model)
    structure.setup_entities()
    structure.write_pdb(str(pdb_path))

    prepared_atoms = provenance.parse_atom_records(pdb_path)
    after = provenance.pocket_census(
        prepared_atoms, args.protein_author_chain, pocket_residues)
    print(f"MSE -> MET       : "
          f"{transformation_record['record']['MSE_to_MET']['count']}")
    print(f"KCX -> LYS       : "
          f"{transformation_record['record']['KCX_to_LYS']['count']}")
    print(f"removed          : "
          f"{component_record['removed']['components_in_selected_chain']}")
    print(f"other copies     : "
          f"{sorted(component_record['removed']['other_chains_discarded'])}")
    print(f"retained         : "
          f"{component_record['retained']['by_component'] or 'none'}")
    print(f"wrote            : {pdb_path}")

    # Atoms deliberately absent from the cleaned PDB: everything whose
    # component was removed, plus the carbamyl atoms of any KCX.
    removed_names = set(
        component_record["removed"]["components_in_selected_chain"])
    deliberately_removed = {
        key for key in source_atoms
        if key.chain != args.protein_author_chain
        or key.residue_name in removed_names & (WATER_NAMES | set(ALWAYS_REMOVED)
                                                | {"ANP"})
        or (key.residue_name == "KCX" and key.atom_name in KCX_EXTRA_ATOMS)
    }
    pocket_keys = provenance.pocket_atom_keys(
        source_atoms, args.protein_author_chain, pocket_residues)
    source_to_prepared = provenance.compare_atom_sets(
        provenance.heavy_atoms(source_atoms),
        provenance.heavy_atoms(prepared_atoms),
        residue_transformations=transformation_record["transformations"],
        deliberately_removed=deliberately_removed,
        subset_keys=pocket_keys,
        subset_name="pocket heavy atoms",
    )
    report_correspondence("source -> prepared PDB", source_to_prepared)
    if after["residues_absent"] or after["residues_incomplete"]:
        raise PreparationError(
            f"the pocket lost atoms during cleaning: absent "
            f"{after['residues_absent']}, incomplete "
            f"{after['residues_incomplete']}"
        )
    if source_to_prepared.subset_n_missing:
        raise PreparationError(
            f"{source_to_prepared.subset_n_missing} pocket heavy atoms are "
            f"missing from the cleaned PDB: "
            f"{source_to_prepared.missing_atoms[:10]}"
        )
    if not source_to_prepared.coordinates_preserved:
        raise PreparationError(
            f"coordinates changed between the source and the cleaned PDB, "
            f"beyond the {provenance.COORDINATE_TOLERANCE_ANGSTROM} A rounding "
            f"tolerance: {source_to_prepared.moved_atoms[:10]}"
        )
    source_pdb.unlink()

    if args.skip_meeko:
        print("\n--skip-meeko: stopped after the cleaned PDB. No receptor was "
              "produced and no manifest was written.")
        return

    # ---- Meeko ----------------------------------------------------------
    print("\nrunning mk_prepare_receptor.py")
    try:
        meeko_record = run_meeko(pdb_path, prefix, args.meeko_argument)
        receptor_path = locate_receptor_pdbqt(prefix)
        receptor_atoms = provenance.parse_atom_records(receptor_path)
        prepared_to_receptor = provenance.compare_atom_sets(
            provenance.heavy_atoms(prepared_atoms),
            provenance.heavy_atoms(receptor_atoms),
            subset_keys=provenance.pocket_atom_keys(
                prepared_atoms, args.protein_author_chain, pocket_residues),
            subset_name="pocket heavy atoms",
        )
        report_correspondence("prepared PDB -> receptor PDBQT",
                              prepared_to_receptor)
        if prepared_to_receptor.subset_n_missing:
            raise PreparationError(
                f"{prepared_to_receptor.subset_n_missing} pocket heavy atoms "
                f"are missing from {receptor_path}: "
                f"{prepared_to_receptor.missing_atoms[:10]}. The receptor is "
                f"incomplete where it matters most."
            )
        if not prepared_to_receptor.coordinates_preserved:
            raise PreparationError(
                f"coordinates changed between the cleaned PDB and "
                f"{receptor_path}: {prepared_to_receptor.moved_atoms[:10]}"
            )
        check_retained_components(receptor_atoms, args.condition, receptor_path)
    except (PreparationError, provenance.ReceptorError):
        mark_failed(prefix, pdb_path)
        raise

    receptor_census = provenance.pocket_census(
        receptor_atoms, args.protein_author_chain, pocket_residues)
    manifest = provenance.build_manifest(
        condition=args.condition,
        selections={
            **mapping,
            "pocket_author_residue_ids": [str(value) for value in pocket_residues],
            "pocket_cutoff_angstrom": cutoff,
            "box_record": None if args.box_record is None else str(
                args.box_record.resolve()),
            "box_record_sha256": None if args.box_record is None else
            box_geometry.file_sha256(args.box_record),
        },
        source={
            "path": str(structure_path),
            "sha256": box_geometry.file_sha256(structure_path),
            "model": 1,
            "coordinate_frame": "source structure as read; no transformation, "
                                "recentring or symmetry operation applied",
        },
        outputs={
            "prepared_pdb": str(pdb_path),
            "prepared_pdb_sha256": box_geometry.file_sha256(pdb_path),
            "receptor_pdbqt": str(receptor_path),
            "receptor_pdbqt_sha256": box_geometry.file_sha256(receptor_path),
            "receptor_heavy_atoms": len(provenance.heavy_atoms(receptor_atoms)),
        },
        transformations=transformation_record["record"],
        removed=component_record["removed"],
        retained={**component_record["retained"],
                  "in_receptor_pdbqt": sorted(
                      {key.residue_name for key in receptor_atoms
                       if key.residue_name not in AMINO_ACIDS})},
        checks={
            "pocket_before": before,
            "pocket_after_cleaning": after,
            "pocket_in_receptor_pdbqt": receptor_census,
            "source_to_prepared": source_to_prepared.as_dict(),
            "prepared_to_receptor": prepared_to_receptor.as_dict(),
            "coordinate_tolerance_angstrom":
                provenance.COORDINATE_TOLERANCE_ANGSTROM,
            "tolerance_basis": "PDB/PDBQT store three decimal places, so two "
                               "representations of one coordinate differ by at "
                               "most sqrt(3) x 0.001 A",
        },
        commands=[meeko_record],
        tools={
            "gemmi": gemmi.__version__,
            "mk_prepare_receptor": provenance.tool_version(
                "mk_prepare_receptor.py", "--help"),
            "python": sys.version.split()[0],
        },
        unresolved=unresolved_decisions(args.condition),
    )
    provenance.write_manifest(manifest_path, manifest)

    print(f"\nreceptor ready   : {receptor_path}")
    print(f"manifest         : {manifest_path}")
    print(f"pocket in PDBQT  : {receptor_census['n_found']} residues, "
          f"{receptor_census['total_heavy_atoms']} heavy atoms")
    print("\nUnresolved decisions recorded in the manifest:")
    for item in unresolved_decisions(args.condition):
        print(f"  - {item}")


def check_retained_components(receptor_atoms, condition: str,
                              receptor_path: Path) -> None:
    """Confirm the condition's retained components survived into the PDBQT.

    For `anp-present` this is the check that matters: Meeko must have typed
    ANP as a non-standard residue. If its atoms are not in the receptor, the
    run fails. Appending the records by hand would produce a file Vina reads
    without the atom types and charges that make it meaningful.
    """
    expected = CONDITIONS[condition]["retained"]
    present = {key.residue_name for key in receptor_atoms}
    missing = [name for name in expected if name not in present]
    if missing:
        raise PreparationError(
            f"condition {condition!r} requires {missing} in the receptor, but "
            f"{receptor_path} does not contain those residues. Meeko did not "
            f"type them.\n"
            f"       Do not append the records by hand: a PDBQT needs AutoDock "
            f"atom types and partial charges, and hand-inserted lines have "
            f"neither.\n"
            f"       See docs/receptor_preparation.md for the template-based "
            f"route (--set_template / --add_templates) and the integration "
            f"check it needs."
        )


def unresolved_decisions(condition: str) -> list[str]:
    """Return the preparation decisions this project has not settled.

    These are recorded in every manifest. They are genuinely open, and a
    receptor that passes every check in this script is still subject to them.
    """
    items = [
        "Manganese omission is a documented approximation carried over from a "
        "generation-stage compatibility problem; its effect on the pocket's "
        "electrostatics has not been assessed.",
        "Removing ligands and metals does not produce an experimentally apo "
        "structure; the coordinates remain those of the complex.",
        "MSE to MET replaces selenium with sulphur without re-optimising "
        "geometry; pocket residue 31 is affected and its geometry has not "
        "been inspected.",
        "Protonation states, tautomers and partial charges come from Meeko's "
        "defaults and have not been reviewed for this pocket.",
        "No comparison against a chemical component dictionary is performed; "
        "heavy-atom completeness is judged against a small internal table.",
    ]
    if condition == "anp-present":
        items.append(
            "ANP retention depends on Meeko's non-standard-residue typing. "
            "Its atom types, charges and protonation have not been validated "
            "for this project.")
    else:
        items.append(
            "This receptor has no ANP. The paired ANP-present comparison "
            "requires a second receptor prepared with --condition anp-present, "
            "and the same box.")
    return items


def report_correspondence(stage: str,
                          report: provenance.CorrespondenceReport) -> None:
    """Print one correspondence report, pocket subset included."""
    print(f"\n{stage}")
    print(f"  atoms paired            : {report.n_paired}")
    print(f"  missing                 : {report.n_missing}")
    print(f"  unexpected in output    : {report.n_unexpected}")
    print(f"  max deviation           : "
          f"{report.max_deviation_angstrom:.4f} A "
          f"(tolerance {provenance.COORDINATE_TOLERANCE_ANGSTROM} A)"
          + (f" at {report.max_deviation_atom}"
             if report.max_deviation_atom else ""))
    if report.subset_name:
        print(f"  {report.subset_name:<22}: {report.subset_n_paired} paired, "
              f"{report.subset_n_missing} missing, max deviation "
              f"{report.subset_max_deviation_angstrom:.4f} A")
    for transformation in report.applied_transformations:
        print(f"  transformation applied  : {transformation}")


def mark_failed(prefix: Path, pdb_path: Path) -> None:
    """Rename outputs so a failed preparation leaves nothing looking usable."""
    candidates = [pdb_path, prefix.with_suffix(".pdbqt"),
                  prefix.with_name(prefix.name + "_rigid.pdbqt"),
                  prefix.with_name(prefix.name + "_flex.pdbqt")]
    for path in candidates:
        if path.is_file():
            failed = path.with_name(path.name + ".FAILED")
            path.replace(failed)
            print(f"  renamed {path.name} -> {failed.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
