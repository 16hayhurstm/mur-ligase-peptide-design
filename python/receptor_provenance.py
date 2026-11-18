"""Atom-level provenance checks and the manifest for a prepared receptor.

This module holds the parts of receptor preparation that can be checked
without Gemmi, Meeko or Vina: reading atom records out of PDB and PDBQT files,
pairing atoms between two files by identity rather than by order, comparing
their coordinates, taking a pocket census, and writing and reading the
preparation manifest.

Why atom-level, and not a global RMSD
-------------------------------------
A single global RMSD over thousands of atoms can sit near zero while a handful
of pocket atoms have moved, been renamed or disappeared. Every comparison here
is therefore per atom: atoms are paired on their full identity, the **maximum**
per-atom deviation is reported rather than a mean, and the pocket subset is
reported separately from the whole receptor. Missing and unexpected atoms are
counted explicitly, because an atom that vanished has no deviation at all.

Coordinate tolerance
--------------------
PDB and PDBQT store coordinates as fixed-point numbers with three decimal
places, so a value is rounded to the nearest 0.001 A. Two independently
written representations of the same coordinate can therefore differ by up to
0.001 A per axis, and the distance between them by up to sqrt(3) x 0.001 ~=
0.0017 A. `COORDINATE_TOLERANCE_ANGSTROM` is 0.002 A: large enough that
rounding alone never trips it, and far smaller than any real movement of an
atom. It is a check that coordinates were *not transformed*; it is not a
structural similarity measure and it says nothing about whether the structure
is correct.

Limitations
-----------
- Atom identity here is (author chain, residue number, insertion code,
  residue name, atom name, altloc). Two files that disagree about any of those
  will report an unpaired atom rather than a moved one, which is the intended
  behaviour.
- Expected heavy-atom counts are a small table for the standard residues plus
  MSE. They detect an incomplete side chain, not every chemical problem, and
  they are not a substitute for a chemical component dictionary.
- Nothing here validates atom typing, partial charges, protonation or
  stereochemistry. Those are properties of the preparation tools and must be
  checked where those tools are installed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

MANIFEST_SCHEMA = "mur-ligase-receptor/1"

# Derived in the module docstring: sqrt(3) x 0.001 A ~= 0.0017 A, rounded up.
COORDINATE_TOLERANCE_ANGSTROM = 0.002

# Heavy (non-hydrogen) atom counts for a complete residue, used to detect
# incomplete side chains in the source structure. MSE has the same count as
# MET, with selenium in place of the sulphur.
EXPECTED_HEAVY_ATOMS = {
    "ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6, "GLN": 9, "GLU": 9,
    "GLY": 4, "HIS": 10, "ILE": 8, "LEU": 8, "LYS": 9, "MET": 8, "MSE": 8,
    "PHE": 11, "PRO": 7, "SER": 6, "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7,
}

# Residue-level chemical transformations this project has agreed to, as
# (source name, prepared name) -> (source atom, prepared atom, note). These
# are recorded as transformations, never as atoms that happen to match.
AGREED_RESIDUE_TRANSFORMATIONS = {
    ("MSE", "MET"): (
        "SE", "SD",
        "selenomethionine to methionine: selenium replaced by sulphur, "
        "coordinates not re-optimised",
    ),
    ("KCX", "LYS"): (
        None, None,
        "carbamylated lysine to lysine: the carbamyl atoms are deleted, "
        "removing that chemistry from the model",
    ),
}


class ReceptorError(RuntimeError):
    """Raised when a receptor or its provenance is not usable as given."""


# ---------------------------------------------------------------------------
# Reading atom records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomKey:
    """An atom's full identity, used for pairing between files.

    Author-numbered, because that is what PDB and PDBQT carry; the canonical
    mmCIF identifiers are recorded in the manifest alongside, since they
    cannot be represented in these formats.
    """
    chain: str
    residue_seq: int
    insertion_code: str
    residue_name: str
    atom_name: str
    altloc: str

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain, self.residue_seq, self.insertion_code)

    def __str__(self) -> str:
        return (f"{self.chain}:{self.residue_seq}{self.insertion_code}"
                f" {self.residue_name} {self.atom_name}"
                + (f" altloc {self.altloc}" if self.altloc else ""))


@dataclass(frozen=True)
class AtomRecord:
    key: AtomKey
    position: tuple[float, float, float]
    element: str


def parse_atom_records(path: Path | str) -> dict[AtomKey, AtomRecord]:
    """Read ATOM/HETATM records from a PDB or PDBQT file by column position.

    Columns 1-54 are identical in both formats, which is why one reader serves
    for each: name 13-16, altloc 17, residue name 18-20, chain 22, residue
    number 23-26, insertion code 27, and x, y, z in 31-54. PDBQT's partial
    charge and atom type sit beyond column 54 and are not read here.

    Raises `ReceptorError` on a malformed coordinate, a duplicated atom
    identity, or a file with no atom records: each of those makes a
    correspondence check meaningless rather than merely incomplete.
    """
    source = Path(path)
    try:
        text = source.read_text()
    except OSError as error:
        raise ReceptorError(f"cannot read {source}: {error}") from error

    records: dict[AtomKey, AtomRecord] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            position = (float(line[30:38]), float(line[38:46]),
                        float(line[46:54]))
            residue_seq = int(line[22:26])
        except (ValueError, IndexError) as error:
            raise ReceptorError(
                f"{source} line {number}: malformed atom record ({error})"
            ) from error
        key = AtomKey(
            chain=line[21:22].strip(),
            residue_seq=residue_seq,
            insertion_code=line[26:27].strip(),
            residue_name=line[17:20].strip(),
            atom_name=line[12:16].strip(),
            altloc=line[16:17].strip(),
        )
        if key in records:
            raise ReceptorError(
                f"{source} line {number}: duplicated atom identity {key}; "
                f"atoms cannot be paired unambiguously"
            )
        records[key] = AtomRecord(key=key, position=position,
                                  element=element_of(line, key))
    if not records:
        raise ReceptorError(f"{source} contains no ATOM or HETATM records")
    return records


def element_of(line: str, key: AtomKey) -> str:
    """Return an atom's element symbol, from columns 77-78 or from its name.

    PDB puts the element in columns 77-78. PDBQT replaces that field with an
    AutoDock atom type, which is related but not identical ("A" for aromatic
    carbon, "NA" for a hydrogen-bond-accepting nitrogen), so it is not read as
    an element. In both cases the leading alphabetic characters of the atom
    name are used as the fallback.
    """
    field = line[76:78].strip() if len(line) >= 78 else ""
    if field and field.isalpha() and len(field) <= 2:
        return field.upper()
    name = key.atom_name.lstrip("0123456789")
    return name[:1].upper() if name else ""


def is_hydrogen(record: AtomRecord) -> bool:
    """True for a hydrogen or deuterium atom."""
    return record.element in ("H", "D")


def heavy_atoms(records: dict[AtomKey, AtomRecord]) -> dict[AtomKey, AtomRecord]:
    """Return only the non-hydrogen atoms of a record set."""
    return {key: record for key, record in records.items()
            if not is_hydrogen(record)}


# ---------------------------------------------------------------------------
# Correspondence and coordinate preservation
# ---------------------------------------------------------------------------

def distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Euclidean distance between two coordinate triples, in Angstroms."""
    return sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5


def renamed_key(key: AtomKey, transformation: tuple[str, str]) -> AtomKey:
    """Return `key` as it should appear after an agreed residue transformation."""
    source_name, prepared_name = transformation
    atom_name = key.atom_name
    mapping = AGREED_RESIDUE_TRANSFORMATIONS.get((source_name, prepared_name))
    if mapping and mapping[0] and key.atom_name == mapping[0]:
        atom_name = mapping[1]
    return AtomKey(
        chain=key.chain, residue_seq=key.residue_seq,
        insertion_code=key.insertion_code, residue_name=prepared_name,
        atom_name=atom_name, altloc=key.altloc,
    )


@dataclass
class CorrespondenceReport:
    """The result of pairing two atom sets and comparing their coordinates."""
    n_paired: int
    n_missing: int
    n_unexpected: int
    max_deviation_angstrom: float
    max_deviation_atom: str | None
    missing_atoms: list[str]
    unexpected_atoms: list[str]
    moved_atoms: list[str]
    applied_transformations: list[str]
    subset_name: str | None = None
    subset_n_paired: int = 0
    subset_n_missing: int = 0
    subset_max_deviation_angstrom: float = 0.0
    subset_max_deviation_atom: str | None = None

    @property
    def coordinates_preserved(self) -> bool:
        return not self.moved_atoms

    @property
    def complete(self) -> bool:
        return not self.missing_atoms

    def as_dict(self) -> dict[str, object]:
        return {
            "n_paired": self.n_paired,
            "n_missing": self.n_missing,
            "n_unexpected": self.n_unexpected,
            "max_deviation_angstrom": round(self.max_deviation_angstrom, 6),
            "max_deviation_atom": self.max_deviation_atom,
            "tolerance_angstrom": COORDINATE_TOLERANCE_ANGSTROM,
            "coordinates_preserved": self.coordinates_preserved,
            "missing_atoms": self.missing_atoms[:50],
            "unexpected_atoms": self.unexpected_atoms[:50],
            "moved_atoms": self.moved_atoms[:50],
            "applied_transformations": self.applied_transformations,
            "subset": None if self.subset_name is None else {
                "name": self.subset_name,
                "n_paired": self.subset_n_paired,
                "n_missing": self.subset_n_missing,
                "max_deviation_angstrom": round(
                    self.subset_max_deviation_angstrom, 6),
                "max_deviation_atom": self.subset_max_deviation_atom,
            },
        }


def compare_atom_sets(
        source: dict[AtomKey, AtomRecord],
        prepared: dict[AtomKey, AtomRecord],
        *,
        residue_transformations: dict[tuple[str, int, str], tuple[str, str]] | None = None,
        deliberately_removed: Iterable[AtomKey] = (),
        subset_keys: Iterable[AtomKey] | None = None,
        subset_name: str | None = None,
        tolerance: float = COORDINATE_TOLERANCE_ANGSTROM,
) -> CorrespondenceReport:
    """Pair two atom sets by identity and compare their coordinates.

    `residue_transformations` maps a residue key to an agreed (source name,
    prepared name) pair, so a renamed residue's atoms are followed rather than
    reported as missing. Only the transformations in
    `AGREED_RESIDUE_TRANSFORMATIONS` are accepted; anything else raises,
    because an unrecorded residue change is exactly what these checks exist to
    catch.

    `deliberately_removed` lists atoms expected to be absent from `prepared`
    (waters, stripped components, deleted carbamyl atoms). They are excluded
    from the comparison rather than counted as losses, and the caller is
    responsible for recording them in the manifest.

    `subset_keys` names a subset — in practice the pocket — reported separately
    so that a local change cannot be averaged away against the whole receptor.
    """
    transformations = residue_transformations or {}
    for residue_key, pair in transformations.items():
        if pair not in AGREED_RESIDUE_TRANSFORMATIONS:
            raise ReceptorError(
                f"residue {residue_key} carries an unrecorded transformation "
                f"{pair[0]} -> {pair[1]}; only "
                f"{sorted(AGREED_RESIDUE_TRANSFORMATIONS)} are agreed"
            )

    removed = set(deliberately_removed)
    subset = set(subset_keys) if subset_keys is not None else None

    paired = 0
    missing: list[str] = []
    moved: list[str] = []
    applied: set[str] = set()
    max_deviation, max_atom = 0.0, None
    subset_paired = subset_missing = 0
    subset_max, subset_max_atom = 0.0, None
    matched_prepared: set[AtomKey] = set()

    for key, record in source.items():
        if key in removed:
            continue
        expected_key = key
        transformation = transformations.get(key.residue_key)
        if transformation is not None:
            expected_key = renamed_key(key, transformation)
            mapping = AGREED_RESIDUE_TRANSFORMATIONS[transformation]
            applied.add(f"{transformation[0]}->{transformation[1]}: {mapping[2]}")
        counterpart = prepared.get(expected_key)
        in_subset = subset is not None and key in subset
        if counterpart is None:
            missing.append(str(key))
            if in_subset:
                subset_missing += 1
            continue
        matched_prepared.add(expected_key)
        paired += 1
        deviation = distance(record.position, counterpart.position)
        if deviation > max_deviation:
            max_deviation, max_atom = deviation, str(key)
        if deviation > tolerance:
            moved.append(f"{key} moved {deviation:.4f} A")
        if in_subset:
            subset_paired += 1
            if deviation > subset_max:
                subset_max, subset_max_atom = deviation, str(key)

    unexpected = [str(key) for key in prepared if key not in matched_prepared]
    return CorrespondenceReport(
        n_paired=paired,
        n_missing=len(missing),
        n_unexpected=len(unexpected),
        max_deviation_angstrom=max_deviation,
        max_deviation_atom=max_atom,
        missing_atoms=sorted(missing),
        unexpected_atoms=sorted(unexpected),
        moved_atoms=sorted(moved),
        applied_transformations=sorted(applied),
        subset_name=subset_name,
        subset_n_paired=subset_paired,
        subset_n_missing=subset_missing,
        subset_max_deviation_angstrom=subset_max,
        subset_max_deviation_atom=subset_max_atom,
    )


# ---------------------------------------------------------------------------
# Pocket census
# ---------------------------------------------------------------------------

def pocket_census(records: dict[AtomKey, AtomRecord], chain: str,
                  residue_numbers: Iterable[int]) -> dict[str, object]:
    """Count the pocket's residues and heavy atoms, and flag incomplete ones.

    Residues are selected by author number on `chain`, which is how the box
    record lists them. A residue carrying fewer heavy atoms than
    `EXPECTED_HEAVY_ATOMS` allows is reported as incomplete: an incomplete
    pocket side chain in the source is a blocking condition under this
    project's rules, not a detail to note later.

    Insertion codes are counted but cannot be selected for here, because the
    pocket list is plain residue numbers; any pocket residue carrying an
    insertion code is reported so the ambiguity is visible.
    """
    wanted = set(residue_numbers)
    per_residue: dict[tuple[int, str], dict[str, object]] = {}
    for key, record in heavy_atoms(records).items():
        if key.chain != chain or key.residue_seq not in wanted:
            continue
        entry = per_residue.setdefault(
            (key.residue_seq, key.insertion_code),
            {"residue_name": key.residue_name, "heavy_atoms": 0,
             "atom_names": [], "insertion_code": key.insertion_code})
        entry["heavy_atoms"] = int(entry["heavy_atoms"]) + 1
        entry["atom_names"].append(key.atom_name)

    found = sorted(number for number, _ in per_residue)
    absent = sorted(wanted - set(found))
    incomplete, with_insertion_codes = [], []
    for (number, code), entry in sorted(per_residue.items()):
        expected = EXPECTED_HEAVY_ATOMS.get(str(entry["residue_name"]))
        if expected is not None and int(entry["heavy_atoms"]) < expected:
            incomplete.append(
                f"{chain}:{number}{code} {entry['residue_name']} "
                f"{entry['heavy_atoms']}/{expected} heavy atoms")
        if code:
            with_insertion_codes.append(f"{chain}:{number}{code}")
    return {
        "chain": chain,
        "n_requested": len(wanted),
        "n_found": len(per_residue),
        "residues_absent": absent,
        "residues_incomplete": incomplete,
        "residues_with_insertion_codes": with_insertion_codes,
        "total_heavy_atoms": sum(
            int(entry["heavy_atoms"]) for entry in per_residue.values()),
        "per_residue": {
            f"{number}{code}": {
                "residue_name": entry["residue_name"],
                "heavy_atoms": entry["heavy_atoms"],
                "expected_heavy_atoms": EXPECTED_HEAVY_ATOMS.get(
                    str(entry["residue_name"])),
            }
            for (number, code), entry in sorted(per_residue.items())
        },
    }


def pocket_atom_keys(records: dict[AtomKey, AtomRecord], chain: str,
                     residue_numbers: Iterable[int]) -> set[AtomKey]:
    """Return the heavy-atom keys of the pocket residues in a record set."""
    wanted = set(residue_numbers)
    return {key for key, record in heavy_atoms(records).items()
            if key.chain == chain and key.residue_seq in wanted}


# ---------------------------------------------------------------------------
# The preparation manifest
# ---------------------------------------------------------------------------

def tool_version(executable: str, *arguments: str) -> dict[str, object]:
    """Return a tool's reported version, or why it could not be determined.

    Absence is recorded as absence. A missing version is never silently
    omitted, because "the manifest does not say" and "the tool was not there"
    are different facts.
    """
    located = shutil.which(executable)
    if located is None:
        return {"executable": executable, "path": None, "version": None,
                "note": "not found on PATH"}
    try:
        finished = subprocess.run([located, *arguments], capture_output=True,
                                  text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return {"executable": executable, "path": located, "version": None,
                "note": f"version query failed: {error}"}
    output = (finished.stdout or finished.stderr or "").strip().splitlines()
    return {"executable": executable, "path": located,
            "version": output[0].strip() if output else None,
            "returncode": finished.returncode}


def build_manifest(*, condition: str, selections: dict[str, object],
                   source: dict[str, object], outputs: dict[str, object],
                   transformations: dict[str, object],
                   removed: dict[str, object], retained: dict[str, object],
                   checks: dict[str, object], commands: list[dict[str, object]],
                   tools: dict[str, object],
                   unresolved: Iterable[str]) -> dict[str, object]:
    """Assemble the receptor preparation manifest.

    The manifest is the record that lets a later stage establish what this
    receptor is: which structure and protein copy it came from, what was
    removed and retained, which chemistry was changed, which checks were run
    and what they found, and which decisions remain unresolved. `dock.py`
    reads it to confirm that a box and a receptor describe the same protein
    copy of the same structure.
    """
    return {
        "schema": MANIFEST_SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "receptor_condition": condition,
        "selections": dict(selections),
        "source_structure": dict(source),
        "outputs": dict(outputs),
        "residue_transformations": dict(transformations),
        "removed_components": dict(removed),
        "retained_components": dict(retained),
        "checks": dict(checks),
        "commands": list(commands),
        "tools": dict(tools),
        "unresolved_decisions": list(unresolved),
    }


def write_manifest(path: Path | str, manifest: dict[str, object],
                   overwrite: bool = False) -> Path:
    """Write the manifest as JSON, refusing to overwrite unless asked."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise ReceptorError(
            f"{destination} already exists; choose a fresh path or overwrite "
            f"deliberately."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def read_manifest(path: Path | str) -> dict[str, object]:
    """Read and validate a receptor manifest written by `write_manifest`."""
    source = Path(path)
    try:
        manifest = json.loads(source.read_text())
    except OSError as error:
        raise ReceptorError(f"cannot read the manifest {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise ReceptorError(f"{source} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ReceptorError(f"{source} does not contain a manifest object.")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ReceptorError(
            f"{source}: expected schema {MANIFEST_SCHEMA!r}, "
            f"found {manifest.get('schema')!r}."
        )
    for field in ("receptor_condition", "selections", "source_structure",
                  "outputs", "checks"):
        if field not in manifest:
            raise ReceptorError(f"{source}: the manifest has no {field!r} field.")
    return manifest


# ---------------------------------------------------------------------------
# Agreement between a box record and a receptor manifest
# ---------------------------------------------------------------------------

def box_and_receptor_disagreements(box_record: dict[str, object],
                                   manifest: dict[str, object],
                                   receptor_sha256: str | None = None) -> list[str]:
    """Return the reasons a box record and a receptor manifest do not match.

    Checked: the source structure's SHA-256, the author protein chain, the
    pocket cutoff and residue list, and, when given, that the receptor file
    being docked is the one the manifest describes. An empty list means these
    particular facts agree.

    What this cannot establish: that the preparation was scientifically
    appropriate, that the pocket is the right one, or that the receptor's atom
    typing and protonation are correct. It establishes that the box and the
    receptor refer to the same protein copy of the same structure, which is a
    prerequisite, not a validation.
    """
    problems: list[str] = []
    box_source = box_record.get("source_structure", {}) or {}
    manifest_source = manifest.get("source_structure", {}) or {}
    box_selection = box_record.get("selection", {}) or {}
    manifest_selection = manifest.get("selections", {}) or {}

    box_hash, manifest_hash = box_source.get("sha256"), manifest_source.get("sha256")
    if box_hash and manifest_hash and box_hash != manifest_hash:
        problems.append(
            f"different source structures: the box was mapped on "
            f"{box_source.get('path')!r} (sha256 {box_hash[:12]}...) but the "
            f"receptor was prepared from {manifest_source.get('path')!r} "
            f"(sha256 {manifest_hash[:12]}...)"
        )
    elif not box_hash or not manifest_hash:
        problems.append(
            "source structure hashes cannot be compared: "
            f"box record {'has' if box_hash else 'lacks'} one, manifest "
            f"{'has' if manifest_hash else 'lacks'} one"
        )

    box_chain = box_selection.get("protein_author_chain_id")
    manifest_chain = manifest_selection.get("protein_author_chain_id")
    if box_chain and manifest_chain and box_chain != manifest_chain:
        problems.append(
            f"different protein copies: the box was mapped on author chain "
            f"{box_chain!r}, the receptor prepared from {manifest_chain!r}"
        )

    box_cutoff = box_selection.get("pocket_cutoff_angstrom")
    manifest_cutoff = manifest_selection.get("pocket_cutoff_angstrom")
    if (box_cutoff is not None and manifest_cutoff is not None
            and float(box_cutoff) != float(manifest_cutoff)):
        problems.append(
            f"different pocket cutoffs: box {box_cutoff} A, "
            f"receptor {manifest_cutoff} A"
        )

    box_pocket = [str(value) for value in
                  box_selection.get("pocket_author_residue_ids", [])]
    manifest_pocket = [str(value) for value in
                       manifest_selection.get("pocket_author_residue_ids", [])]
    if box_pocket and manifest_pocket and box_pocket != manifest_pocket:
        only_box = sorted(set(box_pocket) - set(manifest_pocket))
        only_manifest = sorted(set(manifest_pocket) - set(box_pocket))
        problems.append(
            f"different pocket residue lists: {len(only_box)} only in the box "
            f"record ({only_box[:5]}), {len(only_manifest)} only in the "
            f"manifest ({only_manifest[:5]})"
        )

    if receptor_sha256 is not None:
        recorded = (manifest.get("outputs", {}) or {}).get("receptor_pdbqt_sha256")
        if recorded and recorded != receptor_sha256:
            problems.append(
                f"the receptor being docked is not the one the manifest "
                f"describes: file sha256 {receptor_sha256[:12]}..., manifest "
                f"records {recorded[:12]}..."
            )
        elif not recorded:
            problems.append(
                "the manifest records no receptor PDBQT hash, so the file "
                "being docked cannot be tied to it"
            )
    return problems
