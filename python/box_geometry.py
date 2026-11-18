"""Construct, record and read an AutoDock Vina search box.

This module holds the geometry and the on-disk record so that the stage that
*defines* the box (`map_pocket.py`) and the stage that *uses* it (`dock.py`)
cannot disagree, and so that no centre or dimension is ever transcribed by
hand. It deliberately depends only on NumPy and the standard library, which
keeps it importable, and testable, in an environment without Gemmi or Vina.

Geometry conventions
--------------------
All coordinates and lengths are in Angstroms, in the coordinate frame of the
source structure as deposited (mmCIF `Cartn_x/y/z`). No transformation,
recentring or symmetry operation is applied anywhere in this module, so a box
is only meaningful against a receptor prepared from the same structure and the
same protein copy. The record carries that provenance; it does not enforce it.

A box is an axis-aligned rectangular cuboid, defined by a centre and three
independent dimensions, matching Vina's `center_x/y/z` and `size_x/y/z`.

Centring modes
--------------
`bounds-midpoint`
    The midpoint of the selection's coordinate bounds. Padding then means the
    same thing on opposite faces.
`ligand-atom-mean`
    The unweighted mean of the selection's atom coordinates. This reproduces
    the historical behaviour of `map_pocket.py` and is retained only so that
    older boxes can be reconstructed. It shifts the box towards whichever part
    of the selection carries more atoms, so the clearance on one face is
    reduced by exactly the amount added to the opposite face.

    Because of that displacement, atom-mean centring **cannot** honour a
    requested per-face padding: the padding would be met on one face and
    breached on the opposite one. `validated_centre_and_sizing` therefore
    rejects the combination, and atom-mean centring is permitted only with
    explicit dimensions, where the caller has stated the size outright and no
    clearance is being promised.

Sizing modes
------------
`explicit-size`
    Dimensions given directly, used exactly as supplied. One value gives a
    cube; three give independent axes.
`padding-per-face`
    Dimensions derived as `extent + 2 x padding` per axis, where `extent` is
    the per-axis span of the selected atoms. The result is rounded *up* to
    0.1 A so that the stated padding is a guaranteed minimum rather than a
    nominal figure that rounding can erode.

Limitations
-----------
- The box is an axis-aligned estimate of a search region. It is not a pocket
  definition, and covering a region is not evidence that the region is the
  right one.
- Vina's box restricts the positions of *movable* atoms only. Receptor atoms
  outside the box are not excluded from the receptor; whether they still
  contribute to the scoring grid inside the box has not been verified here.
- No check in this module establishes that the centre lies in a pocket, or
  that the selection used to derive it is the intended one.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SCHEMA = "mur-ligase-docking-box/1"
CENTRE_MODES = ("bounds-midpoint", "ligand-atom-mean")
SIZING_MODES = ("explicit-size", "padding-per-face")
DIMENSION_ROUNDING_ANGSTROM = 0.1

# A search box far larger than the structure it was derived from is almost
# always a transcription or selection error rather than an intended choice.
MAXIMUM_DIMENSION_ANGSTROM = 1000.0


class BoxError(ValueError):
    """Raised when a box, or a box record, is not usable as given."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def validated_points(points: np.ndarray, description: str) -> np.ndarray:
    """Return an (n, 3) float array of finite coordinates, or raise.

    `description` names the selection in the diagnostic, because an empty or
    non-finite selection is nearly always a selection mistake rather than a
    corrupt file.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise BoxError(
            f"{description}: expected an (n, 3) coordinate array, "
            f"got shape {array.shape}."
        )
    if array.shape[0] == 0:
        raise BoxError(f"{description}: the selection contains no atoms.")
    if not np.isfinite(array).all():
        raise BoxError(f"{description}: the selection contains non-finite coordinates.")
    return array


def centre_from_points(points: np.ndarray, mode: str) -> np.ndarray:
    """Return the box centre in Angstroms for one of `CENTRE_MODES`."""
    array = validated_points(points, "box centring")
    if mode == "bounds-midpoint":
        return (array.max(0) + array.min(0)) / 2.0
    if mode == "ligand-atom-mean":
        return array.mean(0)
    raise BoxError(f"Unknown centring mode {mode!r}; expected one of {CENTRE_MODES}.")


def validated_centre_and_sizing(centre_mode: str, sizing_mode: str) -> None:
    """Reject centring and sizing combinations that cannot deliver the padding.

    Per-face padding is a guarantee: the clearance between the selection's
    bounds and every one of the six walls must be at least the value asked
    for. Only `bounds-midpoint` centring can deliver that, because any other
    centre is displaced from the bounds and subtracts from one face exactly
    what it adds to the opposite one. `ligand-atom-mean` is therefore allowed
    only with explicit dimensions.

    Raises `BoxError` for an unknown mode or a combination that would quietly
    breach the stated padding.
    """
    if centre_mode not in CENTRE_MODES:
        raise BoxError(f"Unknown centring mode {centre_mode!r}; expected {CENTRE_MODES}.")
    if sizing_mode not in SIZING_MODES:
        raise BoxError(f"Unknown sizing mode {sizing_mode!r}; expected {SIZING_MODES}.")
    if sizing_mode == "padding-per-face" and centre_mode != "bounds-midpoint":
        raise BoxError(
            f"Centring mode {centre_mode!r} cannot honour a per-face padding: it "
            f"displaces the box from the selection's bounds, so the padding would "
            f"be breached on one face and exceeded on the opposite one. Use "
            f"'bounds-midpoint' with padding, or give explicit dimensions."
        )


def extent_from_points(points: np.ndarray) -> np.ndarray:
    """Return the per-axis span (max - min) of a selection, in Angstroms."""
    array = validated_points(points, "box extent")
    return array.max(0) - array.min(0)


def round_up(values: np.ndarray, step: float = DIMENSION_ROUNDING_ANGSTROM) -> np.ndarray:
    """Round each value up to a multiple of `step`.

    Rounding up, rather than to nearest, is what makes a requested padding a
    floor: rounding to nearest can remove up to half a step from the clearance
    the caller asked for.
    """
    # Values are rounded to 10 decimals first so that a representation error
    # such as 23.500000000000004 does not push the result to the next step.
    return np.ceil(np.round(np.asarray(values, dtype=float) / step, 10)) * step


def validated_dimensions(values: Sequence[float]) -> np.ndarray:
    """Return three finite, strictly positive box dimensions, or raise."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != 3:
        raise BoxError(f"Expected three box dimensions, got {array.size}.")
    if not np.isfinite(array).all():
        raise BoxError(f"Box dimensions must be finite, got {array.tolist()}.")
    if (array <= 0).any():
        raise BoxError(f"Box dimensions must be strictly positive, got {array.tolist()}.")
    if (array > MAXIMUM_DIMENSION_ANGSTROM).any():
        raise BoxError(
            f"Box dimensions exceed {MAXIMUM_DIMENSION_ANGSTROM:g} A "
            f"({array.tolist()}); check the units and the selection."
        )
    return array


def dimensions_from_padding(points: np.ndarray,
                            padding_per_face: Sequence[float]) -> np.ndarray:
    """Return `extent + 2 x padding` per axis, rounded up to 0.1 A.

    `padding_per_face` is the clearance requested between the selection's
    bounds and each box wall, in Angstroms, per axis. It must be finite and
    non-negative; zero is allowed and means a box that just contains the
    selection's bounds.
    """
    padding = np.asarray(padding_per_face, dtype=float).reshape(-1)
    if padding.size != 3:
        raise BoxError(f"Expected three padding values, got {padding.size}.")
    if not np.isfinite(padding).all() or (padding < 0).any():
        raise BoxError(
            f"Padding per face must be finite and non-negative, got {padding.tolist()}."
        )
    return validated_dimensions(round_up(extent_from_points(points) + 2.0 * padding))


def face_clearances(centre: Sequence[float], size: Sequence[float],
                    points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed gaps from a point cloud's bounds to the six box walls, in A.

    Returns `(low, high)` arrays over x, y, z. A negative entry means the cloud
    extends past that wall. This is a property of the cloud's bounding box, not
    of individual atoms; see `distance_to_box` for the per-atom quantity.
    """
    array = validated_points(points, "face clearance")
    centre = np.asarray(centre, dtype=float)
    half = validated_dimensions(size) / 2.0
    return array.min(0) - (centre - half), (centre + half) - array.max(0)


def distance_to_box(centre: Sequence[float], size: Sequence[float],
                    points: np.ndarray) -> np.ndarray:
    """Euclidean distance from each atom to the nearest point of the box volume.

    Zero for atoms inside the box. For a rigid receptor atom this, rather than
    face clearance, is the quantity that bears on whether a movable ligand atom
    placed inside the box could come within interaction distance of it.
    """
    array = validated_points(points, "distance to box")
    centre = np.asarray(centre, dtype=float)
    half = validated_dimensions(size) / 2.0
    return np.linalg.norm(np.clip(np.abs(array - centre) - half, 0.0, None), axis=-1)


def points_inside(centre: Sequence[float], size: Sequence[float],
                  points: np.ndarray) -> np.ndarray:
    """Boolean mask of atoms lying within the closed box volume."""
    array = validated_points(points, "box membership")
    centre = np.asarray(centre, dtype=float)
    half = validated_dimensions(size) / 2.0
    return ((array >= centre - half) & (array <= centre + half)).all(1)


# ---------------------------------------------------------------------------
# The box record
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository_root: Path) -> dict[str, object]:
    """Return the current commit and whether the tree is dirty, if available.

    A commit hash alone cannot identify modified working code, so the dirty
    flag is recorded beside it. Both are reported as unknown rather than
    guessed when Git is unavailable.
    """
    def run(*arguments: str) -> str | None:
        try:
            finished = subprocess.run(
                ["git", "-C", str(repository_root), *arguments],
                capture_output=True, text=True, check=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return finished.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "working_tree_dirty": None if status is None else bool(status),
    }


def build_record(*,
                 centre: Sequence[float],
                 size: Sequence[float],
                 centre_mode: str,
                 sizing: dict[str, object],
                 selection: dict[str, object],
                 source_structure_path: Path | str | None,
                 software: dict[str, object] | None = None,
                 notes: Iterable[str] = ()) -> dict[str, object]:
    """Assemble a machine-readable box record.

    The record is the only intended route from box definition to docking: it
    carries the centre, the three dimensions, the selection the box was derived
    from, and the structure and coordinate frame it belongs to. Everything
    needed to notice that a box has been paired with the wrong protein copy is
    therefore in one place.
    """
    centre_array = np.asarray(centre, dtype=float).reshape(-1)
    if centre_array.size != 3 or not np.isfinite(centre_array).all():
        raise BoxError(f"The box centre must be three finite values, got {centre!r}.")
    if centre_mode not in CENTRE_MODES:
        raise BoxError(f"Unknown centring mode {centre_mode!r}; expected {CENTRE_MODES}.")
    size_array = validated_dimensions(size)

    source: dict[str, object] = {
        "path": None if source_structure_path is None else str(source_structure_path),
        "sha256": None,
        "model": 1,
        "coordinate_frame": (
            "source structure as read (mmCIF Cartn_x/y/z); no transformation, "
            "recentring or symmetry operation applied"
        ),
    }
    if source_structure_path is not None and Path(source_structure_path).is_file():
        source["sha256"] = file_sha256(Path(source_structure_path))

    return {
        "schema": SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "units": "angstrom",
        "centre_mode": centre_mode,
        "centre_angstrom": [float(value) for value in centre_array],
        "size_angstrom": [float(value) for value in size_array],
        "sizing": dict(sizing),
        "selection": dict(selection),
        "source_structure": source,
        "software": dict(software or {}),
        "notes": list(notes),
    }


def write_record(path: Path | str, record: dict[str, object],
                 overwrite: bool = False) -> Path:
    """Write a box record as JSON, refusing to overwrite unless asked.

    Collisions are refused because a stale box silently reused is exactly the
    failure this record exists to prevent.
    """
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise BoxError(
            f"{destination} already exists; choose a fresh path or pass the "
            f"overwrite option deliberately."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return destination


def read_record(path: Path | str) -> dict[str, object]:
    """Read and validate a box record written by `write_record`.

    Validates the schema name, the units, the centre and the three dimensions.
    It cannot validate that the box is scientifically appropriate, nor that the
    receptor it will be used with came from the recorded structure; `dock.py`
    performs the receptor consistency checks it can.
    """
    source = Path(path)
    try:
        record = json.loads(source.read_text())
    except OSError as error:
        raise BoxError(f"Cannot read the box record {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise BoxError(f"{source} is not valid JSON: {error}") from error
    if not isinstance(record, dict):
        raise BoxError(f"{source} does not contain a box record object.")

    schema = record.get("schema")
    if schema != SCHEMA:
        raise BoxError(f"{source}: expected schema {SCHEMA!r}, found {schema!r}.")
    units = record.get("units")
    if units != "angstrom":
        raise BoxError(f"{source}: expected units 'angstrom', found {units!r}.")

    centre = np.asarray(record.get("centre_angstrom", []), dtype=float).reshape(-1)
    if centre.size != 3 or not np.isfinite(centre).all():
        raise BoxError(f"{source}: centre_angstrom must be three finite values.")
    validated_dimensions(record.get("size_angstrom", []))
    return record


def record_centre(record: dict[str, object]) -> np.ndarray:
    """Return the record's centre as a (3,) array in Angstroms."""
    return np.asarray(record["centre_angstrom"], dtype=float)


def record_size(record: dict[str, object]) -> np.ndarray:
    """Return the record's three dimensions as a (3,) array in Angstroms."""
    return validated_dimensions(record["size_angstrom"])


def describe(record: dict[str, object]) -> str:
    """Return a short human-readable summary of a box record."""
    centre = record_centre(record)
    size = record_size(record)
    selection = record.get("selection", {})
    return (
        f"centre {np.round(centre, 3).tolist()} A ({record.get('centre_mode')}), "
        f"size {np.round(size, 2).tolist()} A, "
        f"selection {selection.get('description', 'unspecified')}, "
        f"source {record.get('source_structure', {}).get('path')}"
    )


# ---------------------------------------------------------------------------
# ChimeraX visualisation
# ---------------------------------------------------------------------------

def bild_lines(record: dict[str, object], *,
               edge_radius_angstrom: float = 0.15,
               transparency: float = 0.8) -> list[str]:
    """Return ChimeraX BILD lines drawing the box as a wireframe cuboid.

    The box is drawn as twelve edge cylinders plus a transparent solid body and
    a small sphere at the centre, in the record's own coordinate frame. Open it
    alongside a receptor built from the recorded structure; if the box appears
    far from the protein, the box and the receptor do not belong together.
    """
    centre = record_centre(record)
    half = record_size(record) / 2.0
    low, high = centre - half, centre + half
    if not math.isfinite(edge_radius_angstrom) or edge_radius_angstrom <= 0:
        raise BoxError("The edge radius must be finite and positive.")

    corners = [
        np.array([x, y, z])
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    lines = [
        "# ChimeraX BILD: AutoDock Vina search box",
        f"# schema {record.get('schema')}  written {record.get('written_utc')}",
        f"# {describe(record)}",
        "# Coordinates are in the frame of the recorded source structure.",
        ".color cornflower blue",
    ]
    for start, end in itertools.combinations(corners, 2):
        # An edge joins two corners differing along exactly one axis.
        if (np.abs(start - end) > 1e-9).sum() != 1:
            continue
        lines.append(
            ".cylinder "
            f"{start[0]:.3f} {start[1]:.3f} {start[2]:.3f} "
            f"{end[0]:.3f} {end[1]:.3f} {end[2]:.3f} "
            f"{edge_radius_angstrom:.3f}"
        )
    lines.append(f".transparency {transparency:.2f}")
    lines.append(
        ".box "
        f"{low[0]:.3f} {low[1]:.3f} {low[2]:.3f} "
        f"{high[0]:.3f} {high[1]:.3f} {high[2]:.3f}"
    )
    lines.append(".transparency 0.0")
    lines.append(".color red")
    lines.append(f".sphere {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f} 0.4")
    return lines
