#!/usr/bin/env python3
"""Validate a BoltzGen refolded design and place it in the receptor frame.

Purpose
-------
For one refolded complex produced by BoltzGen, establish the facts that must
hold before the designed peptide can be prepared as a docking ligand:

1. which chain is the designed peptide, identified by **sequence** and not by
   a chain-length heuristic;
2. that its residues carry a complete set of standard heavy atoms;
3. that every alpha carbon is L-configured;
4. the rigid transform that superposes the complex's **protein** chain onto
   the deposited receptor, together with its RMSD and residue count;
5. the designed peptide's position in that receptor frame, its minimum
   distances to named receptor components such as ANP, and whether it lies
   inside a recorded docking box.

The peptide is never superposed on anything. Only the protein chain is
fitted, and the resulting transform is applied to the peptide unchanged, so
the peptide's placement relative to the receptor is preserved exactly as
BoltzGen predicted it. Fitting the peptide would destroy the quantity being
measured.

Units, indexing and conventions
-------------------------------
All distances are Angstroms. Coordinates come from the first model of each
structure. Protein correspondence is established on (author residue number,
insertion code) with residue-name equality, and measured on CA atoms only.
Box containment uses the half-open convention that an atom is inside when its
displacement from the box centre is at most half the box dimension on every
axis.

Alpha-carbon configuration is the sign of

    det = (N - CA) x (C - CA) . (CB - CA)

**The sign convention is empirical, not assumed.** Over 1P3D, all 839
amino-acid residues bearing a CB give a strictly positive determinant, so
positive is taken as L and negative as D. Glycine has no CB and is reported
as having no centre rather than as a failure.

Assumptions
-----------
- The designed peptide's one-letter sequence is recorded in the campaign
  metrics as `designed_sequence` and matches exactly one chain.
- Residues are standard amino acids. A non-standard residue is reported, not
  silently accepted or substituted.
- The deposited receptor is the authoritative frame; the refolded complex's
  protein chain is re-predicted and so will not match it exactly.

Limitations
-----------
Nothing here validates binding, scores a pose, or establishes that a design
is a binder. A peptide that passes every check is a peptide that can be
*prepared*, nothing more. The ANP distances are geometric contacts under the
unvalidated ANP representation already recorded as outstanding; they are not
an interaction energy. Box containment is a property of the recorded box, not
evidence that the box is correctly placed.

Failure conditions
------------------
Raises `GeometryError` when a structure cannot be read, when the designed
sequence matches no chain or more than one, when a required backbone atom is
absent for the superposition, or when fewer than three protein residues
correspond. Per-residue atom and chirality problems are reported in the
record rather than raised, so that a whole cohort can be screened in one
pass and every failing design retained with its reason.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np

# Standard heavy (non-hydrogen) atoms of each amino acid, backbone included.
# OXT is permitted additionally on a C-terminal residue and is not required.
BACKBONE = ("N", "CA", "C", "O")
SIDE_CHAIN_ATOMS: Dict[str, Tuple[str, ...]] = {
    "GLY": (),
    "ALA": ("CB",),
    "SER": ("CB", "OG"),
    "CYS": ("CB", "SG"),
    "THR": ("CB", "OG1", "CG2"),
    "VAL": ("CB", "CG1", "CG2"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PRO": ("CB", "CG", "CD"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
}
OPTIONAL_ATOMS = frozenset({"OXT"})

# Empirically calibrated on 1P3D: every L-residue with a CB gives det > 0.
L_CONFIGURATION_SIGN = 1.0
MINIMUM_SUPERPOSITION_RESIDUES = 3


class GeometryError(ValueError):
    """A condition that makes the reported geometry untrustworthy."""


@dataclass
class ResidueFinding:
    """Per-residue atom-completeness and configuration result."""

    sequence_index: int
    name: str
    author_seqid: str
    missing_atoms: List[str] = field(default_factory=list)
    unexpected_atoms: List[str] = field(default_factory=list)
    configuration: Optional[str] = None
    chirality_determinant: Optional[float] = None
    note: str = ""

    @property
    def acceptable(self) -> bool:
        """True when nothing about this residue blocks ligand preparation."""
        return (
            not self.missing_atoms
            and self.configuration in ("L", None)
            and not self.note
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "sequence_index": self.sequence_index,
            "name": self.name,
            "author_seqid": self.author_seqid,
            "missing_atoms": self.missing_atoms,
            "unexpected_atoms": self.unexpected_atoms,
            "configuration": self.configuration,
            "chirality_determinant": self.chirality_determinant,
            "note": self.note,
            "acceptable": self.acceptable,
        }


def read_structure(path: Path) -> gemmi.Structure:
    """Read a structure's first model, with entity setup, or fail loudly."""
    if not path.is_file():
        raise GeometryError(f"Structure not found: {path}")
    try:
        structure = gemmi.read_structure(str(path))
    except Exception as error:  # gemmi raises assorted exception types
        raise GeometryError(f"Could not read {path}: {error}") from error
    structure.setup_entities()
    if len(structure) == 0:
        raise GeometryError(f"Structure contains no model: {path}")
    return structure


def one_letter_sequence(chain: gemmi.Chain) -> str:
    """Return a chain's amino-acid sequence in one-letter code.

    Non-amino-acid residues are skipped, so a chain carrying waters or ions
    alongside its polymer still yields its peptide sequence. An unrecognised
    amino acid contributes 'X', which will prevent an exact sequence match and
    so surface as a mismatch rather than as a silent substitution.
    """
    letters = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is None or not info.is_amino_acid():
            continue
        letters.append(gemmi.find_tabulated_residue(residue.name).one_letter_code.upper())
    return "".join(letters)


def amino_acid_residues(chain: gemmi.Chain) -> List[gemmi.Residue]:
    """Return a chain's amino-acid residues in order."""
    out = []
    for residue in chain:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is not None and info.is_amino_acid():
            out.append(residue)
    return out


def identify_chain_by_sequence(
    model: gemmi.Model, designed_sequence: str
) -> gemmi.Chain:
    """Find the single chain whose sequence equals `designed_sequence`.

    Identification is by sequence precisely because chain-length heuristics
    are unreliable here: a retained cofactor or a short receptor fragment can
    make "the shortest chain" the wrong answer. Zero or several matches are
    both fatal, because the design's identity would be ambiguous.
    """
    wanted = designed_sequence.strip().upper()
    if not wanted:
        raise GeometryError("designed_sequence is empty")
    matches = [c for c in model if one_letter_sequence(c) == wanted]
    if not matches:
        found = {c.name: one_letter_sequence(c) for c in model}
        raise GeometryError(
            f"No chain matches the designed sequence {wanted!r}. "
            f"Chain sequences present: {found}"
        )
    if len(matches) > 1:
        raise GeometryError(
            f"Designed sequence {wanted!r} matches several chains: "
            f"{[c.name for c in matches]}. The design cannot be identified."
        )
    return matches[0]


def heavy_atom_names(residue: gemmi.Residue) -> List[str]:
    """Return the residue's non-hydrogen atom names, in file order."""
    return [
        atom.name
        for atom in residue
        if atom.element != gemmi.Element("H") and atom.element != gemmi.Element("D")
    ]


def calpha_configuration(
    residue: gemmi.Residue,
) -> Tuple[Optional[str], Optional[float]]:
    """Classify a residue's alpha carbon as 'L' or 'D'.

    Returns (None, None) when the residue has no chiral alpha carbon, which is
    the case for glycine, or when a required atom is absent. The determinant
    is returned so a borderline value can be inspected rather than trusted.
    """
    try:
        n = np.asarray(residue["N"][0].pos.tolist(), dtype=float)
        ca = np.asarray(residue["CA"][0].pos.tolist(), dtype=float)
        c = np.asarray(residue["C"][0].pos.tolist(), dtype=float)
        cb = np.asarray(residue["CB"][0].pos.tolist(), dtype=float)
    except (IndexError, RuntimeError, KeyError):
        return None, None
    determinant = float(np.dot(np.cross(n - ca, c - ca), cb - ca))
    if determinant == 0.0:
        return None, determinant
    sign = math.copysign(1.0, determinant)
    return ("L" if sign == L_CONFIGURATION_SIGN else "D"), determinant


def inspect_peptide_residues(chain: gemmi.Chain) -> List[ResidueFinding]:
    """Check atom completeness and alpha-carbon configuration residue by residue.

    Problems are recorded, not raised: screening a cohort needs every failing
    design and its reason, not an abort on the first one.
    """
    findings: List[ResidueFinding] = []
    residues = amino_acid_residues(chain)
    for index, residue in enumerate(residues, start=1):
        seqid = f"{residue.seqid.num}{residue.seqid.icode.strip()}"
        finding = ResidueFinding(
            sequence_index=index, name=residue.name, author_seqid=seqid
        )
        if residue.name not in SIDE_CHAIN_ATOMS:
            finding.note = (
                f"{residue.name} is not a standard amino acid; its expected "
                "heavy atoms and chirality are not defined here"
            )
            findings.append(finding)
            continue
        expected = set(BACKBONE) | set(SIDE_CHAIN_ATOMS[residue.name])
        present = set(heavy_atom_names(residue))
        finding.missing_atoms = sorted(expected - present)
        finding.unexpected_atoms = sorted(present - expected - OPTIONAL_ATOMS)
        finding.configuration, finding.chirality_determinant = calpha_configuration(
            residue
        )
        findings.append(finding)
    return findings


def ca_correspondence(
    mobile: gemmi.Chain, reference: gemmi.Chain
) -> Tuple[List[gemmi.Position], List[gemmi.Position], List[str]]:
    """Pair CA atoms of two protein chains on identity and numbering.

    Correspondence is on (author residue number, insertion code) with
    residue-name equality. A residue present in one chain only, or renamed, is
    skipped and reported, so a silent mismatch cannot inflate the fit.
    """
    def index(chain: gemmi.Chain) -> Dict[Tuple[int, str], gemmi.Residue]:
        out = {}
        for residue in amino_acid_residues(chain):
            out[(residue.seqid.num, residue.seqid.icode.strip())] = residue
        return out

    mobile_index, reference_index = index(mobile), index(reference)
    mobile_positions: List[gemmi.Position] = []
    reference_positions: List[gemmi.Position] = []
    skipped: List[str] = []
    for key in sorted(set(mobile_index) & set(reference_index)):
        a, b = mobile_index[key], reference_index[key]
        if a.name != b.name:
            skipped.append(f"{key[0]}{key[1]}: {a.name} vs {b.name}")
            continue
        try:
            mobile_positions.append(a["CA"][0].pos)
            reference_positions.append(b["CA"][0].pos)
        except (IndexError, RuntimeError, KeyError):
            skipped.append(f"{key[0]}{key[1]}: CA absent")
    return mobile_positions, reference_positions, skipped


def superpose_protein(
    mobile: gemmi.Chain, reference: gemmi.Chain
) -> Tuple[gemmi.Transform, Dict[str, object]]:
    """Fit the mobile protein chain onto the reference chain on CA atoms.

    Returns the rigid transform and a report of what the fit was based on.
    This transform is the only thing that may be applied to the designed
    peptide.
    """
    mobile_positions, reference_positions, skipped = ca_correspondence(
        mobile, reference
    )
    if len(mobile_positions) < MINIMUM_SUPERPOSITION_RESIDUES:
        raise GeometryError(
            f"Only {len(mobile_positions)} corresponding CA atoms between "
            f"chains {mobile.name!r} and {reference.name!r}; at least "
            f"{MINIMUM_SUPERPOSITION_RESIDUES} are required to define a frame."
        )
    result = gemmi.superpose_positions(reference_positions, mobile_positions)
    report = {
        "matched_ca_atoms": len(mobile_positions),
        "rmsd_angstrom": round(float(result.rmsd), 4),
        "skipped_residues": skipped,
        "mobile_chain": mobile.name,
        "reference_chain": reference.name,
        "note": "Protein CA fit only. The peptide is never fitted.",
    }
    return result.transform, report


def transformed_positions(
    residues: Sequence[gemmi.Residue], transform: Optional[gemmi.Transform]
) -> np.ndarray:
    """Return heavy-atom coordinates, optionally mapped by a rigid transform."""
    rows: List[List[float]] = []
    for residue in residues:
        for atom in residue:
            if atom.element in (gemmi.Element("H"), gemmi.Element("D")):
                continue
            position = (
                transform.apply(atom.pos) if transform is not None else atom.pos
            )
            rows.append([position.x, position.y, position.z])
    if not rows:
        raise GeometryError("No heavy atoms to transform")
    return np.asarray(rows, dtype=float)


def component_positions(
    model: gemmi.Model,
    component: str,
    author_chains: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return heavy-atom coordinates of residues named `component`.

    `author_chains` restricts the search to the named author chains, and
    **must be supplied when the structure holds more than one crystallographic
    copy**. 1P3D contains two copies of the whole assembly, in author chains A
    and B, each with its own UMA, ANP and pair of manganese ions.

    Pooling them does not fabricate a contact across copies: the two sites are
    about 89 A apart. What it does is report whichever copy happens to be
    closer, which may not be the copy under assessment. Measured here, the
    UMA-to-ANP minimum is 3.96 A within copy A and 3.02 A within copy B, so a
    pooled search while assessing copy A returns copy B's 3.02 A -- a real
    distance belonging to the wrong site. The two copies genuinely differ, so
    there is no copy-independent answer to fall back on.

    A copy count is not inferred here: the caller states which chains belong
    to the site being assessed, because a shared residue name does not
    establish which ligand belongs to which protein site.
    """
    wanted = None if author_chains is None else {str(c) for c in author_chains}
    rows: List[List[float]] = []
    for chain in model:
        if wanted is not None and chain.name not in wanted:
            continue
        for residue in chain:
            if residue.name != component:
                continue
            for atom in residue:
                if atom.element in (gemmi.Element("H"), gemmi.Element("D")):
                    continue
                rows.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def minimum_distance(
    first: np.ndarray, second: np.ndarray, contact_cutoff: float
) -> Dict[str, object]:
    """Report the minimum heavy-atom separation and the contact count.

    `contact_cutoff` is a stated geometric threshold in Angstroms, strictly
    less-than, matching the pocket-mapping convention used elsewhere in this
    project. A contact is a geometric proximity, never an interaction.
    """
    if first.size == 0 or second.size == 0:
        return {
            "minimum_angstrom": None,
            "contacts_below_cutoff": None,
            "cutoff_angstrom": contact_cutoff,
            "note": "One of the atom sets is empty; no distance is defined.",
        }
    deltas = first[:, None, :] - second[None, :, :]
    distances = np.sqrt((deltas * deltas).sum(axis=-1))
    return {
        "minimum_angstrom": round(float(distances.min()), 4),
        "contacts_below_cutoff": int((distances < contact_cutoff).sum()),
        "cutoff_angstrom": contact_cutoff,
    }


def box_containment(points: np.ndarray, box_record: Dict) -> Dict[str, object]:
    """Report how much of a peptide lies inside a recorded docking box.

    An atom is inside when its absolute displacement from the box centre is at
    most half the box dimension on every axis. The per-axis overshoot is the
    largest excess beyond the half-dimension, which says how much larger the
    box would have to be to contain the peptide about the same centre.
    """
    for key in ("centre_angstrom", "size_angstrom"):
        if key not in box_record:
            raise GeometryError(f"Box record has no {key!r}")
    centre = np.asarray(box_record["centre_angstrom"], dtype=float)
    size = np.asarray(box_record["size_angstrom"], dtype=float)
    if centre.shape != (3,) or size.shape != (3,):
        raise GeometryError("Box centre and size must each have three values")
    if not np.all(np.isfinite(centre)) or not np.all(size > 0):
        raise GeometryError("Box centre must be finite and dimensions positive")
    half = size / 2.0
    displacement = np.abs(points - centre)
    inside_per_axis = displacement <= half
    inside = inside_per_axis.all(axis=1)
    overshoot = np.maximum(displacement.max(axis=0) - half, 0.0)
    required = 2.0 * displacement.max(axis=0)
    return {
        "heavy_atoms": int(points.shape[0]),
        "atoms_inside": int(inside.sum()),
        "fraction_inside": round(float(inside.mean()), 4),
        "fully_contained": bool(inside.all()),
        "per_axis_overshoot_angstrom": [round(float(v), 4) for v in overshoot],
        "required_size_about_this_centre_angstrom": [
            round(float(v), 4) for v in required
        ],
        "box_centre_angstrom": [float(v) for v in centre],
        "box_size_angstrom": [float(v) for v in size],
    }


def assess_design(
    refolded_cif: Path,
    designed_sequence: str,
    reference_structure: Path,
    reference_chain_name: str,
    box_record: Optional[Dict] = None,
    clash_components: Sequence[str] = ("ANP",),
    contact_cutoff: float = 4.0,
    design_id: str = "",
    component_author_chains: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Run every check for one refolded design and return a single record.

    The protein chain of the refolded complex is taken to be the longest
    amino-acid chain that is *not* the designed peptide. That is a heuristic,
    and it is reported as such in the record, but it is applied only after the
    peptide has been identified by exact sequence, so it cannot misidentify
    the design itself.
    """
    structure = read_structure(refolded_cif)
    model = structure[0]
    peptide_chain = identify_chain_by_sequence(model, designed_sequence)
    peptide_residues = amino_acid_residues(peptide_chain)
    findings = inspect_peptide_residues(peptide_chain)

    other_chains = [
        c
        for c in model
        if c.name != peptide_chain.name and amino_acid_residues(c)
    ]
    if not other_chains:
        raise GeometryError(
            f"{refolded_cif}: no protein chain besides the designed peptide, "
            "so no receptor frame can be established."
        )
    protein_chain = max(other_chains, key=lambda c: len(amino_acid_residues(c)))

    reference = read_structure(reference_structure)
    reference_chains = [
        c for c in reference[0] if c.name == reference_chain_name
    ]
    if not reference_chains:
        raise GeometryError(
            f"Reference chain {reference_chain_name!r} not found in "
            f"{reference_structure}"
        )
    transform, fit = superpose_protein(protein_chain, reference_chains[0])

    peptide_points = transformed_positions(peptide_residues, transform)
    # Restrict cofactor searches to the receptor copy being assessed. Where the
    # caller names no chains, default to the reference protein chain itself,
    # which is the copy the peptide was placed against.
    component_chains = (
        list(component_author_chains)
        if component_author_chains is not None
        else [reference_chain_name]
    )
    clashes = {}
    for component in clash_components:
        component_points = component_positions(
            reference[0], component, component_chains
        )
        clashes[component] = minimum_distance(
            peptide_points, component_points, contact_cutoff
        )
        clashes[component]["author_chains_searched"] = component_chains
    receptor_points = transformed_positions(
        amino_acid_residues(reference_chains[0]), None
    )
    clashes["receptor_protein"] = minimum_distance(
        peptide_points, receptor_points, contact_cutoff
    )

    failures = [f.as_dict() for f in findings if not f.acceptable]
    record: Dict[str, object] = {
        "schema": "mur-ligase-peptide-geometry/1",
        "design_id": design_id,
        "refolded_cif": str(refolded_cif),
        "designed_sequence": designed_sequence,
        "residues": len(peptide_residues),
        "peptide_chain": peptide_chain.name,
        "protein_chain": protein_chain.name,
        "protein_chain_selection": (
            "longest amino-acid chain other than the sequence-identified peptide"
        ),
        "reference_structure": str(reference_structure),
        "reference_chain": reference_chain_name,
        "protein_superposition": fit,
        "peptide_heavy_atoms": int(peptide_points.shape[0]),
        "residue_findings": [f.as_dict() for f in findings],
        "residue_failures": failures,
        "atoms_complete": all(not f.missing_atoms for f in findings),
        "all_calpha_L": all(f.configuration in ("L", None) for f in findings),
        "d_configured_residues": [
            f.sequence_index for f in findings if f.configuration == "D"
        ],
        "distances": clashes,
        "interpretation": (
            "Geometry and chemistry checks only. Passing them means the design "
            "can be prepared as a ligand; it is not evidence of binding. "
            "Distances to ANP are geometric and use the unvalidated ANP "
            "representation recorded as outstanding."
        ),
    }
    if box_record is not None:
        record["box_containment"] = box_containment(peptide_points, box_record)
    record["usable_for_preparation"] = bool(
        record["atoms_complete"] and record["all_calpha_L"] and not failures
    )
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("refolded_cif", type=Path)
    parser.add_argument(
        "--designed-sequence",
        required=True,
        help="One-letter sequence recorded for this design in the metrics CSV",
    )
    parser.add_argument(
        "--reference", type=Path, required=True, help="Deposited receptor, e.g. 1P3D.cif"
    )
    parser.add_argument(
        "--reference-chain",
        default="A",
        help="Author chain name of the receptor protein (default: A)",
    )
    parser.add_argument("--design-id", default="")
    parser.add_argument("--box-record", type=Path, default=None)
    parser.add_argument(
        "--clash-component",
        action="append",
        default=None,
        help="Receptor component to measure against, repeatable (default: ANP)",
    )
    parser.add_argument(
        "--component-author-chain",
        action="append",
        default=None,
        help="Author chain(s) holding the components to measure against, "
        "repeatable. Required when the reference holds several "
        "crystallographic copies; defaults to --reference-chain.",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=4.0,
        help="Strict less-than contact threshold in Angstroms (default: 4.0)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.contact_cutoff <= 0:
        parser.error("--contact-cutoff must be positive")
    if args.json_out is not None and args.json_out.exists():
        parser.error(f"Refusing to overwrite an existing file: {args.json_out}")

    box_record = None
    if args.box_record is not None:
        box_record = json.loads(args.box_record.read_text())

    try:
        record = assess_design(
            args.refolded_cif,
            args.designed_sequence,
            args.reference,
            args.reference_chain,
            box_record=box_record,
            clash_components=tuple(args.clash_component or ("ANP",)),
            contact_cutoff=args.contact_cutoff,
            design_id=args.design_id,
            component_author_chains=args.component_author_chain,
        )
    except GeometryError as error:
        print(f"assessment failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(record, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"record: {args.json_out}", file=sys.stderr)
    return 0 if record["usable_for_preparation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
