"""Place one BoltzGen refolded peptide into the standard receptor frame.

This is stage one of the peptide scoring pilot: everything that can be done
with Gemmi and NumPy alone, before Meeko or Vina are involved. It takes a
BoltzGen refolded complex, extracts the designed peptide, checks that the
peptide is chemically and stereochemically intact, moves it into the frame of
the project's standard receptor using **the receptor protein only**, measures
what it then clashes with, and writes the placed coordinates with a provenance
record.

    python python/prepare_peptide_pose.py \
        <refolded.cif> --designed-sequence SRGSGTSF --design-id spec_0574 \
        --reference <murc_anp_prepared.pdb> --reference-chain A \
        --box-record examples/box_murc_uma_chainE_30A.json \
        --out-dir poses/spec_0574

The alignment question, and why the peptide is never fitted
-----------------------------------------------------------
BoltzGen predicted the peptide in the frame of *its own* copy of the receptor.
The project's docking box, prepared receptor and every recorded UMA result
live in the frame of the deposited structure. Something has to move, and which
thing moves is a scientific choice, not a technical one.

The receptor protein is superposed onto the reference receptor protein, and the
single rigid transform that results is then applied to the peptide. The peptide
is never fitted to anything. This matters because fitting the peptide
independently would optimise the very quantity under test: the pose BoltzGen
predicted *relative to the receptor*. Carrying the receptor's transform across
preserves that relationship exactly, and any residual clash is then a property
of the prediction rather than of the alignment.

The consequence is that the peptide inherits the protein fit's error. The fit
RMSD is recorded for exactly that reason: a 0.44 A protein fit means the
peptide's placement carries roughly that much uncertainty, which is not
negligible beside a 1.5 A clash.

What is checked, and what each check can fail to catch
------------------------------------------------------
- **Sequence.** The peptide chain is identified by exact one-letter sequence
  match against the sequence recorded in the campaign metrics, never by chain
  length or chain order. A retained cofactor or a receptor fragment can make
  "the shortest chain" the wrong answer.
- **Heavy atoms.** Every standard residue's expected heavy atoms must be
  present. This catches truncated side chains. It cannot catch an atom that is
  present but misnamed, nor one placed at an implausible position within its
  own residue.
- **Covalent connectivity.** Consecutive C-N distances are measured. This
  catches a chain break or a grossly distorted peptide bond. It does not
  validate intra-residue bonding, ring geometry or planarity.
- **Stereochemistry.** Each alpha carbon is classified L or D by the signed
  volume of its substituents. This catches an inverted centre. It says nothing
  about side-chain rotamer plausibility.
- **Termini.** The presence of a terminal carboxyl oxygen (OXT) is reported.
  BoltzGen output generally lacks it, which leaves the C-terminal carboxyl
  group chemically incomplete; see the warning below.

Termini are reported, never silently repaired
---------------------------------------------
A C-terminal residue carrying C and O but no OXT is not a complete carboxyl
group. Any toolkit asked to perceive chemistry from such a file has to guess,
and the guess is not a carboxylate. This is recorded as
`termini_require_a_decision`, and it is a **blocking** condition for ligand
preparation rather than something this script fixes: adding OXT means choosing
a position, and choosing a protonation state means choosing a charge. Both are
scientific decisions belonging to the project record, not defaults.

Output
------
`<out-dir>/<design-id>_aligned.pdb` holds the placed peptide. Coordinates are
the original BoltzGen coordinates with one rigid transform applied, and
nothing else: no minimisation, no idealisation, no hydrogen addition.
`<out-dir>/<design-id>_pose.json` holds the record, including input checksums,
software versions, the transform itself, displacement, every check and every
failure.

What this establishes
---------------------
That a given design can or cannot be carried into the standard receptor frame
and prepared as a ligand, and what it collides with when it is. Nothing here
is evidence of binding, of a correct pose, or of a usable score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gemmi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import peptide_contacts  # noqa: E402
import peptide_geometry  # noqa: E402

# Ideal trans peptide bond C-N length, Engh & Huber (1991) restraint library,
# 1.329 A. The acceptance window is deliberately wide: a structure predictor's
# output is not restrained to a crystallographic library, and the purpose here
# is to catch a break or a gross distortion, not to grade bond geometry.
IDEAL_PEPTIDE_BOND_ANGSTROM = 1.329
PEPTIDE_BOND_MINIMUM_ANGSTROM = 1.20
PEPTIDE_BOND_MAXIMUM_ANGSTROM = 1.45

# Beyond this separation the residues are not bonded at all: the chain is
# broken and the peptide is not one molecule.
CHAIN_BREAK_ANGSTROM = 1.80

# Consecutive CA-CA separation, about 3.80 A for a trans peptide and about
# 2.90 A for a cis one. Reported for information; a cis bond is unusual but
# not in itself an error.
TRANS_CA_CA_ANGSTROM = 3.80

SCHEMA = "mur-ligase-peptide-pose/1"


class PoseError(ValueError):
    """Raised when a peptide cannot be placed or recorded as specified."""


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256, reusing the project's box-record helper."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(131072), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> Dict[str, object]:
    """Record the versions of every library that touched the coordinates."""
    return {
        "python": sys.version.split()[0],
        "gemmi": gemmi.__version__,
        "numpy": np.__version__,
        "note": (
            "Meeko, Vina and any minimiser are not used at this stage and so "
            "are not recorded here; the preparation record written later must "
            "record them itself."
        ),
    }


def peptide_bond_findings(
    residues: Sequence[gemmi.Residue],
) -> Dict[str, object]:
    """Measure consecutive C-N distances along the peptide.

    Returns a record listing every bond, those outside the acceptance window,
    and any chain break. A missing C or N is itself reported as a bond that
    could not be measured, rather than skipped.
    """
    bonds: List[Dict[str, object]] = []
    unmeasurable: List[str] = []
    for index in range(len(residues) - 1):
        first, second = residues[index], residues[index + 1]
        label = (
            f"{first.name}{first.seqid.num}-{second.name}{second.seqid.num}"
        )
        try:
            carbon = first["C"][0].pos
            nitrogen = second["N"][0].pos
        except (IndexError, RuntimeError, KeyError):
            unmeasurable.append(f"{label}: C or N absent")
            continue
        distance = carbon.dist(nitrogen)
        ca_ca: Optional[float] = None
        try:
            ca_ca = first["CA"][0].pos.dist(second["CA"][0].pos)
        except (IndexError, RuntimeError, KeyError):
            ca_ca = None
        bonds.append(
            {
                "bond": label,
                "c_to_n_angstrom": round(float(distance), 4),
                "deviation_from_ideal_angstrom": round(
                    float(distance - IDEAL_PEPTIDE_BOND_ANGSTROM), 4
                ),
                "ca_to_ca_angstrom": (
                    None if ca_ca is None else round(float(ca_ca), 4)
                ),
                "within_window": bool(
                    PEPTIDE_BOND_MINIMUM_ANGSTROM
                    <= distance
                    <= PEPTIDE_BOND_MAXIMUM_ANGSTROM
                ),
                "chain_break": bool(distance > CHAIN_BREAK_ANGSTROM),
            }
        )
    outside = [b["bond"] for b in bonds if not b["within_window"]]
    breaks = [b["bond"] for b in bonds if b["chain_break"]]
    lengths = [float(b["c_to_n_angstrom"]) for b in bonds]
    return {
        "expected_bonds": max(len(residues) - 1, 0),
        "measured_bonds": len(bonds),
        "unmeasurable_bonds": unmeasurable,
        "bonds": bonds,
        "bonds_outside_window": outside,
        "chain_breaks": breaks,
        "acceptance_window_angstrom": [
            PEPTIDE_BOND_MINIMUM_ANGSTROM,
            PEPTIDE_BOND_MAXIMUM_ANGSTROM,
        ],
        "ideal_angstrom": IDEAL_PEPTIDE_BOND_ANGSTROM,
        "minimum_angstrom": round(min(lengths), 4) if lengths else None,
        "maximum_angstrom": round(max(lengths), 4) if lengths else None,
        "connectivity_intact": bool(
            not unmeasurable and not breaks and len(bonds) == max(len(residues) - 1, 0)
        ),
        "limitation": (
            "Consecutive C-N distances only. Intra-residue bonding, bond "
            "angles, planarity and ring geometry are not validated."
        ),
    }


def terminus_findings(residues: Sequence[gemmi.Residue]) -> Dict[str, object]:
    """Report the chemical completeness of the two termini.

    The C-terminal carboxyl is complete only when a second oxygen (OXT) is
    present. Its absence is reported as requiring a decision, because the
    remedy is to place an atom and choose a charge.
    """
    if not residues:
        raise PoseError("No residues from which to assess termini")
    first, last = residues[0], residues[-1]
    last_atoms = {atom.name for atom in last}
    first_atoms = {atom.name for atom in first}
    has_oxt = "OXT" in last_atoms
    return {
        "n_terminal_residue": f"{first.name}{first.seqid.num}",
        "c_terminal_residue": f"{last.name}{last.seqid.num}",
        "c_terminal_has_oxt": has_oxt,
        "n_terminal_atoms": sorted(first_atoms),
        "c_terminal_atoms": sorted(last_atoms),
        "termini_require_a_decision": not has_oxt,
        "reason": (
            "C-terminal carboxyl carries C and O but no OXT, so the group is "
            "chemically incomplete and any toolkit reading this file must "
            "guess its chemistry. Placing OXT and choosing a protonation "
            "state are recorded decisions, not defaults."
            if not has_oxt
            else "C-terminal carboxyl has both oxygens present."
        ),
        "limitation": (
            "Hydrogens are absent throughout, so the N-terminal protonation "
            "state is not represented in the file either way."
        ),
    }


def transform_record(
    transform: gemmi.Transform, before: np.ndarray, after: np.ndarray
) -> Dict[str, object]:
    """Record the rigid transform and the displacement it produced.

    The rotation angle comes from the trace of the rotation matrix. The
    displacement statistics are over the peptide's own heavy atoms, which is
    the quantity a reader cares about: how far this peptide moved in being
    carried into the reference frame.
    """
    matrix = np.asarray(transform.mat.tolist(), dtype=float)
    translation = np.asarray(
        [transform.vec.x, transform.vec.y, transform.vec.z], dtype=float
    )
    cosine = (float(np.trace(matrix)) - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, cosine))
    displacement = np.linalg.norm(after - before, axis=1)
    return {
        "rotation_matrix": [[round(float(v), 8) for v in row] for row in matrix],
        "translation_angstrom": [round(float(v), 6) for v in translation],
        "rotation_angle_degrees": round(math.degrees(math.acos(cosine)), 4),
        "translation_norm_angstrom": round(float(np.linalg.norm(translation)), 4),
        "peptide_displacement_angstrom": {
            "mean": round(float(displacement.mean()), 4),
            "maximum": round(float(displacement.max()), 4),
            "minimum": round(float(displacement.min()), 4),
        },
        "note": (
            "One rigid transform derived from the receptor protein fit, "
            "applied to the whole peptide. The peptide was not fitted."
        ),
    }


def aligned_peptide_structure(
    chain: gemmi.Chain,
    transform: gemmi.Transform,
    design_id: str,
    chain_name: str = "P",
) -> gemmi.Structure:
    """Build a single-chain structure holding the transformed peptide.

    Atom names, residue names, author numbering, elements, occupancies and
    B-factors are copied unchanged. Only the coordinates are transformed, and
    every atom is transformed by the same transform.
    """
    structure = gemmi.Structure()
    structure.name = design_id or "peptide"
    structure.spacegroup_hm = "P 1"
    structure.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    model = gemmi.Model("1")
    new_chain = gemmi.Chain(chain_name)
    for residue in peptide_geometry.amino_acid_residues(chain):
        new_residue = gemmi.Residue()
        new_residue.name = residue.name
        new_residue.seqid = residue.seqid
        new_residue.het_flag = "A"
        for atom in residue:
            new_atom = gemmi.Atom()
            new_atom.name = atom.name
            new_atom.element = atom.element
            new_atom.occ = atom.occ
            new_atom.b_iso = atom.b_iso
            moved = transform.apply(atom.pos)
            new_atom.pos = gemmi.Position(moved.x, moved.y, moved.z)
            new_residue.add_atom(new_atom)
        new_chain.add_residue(new_residue)
    model.add_chain(new_chain)
    structure.add_model(model)
    structure.setup_entities()
    return structure


def place_peptide(
    refolded_cif: Path,
    designed_sequence: str,
    reference_structure: Path,
    reference_chain_name: str,
    design_id: str,
    box_record: Optional[Dict] = None,
    clash_components: Sequence[str] = ("ANP",),
    contact_cutoff: float = peptide_contacts.DEFAULT_CONTACT_CUTOFF,
    clash_overlap_tolerance: float = (
        peptide_contacts.DEFAULT_CLASH_OVERLAP_TOLERANCE
    ),
) -> Dict[str, object]:
    """Carry out every stage-one check for one design and return the record.

    The returned record is self-describing: it names its inputs and their
    checksums, the transform applied, every check and its result, and whether
    the design may proceed to ligand preparation.
    """
    structure = peptide_geometry.read_structure(refolded_cif)
    model = structure[0]
    peptide_chain = peptide_geometry.identify_chain_by_sequence(
        model, designed_sequence
    )
    peptide_residues = peptide_geometry.amino_acid_residues(peptide_chain)
    residue_findings = peptide_geometry.inspect_peptide_residues(peptide_chain)

    design_protein_chains = [
        c
        for c in model
        if c.name != peptide_chain.name and peptide_geometry.amino_acid_residues(c)
    ]
    if not design_protein_chains:
        raise PoseError(
            f"{refolded_cif}: the refolded complex has no protein chain "
            "besides the designed peptide, so no receptor frame exists to "
            "align from."
        )
    if len(design_protein_chains) > 1:
        names = [c.name for c in design_protein_chains]
        raise PoseError(
            f"{refolded_cif}: several protein chains besides the peptide "
            f"({names}); which one is the receptor is ambiguous and must be "
            "stated rather than guessed."
        )
    design_protein = design_protein_chains[0]

    reference = peptide_geometry.read_structure(reference_structure)
    reference_chains = [
        c for c in reference[0] if c.name == reference_chain_name
    ]
    if not reference_chains:
        raise PoseError(
            f"Reference chain {reference_chain_name!r} not found in "
            f"{reference_structure}"
        )
    reference_chain = reference_chains[0]

    transform, fit = peptide_geometry.superpose_protein(
        design_protein, reference_chain
    )

    before = peptide_geometry.transformed_positions(peptide_residues, None)
    peptide_sites = peptide_contacts.collect_sites(peptide_chain, transform)
    after = peptide_contacts.positions_of(peptide_sites)

    # The reference is partitioned by residue class, never by chain letter:
    # in the prepared receptor ANP shares author chain A with the protein.
    reference_sites = peptide_contacts.collect_sites(reference_chain)
    protein_sites, non_protein_sites, census = peptide_contacts.partition_sites(
        reference_sites
    )

    reports = [
        peptide_contacts.contact_report(
            peptide_sites,
            protein_sites,
            target_description=(
                f"MurC protein, author chain {reference_chain_name}, "
                "amino-acid residues only"
            ),
            contact_cutoff=contact_cutoff,
            clash_overlap_tolerance=clash_overlap_tolerance,
        )
    ]
    for component in clash_components:
        component_sites = peptide_contacts.sites_for_component(
            non_protein_sites, component
        )
        reports.append(
            peptide_contacts.contact_report(
                peptide_sites,
                component_sites,
                target_description=(
                    f"{component} cofactor, author chain "
                    f"{reference_chain_name}"
                ),
                contact_cutoff=contact_cutoff,
                clash_overlap_tolerance=clash_overlap_tolerance,
            )
        )

    bonds = peptide_bond_findings(peptide_residues)
    termini = terminus_findings(peptide_residues)
    failures = [f.as_dict() for f in residue_findings if not f.acceptable]
    atoms_complete = all(not f.missing_atoms for f in residue_findings)
    all_l = all(f.configuration in ("L", None) for f in residue_findings)

    record: Dict[str, object] = {
        "schema": SCHEMA,
        "design_id": design_id,
        "designed_sequence": designed_sequence,
        "residues": len(peptide_residues),
        "peptide_heavy_atoms": len(peptide_sites),
        "inputs": {
            "refolded_cif": str(refolded_cif),
            "refolded_cif_sha256": file_sha256(refolded_cif),
            "reference_structure": str(reference_structure),
            "reference_structure_sha256": file_sha256(reference_structure),
            "reference_chain": reference_chain_name,
        },
        "chain_assignment": {
            "peptide_chain": peptide_chain.name,
            "peptide_chain_selection": "exact one-letter sequence match",
            "design_receptor_chain": design_protein.name,
            "design_receptor_selection": (
                "the single protein chain other than the sequence-identified "
                "peptide; several would be fatal"
            ),
        },
        "reference_partition": census.as_dict(),
        "protein_superposition": fit,
        "transform": transform_record(transform, before, after),
        "peptide_bonds": bonds,
        "termini": termini,
        "residue_findings": [f.as_dict() for f in residue_findings],
        "residue_failures": failures,
        "atoms_complete": atoms_complete,
        "all_calpha_L": all_l,
        "d_configured_residues": [
            f.sequence_index for f in residue_findings if f.configuration == "D"
        ],
        "contacts": reports,
        "clash_summary": peptide_contacts.clash_summary(reports),
        "software": software_versions(),
    }
    if box_record is not None:
        record["box_containment"] = peptide_geometry.box_containment(
            after, box_record
        )

    structure_intact = bool(
        atoms_complete
        and all_l
        and not failures
        and bonds["connectivity_intact"]
    )
    record["structure_intact"] = structure_intact
    record["ready_for_ligand_preparation"] = bool(
        structure_intact and not termini["termini_require_a_decision"]
    )
    record["blocking_issues"] = blocking_issues(record)
    record["interpretation"] = (
        "Stage-one geometry and chemistry only. A design that passes may be "
        "prepared as a ligand; this is not evidence of binding, of a correct "
        "pose, or of a usable score. Distances to ANP are measured against a "
        "representation this project records as chemically unvalidated."
    )
    return record


def blocking_issues(record: Dict[str, object]) -> List[str]:
    """List, in plain words, every reason this design cannot proceed.

    A design with residual clashes is **not** blocked here: a clash is a
    finding to be acted on, and removing a clashing peptide silently is
    exactly what this pilot must not do.
    """
    issues: List[str] = []
    if not record["atoms_complete"]:
        issues.append("heavy atoms missing from one or more residues")
    if not record["all_calpha_L"]:
        issues.append(
            "non-L alpha carbon at residue(s) "
            f"{record['d_configured_residues']}"
        )
    bonds = record["peptide_bonds"]
    if bonds["chain_breaks"]:
        issues.append(f"peptide chain broken at {bonds['chain_breaks']}")
    if bonds["unmeasurable_bonds"]:
        issues.append(f"unmeasurable peptide bond(s): {bonds['unmeasurable_bonds']}")
    if record["termini"]["termini_require_a_decision"]:
        issues.append(
            "C-terminal carboxyl incomplete (no OXT): a position and a "
            "protonation state must be decided before ligand preparation"
        )
    return issues


def write_outputs(
    record: Dict[str, object],
    peptide_chain: gemmi.Chain,
    transform: gemmi.Transform,
    out_dir: Path,
    design_id: str,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Write the aligned peptide and its record, refusing to clobber by default."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = out_dir / f"{design_id}_aligned.pdb"
    json_path = out_dir / f"{design_id}_pose.json"
    for path in (pdb_path, json_path):
        if path.exists() and not overwrite:
            raise PoseError(
                f"Refusing to overwrite {path}. Choose a fresh output "
                "directory, or pass --overwrite deliberately."
            )
    structure = aligned_peptide_structure(peptide_chain, transform, design_id)
    structure.write_pdb(str(pdb_path))
    record = dict(record)
    record["outputs"] = {
        "aligned_peptide_pdb": str(pdb_path),
        "coordinate_frame": (
            "reference receptor frame, reached by one rigid transform from the "
            "BoltzGen refolded frame; no minimisation or idealisation applied"
        ),
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    record["outputs"]["aligned_peptide_pdb_sha256"] = file_sha256(pdb_path)
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    return {
        "aligned_peptide_pdb": str(pdb_path),
        "pose_record": str(json_path),
    }


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
    parser.add_argument("--design-id", required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Prepared standard receptor, in whose frame the box is defined",
    )
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--box-record", type=Path, default=None)
    parser.add_argument(
        "--clash-component",
        action="append",
        default=None,
        help="Retained component to measure separately, repeatable "
        "(default: ANP)",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=peptide_contacts.DEFAULT_CONTACT_CUTOFF,
        help="Strict less-than contact threshold in Angstroms "
        f"(default: {peptide_contacts.DEFAULT_CONTACT_CUTOFF})",
    )
    parser.add_argument(
        "--clash-overlap-tolerance",
        type=float,
        default=peptide_contacts.DEFAULT_CLASH_OVERLAP_TOLERANCE,
        help="Van der Waals overlap in Angstroms above which a pair is called "
        f"a clash (default: {peptide_contacts.DEFAULT_CLASH_OVERLAP_TOLERANCE})",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.contact_cutoff <= 0:
        parser.error("--contact-cutoff must be positive")

    box_record = None
    if args.box_record is not None:
        box_record = json.loads(args.box_record.read_text())

    try:
        record = place_peptide(
            args.refolded_cif,
            args.designed_sequence,
            args.reference,
            args.reference_chain,
            args.design_id,
            box_record=box_record,
            clash_components=tuple(args.clash_component or ("ANP",)),
            contact_cutoff=args.contact_cutoff,
            clash_overlap_tolerance=args.clash_overlap_tolerance,
        )
        # Re-read only what writing needs; place_peptide owns the science.
        structure = peptide_geometry.read_structure(args.refolded_cif)
        peptide_chain = peptide_geometry.identify_chain_by_sequence(
            structure[0], args.designed_sequence
        )
        design_protein = [
            c
            for c in structure[0]
            if c.name != peptide_chain.name
            and peptide_geometry.amino_acid_residues(c)
        ][0]
        reference = peptide_geometry.read_structure(args.reference)
        reference_chain = [
            c for c in reference[0] if c.name == args.reference_chain
        ][0]
        transform, _ = peptide_geometry.superpose_protein(
            design_protein, reference_chain
        )
        written = write_outputs(
            record,
            peptide_chain,
            transform,
            args.out_dir,
            args.design_id,
            overwrite=args.overwrite,
        )
    except (PoseError, peptide_geometry.GeometryError, peptide_contacts.ContactError) as error:
        print(f"peptide placement failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(record, indent=2))
    for name, path in written.items():
        print(f"{name}: {path}", file=sys.stderr)
    if record["blocking_issues"]:
        print(
            "blocking issues: " + "; ".join(record["blocking_issues"]),
            file=sys.stderr,
        )
    return 0 if record["ready_for_ligand_preparation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
