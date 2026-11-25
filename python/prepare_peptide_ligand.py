"""Turn an aligned peptide PDB into a chemically validated docking ligand.

This is stage two of the peptide scoring pilot. It takes the aligned peptide
written by `prepare_peptide_pose.py` and produces the SDF that Meeko needs,
together with the evidence that the chemistry in that SDF is the chemistry
that was intended.

    python python/prepare_peptide_ligand.py \
        poses/spec_0574/spec_0574_aligned.pdb \
        --designed-sequence SRGSGTSF --design-id spec_0574 \
        --place-c-terminal-oxt \
        --ionisation-state physiological \
        --protonation-justification "..." \
        --out-dir ligands/spec_0574

Why an SDF, and why this is not optional
----------------------------------------
Meeko's ligand preparation accepts SDF, MOL2 and MOL only. Read from the
Meeko source, the accepted extensions are exactly those three, and a PDB is
rejected with "Format [pdb] not in supported formats". A PDB carries no bond
orders, so the conversion is not a format change: it is the step where the
molecule's chemistry gets decided. Doing it explicitly here, from an
authoritative template, is what stops that decision being made implicitly by
whichever perception algorithm happens to read the file.

Meeko also does not add hydrogens. Its source reads input with
`removeHs=False` and comments "input must have explicit H", and there is no
add-hydrogen option. Hydrogens are therefore added here, at a stated
ionisation state.

Where the bond orders come from
-------------------------------
The designed sequence is built into a template molecule with
`Chem.MolFromSequence`, and the template's bond orders are transferred onto
the experimental coordinates with `AssignBondOrdersFromTemplate`. The
coordinates never come from the template: only the connectivity and bond
orders do. If the atoms do not correspond exactly, the transfer fails, and
that failure is treated as fatal rather than worked around, because every
alternative is a guess about the molecule's chemistry.

The C-terminal oxygen, and why it blocks everything
---------------------------------------------------
BoltzGen refolded output carries no OXT, so its C-terminal residue has one
carboxyl oxygen rather than two. For `spec_0574` the consequence is exact and
measurable: the structure has 55 heavy atoms and formula C32H51N11O12, the
template has 56 and C32H51N11O13, and the bond-order transfer fails with "No
matching found". The missing atom is not cosmetic; without it the molecule
cannot be matched to its own sequence.

`--place-c-terminal-oxt` places it, at 1.25 A from the carbonyl carbon along
the bisector of the external CA-C-O angle, which puts it in the carboxyl plane
at about 120 degrees to both neighbours. This is idealised geometry, not
predicted geometry, and it is recorded as such. The flag is required rather
than implied, because adding an atom to a predicted structure is a decision
about that structure.

Ionisation is a stated decision, never a default
------------------------------------------------
`--ionisation-state` is required and has no default.

`neutral` keeps the template's state: a protonated carboxyl, a neutral
guanidinium, a neutral amine. This is the state a chemical dictionary
describes and the wrong thing to dock at physiological pH.

`physiological` models pH about 7.4: the N-terminal amine protonated, the
C-terminal carboxyl deprotonated, arginine and lysine protonated, aspartate
and glutamate deprotonated. Histidine and cysteine are **refused**, because a
histidine needs a tautomer chosen and a cysteine needs its thiol state
chosen, and neither follows from pH alone.

This matters for scoring and not only for tidiness. Gasteiger charges are
computed from the structure handed to Meeko, so a neutral guanidinium and a
protonated one give different charges on every atom of that side chain, and
AutoDock Vina's scoring function reads the resulting atom types.

What is checked
---------------
- Heavy-atom count and molecular formula against the template.
- Every heavy atom's coordinates against the input PDB, to 0.001 A.
- Stereochemistry, by CIP code from the 3D coordinates, against the
  configuration expected for L-amino acids. This is an independent route to
  the alpha-carbon chirality that `peptide_geometry` computes by signed
  volume, so the two can disagree and be investigated.
- Rotatable-bond count, as an estimate of torsional degrees of freedom.
- An atom index to PDB atom name mapping, so Meeko's own `--add_index_map`
  output can later be joined back to the original structure.

What this cannot establish
--------------------------
That Meeko will interpret the molecule as intended. Meeko's documented remit
is small organic molecules as ligands and proteins as *receptors*; its source
has no peptide or polymer path on the ligand side. Treating an eight-residue
peptide as a Meeko ligand is therefore off the documented path, and the PDBQT
it writes must be checked independently, on the machine where Meeko runs.

Nothing here is evidence of binding, and the rotatable-bond count is RDKit's,
not Meeko's TORSDOF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where RDKit is absent
    RDKIT_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

import peptide_geometry  # noqa: E402

SCHEMA = "mur-ligase-peptide-ligand/1"

# Idealised carboxyl geometry for a placed OXT. The C-O bond length of a
# carboxylate is about 1.25 A, and the two oxygens sit at about 120 degrees to
# the alpha carbon. These are idealised values: the placed atom's geometry is
# chosen, not predicted.
OXT_BOND_LENGTH_ANGSTROM = 1.25

# Coordinates are compared at the precision a PDB file stores, three decimal
# places, plus a small margin.
COORDINATE_TOLERANCE_ANGSTROM = 0.002

# CIP code expected at the alpha carbon of each L-amino acid. Every L residue
# is (S) at CA except cysteine, which is (R) because sulphur outranks the
# carboxyl carbon. Glycine has no chiral alpha carbon.
EXPECTED_ALPHA_CIP: Dict[str, Optional[str]] = {
    "ALA": "S", "ARG": "S", "ASN": "S", "ASP": "S", "CYS": "R",
    "GLN": "S", "GLU": "S", "GLY": None, "HIS": "S", "ILE": "S",
    "LEU": "S", "LYS": "S", "MET": "S", "PHE": "S", "PRO": "S",
    "SER": "S", "THR": "S", "TRP": "S", "TYR": "S", "VAL": "S",
}

# Second chiral centre in the side chain, for the two residues that have one.
EXPECTED_BETA_CIP: Dict[str, str] = {"THR": "R", "ILE": "S"}

# Residues whose physiological state cannot be settled by pH alone.
IONISATION_REFUSED = {
    "HIS": (
        "histidine needs a tautomer chosen (ND1-H or NE2-H), which pH does "
        "not determine"
    ),
    "CYS": (
        "cysteine needs its thiol state chosen; pKa about 8.3 makes neither "
        "state the obvious default at pH 7.4"
    ),
}


class LigandPreparationError(ValueError):
    """Raised when a peptide cannot be prepared as a validated ligand."""


def require_rdkit() -> None:
    if not RDKIT_AVAILABLE:
        raise LigandPreparationError(
            "RDKit is required for ligand preparation and is not importable "
            "in this interpreter. This is a blocker, not a check that can be "
            "skipped."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(131072), b""):
            digest.update(block)
    return digest.hexdigest()


def place_c_terminal_oxt(
    structure: gemmi.Structure,
) -> Dict[str, object]:
    """Add OXT to the C-terminal residue, in idealised carboxyl geometry.

    The atom is placed at `OXT_BOND_LENGTH_ANGSTROM` from the carbonyl carbon,
    along the negated bisector of the CA-C and O-C directions. That direction
    lies in the CA/C/O plane and makes equal angles, close to 120 degrees, with
    both existing substituents.

    Returns a record of what was placed and the geometry achieved, so the
    idealisation is visible in the provenance. Fails if OXT is already present,
    rather than moving an existing atom.
    """
    chains = [c for c in structure[0] if peptide_geometry.amino_acid_residues(c)]
    if len(chains) != 1:
        raise LigandPreparationError(
            f"Expected exactly one peptide chain, found {len(chains)}. The "
            "aligned peptide file should hold the peptide alone."
        )
    residues = peptide_geometry.amino_acid_residues(chains[0])
    last = residues[-1]
    if any(atom.name == "OXT" for atom in last):
        raise LigandPreparationError(
            f"{last.name}{last.seqid.num} already has OXT; refusing to place "
            "a second one or move the existing atom."
        )
    try:
        ca = np.asarray(last["CA"][0].pos.tolist(), dtype=float)
        carbon = np.asarray(last["C"][0].pos.tolist(), dtype=float)
        oxygen = np.asarray(last["O"][0].pos.tolist(), dtype=float)
    except (IndexError, RuntimeError, KeyError) as error:
        raise LigandPreparationError(
            f"{last.name}{last.seqid.num} lacks CA, C or O, so the carboxyl "
            "plane is undefined and OXT cannot be placed."
        ) from error

    to_ca = (ca - carbon) / np.linalg.norm(ca - carbon)
    to_oxygen = (oxygen - carbon) / np.linalg.norm(oxygen - carbon)
    direction = -(to_ca + to_oxygen)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise LigandPreparationError(
            "CA, C and O are collinear, so the carboxyl bisector is undefined."
        )
    direction = direction / norm
    position = carbon + OXT_BOND_LENGTH_ANGSTROM * direction

    atom = gemmi.Atom()
    atom.name = "OXT"
    atom.element = gemmi.Element("O")
    atom.pos = gemmi.Position(*position)
    atom.occ = 1.0
    atom.b_iso = 0.0
    last.add_atom(atom)

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    return {
        "placed": True,
        "residue": f"{last.name}{last.seqid.num}",
        "bond_length_angstrom": OXT_BOND_LENGTH_ANGSTROM,
        "angle_ca_c_oxt_degrees": round(angle(to_ca, direction), 2),
        "angle_o_c_oxt_degrees": round(angle(to_oxygen, direction), 2),
        "position_angstrom": [round(float(v), 4) for v in position],
        "geometry": (
            "idealised: 1.25 A along the negated bisector of the CA-C and O-C "
            "directions, placing OXT in the carboxyl plane"
        ),
        "caveat": (
            "This atom was constructed, not predicted. BoltzGen did not place "
            "it, and its position is a geometric convention."
        ),
    }


def assign_bond_orders(
    pdb_path: Path, designed_sequence: str
) -> "Chem.Mol":
    """Transfer template bond orders onto the experimental coordinates.

    The template is built from the designed sequence, so the chemistry asserted
    is the chemistry the campaign recorded. A failure to match is fatal, and
    the diagnostic names the heavy-atom and formula difference, because that
    difference is almost always the real problem.
    """
    require_rdkit()
    molecule = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=True)
    if molecule is None:
        raise LigandPreparationError(
            f"RDKit could not read {pdb_path} as a molecule."
        )
    template = Chem.MolFromSequence(designed_sequence)
    if template is None:
        raise LigandPreparationError(
            f"RDKit could not build a template from the sequence "
            f"{designed_sequence!r}. A non-standard residue would do this."
        )
    try:
        return AllChem.AssignBondOrdersFromTemplate(template, molecule)
    except Exception as error:
        structure_heavy = sum(
            1 for a in molecule.GetAtoms() if a.GetAtomicNum() > 1
        )
        template_heavy = sum(
            1 for a in template.GetAtoms() if a.GetAtomicNum() > 1
        )
        raise LigandPreparationError(
            f"Bond orders could not be transferred from the sequence template "
            f"to {pdb_path}: {error}. "
            f"Structure has {structure_heavy} heavy atoms "
            f"({rdMolDescriptors.CalcMolFormula(molecule)}); the template for "
            f"{designed_sequence!r} has {template_heavy} "
            f"({rdMolDescriptors.CalcMolFormula(template)}). "
            "A difference of one oxygen is the missing C-terminal OXT; see "
            "--place-c-terminal-oxt."
        ) from error


def find_atom(
    molecule: "Chem.Mol", residue_seqid: int, atom_name: str
) -> Optional["Chem.Atom"]:
    """Locate an atom by author residue number and PDB atom name."""
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if (
            info.GetResidueNumber() == residue_seqid
            and info.GetName().strip() == atom_name
        ):
            return atom
    return None


def residue_names_in_order(molecule: "Chem.Mol") -> List[Tuple[int, str]]:
    """Return (seqid, residue name) pairs in first-appearance order."""
    seen: List[Tuple[int, str]] = []
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (info.GetResidueNumber(), info.GetResidueName().strip())
        if key not in seen:
            seen.append(key)
    return seen


def apply_physiological_ionisation(
    molecule: "Chem.Mol",
) -> Tuple["Chem.Mol", Dict[str, object]]:
    """Set formal charges for pH about 7.4, or refuse where pH is not enough.

    Returns the modified molecule and a record naming every group whose charge
    was set. Histidine and cysteine cause a refusal rather than a silent
    choice; see `IONISATION_REFUSED`.
    """
    require_rdkit()
    residues = residue_names_in_order(molecule)
    refusals = [
        f"{name}{seqid}: {IONISATION_REFUSED[name]}"
        for seqid, name in residues
        if name in IONISATION_REFUSED
    ]
    if refusals:
        raise LigandPreparationError(
            "Physiological ionisation refused for this peptide: "
            + "; ".join(refusals)
            + ". Decide the state explicitly and record it, or prepare the "
            "neutral form."
        )

    editable = Chem.RWMol(molecule)
    changes: List[Dict[str, object]] = []

    first_seqid = residues[0][0]
    last_seqid = residues[-1][0]

    amine = find_atom(editable, first_seqid, "N")
    if amine is None:
        raise LigandPreparationError("N-terminal nitrogen not found")
    amine.SetFormalCharge(1)
    changes.append(
        {
            "group": "N-terminal amine",
            "residue": f"{residues[0][1]}{first_seqid}",
            "atom": "N",
            "formal_charge": 1,
            "basis": "primary amine, pKa about 9, protonated at pH 7.4",
        }
    )

    carboxyl = find_atom(editable, last_seqid, "OXT")
    if carboxyl is None:
        raise LigandPreparationError(
            "C-terminal OXT not found, so the carboxylate cannot be formed. "
            "Place OXT first with --place-c-terminal-oxt."
        )
    carboxyl.SetFormalCharge(-1)
    carboxyl.SetNoImplicit(True)
    carboxyl.SetNumExplicitHs(0)
    changes.append(
        {
            "group": "C-terminal carboxyl",
            "residue": f"{residues[-1][1]}{last_seqid}",
            "atom": "OXT",
            "formal_charge": -1,
            "basis": "carboxylic acid, pKa about 3, deprotonated at pH 7.4",
        }
    )

    for seqid, name in residues:
        if name == "ARG":
            centre = find_atom(editable, seqid, "CZ")
            if centre is None:
                raise LigandPreparationError(f"ARG{seqid} has no CZ atom")
            target = None
            for bond in centre.GetBonds():
                other = bond.GetOtherAtom(centre)
                if (
                    bond.GetBondType() == Chem.BondType.DOUBLE
                    and other.GetSymbol() == "N"
                ):
                    target = other
                    break
            if target is None:
                raise LigandPreparationError(
                    f"ARG{seqid}: no N double-bonded to CZ, so the "
                    "guanidinium cannot be protonated as intended."
                )
            target.SetFormalCharge(1)
            changes.append(
                {
                    "group": "arginine guanidinium",
                    "residue": f"{name}{seqid}",
                    "atom": target.GetPDBResidueInfo().GetName().strip(),
                    "formal_charge": 1,
                    "basis": "guanidinium, pKa about 12.5, protonated at pH 7.4",
                }
            )
        elif name == "LYS":
            nitrogen = find_atom(editable, seqid, "NZ")
            if nitrogen is None:
                raise LigandPreparationError(f"LYS{seqid} has no NZ atom")
            nitrogen.SetFormalCharge(1)
            changes.append(
                {
                    "group": "lysine side-chain amine",
                    "residue": f"{name}{seqid}",
                    "atom": "NZ",
                    "formal_charge": 1,
                    "basis": "alkyl amine, pKa about 10.5, protonated at pH 7.4",
                }
            )
        elif name in ("ASP", "GLU"):
            atom_name = "OD2" if name == "ASP" else "OE2"
            oxygen = find_atom(editable, seqid, atom_name)
            if oxygen is None:
                raise LigandPreparationError(
                    f"{name}{seqid} has no {atom_name} atom"
                )
            oxygen.SetFormalCharge(-1)
            oxygen.SetNoImplicit(True)
            oxygen.SetNumExplicitHs(0)
            changes.append(
                {
                    "group": f"{name.lower()} side-chain carboxyl",
                    "residue": f"{name}{seqid}",
                    "atom": atom_name,
                    "formal_charge": -1,
                    "basis": "carboxylic acid, pKa about 4, deprotonated at pH 7.4",
                }
            )

    result = editable.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception as error:
        raise LigandPreparationError(
            f"The ionised molecule does not sanitise: {error}"
        ) from error
    return result, {
        "state": "physiological",
        "ph_modelled": 7.4,
        "changes": changes,
        "net_formal_charge": Chem.GetFormalCharge(result),
        "limitation": (
            "Formal charges follow residue identity and nominal pKa values, "
            "not the local electrostatic environment of this binding site. A "
            "pKa shifted by the pocket is not modelled."
        ),
    }


def stereochemistry_report(molecule: "Chem.Mol") -> Dict[str, object]:
    """Check CIP codes from 3D against the configuration expected for L residues.

    This is an independent route to the result `peptide_geometry` obtains from
    the signed volume of the alpha-carbon substituents. Agreement between two
    unrelated methods is worth more than either alone, so both are recorded.
    """
    require_rdkit()
    working = Chem.Mol(molecule)
    Chem.AssignStereochemistryFrom3D(working)
    observed: Dict[str, str] = {}
    for atom in working.GetAtoms():
        if not atom.HasProp("_CIPCode"):
            continue
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (
            f"{info.GetResidueName().strip()}{info.GetResidueNumber()}"
            f"/{info.GetName().strip()}"
        )
        observed[key] = atom.GetProp("_CIPCode")

    mismatches: List[str] = []
    checked = 0
    for seqid, name in residue_names_in_order(working):
        expected_alpha = EXPECTED_ALPHA_CIP.get(name)
        if expected_alpha is not None:
            key = f"{name}{seqid}/CA"
            seen = observed.get(key)
            checked += 1
            if seen is None:
                mismatches.append(f"{key}: no CIP code assigned")
            elif seen != expected_alpha:
                mismatches.append(
                    f"{key}: {seen}, expected {expected_alpha} for an "
                    "L-amino acid"
                )
        expected_beta = EXPECTED_BETA_CIP.get(name)
        if expected_beta is not None:
            key = f"{name}{seqid}/CB"
            seen = observed.get(key)
            checked += 1
            if seen is not None and seen != expected_beta:
                mismatches.append(
                    f"{key}: {seen}, expected {expected_beta} for "
                    f"L-{name.title()}"
                )
    return {
        "observed_cip_codes": dict(sorted(observed.items())),
        "centres_checked": checked,
        "mismatches": mismatches,
        "all_l_configuration": not mismatches,
        "method": (
            "RDKit AssignStereochemistryFrom3D, compared against the CIP code "
            "expected for each L-amino acid: (S) at CA for all but cysteine, "
            "which is (R); second centres checked for threonine and isoleucine"
        ),
    }


def rotatable_bond_report(molecule: "Chem.Mol") -> Dict[str, object]:
    """Count rotatable bonds as an estimate of torsional degrees of freedom."""
    require_rdkit()
    strict = rdMolDescriptors.CalcNumRotatableBonds(
        molecule, rdMolDescriptors.NumRotatableBondsOptions.Strict
    )
    non_strict = rdMolDescriptors.CalcNumRotatableBonds(
        molecule, rdMolDescriptors.NumRotatableBondsOptions.NonStrict
    )
    return {
        "rdkit_rotatable_bonds_strict": int(strict),
        "rdkit_rotatable_bonds_non_strict": int(non_strict),
        "caveat": (
            "This is RDKit's count, not Meeko's TORSDOF. RDKit's strict "
            "definition excludes amide bonds, as Meeko does by default, but "
            "the two need not agree, and only the PDBQT's own TORSDOF line "
            "governs what Vina searches. Measure it where Meeko runs."
        ),
        "relevance": (
            "AutoDock 4's documented practical limit is 32 torsions. That "
            "limit belongs to AutoDock 4, not to Vina, and this project's "
            "earlier attribution of it to Vina was incorrect. A high torsion "
            "count is still a reason to prefer fixed-pose scoring and local "
            "optimisation over an unrestrained search."
        ),
    }


def atom_mapping(molecule: "Chem.Mol") -> List[Dict[str, object]]:
    """Map molecule atom indices to PDB atom identities.

    This is the join key for Meeko's `--add_index_map` output: Meeko maps
    input index to PDBQT index, and this maps input index to the original
    atom's chain, residue and name, so a PDBQT atom can be traced back to the
    BoltzGen structure.
    """
    mapping: List[Dict[str, object]] = []
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        entry: Dict[str, object] = {
            "molecule_index_zero_based": atom.GetIdx(),
            "element": atom.GetSymbol(),
        }
        if info is not None:
            entry.update(
                {
                    "author_chain": info.GetChainId().strip(),
                    "residue_name": info.GetResidueName().strip(),
                    "residue_seqid": info.GetResidueNumber(),
                    "atom_name": info.GetName().strip(),
                }
            )
        mapping.append(entry)
    return mapping


def heavy_atom_coordinate_check(
    molecule: "Chem.Mol", source_pdb: Path
) -> Dict[str, object]:
    """Confirm every heavy atom still sits where the input PDB put it.

    Atoms are paired on residue number and atom name, never on order, and the
    **maximum** deviation is reported rather than a mean, because a mean over
    fifty atoms hides one atom that moved.
    """
    structure = peptide_geometry.read_structure(source_pdb)
    expected: Dict[Tuple[int, str], np.ndarray] = {}
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                if atom.element == gemmi.Element("H"):
                    continue
                expected[(residue.seqid.num, atom.name)] = np.asarray(
                    atom.pos.tolist(), dtype=float
                )

    conformer = molecule.GetConformer()
    compared = 0
    missing: List[str] = []
    worst = 0.0
    worst_atom = None
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        info = atom.GetPDBResidueInfo()
        if info is None:
            missing.append(f"index {atom.GetIdx()}: no PDB identity")
            continue
        key = (info.GetResidueNumber(), info.GetName().strip())
        if key not in expected:
            missing.append(f"{key[1]} in residue {key[0]}: absent from input")
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        deviation = float(
            np.linalg.norm(
                np.asarray([position.x, position.y, position.z]) - expected[key]
            )
        )
        compared += 1
        if deviation > worst:
            worst = deviation
            worst_atom = f"{key[1]} in residue {key[0]}"
    return {
        "heavy_atoms_compared": compared,
        "heavy_atoms_in_input": len(expected),
        "unmatched": missing,
        "maximum_deviation_angstrom": round(worst, 6),
        "worst_atom": worst_atom,
        "tolerance_angstrom": COORDINATE_TOLERANCE_ANGSTROM,
        "coordinates_preserved": bool(
            not missing
            and compared == len(expected)
            and worst <= COORDINATE_TOLERANCE_ANGSTROM
        ),
    }


def prepare_ligand(
    aligned_pdb: Path,
    designed_sequence: str,
    design_id: str,
    ionisation_state: str,
    protonation_justification: str,
    place_oxt: bool,
    out_dir: Path,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Run every stage-two step and return the record.

    The SDF is written only when every chemical check passes. A design whose
    chemistry cannot be validated must not reach Meeko with a plausible-looking
    file, because the resulting PDBQT would be indistinguishable from a good
    one.
    """
    require_rdkit()
    if ionisation_state not in ("neutral", "physiological"):
        raise LigandPreparationError(
            f"Unknown ionisation state {ionisation_state!r}"
        )
    if not protonation_justification.strip():
        raise LigandPreparationError(
            "A protonation justification is required: the modelled state must "
            "be a recorded decision, not an inherited default."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = out_dir / f"{design_id}_ligand.sdf"
    record_path = out_dir / f"{design_id}_ligand.json"
    working_pdb = out_dir / f"{design_id}_with_oxt.pdb"
    for path in (sdf_path, record_path):
        if path.exists() and not overwrite:
            raise LigandPreparationError(
                f"Refusing to overwrite {path}. Choose a fresh output "
                "directory, or pass --overwrite deliberately."
            )

    structure = peptide_geometry.read_structure(aligned_pdb)
    oxt_record: Dict[str, object] = {
        "placed": False,
        "reason": "--place-c-terminal-oxt was not given",
    }
    source_for_rdkit = aligned_pdb
    if place_oxt:
        oxt_record = place_c_terminal_oxt(structure)
        structure.write_pdb(str(working_pdb))
        source_for_rdkit = working_pdb

    molecule = assign_bond_orders(source_for_rdkit, designed_sequence)
    ionisation: Dict[str, object]
    if ionisation_state == "physiological":
        molecule, ionisation = apply_physiological_ionisation(molecule)
    else:
        ionisation = {
            "state": "neutral",
            "changes": [],
            "net_formal_charge": Chem.GetFormalCharge(molecule),
            "limitation": (
                "The template's neutral state: protonated carboxyl, neutral "
                "guanidinium, neutral amine. This is a chemical-dictionary "
                "state, not a physiological one."
            ),
        }

    stereochemistry = stereochemistry_report(molecule)
    coordinates = heavy_atom_coordinate_check(molecule, source_for_rdkit)
    torsions = rotatable_bond_report(molecule)

    with_hydrogens = Chem.AddHs(molecule, addCoords=True)
    heavy_atoms = sum(
        1 for a in with_hydrogens.GetAtoms() if a.GetAtomicNum() > 1
    )

    record: Dict[str, object] = {
        "schema": SCHEMA,
        "design_id": design_id,
        "designed_sequence": designed_sequence,
        "inputs": {
            "aligned_peptide_pdb": str(aligned_pdb),
            "aligned_peptide_pdb_sha256": file_sha256(aligned_pdb),
        },
        "c_terminal_oxt": oxt_record,
        "bond_orders": {
            "source": "RDKit MolFromSequence template for the designed sequence",
            "method": "AllChem.AssignBondOrdersFromTemplate",
            "molecular_formula": rdMolDescriptors.CalcMolFormula(molecule),
            "heavy_atoms": heavy_atoms,
            "note": (
                "Connectivity and bond orders come from the template; every "
                "coordinate comes from the structure."
            ),
        },
        "ionisation": ionisation,
        "protonation_justification": protonation_justification,
        "stereochemistry": stereochemistry,
        "heavy_atom_coordinates": coordinates,
        "torsional_degrees_of_freedom": torsions,
        "atom_mapping": atom_mapping(with_hydrogens),
        "hydrogens": {
            "added_by": "RDKit AddHs with addCoords=True",
            "total_atoms_with_hydrogen": with_hydrogens.GetNumAtoms(),
            "reason": (
                "Meeko does not add hydrogens and its source requires explicit "
                "hydrogens on input."
            ),
            "limitation": (
                "Hydrogen positions are idealised from geometry, not optimised "
                "against the receptor. No hydrogen-bond network was considered."
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "rdkit": __import__("rdkit").__version__,
            "gemmi": gemmi.__version__,
            "numpy": np.__version__,
        },
    }

    failures: List[str] = []
    if not stereochemistry["all_l_configuration"]:
        failures.append(
            f"stereochemistry mismatch: {stereochemistry['mismatches']}"
        )
    if not coordinates["coordinates_preserved"]:
        failures.append(
            "heavy-atom coordinates not preserved: maximum deviation "
            f"{coordinates['maximum_deviation_angstrom']} A, unmatched "
            f"{coordinates['unmatched']}"
        )
    record["chemical_validation_failures"] = failures
    record["validated"] = not failures

    if failures:
        record["outputs"] = {
            "sdf": None,
            "reason": "no SDF written: chemical validation failed",
        }
    else:
        writer = Chem.SDWriter(str(sdf_path))
        with_hydrogens.SetProp("_Name", design_id)
        writer.write(with_hydrogens)
        writer.close()
        record["outputs"] = {
            "sdf": str(sdf_path),
            "sdf_sha256": file_sha256(sdf_path),
            "working_pdb": str(working_pdb) if place_oxt else None,
        }
    record["interpretation"] = (
        "Chemical validation of a docking ligand. It establishes that the SDF "
        "carries the intended chemistry at a stated ionisation state with the "
        "predicted coordinates intact. It does not establish that Meeko will "
        "type the molecule correctly, and it is not evidence of binding."
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    record["outputs"]["record"] = str(record_path)
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("aligned_pdb", type=Path)
    parser.add_argument("--designed-sequence", required=True)
    parser.add_argument("--design-id", required=True)
    parser.add_argument(
        "--ionisation-state",
        required=True,
        choices=("neutral", "physiological"),
        help="Modelled ionisation state. No default: this is a decision.",
    )
    parser.add_argument(
        "--protonation-justification",
        required=True,
        help="Why this state was chosen; recorded in the ligand record",
    )
    parser.add_argument(
        "--place-c-terminal-oxt",
        action="store_true",
        help="Construct the missing C-terminal carboxyl oxygen in idealised "
        "geometry. Required for BoltzGen output, which omits it.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        record = prepare_ligand(
            args.aligned_pdb,
            args.designed_sequence,
            args.design_id,
            args.ionisation_state,
            args.protonation_justification,
            args.place_c_terminal_oxt,
            args.out_dir,
            overwrite=args.overwrite,
        )
    except (LigandPreparationError, peptide_geometry.GeometryError) as error:
        print(f"ligand preparation failed: {error}", file=sys.stderr)
        return 2

    summary = dict(record)
    summary.pop("atom_mapping", None)
    print(json.dumps(summary, indent=2))
    if record["validated"]:
        print(f"sdf: {record['outputs']['sdf']}", file=sys.stderr)
    else:
        print(
            "chemical validation failed: "
            + "; ".join(record["chemical_validation_failures"]),
            file=sys.stderr,
        )
    return 0 if record["validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
