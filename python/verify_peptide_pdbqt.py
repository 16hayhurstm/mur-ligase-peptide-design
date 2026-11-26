"""Verify a Meeko-produced peptide PDBQT against the structure it came from.

Stage 4 of the peptide scoring pilot runs Meeko, which this project cannot
check from the inside: Meeko's documented remit is small organic molecules as
ligands and proteins as *receptors*, and its source carries no peptide or
polymer path on the ligand side. Treating an eight-residue peptide as a Meeko
ligand is therefore off the documented path, and the PDBQT it writes has to be
checked independently.

    python python/verify_peptide_pdbqt.py \
        ligands/spec_0574/spec_0574_bbrigid.pdbqt \
        --source poses/spec_0574/spec_0574_with_oxt.pdb \
        --ligand-record ligands/spec_0574/spec_0574_ligand.json \
        --design-id spec_0574 --expect-backbone-rigid \
        --json-out ligands/spec_0574/spec_0574_bbrigid_check.json

Why a real Meeko PDBQT has no usable atom identities
----------------------------------------------------
Meeko accepts SDF, MOL2 and MOL for ligands, and rejects PDB. Stage 3
therefore writes an SDF, and an SDF carries no PDB residue information, so by
the time Meeko reads the molecule the original chain, residue and atom names
are already gone. Meeko writes what it has, which on the real `spec_0574`
output is a generic residue `UNL` with repeated atom names, giving identities
like `:1 UNL C` several times over.

That is not a Meeko defect and it cannot be fixed by preparing the ligand
differently: it is the cost of the only input format Meeko accepts. The first
version of this module paired atoms on their PDBQT names, which works for a
file with real names and fails closed on a real Meeko output — observed on
Epicac, where no verification record was produced at all.

The route back is the index map
-------------------------------
`--add_index_map` makes Meeko write `REMARK INDEX MAP` lines relating its
output to the molecule it read. From the Meeko v0.7.1 source
(`meeko/writer.py`, `PDBQTWriterLegacy.remark_index_map`), each pair is
emitted as

    string = " %d %d" % (order[key], numbering[key])

where `order = {key: key + 1 for key in numbering}` over the molecule's
0-based atom indices and `numbering[key]` is the PDBQT serial, counted from 1.
So each pair is **(input atom index, PDBQT serial), both 1-based**. The pairs
are wrapped greedily by `break_long_remark_lines` to under 79 characters, with
the full prefix repeated on each line, so there is **no fixed number of pairs
per line** and a parser must not assume one.

Joining that to `atom_mapping` in the stage-3 ligand record, which relates the
same 0-based molecule indices to the original chain, residue and atom names,
recovers a full identity for every PDBQT atom. The join is
`input index - 1 == molecule_index_zero_based`.

Atoms absent from the map
-------------------------
`remark_index_map` skips pseudo atoms, and `numbering` is built only over
atoms that received a PDBQT serial, so an atom Meeko ignored or merged never
appears. For a peptide the expected case is non-polar hydrogens, which Meeko
merges into their heavy atoms by default. A **hydrogen** missing from the map
is therefore recorded and accepted; a **heavy atom** missing from it is a
failure, because a heavy atom that cannot be identified cannot be checked.

Two modes, and failing closed
-----------------------------
With `--ligand-record`, identities come from the index map and the record.
Without it, they are read from the PDBQT's own names, which is only valid when
those names are unique and not generic. If they are repeated or generic, the
verification **fails closed** and says to supply the record, rather than
reporting a comparison it could not actually make.

What is checked
---------------
- Every PDBQT atom resolves to exactly one original atom, and vice versa.
- Heavy-atom coordinates against the source structure, by that verified
  correspondence, reporting the maximum per-atom deviation rather than a mean.
- Which bonds Meeko made rotatable, read from the `BRANCH` records and
  classified as phi, psi, omega or side chain using the *recovered*
  identities. This is the question `grep TORSDOF` cannot answer: a count says
  how many torsions there are, not which, and for a peptide which ones is the
  whole question. If `--rigidify_bonds_smarts` fails to match on the installed
  build, the rigidification silently does nothing and the file looks valid
  while carrying every backbone torsion; `--expect-backbone-rigid` turns that
  into a non-zero exit.
- That no amide became rotatable. `--flexible_amides` was not passed, and
  Meeko's own help calls rotating them "bad". An omega torsion means the
  molecule was not treated as a peptide.
- TORSDOF against the number of `BRANCH` records.
- The atom-type census, and the type and charge on the groups whose typing the
  recorded ionisation decision changes.

`--rename_atoms` must still not be used. It rewrites names, and although this
module no longer depends on them when a ligand record is supplied, it would
make the fallback mode and every by-hand inspection misleading.

What this cannot establish
--------------------------
That the partial charges or atom types are *correct*, only what they are. That
the protonation state is right, only that it is the one recorded. Nothing
about binding.

**The `REMARK INDEX MAP` format here is taken from the Meeko v0.7.1 source,
not from a real output file.** No real Meeko PDBQT was available when this was
written, and the synthetic files in the tests were built to match that source
reading. Confirming the format against a real Epicac output is an outstanding
step, and a passing synthetic test is not validation of the real files.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import receptor_provenance  # noqa: E402

SCHEMA = "mur-ligase-peptide-pdbqt-check/2"

BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

INDEX_MAP_PREFIX = "REMARK INDEX MAP"

# Residue names that carry no chemical identity. Meeko writes a generic
# residue when the molecule it read had no PDB information, which an SDF never
# does. Identities built from these names cannot distinguish one residue from
# another, so they are refused rather than used.
GENERIC_RESIDUE_NAMES = frozenset({"UNL", "LIG", "UNK", "MOL", ""})

# Coordinates are compared at the precision a PDBQT stores, three decimal
# places, plus a small margin for independent rounding.
COORDINATE_TOLERANCE_ANGSTROM = 0.002


class PdbqtVerificationError(ValueError):
    """Raised when a PDBQT cannot be read or verified as specified."""


@dataclass(frozen=True)
class AtomIdentity:
    """One atom's original chemical identity, recovered or read directly."""

    residue_name: str
    residue_seqid: int
    atom_name: str
    author_chain: str = ""
    element: str = ""

    @property
    def label(self) -> str:
        return f"{self.residue_name}{self.residue_seqid}:{self.atom_name}"

    @property
    def is_hydrogen(self) -> bool:
        if self.element:
            return self.element.upper() in ("H", "D")
        return self.atom_name.upper().startswith("H")

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "residue_name": self.residue_name,
            "residue_seqid": self.residue_seqid,
            "atom_name": self.atom_name,
            "author_chain": self.author_chain,
            "element": self.element,
        }


@dataclass
class IdentityRecovery:
    """The correspondence between PDBQT serials and original atom identities.

    `established` is the fail-closed gate: when it is False no comparison that
    depends on identity may be reported, and `reasons` says why.
    """

    by_serial: Dict[int, AtomIdentity] = field(default_factory=dict)
    index_map: Dict[int, int] = field(default_factory=dict)
    method: str = ""
    established: bool = False
    reasons: List[str] = field(default_factory=list)
    hydrogens_absent_from_pdbqt: List[str] = field(default_factory=list)
    heavy_atoms_absent_from_pdbqt: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "established": self.established,
            "reasons": self.reasons,
            "atoms_identified": len(self.by_serial),
            "index_map_pairs": len(self.index_map),
            "hydrogens_absent_from_pdbqt": self.hydrogens_absent_from_pdbqt,
            "heavy_atoms_absent_from_pdbqt": self.heavy_atoms_absent_from_pdbqt,
            "hydrogen_note": (
                "Meeko merges non-polar hydrogens by default, so a hydrogen "
                "absent from the index map is expected. A heavy atom absent "
                "from it is a failure: it cannot be identified, so it cannot "
                "be checked."
            ),
        }


def atoms_by_serial(path: Path) -> Dict[int, Dict[str, object]]:
    """Read a PDBQT's atoms keyed by serial number.

    The serial is the only reliable key in a real Meeko ligand PDBQT: the
    torsion tree refers to atoms by serial, and the names and residues may be
    generic and repeated. PDBQT shares columns 1-54 with PDB, and adds the
    partial charge at 71-76 and the AutoDock atom type at 78-79.
    """
    if not path.is_file():
        raise PdbqtVerificationError(f"PDBQT not found: {path}")
    atoms: Dict[int, Dict[str, object]] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            serial = int(line[6:11])
            name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain = line[21:22].strip()
            residue_seq = int(line[22:26])
            position = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except (ValueError, IndexError) as error:
            raise PdbqtVerificationError(
                f"{path} line {number}: malformed atom record ({error})"
            ) from error
        charge: Optional[float] = None
        if len(line) >= 76:
            try:
                charge = float(line[70:76])
            except ValueError:
                charge = None
        atom_type = line[77:79].strip() if len(line) >= 78 else ""
        if serial in atoms:
            raise PdbqtVerificationError(
                f"{path} line {number}: serial {serial} appears twice, so the "
                "torsion tree and the index map cannot be resolved "
                "unambiguously."
            )
        atoms[serial] = {
            "serial": serial,
            "atom_name": name,
            "residue_name": residue_name,
            "author_chain": chain,
            "residue_seq": residue_seq,
            "atom_type": atom_type,
            "partial_charge": charge,
            "position": position,
        }
    if not atoms:
        raise PdbqtVerificationError(f"{path} contains no atom records")
    return atoms


def parse_index_map(path: Path) -> Dict[int, int]:
    """Parse Meeko's `REMARK INDEX MAP` into {PDBQT serial: input index}.

    Both numbers are 1-based, and each pair is written as
    `(input index, PDBQT serial)` by Meeko v0.7.1. The pairs are wrapped
    greedily to under 79 characters with the prefix repeated, so this reads
    every matching line and makes no assumption about how many pairs each
    carries.

    An odd number of values, a repeated serial or a repeated input index is
    fatal: each would make the correspondence ambiguous, and an ambiguous
    correspondence must not be used.
    """
    values: List[int] = []
    lines_seen = 0
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith(INDEX_MAP_PREFIX):
            continue
        lines_seen += 1
        remainder = line[len(INDEX_MAP_PREFIX):]
        for token in remainder.split():
            try:
                values.append(int(token))
            except ValueError as error:
                raise PdbqtVerificationError(
                    f"{path} line {number}: {INDEX_MAP_PREFIX} contains a "
                    f"non-integer value {token!r} ({error})"
                ) from error
    if lines_seen == 0:
        return {}
    if len(values) % 2 != 0:
        raise PdbqtVerificationError(
            f"{path}: {INDEX_MAP_PREFIX} holds {len(values)} values, which is "
            "odd; the map is written as (input index, PDBQT serial) pairs, so "
            "an odd count means the map is truncated or malformed."
        )

    by_serial: Dict[int, int] = {}
    seen_input: Dict[int, int] = {}
    for position in range(0, len(values), 2):
        input_index, serial = values[position], values[position + 1]
        if input_index < 1 or serial < 1:
            raise PdbqtVerificationError(
                f"{path}: {INDEX_MAP_PREFIX} pair "
                f"({input_index}, {serial}) is not 1-based; both values must "
                "be at least 1."
            )
        if serial in by_serial:
            raise PdbqtVerificationError(
                f"{path}: {INDEX_MAP_PREFIX} maps PDBQT serial {serial} twice "
                f"(to input indices {by_serial[serial]} and {input_index}); "
                "the correspondence is ambiguous."
            )
        if input_index in seen_input:
            raise PdbqtVerificationError(
                f"{path}: {INDEX_MAP_PREFIX} maps input index {input_index} "
                f"twice (to serials {seen_input[input_index]} and {serial}); "
                "the correspondence is ambiguous."
            )
        by_serial[serial] = input_index
        seen_input[input_index] = serial
    return by_serial


def load_ligand_atom_mapping(path: Path) -> Dict[int, AtomIdentity]:
    """Read `atom_mapping` from a stage-3 ligand record.

    Keyed by the 0-based molecule index, which is the index Meeko's map refers
    to after subtracting one. An entry without a chemical identity, or a
    repeated index, is fatal: both would silently lose atoms.
    """
    if not path.is_file():
        raise PdbqtVerificationError(f"Ligand record not found: {path}")
    try:
        record = json.loads(path.read_text())
    except ValueError as error:
        raise PdbqtVerificationError(
            f"Ligand record {path} is not valid JSON: {error}"
        ) from error
    entries = record.get("atom_mapping")
    if not entries:
        raise PdbqtVerificationError(
            f"Ligand record {path} has no 'atom_mapping'. Stage 3 writes it; "
            "a record without it cannot identify PDBQT atoms."
        )

    mapping: Dict[int, AtomIdentity] = {}
    for entry in entries:
        if "molecule_index_zero_based" not in entry:
            raise PdbqtVerificationError(
                f"Ligand record {path}: an atom_mapping entry has no "
                f"'molecule_index_zero_based': {entry}"
            )
        index = int(entry["molecule_index_zero_based"])
        if index in mapping:
            raise PdbqtVerificationError(
                f"Ligand record {path}: molecule index {index} appears twice "
                "in atom_mapping; the correspondence is ambiguous."
            )
        element = str(entry.get("element", ""))
        if "atom_name" not in entry:
            # Hydrogens added after the PDB stage carry no PDB identity. That
            # is expected and harmless, because Meeko merges them; a heavy
            # atom without an identity is not, and is caught below.
            if element.upper() in ("H", "D"):
                mapping[index] = AtomIdentity(
                    residue_name="", residue_seqid=0,
                    atom_name=f"H@{index}", element=element,
                )
                continue
            raise PdbqtVerificationError(
                f"Ligand record {path}: heavy atom at molecule index {index} "
                f"has no 'atom_name', so it cannot be identified: {entry}"
            )
        mapping[index] = AtomIdentity(
            residue_name=str(entry.get("residue_name", "")),
            residue_seqid=int(entry.get("residue_seqid", 0)),
            atom_name=str(entry["atom_name"]),
            author_chain=str(entry.get("author_chain", "")),
            element=element,
        )
    return mapping


def recover_identities(
    atoms: Dict[int, Dict[str, object]],
    index_map: Dict[int, int],
    ligand_mapping: Dict[int, AtomIdentity],
) -> IdentityRecovery:
    """Recover each PDBQT atom's original identity via the index map.

    Fails closed: when any PDBQT atom cannot be resolved, or a heavy atom of
    the original molecule has no serial, `established` is False and the
    reasons say which.
    """
    recovery = IdentityRecovery(
        index_map=dict(index_map),
        method=(
            "Meeko REMARK INDEX MAP joined to atom_mapping in the stage-3 "
            "ligand record; input index - 1 == molecule_index_zero_based"
        ),
    )
    if not index_map:
        recovery.reasons.append(
            f"the PDBQT carries no {INDEX_MAP_PREFIX} lines; rerun "
            "mk_prepare_ligand.py with --add_index_map"
        )
        return recovery

    unresolved: List[str] = []
    for serial in sorted(atoms):
        input_index = index_map.get(serial)
        if input_index is None:
            unresolved.append(
                f"serial {serial} has no entry in {INDEX_MAP_PREFIX}"
            )
            continue
        identity = ligand_mapping.get(input_index - 1)
        if identity is None:
            unresolved.append(
                f"serial {serial} maps to input index {input_index}, which is "
                "absent from the ligand record's atom_mapping"
            )
            continue
        recovery.by_serial[serial] = identity

    # The map must describe this file and nothing else. A pair naming a serial
    # that does not exist is not harmless: the "which molecule atoms reached
    # the PDBQT" test below reads the map, so a stale or invented pair would
    # mark a molecule atom as present when no atom record carries it, masking
    # a dropped heavy atom.
    extra_serials = sorted(set(index_map) - set(atoms))
    if extra_serials:
        unresolved.append(
            f"{INDEX_MAP_PREFIX} names PDBQT serial(s) {extra_serials[:10]} "
            "for which the file has no atom record; the map does not describe "
            "this PDBQT"
        )

    mapped_indices = {
        input_index - 1
        for serial, input_index in index_map.items()
        if serial in atoms
    }
    for index in sorted(set(ligand_mapping) - mapped_indices):
        identity = ligand_mapping[index]
        if identity.is_hydrogen:
            recovery.hydrogens_absent_from_pdbqt.append(identity.label)
        else:
            recovery.heavy_atoms_absent_from_pdbqt.append(identity.label)

    # Two PDBQT atoms resolving to one original atom is a false-positive
    # hazard rather than a curiosity: both would be compared against the same
    # source atom, so one of them is never actually checked while the
    # comparison reports a clean pass. The key is chain, residue number and
    # atom name, which is what identifies an atom within a structure; the
    # residue name is deliberately not part of it, so that two entries
    # disagreeing only on residue name are still caught as one atom.
    seen_identities: Dict[Tuple[str, int, str], int] = {}
    duplicate_identities: List[str] = []
    for serial in sorted(recovery.by_serial):
        identity = recovery.by_serial[serial]
        if identity.is_hydrogen:
            continue
        key = (identity.author_chain, identity.residue_seqid, identity.atom_name)
        if key in seen_identities:
            duplicate_identities.append(
                f"{identity.label} is claimed by serials "
                f"{seen_identities[key]} and {serial}"
            )
            continue
        seen_identities[key] = serial

    if unresolved:
        recovery.reasons.extend(unresolved)
    if duplicate_identities:
        recovery.reasons.append(
            "distinct PDBQT atoms resolve to the same original heavy atom: "
            f"{duplicate_identities[:10]}"
        )
    if recovery.heavy_atoms_absent_from_pdbqt:
        recovery.reasons.append(
            "heavy atoms of the prepared molecule have no PDBQT serial: "
            f"{recovery.heavy_atoms_absent_from_pdbqt[:10]}"
        )

    # The source join is keyed on residue number and atom name, so identities
    # spanning several chains would make it ambiguous.
    chains = {
        i.author_chain for i in recovery.by_serial.values() if i.author_chain
    }
    if len(chains) > 1:
        recovery.reasons.append(
            f"recovered identities span several chains {sorted(chains)}; the "
            "source join is keyed on residue number and atom name only, so "
            "the correspondence would be ambiguous"
        )

    recovery.established = not recovery.reasons
    return recovery


def identities_from_pdbqt_names(
    atoms: Dict[int, Dict[str, object]]
) -> IdentityRecovery:
    """Read identities straight from the PDBQT, when that is legitimate.

    Only valid when the names are unique and not generic. A real Meeko ligand
    PDBQT satisfies neither, so this path refuses and says to supply the
    ligand record rather than silently comparing the wrong atoms.
    """
    recovery = IdentityRecovery(
        method="PDBQT atom names read directly (no ligand record supplied)"
    )
    generic = sorted(
        {
            str(a["residue_name"])
            for a in atoms.values()
            if str(a["residue_name"]).upper() in GENERIC_RESIDUE_NAMES
        }
    )
    if generic:
        recovery.reasons.append(
            f"the PDBQT uses generic residue name(s) {generic}, which carry "
            "no chemical identity. Supply --ligand-record so identities can "
            "be recovered from the index map."
        )

    seen: Dict[Tuple[str, int, str], int] = {}
    duplicates: List[str] = []
    for serial in sorted(atoms):
        atom = atoms[serial]
        key = (
            str(atom["residue_name"]),
            int(atom["residue_seq"]),
            str(atom["atom_name"]),
        )
        if key in seen:
            duplicates.append(
                f"{key[0]}{key[1]}:{key[2]} at serials {seen[key]} and {serial}"
            )
            continue
        seen[key] = serial
        recovery.by_serial[serial] = AtomIdentity(
            residue_name=key[0],
            residue_seqid=key[1],
            atom_name=key[2],
            author_chain=str(atom["author_chain"]),
        )
    if duplicates:
        recovery.reasons.append(
            "the PDBQT repeats atom identities, so they cannot be used as "
            f"keys: {duplicates[:10]}. Supply --ligand-record so identities "
            "can be recovered from the index map."
        )
    recovery.established = not recovery.reasons
    return recovery


def parse_torsion_tree(path: Path) -> List[Tuple[int, int]]:
    """Return the rotatable bonds a PDBQT declares, as atom-serial pairs.

    Each `BRANCH i j` record names one rotatable bond. This reads the bonds
    themselves rather than the TORSDOF count, because the count says how many
    there are and not which ones, and for a peptide which ones is the whole
    question.
    """
    bonds: List[Tuple[int, int]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith("BRANCH"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise PdbqtVerificationError(
                f"{path} line {number}: malformed BRANCH record {line!r}"
            )
        try:
            bonds.append((int(parts[1]), int(parts[2])))
        except ValueError as error:
            raise PdbqtVerificationError(
                f"{path} line {number}: BRANCH serials are not integers "
                f"({error})"
            ) from error
    return bonds


def torsdof(path: Path) -> Optional[int]:
    """Read the TORSDOF line, or None when it is absent or unparseable.

    `torsdof_status` distinguishes those two cases; this returns only the
    value, because most callers want the number.
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


def torsdof_status(path: Path) -> str:
    """Report whether TORSDOF is 'present', 'absent' or 'malformed'.

    The distinction matters because both absence and a malformed value give a
    `None` count, and a `None` must not be mistaken for agreement. TORSDOF is
    the number Vina reads, so a file without a usable one is not a file whose
    torsional degrees of freedom have been established.
    """
    for line in path.read_text().splitlines():
        if not line.startswith("TORSDOF"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return "malformed"
        try:
            int(parts[1])
        except ValueError:
            return "malformed"
        return "present"
    return "absent"


def classify_torsions(
    bonds: Sequence[Tuple[int, int]], recovery: IdentityRecovery
) -> Dict[str, object]:
    """Classify each rotatable bond using recovered chemical identities.

    - `N`-`CA` within one residue is phi
    - `CA`-`C` within one residue is psi
    - `C`-`N` between different residues is omega, the amide
    - anything else is a side-chain bond

    A serial with no recovered identity is reported rather than skipped:
    silently dropping it would understate the backbone count, which is the
    number the rigidification check turns on.
    """
    phi_psi: List[Dict[str, object]] = []
    omega: List[Dict[str, object]] = []
    side_chain: List[Dict[str, object]] = []
    unresolved: List[str] = []

    for first_serial, second_serial in bonds:
        first = recovery.by_serial.get(first_serial)
        second = recovery.by_serial.get(second_serial)
        if first is None or second is None:
            unresolved.append(
                f"BRANCH {first_serial} {second_serial}: no recovered "
                "identity for at least one atom"
            )
            continue
        names = {first.atom_name, second.atom_name}
        same_residue = first.residue_seqid == second.residue_seqid
        entry = {
            "bond": f"{first.label} - {second.label}",
            "serials": [first_serial, second_serial],
        }
        if same_residue and names == {"N", "CA"}:
            entry["torsion"] = "phi"
            phi_psi.append(entry)
        elif same_residue and names == {"CA", "C"}:
            entry["torsion"] = "psi"
            phi_psi.append(entry)
        elif not same_residue and names == {"C", "N"}:
            entry["torsion"] = "omega"
            omega.append(entry)
        else:
            side_chain.append(entry)

    return {
        "rotatable_bonds_declared": len(bonds),
        "backbone_phi_psi": phi_psi,
        "backbone_phi_psi_count": len(phi_psi),
        "amide_omega": omega,
        "amide_omega_count": len(omega),
        "side_chain": side_chain,
        "side_chain_count": len(side_chain),
        "unresolved": unresolved,
        "backbone_is_rigid": bool(not phi_psi and not omega and not unresolved),
        "identity_source": recovery.method,
        "note": (
            "Read from the PDBQT's own BRANCH records and classified using "
            "recovered identities. A backbone torsion here means Vina's local "
            "optimisation can reshape the predicted backbone; an omega "
            "torsion means the molecule was not treated as a peptide."
        ),
    }


def atom_type_census(atoms: Dict[int, Dict[str, object]]) -> Dict[str, int]:
    """Count AutoDock atom types, so the typing is visible rather than assumed."""
    census: Dict[str, int] = {}
    for atom in atoms.values():
        key = str(atom["atom_type"]) or "(blank)"
        census[key] = census.get(key, 0) + 1
    return dict(sorted(census.items()))


def types_for_atom_names(
    atoms: Dict[int, Dict[str, object]],
    recovery: IdentityRecovery,
    wanted: Sequence[str],
) -> Dict[str, object]:
    """Report the type and charge assigned to each named atom.

    Names come from the recovered identities, so this works on a PDBQT whose
    own names are generic. Used for the groups the ionisation decision
    changes: the arginine guanidinium and the C-terminal carboxylate.
    """
    out: Dict[str, object] = {}
    for serial, identity in recovery.by_serial.items():
        if identity.atom_name not in wanted:
            continue
        atom = atoms.get(serial)
        if atom is None:
            continue
        out[identity.label] = {
            "serial": serial,
            "atom_type": atom["atom_type"],
            "partial_charge": atom["partial_charge"],
        }
    return dict(sorted(out.items()))


def compare_against_source(
    atoms: Dict[int, Dict[str, object]],
    recovery: IdentityRecovery,
    source: Path,
) -> Dict[str, object]:
    """Confirm the PDBQT's heavy atoms sit where the source structure put them.

    Pairing runs PDBQT serial -> recovered identity -> source atom, so it
    never depends on the PDBQT's own names. The **maximum** per-atom deviation
    is reported rather than a mean, because a mean over fifty atoms hides one
    atom that moved.

    Hydrogens are excluded on both sides: Meeko merges non-polar hydrogens by
    default, so their absence is expected rather than a discrepancy.
    """
    try:
        source_records = receptor_provenance.parse_atom_records(source)
    except receptor_provenance.ReceptorError as error:
        raise PdbqtVerificationError(
            f"cannot read source structure {source}: {error}"
        ) from error

    source_heavy: Dict[Tuple[int, str], Tuple[str, Tuple[float, float, float]]] = {}
    for key, record in source_records.items():
        if key.atom_name.upper().startswith("H"):
            continue
        join_key = (key.residue_seq, key.atom_name)
        if join_key in source_heavy:
            raise PdbqtVerificationError(
                f"source {source} has two atoms named {key.atom_name} in "
                f"residue {key.residue_seq}; the join is ambiguous."
            )
        source_heavy[join_key] = (key.residue_name, record.position)

    compared = 0
    worst = 0.0
    worst_atom: Optional[str] = None
    moved: List[Dict[str, object]] = []
    missing: List[str] = []
    residue_name_mismatches: List[str] = []

    for serial in sorted(atoms):
        identity = recovery.by_serial.get(serial)
        if identity is None or identity.is_hydrogen:
            continue
        join_key = (identity.residue_seqid, identity.atom_name)
        if join_key not in source_heavy:
            missing.append(
                f"{identity.label} (serial {serial}) is not in the source"
            )
            continue
        source_residue_name, source_position = source_heavy[join_key]
        if (
            identity.residue_name
            and source_residue_name
            and identity.residue_name != source_residue_name
        ):
            residue_name_mismatches.append(
                f"residue {identity.residue_seqid}: ligand record says "
                f"{identity.residue_name}, source says {source_residue_name}"
            )
        position = atoms[serial]["position"]
        deviation = (
            sum((a - b) ** 2 for a, b in zip(position, source_position)) ** 0.5
        )
        compared += 1
        if deviation > worst:
            worst = deviation
            worst_atom = identity.label
        if deviation > COORDINATE_TOLERANCE_ANGSTROM:
            moved.append(
                {
                    "atom": identity.label,
                    "serial": serial,
                    "moved_angstrom": round(deviation, 4),
                }
            )

    unpaired_source = sorted(
        f"{name}{key[0]}:{key[1]}"
        for key, (name, _) in source_heavy.items()
        if key
        not in {
            (i.residue_seqid, i.atom_name)
            for s, i in recovery.by_serial.items()
            if not i.is_hydrogen and s in atoms
        }
    )
    moved.sort(key=lambda entry: entry["moved_angstrom"], reverse=True)

    return {
        "source": str(source),
        "source_heavy_atoms": len(source_heavy),
        "pdbqt_heavy_atoms_identified": compared,
        "heavy_atoms_compared": compared,
        "max_deviation_angstrom": round(worst, 6),
        "max_deviation_atom": worst_atom,
        "tolerance_angstrom": COORDINATE_TOLERANCE_ANGSTROM,
        "moved_atoms": moved[:25],
        "moved_atom_count": len(moved),
        "missing_atoms": missing,
        "source_atoms_not_in_pdbqt": unpaired_source,
        "residue_name_mismatches": residue_name_mismatches,
        # One comparison per source heavy atom, no more and no fewer. A count
        # larger than the source's is the arithmetic signature of two PDBQT
        # atoms being compared against one source atom, which reads as a
        # clean pass while leaving one of them unchecked.
        "comparison_count_consistent": compared == len(source_heavy),
        "coordinates_preserved": bool(
            not moved
            and not missing
            and not unpaired_source
            and compared == len(source_heavy)
        ),
        "paired_on": (
            "PDBQT serial -> recovered identity -> source (residue number, "
            "atom name); never file order and never the PDBQT's own names"
        ),
        "hydrogens": (
            "excluded: Meeko merges non-polar hydrogens by default, so their "
            "absence from the PDBQT is expected"
        ),
    }


def verify(
    pdbqt: Path,
    source: Optional[Path],
    design_id: str,
    ligand_record: Optional[Path] = None,
    expect_backbone_rigid: bool = False,
    inspect_atom_names: Sequence[str] = (
        "NE", "CZ", "NH1", "NH2", "OXT", "O",
    ),
) -> Dict[str, object]:
    """Run every PDBQT check and return a single record."""
    atoms = atoms_by_serial(pdbqt)
    bonds = parse_torsion_tree(pdbqt)
    declared = torsdof(pdbqt)
    declaration = torsdof_status(pdbqt)
    index_map = parse_index_map(pdbqt)

    if ligand_record is not None:
        ligand_mapping = load_ligand_atom_mapping(ligand_record)
        recovery = recover_identities(atoms, index_map, ligand_mapping)
    else:
        recovery = identities_from_pdbqt_names(atoms)
        # The map is present but unusable without the record that gives the
        # input indices their chemical meaning. Say so in the reasons, since
        # the remedy is one argument away.
        if index_map and not recovery.established:
            recovery.reasons.append(
                f"the PDBQT does carry {len(index_map)} {INDEX_MAP_PREFIX} "
                "pairs, but they name input indices only; --ligand-record "
                "supplies what those indices mean"
            )

    torsions = classify_torsions(bonds, recovery)

    record: Dict[str, object] = {
        "schema": SCHEMA,
        "design_id": design_id,
        "pdbqt": str(pdbqt),
        "ligand_record": None if ligand_record is None else str(ligand_record),
        "atoms_in_pdbqt": len(atoms),
        "heavy_atoms_in_pdbqt": sum(
            1 for a in atoms.values()
            if not str(a["atom_name"]).upper().startswith("H")
        ),
        "identity_recovery": recovery.as_dict(),
        "torsdof": {
            "declared": declared,
            "declaration": declaration,
            "branch_records": len(bonds),
            "agrees_with_branch_count": (
                None if declared is None else declared == len(bonds)
            ),
            "note": (
                "TORSDOF is what Vina reads. A disagreement with the number of "
                "BRANCH records means the file is internally inconsistent, and "
                "an absent or malformed declaration means the torsional "
                "degrees of freedom are not established at all."
            ),
        },
        "torsion_tree": torsions,
        "atom_type_census": atom_type_census(atoms),
        "ionisable_group_typing": types_for_atom_names(
            atoms, recovery, inspect_atom_names
        ),
        "expectations": {
            "backbone_rigid_expected": expect_backbone_rigid,
            "rename_atoms_must_not_be_used": (
                "--rename_atoms rewrites atom names; identities are recovered "
                "from the index map, but renaming would make the fallback "
                "mode and any by-hand inspection misleading"
            ),
        },
        "index_map_format": (
            "Taken from the Meeko v0.7.1 source "
            "(PDBQTWriterLegacy.remark_index_map): pairs of "
            "(input atom index, PDBQT serial), both 1-based, wrapped greedily "
            "below 79 characters with the prefix repeated per line. NOT yet "
            "confirmed against a real Meeko output."
        ),
    }

    failures: List[str] = []

    # Identity recovery gates everything that depends on identity. Without it
    # the coordinate comparison and the torsion classification are not merely
    # incomplete, they are unfounded, so the verification fails closed.
    if not recovery.established:
        failures.append(
            "atom identities could not be established: "
            + "; ".join(recovery.reasons)
        )
        record["coordinate_preservation"] = {
            "checked": False,
            "reason": (
                "atom identities were not established, so no coordinate "
                "comparison was attempted"
            ),
        }
    elif source is not None:
        record["coordinate_preservation"] = compare_against_source(
            atoms, recovery, source
        )
    else:
        record["coordinate_preservation"] = {
            "checked": False,
            "reason": (
                "no --source given, so it is not established that Meeko "
                "preserved the predicted coordinates"
            ),
        }

    if torsions["amide_omega_count"]:
        failures.append(
            f"{torsions['amide_omega_count']} amide (omega) torsion(s) in the "
            "tree: the molecule was not treated as a peptide"
        )
    if torsions["unresolved"]:
        failures.append(
            f"unresolvable torsion serials: {torsions['unresolved']}"
        )
    if expect_backbone_rigid and torsions["backbone_phi_psi_count"]:
        failures.append(
            f"{torsions['backbone_phi_psi_count']} backbone phi/psi torsion(s) "
            "remain although backbone rigidity was expected: the "
            "--rigidify_bonds_smarts pattern did not match on this Meeko build"
        )
    if declaration != "present":
        failures.append(
            f"TORSDOF declaration is {declaration}: the number Vina reads for "
            "this ligand's torsional degrees of freedom is not established"
        )
    elif record["torsdof"]["agrees_with_branch_count"] is False:
        failures.append(
            f"TORSDOF {declared} disagrees with {len(bonds)} BRANCH records"
        )

    preservation = record["coordinate_preservation"]
    if preservation.get("checked") is not False:
        if not preservation.get("coordinates_preserved"):
            failures.append(
                "heavy-atom coordinates not preserved: maximum deviation "
                f"{preservation.get('max_deviation_angstrom')} A at "
                f"{preservation.get('max_deviation_atom')}"
            )
        if preservation.get("missing_atoms"):
            failures.append(
                "heavy atoms absent from the PDBQT: "
                f"{preservation['missing_atoms'][:10]}"
            )
        if preservation.get("source_atoms_not_in_pdbqt"):
            failures.append(
                "source heavy atoms with no PDBQT counterpart: "
                f"{preservation['source_atoms_not_in_pdbqt'][:10]}"
            )
        if preservation.get("residue_name_mismatches"):
            failures.append(
                "residue name disagreement between the ligand record and the "
                f"source: {preservation['residue_name_mismatches'][:5]}"
            )

    record["verification_failures"] = failures
    record["verified"] = not failures
    record["interpretation"] = (
        "Structural and bookkeeping verification of a PDBQT. It establishes "
        "what Meeko produced, not that the charges or types are correct, and "
        "it is not evidence of binding. A pass on a synthetic file is not "
        "validation of a real Meeko output."
    )
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdbqt", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Structure the ligand was prepared from, for the coordinate "
        "comparison. Omitting it leaves coordinate preservation unestablished.",
    )
    parser.add_argument(
        "--ligand-record",
        type=Path,
        default=None,
        help="Stage-3 ligand record holding atom_mapping. Required for a real "
        "Meeko PDBQT, whose own atom identities are generic and repeated.",
    )
    parser.add_argument("--design-id", required=True)
    parser.add_argument(
        "--expect-backbone-rigid",
        action="store_true",
        help="Fail if any backbone phi/psi torsion remains. Use for a PDBQT "
        "prepared with --rigidify_bonds_smarts.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.json_out is not None and args.json_out.exists():
        parser.error(f"Refusing to overwrite an existing file: {args.json_out}")

    try:
        record = verify(
            args.pdbqt,
            args.source,
            args.design_id,
            ligand_record=args.ligand_record,
            expect_backbone_rigid=args.expect_backbone_rigid,
        )
    except PdbqtVerificationError as error:
        print(f"PDBQT verification failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(record, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"record: {args.json_out}", file=sys.stderr)
    if record["verification_failures"]:
        print(
            "verification failures: "
            + "; ".join(record["verification_failures"]),
            file=sys.stderr,
        )
    return 0 if record["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
