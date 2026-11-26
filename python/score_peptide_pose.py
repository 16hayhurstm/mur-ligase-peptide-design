"""Score a placed peptide with AutoDock Vina, and optionally relax it locally.

This is stage five of the peptide scoring pilot, after Meeko has written the
ligand PDBQT and `verify_peptide_pdbqt.py` has verified it. It evaluates the
BoltzGen pose as given, and separately reports what happens when Vina is
allowed to relax it locally. The two results are never merged: one is a
property of the prediction, the other is a property of the prediction plus
Vina's local search.

    python python/score_peptide_pose.py \
        --ligand ligands/spec_0574/spec_0574_backbone_rigid.pdbqt \
        --ligand-record ligands/spec_0574/spec_0574_ligand.json \
        --verification-record \
            ligands/spec_0574/spec_0574_backbone_rigid_pep001_check.json \
        --predicted-structure ligands/spec_0574/spec_0574_with_oxt.pdb \
        --receptor receptors/murc_anp_nomn_try1.pdbqt \
        --receptor-structure receptors/murc_anp_nomn_try1_prepared.pdb \
        --receptor-manifest receptors/murc_anp_nomn_try1.manifest.json \
        --expected-component-atoms 31 \
        --box-record examples/box_murc_uma_chainE_30A.json \
        --design-id spec_0574 --out-dir scores/spec_0574 \
        --seed 1 --cpu 2 --local-optimisation

`--verification-record` and `--predicted-structure` are **required**. Between
them they establish that the file being scored is the file PEP-001 verified
and that it still holds the predicted coordinates; without both, the number
produced would be indistinguishable from one where they held.
`--unvalidated-exploratory` waives the requirement, and the record is then
labelled an UNVALIDATED EXPLORATORY SCORE, carries the blocking issue and
exits 1. That label never means PEP-001 verification failed, and must never be
read as meaning it passed.

`--receptor` and `--receptor-structure` must be the two outputs of one
preparation run of the same receptor: `prepare_receptor.py` writes
`<prefix>.pdbqt` and `<prefix>_prepared.pdb`. The score comes from the first
and the clash distances from the second, so the run checks that they agree and
**refuses the clash comparison** if they do not, rather than reporting clash
counts that belong to another receptor.

A clash comparison also needs the cofactor's completeness established from
outside those two files, by `--expected-component-atoms` (31 heavy atoms for
the ANP-containing MurC model prepared here) or by `--receptor-manifest`,
whose recorded checksums identify the validated receptor bytes. A cofactor
truncated the same way in both files is self-consistent and would otherwise
pass every check that compares them with each other.

**This stage requires AutoDock Vina and runs only where Vina is installed.**
On a machine without it, every function that needs it raises rather than
returning a degraded result, because a missing scoring function is a blocker
and not a check that can be skipped.

A real Meeko ligand PDBQT has no usable atom identities
-------------------------------------------------------
Meeko accepts only SDF, MOL2 and MOL for ligands, and an SDF carries no PDB
residue information. Its ligand PDBQT therefore uses the generic residue
`UNL`, residue number 1, a blank chain and bare element letters for atom
names, so identities such as `:1 UNL C` repeat many times over. Nothing can be
keyed on them, and an earlier version of this module aborted on exactly that,
before Vina was even constructed.

Two different correspondence problems follow, and they are solved differently.

**The starting ligand.** Identities are recovered from Meeko's
`REMARK INDEX MAP` joined to `atom_mapping` in the stage-3 ligand record,
which is the mechanism `verify_peptide_pdbqt.py` validates. `--ligand-record`
is therefore required, not optional: without it there is no chemical identity
for any atom, so no backbone/side-chain split and no named clash.

**The pose Vina writes.** There is no index map joining it to the starting
file, and this module does **not** assume that Vina preserved atom order,
serial numbers or the order in which a group's sibling branches were written.
Correspondence is established from quantities that Vina's local optimisation
cannot change -- the rigid-group partition of the torsion tree, the AutoDock
atom type, and each atom's sorted distances to the other atoms of its own
rigid group -- and is then verified against those same invariants, with the
worst deviation reported.

It is established in two stages, because an atom is identified *within* its
rigid group and so the groups have to be paired first. They are paired on
their own invariants, descending the tree from the root: a group's written
path encodes which sibling came first, and two equivalent sibling branches
written the other way round would be cross-assigned without a single
invariant objecting, since everything inside each branch still agrees. See
`match_rigid_groups`.

Atoms, or whole branches, that no such invariant can separate are reported as
indistinguishable and the comparison fails closed; pairing *atoms* by minimal
displacement is available on an explicit flag and recorded as a convention,
while two equivalent branches are refused either way. Whether the serial
order did in fact survive is *reported as an observation*, never used as an
assumption. See `establish_correspondence`.

**What verification can be relied on.** A stage-4 record is evidence about
the file it was made from, so every quantity it states is re-derived from the
file being scored and required to agree before the record is used at all:
atom count, atom-type census, TORSDOF, BRANCH count, design ID and file
names. That fixes the molecule and its torsion state, which is what separates
one design's flexible and backbone-rigid variants. It does not fix the
coordinates, because the record carries no checksum; `--predicted-structure`
closes that by comparing the PDBQT against the structure stage 3 wrote. See
`bind_verification_record`.

What local optimisation moves, and the trap in it
-------------------------------------------------
Vina relaxes the rigid-body placement **and every torsion in the ligand's
torsion tree**. For a peptide that includes the backbone: phi and psi are
ordinary rotatable single bonds, and Meeko makes them torsions by default. The
real `spec_0574` flexible PDBQT carries 16 backbone phi/psi torsions out of a
TORSDOF of 30; its backbone-rigid counterpart carries none, with TORSDOF 14.

So an unqualified local optimisation **does not preserve the predicted
backbone**. Even with a backbone-rigid torsion tree, the six rigid-body
degrees of freedom remain, so the peptide may translate and rotate as a unit.
Measured as total displacement that legitimate motion is indistinguishable
from the backbone being reshaped, and would wrongly fail a "backbone
preserved" test. The displacement report therefore separates:

1. **absolute displacement** in the receptor frame, per atom;
2. **rigid-body movement** of the peptide as a whole;
3. **internal backbone deformation**, the residual after superposing the
   relaxed backbone onto the predicted one;
4. **side-chain displacement**, in the receptor frame and again in the
   peptide's own frame.

Only (3) says whether the predicted backbone was altered.

The receptor is rigid throughout, as in the campaign. A clash that the real
complex would relieve by moving a receptor side chain cannot be relieved here.

What a score means
------------------
A Vina score is an approximate model score in kcal/mol. It is not a measured
binding free energy, not a dissociation constant and not evidence of binding
or of enzyme inhibition. Vina's own validation error is roughly 2-3 kcal/mol,
so a difference smaller than that does not establish a preference however
reproducibly it is reported. Scores for a peptide are additionally outside the
chemical space Vina's scoring function was fitted on, and an improvement under
local optimisation is a statement about Vina's function, not about affinity.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import peptide_contacts  # noqa: E402
import peptide_geometry  # noqa: E402
import receptor_provenance  # noqa: E402
import verify_peptide_pdbqt as verify  # noqa: E402

try:
    from vina import Vina

    VINA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where Vina is absent
    VINA_AVAILABLE = False

# Correspondence is established twice over, for atoms and for the rigid
# groups they sit in, by the same kind of reasoning. `Key` lets the two
# share the helpers rather than carry two copies of them.
Key = TypeVar("Key")

SCHEMA = "mur-ligase-peptide-score/2"

# Backbone heavy-atom names, including the constructed C-terminal OXT. These
# are PDB atom names from the recovered identities, never the generic names a
# Meeko ligand PDBQT carries.
BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})

# The C-terminal oxygen that BoltzGen does not predict and stage 3 constructs.
# Clash counts are reported with and without it, because the project's earlier
# 24-clash figure was measured on the 55-atom pose before it existed.
CONSTRUCTED_C_TERMINAL_ATOM = "OXT"

# AutoDock atom types that denote hydrogen. Heavy-atom selection uses the atom
# type, not the atom name: in a generic Meeko ligand PDBQT the names are bare
# element letters, so a name-based test both misses heavy atoms and catches
# hydrogens whose name happens to start with H.
HYDROGEN_ATOM_TYPES = frozenset({"H", "HD", "HS"})

# Tolerance, in Angstroms, on distances that Vina's local optimisation cannot
# change: separations within one rigid group, and the lengths of the rotatable
# bonds between groups. Coordinates are written to three decimal places, so
# rounding alone accounts for about 0.002 A; this is deliberately generous
# against that and tight against any real change of internal geometry.
RIGID_DISTANCE_TOLERANCE_ANGSTROM = 0.05

# Tolerance, in Angstroms, below which two atoms of the same type in the same
# rigid group count as *indistinguishable* rather than merely similar. It is
# deliberately much tighter than the tolerance above, because the two serve
# opposite purposes: the generous one must never exclude the true pairing,
# while this one decides when a difference is large enough to pair atoms on.
# Coordinates are written to three decimals, so a distance carries an error of
# about 0.002 A and a signature difference of 0.02 A is an order of magnitude
# above the noise. This matters for real output: measured on the aligned
# `spec_0574` pose, the C-terminal carbonyl oxygen sits 1.2211 A from its
# carbon while OXT is constructed at 1.25 A, a difference of 0.0289 A. Those
# two oxygens are therefore separable, but only by a margin that a single
# tolerance of 0.05 A would have thrown away.
SIGNATURE_DISTINCTNESS_TOLERANCE_ANGSTROM = 0.02

# The largest set of mutually indistinguishable atoms this module will resolve
# by displacement minimisation. Beyond this the permutation count stops being
# small, and a set that large is better reported than resolved.
MAXIMUM_DEGENERATE_SET = 6

# Minimum atoms required to define a rigid-body fit. Three non-collinear
# points fix a frame; fewer cannot, and a fit on fewer would silently report a
# rotation that is not determined.
MINIMUM_SUPERPOSITION_ATOMS = 3


class ScoringError(RuntimeError):
    """Raised when a peptide pose cannot be scored or compared as specified."""


def require_vina() -> None:
    """Refuse to continue without Vina, rather than degrading the result."""
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

    Returns `None` when the file carries no usable TORSDOF line, rather than
    substituting a count computed some other way: the number that matters is
    the one in the file Vina reads. `verify_peptide_pdbqt.torsdof_status`
    distinguishes an absent line from a malformed one, and this module records
    that status beside the count.
    """
    return verify.torsdof(path)


def pose_atoms(path: Path) -> Dict[int, Dict[str, object]]:
    """Read a PDBQT's atoms keyed by serial, translating read errors.

    Reading is delegated to the PEP-001 verifier, so there is one PDBQT reader
    in the project rather than two that could drift apart. The serial is used
    as a *key for reading*, never as evidence of correspondence between two
    files; that is `establish_correspondence`'s job.
    """
    try:
        return verify.atoms_by_serial(path)
    except verify.PdbqtVerificationError as error:
        raise ScoringError(f"cannot read pose {path}: {error}") from error


# ---------------------------------------------------------------------------
# A. Atom correspondence between the starting ligand and the pose Vina writes
# ---------------------------------------------------------------------------


@dataclass
class TorsionTree:
    """A ligand PDBQT's rigid groups and the rotatable bonds between them.

    `group_of_serial` maps each atom serial to its rigid group's path, a
    string such as `root` or `root.2.1` built from nesting depth and order of
    appearance. `bonds` are the `BRANCH` atom pairs, and `bond_groups` the
    (parent path, child path) pairs they connect.

    The partition is the useful part. Vina's ligand model moves a ligand by
    rigid-body placement plus rotation about the bonds in this tree, so atoms
    within one group never move relative to one another. Every distance inside
    a group is therefore an invariant of the optimisation, which is what makes
    correspondence provable rather than assumed.
    """

    group_of_serial: Dict[int, str] = field(default_factory=dict)
    groups: Dict[str, List[int]] = field(default_factory=dict)
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    bond_groups: List[Tuple[str, str]] = field(default_factory=list)

    # There is deliberately no "compare two trees by their group paths"
    # helper here. A path numbers siblings by order of appearance, so such a
    # comparison rejects a legitimately reordered file and accepts a
    # cross-assigned one; `match_rigid_groups` pairs the groups on their own
    # invariants instead.

    def as_dict(self) -> Dict[str, object]:
        return {
            "rigid_groups": len(self.groups),
            "rotatable_bonds": len(self.bonds),
            "group_sizes": {path: len(s) for path, s in sorted(self.groups.items())},
            "bond_groups": [list(pair) for pair in self.bond_groups],
        }


def parse_rigid_groups(path: Path) -> TorsionTree:
    """Partition a ligand PDBQT's atoms into the rigid groups of its tree.

    `ROOT`/`ENDROOT` and each `BRANCH`/`ENDBRANCH` pair open and close a rigid
    group. A group's path records its position in the tree: `root`, then the
    order of appearance among its parent's children, so `root.2.1` is the
    first child of the second child of the root. The path is independent of
    atom serial numbering, which is the point: it is comparable between two
    files whose serials may differ.

    The root's own branches appear *after* `ENDROOT` in the PDBQT layout, so
    the root remains the parent of a top-level branch although its own atom
    block has closed. Atoms are assigned only to a group whose block is open:
    an atom after `ENDROOT` but outside every `BRANCH` belongs to no rigid
    group, and that, like an unbalanced `END` record, is fatal. Such a file is
    not a ligand PDBQT Vina can read, and guessing a partition for it would
    put a wrong correspondence on an apparently successful footing.
    """
    tree = TorsionTree()
    open_branches: List[str] = []
    atom_target: Optional[str] = None
    children: Dict[str, int] = {}
    root_seen = False
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        keyword = line.split()[0] if line.strip() else ""
        if keyword == "ROOT":
            if root_seen or open_branches:
                raise ScoringError(
                    f"{path} line {number}: a second ROOT, or a ROOT inside a "
                    "branch"
                )
            root_seen = True
            atom_target = "root"
            tree.groups.setdefault("root", [])
        elif keyword == "ENDROOT":
            if atom_target != "root":
                raise ScoringError(
                    f"{path} line {number}: ENDROOT with no matching ROOT"
                )
            atom_target = None
        elif keyword == "BRANCH":
            parts = line.split()
            if len(parts) < 3:
                raise ScoringError(
                    f"{path} line {number}: malformed BRANCH record {line!r}"
                )
            try:
                bond = (int(parts[1]), int(parts[2]))
            except ValueError as error:
                raise ScoringError(
                    f"{path} line {number}: BRANCH serials are not integers "
                    f"({error})"
                ) from error
            if not root_seen:
                raise ScoringError(
                    f"{path} line {number}: BRANCH before any ROOT"
                )
            parent = open_branches[-1] if open_branches else "root"
            children[parent] = children.get(parent, 0) + 1
            child = f"{parent}.{children[parent]}"
            open_branches.append(child)
            atom_target = child
            tree.groups.setdefault(child, [])
            tree.bonds.append(bond)
            tree.bond_groups.append((parent, child))
        elif keyword == "ENDBRANCH":
            if not open_branches:
                raise ScoringError(
                    f"{path} line {number}: ENDBRANCH with no matching BRANCH"
                )
            open_branches.pop()
            atom_target = open_branches[-1] if open_branches else None
        elif keyword in ("ATOM", "HETATM"):
            if atom_target is None:
                raise ScoringError(
                    f"{path} line {number}: atom record outside ROOT and "
                    "every BRANCH, so it belongs to no rigid group. A ligand "
                    "PDBQT Vina can read has a ROOT."
                )
            try:
                serial = int(line[6:11])
            except ValueError as error:
                raise ScoringError(
                    f"{path} line {number}: unreadable atom serial ({error})"
                ) from error
            if serial in tree.group_of_serial:
                raise ScoringError(
                    f"{path} line {number}: serial {serial} appears twice, so "
                    "its rigid group is ambiguous"
                )
            tree.group_of_serial[serial] = atom_target
            tree.groups[atom_target].append(serial)
    if open_branches:
        raise ScoringError(
            f"{path}: {len(open_branches)} BRANCH record(s) never closed"
        )
    if not root_seen:
        raise ScoringError(f"{path}: no ROOT record, so there is no rigid-group tree")
    if not tree.group_of_serial:
        raise ScoringError(f"{path}: no atoms inside any rigid group")
    return tree


def intra_group_signatures(
    atoms: Dict[int, Dict[str, object]], tree: TorsionTree
) -> Dict[int, np.ndarray]:
    """Each atom's sorted distances to the other atoms of its rigid group.

    This vector is invariant under rigid-body motion of the whole ligand and
    under rotation about any bond of the torsion tree, because neither changes
    a distance inside a rigid group. It is the evidence that distinguishes two
    atoms of the same element and atom type, which element and type alone
    cannot. For a one-atom group it is empty, and such an atom is identified by
    its group path alone.
    """
    signatures: Dict[int, np.ndarray] = {}
    for path, serials in tree.groups.items():
        present = [s for s in serials if s in atoms]
        positions = {
            s: np.asarray(atoms[s]["position"], dtype=float) for s in present
        }
        for serial in present:
            others = [
                float(np.linalg.norm(positions[serial] - positions[other]))
                for other in present
                if other != serial
            ]
            signatures[serial] = np.asarray(sorted(others), dtype=float)
    return signatures

# ---------------------------------------------------------------------------
# Rigid-group correspondence: which branch of the pose is which branch of the
# starting ligand.
#
# A group's path records its position in the tree by order of appearance, and
# that is *not* an invariant. Nothing established says Vina's writer emits a
# group's sibling branches in the order it read them. Pairing groups by path
# therefore risks exactly the error this module exists to prevent: two
# equivalent sibling branches written in the opposite order would be
# cross-assigned, and because every invariant *inside* each branch still
# agrees, the derived map would pass verification cleanly while reporting the
# distance between the two branches as displacement of atoms that never
# moved. The groups are therefore matched on their own invariants, descending
# the tree from the root, and sibling order is never used as evidence.
# ---------------------------------------------------------------------------


def _distance(
    atoms: Dict[int, Dict[str, object]], first: int, second: int
) -> float:
    """The distance in angstrom between two atoms of one file."""
    return float(
        np.linalg.norm(
            np.asarray(atoms[first]["position"], dtype=float)
            - np.asarray(atoms[second]["position"], dtype=float)
        )
    )


def group_relations(
    tree: TorsionTree,
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[int, int]]]:
    """Each rigid group's child groups, and the bond that attaches each child.

    `BRANCH i j` names the parent-side atom first and the child-side atom
    second, so the attachment is directional and can be described without
    reference to either file's atom numbering.
    """
    children: Dict[str, List[str]] = {path: [] for path in tree.groups}
    attachment: Dict[str, Tuple[int, int]] = {}
    for (parent, child), bond in zip(tree.bond_groups, tree.bonds):
        children.setdefault(parent, []).append(child)
        attachment[child] = bond
    return children, attachment


def group_structural_keys(
    atoms: Dict[int, Dict[str, object]], tree: TorsionTree
) -> Dict[str, Tuple]:
    """A discrete, order-independent description of each group's subtree.

    The key holds the AutoDock atom types of the bond that attaches the group
    to its parent, the multiset of atom types the group contains, and the
    sorted multiset of its children's keys. Two groups with different keys are
    chemically different branches and can be told apart with certainty; two
    with the same key are candidates for one another and are separated, if at
    all, by geometry. Nothing in the key depends on serial numbers, on file
    order or on which sibling was written first.
    """
    children, attachment = group_relations(tree)
    keys: Dict[str, Tuple] = {}

    def build(path: str) -> Tuple:
        bond = attachment.get(path)
        if bond is None:
            anchor: Tuple[str, ...] = ("ROOT",)
        else:
            parent_serial, child_serial = bond
            anchor = (
                str(atoms[parent_serial]["atom_type"])
                if parent_serial in atoms
                else "?",
                str(atoms[child_serial]["atom_type"])
                if child_serial in atoms
                else "?",
            )
        own = tuple(
            sorted(
                str(atoms[serial]["atom_type"])
                for serial in tree.groups[path]
                if serial in atoms
            )
        )
        child_keys = tuple(sorted(build(child) for child in children[path]))
        keys[path] = (anchor, own, child_keys)
        return keys[path]

    build("root")
    return keys


def subtree_signatures(
    atoms: Dict[int, Dict[str, object]], tree: TorsionTree
) -> Dict[str, np.ndarray]:
    """Each group's subtree as a sorted vector of invariant distances.

    The vector collects every intra-group distance of every group in the
    subtree, together with the length of every bond in it including the one
    that attaches the subtree to its parent. None of these can be changed by
    rigid-body motion or by rotation about any bond of the tree, so two
    branches whose geometry differs anywhere differ here, wherever in the
    subtree the difference lies. The vector is sorted, so it carries no
    ordering information of its own.
    """
    children, attachment = group_relations(tree)

    def values(path: str) -> List[float]:
        serials = [serial for serial in tree.groups[path] if serial in atoms]
        collected = [
            _distance(atoms, first, second)
            for first, second in itertools.combinations(serials, 2)
        ]
        bond = attachment.get(path)
        if bond is not None and bond[0] in atoms and bond[1] in atoms:
            collected.append(_distance(atoms, bond[0], bond[1]))
        for child in children[path]:
            collected.extend(values(child))
        return collected

    return {
        path: np.asarray(sorted(values(path)), dtype=float)
        for path in tree.groups
    }


def _unique_assignment(
    candidates: Dict[Key, List[Key]]
) -> Tuple[Dict[Key, Key], Dict[Key, List[Key]]]:
    """Fix every mutually unique candidate pair, repeatedly.

    A source with one candidate which is in turn the only source claiming that
    candidate is determined. Removing it can make a neighbour unique, so the
    pass repeats until nothing more is settled. What is left over is genuinely
    contested, and is returned for the caller to resolve or refuse.
    """
    remaining = {key: list(options) for key, options in candidates.items()}
    assignment: Dict[Key, Key] = {}
    progress = True
    while progress:
        progress = False
        reverse: Dict[Key, List[Key]] = {}
        for source, options in remaining.items():
            for option in options:
                reverse.setdefault(option, []).append(source)
        for source in sorted(remaining):
            options = remaining[source]
            if len(options) != 1:
                continue
            target = options[0]
            if len(reverse.get(target, [])) != 1:
                continue
            assignment[source] = target
            del remaining[source]
            for other in remaining.values():
                if target in other:
                    other.remove(target)
            progress = True
            break
    return assignment, remaining


def match_rigid_groups(
    before_atoms: Dict[int, Dict[str, object]],
    before_tree: TorsionTree,
    after_atoms: Dict[int, Dict[str, object]],
    after_tree: TorsionTree,
    tolerance: float = RIGID_DISTANCE_TOLERANCE_ANGSTROM,
) -> Tuple[Dict[str, str], List[str], List[Dict[str, object]]]:
    """Pair the two files' rigid groups without assuming sibling-branch order.

    The root is matched to the root, and from there each matched pair's
    children are matched among themselves on their structural keys and their
    subtree distance vectors. Two sibling branches that differ chemically or
    geometrically are therefore paired correctly however they were written,
    which an order-dependent scheme gets wrong in both directions: it
    cross-assigns a swapped pair of equivalent branches, and it refuses a
    swapped pair of unequal ones.

    Where two sibling branches cannot be told apart at all, the match **fails
    closed**. Minimal-displacement resolution is deliberately not offered for
    whole branches: unlike two equivalent atoms, two equivalent subtrees also
    fix the assignment of every atom beneath them, and choosing that on
    proximity alone would put a convention underneath a measurement rather
    than beside it. Such a case is reported, with the branches named.

    Returns the group map, the reasons it is not usable, and any pairings that
    needed geometry rather than structure to settle.
    """
    reasons: List[str] = []
    notes: List[Dict[str, object]] = []
    before_children, _ = group_relations(before_tree)
    after_children, _ = group_relations(after_tree)
    before_keys = group_structural_keys(before_atoms, before_tree)
    after_keys = group_structural_keys(after_atoms, after_tree)
    before_signatures = subtree_signatures(before_atoms, before_tree)
    after_signatures = subtree_signatures(after_atoms, after_tree)

    if before_keys["root"] != after_keys["root"]:
        reasons.append(
            "the two torsion trees are not the same tree: their rigid-group "
            "structure differs in atom types, in branch topology or in both, "
            "compared independently of the order the branches were written in"
        )
        return {}, reasons, notes

    group_map: Dict[str, str] = {"root": "root"}
    queue: List[Tuple[str, str]] = [("root", "root")]
    while queue:
        parent_before, parent_after = queue.pop(0)
        sources = sorted(before_children[parent_before])
        targets = sorted(after_children[parent_after])
        if len(sources) != len(targets):
            reasons.append(
                f"rigid group {parent_before} carries {len(sources)} branch(es) "
                f"in the starting ligand and {len(targets)} in the written "
                "pose, so the trees are not the same tree"
            )
            continue
        candidates: Dict[str, List[str]] = {}
        for source in sources:
            allowed = [
                target
                for target in targets
                if after_keys[target] == before_keys[source]
                and before_signatures[source].shape
                == after_signatures[target].shape
                and _signature_mismatch(
                    before_signatures[source], after_signatures[target]
                )
                <= tolerance
            ]
            if not allowed:
                reasons.append(
                    f"rigid group {source} of the starting ligand matches no "
                    f"branch of the written pose below {parent_after}: no "
                    "branch there has its structure and its invariant "
                    f"distances to within {tolerance} A"
                )
            candidates[source] = allowed
        if any(not options for options in candidates.values()):
            continue
        assignment, contested = _unique_assignment(candidates)
        if contested:
            nearest, details = _resolve_by_nearest_signature(
                sorted(contested),
                contested,
                before_signatures,
                after_signatures,
                SIGNATURE_DISTINCTNESS_TOLERANCE_ANGSTROM,
                labels=("starting_group", "pose_group"),
            )
            if nearest is None:
                named = ", ".join(sorted(contested))
                reasons.append(
                    f"rigid groups [{named}] cannot be told apart from one "
                    f"another below {parent_before}: they attach to the same "
                    "kind of atom, hold the same atom types and the same "
                    f"branch structure, and their invariant distances agree "
                    f"to within {tolerance} A. Which branch of the written "
                    "pose is which is therefore not established. The order "
                    "the branches appear in is not evidence, because it is "
                    "not established that Vina's writer preserves it, and "
                    "--resolve-symmetric-atoms does not extend to whole "
                    "branches"
                )
                continue
            assignment.update(nearest)
            notes.extend(details)
        for source in sources:
            group_map[source] = assignment[source]
            queue.append((source, assignment[source]))

    if reasons:
        return {}, reasons, notes
    if len(group_map) != len(before_tree.groups) or len(
        set(group_map.values())
    ) != len(group_map):
        reasons.append(
            f"the rigid-group map pairs {len(group_map)} of "
            f"{len(before_tree.groups)} groups onto "
            f"{len(set(group_map.values()))} distinct groups of the written "
            "pose, so it is not a bijection"
        )
        return {}, reasons, notes
    return group_map, reasons, notes



@dataclass
class Correspondence:
    """Which atom of a written pose is which atom of the starting ligand.

    `established` is the fail-closed gate: when it is False no displacement,
    clash or containment comparison between the two files may be reported, and
    `reasons` says why not.
    """

    pairs: Dict[int, int] = field(default_factory=dict)
    established: bool = False
    reasons: List[str] = field(default_factory=list)
    group_pairs: Dict[str, str] = field(default_factory=dict)
    groups_resolved_by_nearest_signature: List[Dict[str, object]] = field(
        default_factory=list
    )
    determined_by_signature: List[int] = field(default_factory=list)
    resolved_by_nearest_signature: List[Dict[str, object]] = field(
        default_factory=list
    )
    symmetry_resolved: List[Dict[str, object]] = field(default_factory=list)
    serial_order_preserved: Optional[bool] = None
    max_intra_group_deviation_angstrom: Optional[float] = None
    max_bond_length_deviation_angstrom: Optional[float] = None
    checks: Dict[str, object] = field(default_factory=dict)

    @property
    def method(self) -> str:
        return (
            "Rigid-group invariants, in two stages. First the two files' "
            "rigid groups are matched to one another, descending the torsion "
            "tree from the root and pairing each matched group's branches on "
            "their atom types, their branch structure and their subtree "
            "distance vectors. Branch order in the file is not used, because "
            "it is not established that Vina's writer preserves it, and two "
            "equivalent branches written the other way round would otherwise "
            "be cross-assigned without any invariant objecting. Then each "
            "atom is identified within its matched group by its AutoDock atom "
            "type and its sorted vector of distances to the other atoms of "
            "that group, both invariant under rigid-body motion and under "
            "rotation about any tree bond. Candidates are admitted on a "
            "generous tolerance so the true pairing cannot be excluded, and "
            "where more than one survives, the atom is paired with the "
            "signature it matches most closely, but only when that match "
            "beats its runner-up by a margin well above coordinate noise. "
            "Atom serials and file order are never used as evidence; whether "
            "they happened to survive is reported, not assumed."
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "established": self.established,
            "reasons": self.reasons,
            "atoms_paired": len(self.pairs),
            "determined_by_invariant_signature": len(
                self.determined_by_signature
            ),
            "resolved_by_nearest_signature": self.resolved_by_nearest_signature,
            "signature_distinctness_tolerance_angstrom": (
                SIGNATURE_DISTINCTNESS_TOLERANCE_ANGSTROM
            ),
            "resolved_by_symmetry_convention": self.symmetry_resolved,
            "rigid_group_pairs": dict(sorted(self.group_pairs.items())),
            "rigid_groups_resolved_by_nearest_signature": (
                self.groups_resolved_by_nearest_signature
            ),
            "sibling_branch_order_assumed": False,
            "serial_order_preserved": self.serial_order_preserved,
            "serial_order_note": (
                "An observation about the installed Vina's writer, recorded "
                "so the assumption this module refuses to make can be tested "
                "against real output. Nothing here depends on it."
            ),
            "max_intra_group_deviation_angstrom": (
                self.max_intra_group_deviation_angstrom
            ),
            "max_bond_length_deviation_angstrom": (
                self.max_bond_length_deviation_angstrom
            ),
            "rigid_distance_tolerance_angstrom": (
                RIGID_DISTANCE_TOLERANCE_ANGSTROM
            ),
            "checks": self.checks,
            "limitation": (
                "Atoms that are indistinguishable under every invariant "
                "available in a PDBQT are reported, not silently paired. "
                "Where the symmetry convention was applied, the displacement "
                "of those atoms is a lower bound: the assignment chosen is "
                "the one minimising their total squared displacement. Atoms "
                "listed under resolved_by_nearest_signature are a different "
                "case: those were paired on the evidence, each by a margin "
                "the record states. Two sibling branches that no invariant "
                "separates are refused outright rather than resolved by "
                "convention, because the assignment of a branch fixes the "
                "assignment of every atom beneath it."
            ),
        }


def _composition_multiset(
    atoms: Dict[int, Dict[str, object]]
) -> Dict[Tuple[str, Optional[float]], int]:
    """Count atoms by (AutoDock type, partial charge), as a molecule fingerprint."""
    counts: Dict[Tuple[str, Optional[float]], int] = {}
    for atom in atoms.values():
        charge = atom["partial_charge"]
        key = (
            str(atom["atom_type"]),
            None if charge is None else round(float(charge), 3),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _signature_candidates(
    before_atoms: Dict[int, Dict[str, object]],
    after_atoms: Dict[int, Dict[str, object]],
    before_tree: TorsionTree,
    after_tree: TorsionTree,
    before_signatures: Dict[int, np.ndarray],
    after_signatures: Dict[int, np.ndarray],
    group_map: Dict[str, str],
    tolerance: float,
) -> Tuple[Dict[int, List[int]], List[str]]:
    """For each starting atom, the pose atoms its invariants allow it to be.

    Candidacy requires membership of the *matched* rigid group, the same
    AutoDock atom type, and sorted intra-group distance vectors agreeing
    elementwise to within `tolerance`. The group comes from
    `match_rigid_groups` rather than from the group's written path, because a
    path encodes the order the branches were written in and that is not an
    invariant of Vina's writer. The tolerance is the generous one: its job is to make certain
    the true pairing is never excluded, not to decide which pairing is right.
    Returns the candidate lists and any reasons why an atom has no candidate
    at all.
    """
    by_group_after: Dict[str, List[int]] = {}
    for serial, path in after_tree.group_of_serial.items():
        by_group_after.setdefault(path, []).append(serial)

    candidates: Dict[int, List[int]] = {}
    reasons: List[str] = []
    for serial in sorted(before_atoms):
        path = before_tree.group_of_serial[serial]
        signature = before_signatures[serial]
        allowed: List[int] = []
        for other in sorted(by_group_after.get(group_map[path], [])):
            if str(after_atoms[other]["atom_type"]) != str(
                before_atoms[serial]["atom_type"]
            ):
                continue
            candidate_signature = after_signatures[other]
            if candidate_signature.shape != signature.shape:
                continue
            if signature.size == 0:
                allowed.append(other)
                continue
            if float(np.max(np.abs(signature - candidate_signature))) <= tolerance:
                allowed.append(other)
        if not allowed:
            reasons.append(
                f"starting atom serial {serial} (type "
                f"{before_atoms[serial]['atom_type']}, group {path}) matches "
                "no atom of the written pose: no pose atom in the matched "
                f"rigid group {group_map[path]} has its atom type and its "
                f"invariant distance signature to within {tolerance} A"
            )
        candidates[serial] = allowed
    return candidates, reasons


def _signature_mismatch(first: np.ndarray, second: np.ndarray) -> float:
    """The largest elementwise disagreement between two signature vectors."""
    if first.size == 0:
        return 0.0
    return float(np.max(np.abs(first - second)))


def _resolve_by_nearest_signature(
    sources: Sequence[Key],
    candidates: Dict[Key, List[Key]],
    before_signatures: Dict[Key, np.ndarray],
    after_signatures: Dict[Key, np.ndarray],
    margin: float,
    labels: Tuple[str, str] = ("starting_serial", "pose_serial"),
) -> Tuple[Optional[Dict[Key, Key]], List[Dict[str, object]]]:
    """Pair things on which invariant signature they match most closely.

    Used for atoms, with their intra-group distance vectors, and for rigid
    groups, with their subtree distance vectors; `labels` names the two sides
    in the record so the one report does not read as the other.

    This is not a convention: it is a resolution on the evidence, used where
    the generous candidacy tolerance admitted more than one pose atom but the
    signatures still separate them. Every atom must have a *clear* best match,
    better than its runner-up by at least `margin`, and the result must be a
    bijection. If either fails, `None` is returned and the caller treats the
    atoms as genuinely indistinguishable.
    """
    assignment: Dict[Key, Key] = {}
    details: List[Dict[str, object]] = []
    for source in sources:
        scored = sorted(
            (
                _signature_mismatch(
                    before_signatures[source], after_signatures[target]
                ),
                target,
            )
            for target in candidates[source]
        )
        if not scored:
            return None, []
        best_cost, best_target = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else None
        if runner_up is not None and (runner_up - best_cost) < margin:
            return None, []
        assignment[source] = best_target
        details.append(
            {
                labels[0]: source,
                labels[1]: best_target,
                "signature_mismatch_angstrom": round(best_cost, 4),
                "next_best_mismatch_angstrom": (
                    None if runner_up is None else round(runner_up, 4)
                ),
                "margin_required_angstrom": margin,
            }
        )
    if len(set(assignment.values())) != len(assignment):
        return None, []
    return assignment, details


def _resolve_degenerate_set(
    before_serials: Sequence[int],
    after_serials: Sequence[int],
    before_atoms: Dict[int, Dict[str, object]],
    after_atoms: Dict[int, Dict[str, object]],
    candidates: Dict[int, List[int]],
) -> Dict[int, int]:
    """Pair indistinguishable atoms by minimising total squared displacement.

    This is the project's existing convention for atoms that a structure
    cannot tell apart, as used for the equivalent oxygens of the UMA control.
    It is applied only when the caller has asked for it explicitly, and it is
    recorded, because it makes the displacement of these atoms a lower bound
    rather than a measurement.
    """
    best: Optional[Dict[int, int]] = None
    best_cost = float("inf")
    for permutation in itertools.permutations(after_serials):
        if any(
            target not in candidates[source]
            for source, target in zip(before_serials, permutation)
        ):
            continue
        cost = 0.0
        for source, target in zip(before_serials, permutation):
            delta = np.asarray(
                after_atoms[target]["position"], dtype=float
            ) - np.asarray(before_atoms[source]["position"], dtype=float)
            cost += float(delta @ delta)
        if cost < best_cost:
            best_cost = cost
            best = dict(zip(before_serials, permutation))
    if best is None:
        raise ScoringError(
            "no assignment of an indistinguishable atom set is consistent "
            "with the invariants; this should have been caught as a "
            "prerequisite failure"
        )
    return best


def establish_correspondence(
    before_atoms: Dict[int, Dict[str, object]],
    before_tree: TorsionTree,
    after_atoms: Dict[int, Dict[str, object]],
    after_tree: TorsionTree,
    resolve_symmetric_atoms: bool = False,
    tolerance: float = RIGID_DISTANCE_TOLERANCE_ANGSTROM,
) -> Correspondence:
    """Establish which pose atom is which starting atom, or fail closed.

    Vina's written pose carries no index map and no usable atom identities, so
    the correspondence has to be derived. It is derived from quantities Vina's
    local optimisation cannot change:

    1. **the rigid-group partition** of the torsion tree, by nesting path;
    2. **the AutoDock atom type**, which Vina reads and does not rewrite;
    3. **the sorted intra-group distance vector**, which no rigid-body motion
       and no torsion rotation can alter.

    Atom count, element and atom type are necessary but not sufficient, which
    is why (3) is present: it is the only available quantity that separates two
    atoms of the same type in the same group.

    Two tolerances are used, for opposite purposes. `tolerance` admits
    candidates and is generous, so the true pairing can never be excluded by
    writer round-off. `SIGNATURE_DISTINCTNESS_TOLERANCE_ANGSTROM` decides when
    a signature difference is large enough to pair atoms on, and is tight.
    Between them sits the case that matters for the real target: a C-terminal
    carboxylate whose two oxygens differ by 0.029 A, which a single generous
    tolerance would have discarded as indistinguishable.

    Where even (3) cannot separate two atoms, they are **reported as
    indistinguishable and the correspondence is not established**, unless
    `resolve_symmetric_atoms` is set, in which case the minimal-displacement
    assignment is used and recorded as a convention rather than a measurement.

    The derived map is then *verified* against the same invariants, and the
    maximum deviations are reported, so a correspondence that happens to be
    consistent is distinguishable from one that is merely plausible.

    Serial numbers and file order are used nowhere as evidence.
    `serial_order_preserved` records whether the result agrees with serial
    equality, as an observation about the installed Vina.
    """
    correspondence = Correspondence()
    checks: Dict[str, object] = {
        "starting_atoms": len(before_atoms),
        "pose_atoms": len(after_atoms),
        "rigid_distance_tolerance_angstrom": tolerance,
    }

    reasons: List[str] = []
    if len(before_atoms) != len(after_atoms):
        reasons.append(
            f"the written pose has {len(after_atoms)} atom records and the "
            f"starting ligand has {len(before_atoms)}; a pose with a "
            "different atom count is not a pose of this ligand"
        )

    before_composition = _composition_multiset(before_atoms)
    after_composition = _composition_multiset(after_atoms)
    charges_available = all(
        atom["partial_charge"] is not None
        for atom in list(before_atoms.values()) + list(after_atoms.values())
    )
    checks["partial_charges_available"] = charges_available
    if before_composition != after_composition:
        if charges_available:
            reasons.append(
                "the two files disagree on the multiset of (atom type, "
                "partial charge), so they are not the same prepared molecule"
            )
        else:
            before_types = sorted(
                verify.atom_type_census(before_atoms).items()
            )
            after_types = sorted(verify.atom_type_census(after_atoms).items())
            if before_types != after_types:
                reasons.append(
                    "the two files disagree on their AutoDock atom-type "
                    f"census: {before_types} against {after_types}"
                )

    # Deliberately *not* a comparison of the two trees' group paths. A path
    # records which sibling was written first, so comparing paths rejects a
    # legitimately reordered file and accepts a cross-assigned one. Only
    # order-independent quantities are compared here; the branches themselves
    # are matched below.
    before_sizes = sorted(len(s) for s in before_tree.groups.values())
    after_sizes = sorted(len(s) for s in after_tree.groups.values())
    if len(before_tree.groups) != len(after_tree.groups) or len(
        before_tree.bonds
    ) != len(after_tree.bonds):
        reasons.append(
            f"the torsion trees differ in size: {len(before_tree.groups)} "
            f"rigid groups and {len(before_tree.bonds)} rotatable bonds "
            f"against {len(after_tree.groups)} and {len(after_tree.bonds)}. "
            "The rigid-group partition is the frame the correspondence is "
            "built in, so a different tree makes the comparison undefined"
        )
    elif before_sizes != after_sizes:
        reasons.append(
            f"the trees' rigid groups hold {before_sizes} atoms against "
            f"{after_sizes}, so they are not the same partition however the "
            "branches are ordered"
        )
    checks["starting_tree"] = before_tree.as_dict()
    checks["pose_tree"] = after_tree.as_dict()

    if reasons:
        correspondence.reasons = reasons
        correspondence.checks = checks
        return correspondence

    # Which branch of the pose is which branch of the starting ligand. This
    # has to be settled before any atom can be identified, because an atom is
    # identified within its group, and it cannot be settled from the paths.
    group_map, group_reasons, group_notes = match_rigid_groups(
        before_atoms, before_tree, after_atoms, after_tree, tolerance
    )
    correspondence.groups_resolved_by_nearest_signature = group_notes
    checks["rigid_groups_matched"] = len(group_map)
    checks["rigid_group_matching"] = (
        "on atom types, branch structure and subtree distance vectors, "
        "descending from the root; the order the branches were written in is "
        "not used"
    )
    if group_reasons:
        correspondence.reasons = group_reasons
        correspondence.checks = checks
        return correspondence
    correspondence.group_pairs = group_map

    before_signatures = intra_group_signatures(before_atoms, before_tree)
    after_signatures = intra_group_signatures(after_atoms, after_tree)
    candidates, candidate_reasons = _signature_candidates(
        before_atoms, after_atoms, before_tree, after_tree,
        before_signatures, after_signatures, group_map, tolerance,
    )
    reasons.extend(candidate_reasons)
    if reasons:
        correspondence.reasons = reasons
        correspondence.checks = checks
        return correspondence

    # Fix every atom whose candidate is unique and which is in turn the unique
    # candidate of that pose atom, repeatedly. Only what survives that is
    # genuinely degenerate.
    settled, remaining = _unique_assignment(candidates)
    pairs: Dict[int, int] = dict(settled)
    determined: List[int] = sorted(settled)

    # What is left forms connected components of indistinguishable atoms.
    degenerate_sets: List[Tuple[List[int], List[int]]] = []
    unassigned = set(remaining)
    while unassigned:
        seed = min(unassigned)
        sources = {seed}
        targets = set(remaining[seed])
        changed = True
        while changed:
            changed = False
            for serial in sorted(unassigned - sources):
                if targets.intersection(remaining[serial]):
                    sources.add(serial)
                    targets.update(remaining[serial])
                    changed = True
        unassigned -= sources
        degenerate_sets.append((sorted(sources), sorted(targets)))

    for sources, targets in degenerate_sets:
        labels = ", ".join(
            f"serial {s} (type {before_atoms[s]['atom_type']}, group "
            f"{before_tree.group_of_serial[s]})"
            for s in sources
        )
        # Before treating these atoms as indistinguishable, try the evidence
        # once more at the tighter threshold. The candidacy tolerance is
        # generous by design, so it can admit a second candidate whose
        # signature is in fact clearly further away.
        nearest, details = _resolve_by_nearest_signature(
            sources, remaining, before_signatures, after_signatures,
            SIGNATURE_DISTINCTNESS_TOLERANCE_ANGSTROM,
        )
        if nearest is not None:
            pairs.update(nearest)
            determined.extend(sources)
            correspondence.resolved_by_nearest_signature.extend(details)
            continue
        if len(sources) != len(targets):
            reasons.append(
                f"{len(sources)} starting atom(s) [{labels}] share "
                f"{len(targets)} candidate pose atom(s); the correspondence "
                "cannot be a bijection"
            )
            continue
        if len(sources) > MAXIMUM_DEGENERATE_SET:
            reasons.append(
                f"{len(sources)} atoms are mutually indistinguishable "
                f"[{labels}], more than the {MAXIMUM_DEGENERATE_SET} this "
                "module will resolve by convention"
            )
            continue
        if not resolve_symmetric_atoms:
            reasons.append(
                f"{len(sources)} atoms cannot be told apart by any invariant "
                f"a PDBQT carries [{labels}]: same rigid group, same atom "
                "type, and intra-group distance signatures agreeing to within "
                f"{tolerance} A. Pass --resolve-symmetric-atoms to pair them "
                "by minimal displacement, which is recorded as a convention "
                "and makes their displacement a lower bound"
            )
            continue
        resolved = _resolve_degenerate_set(
            sources, targets, before_atoms, after_atoms, remaining
        )
        pairs.update(resolved)
        correspondence.symmetry_resolved.append(
            {
                "starting_serials": sources,
                "pose_serials": targets,
                "rigid_group": before_tree.group_of_serial[sources[0]],
                "atom_type": str(before_atoms[sources[0]]["atom_type"]),
                "assignment": {str(k): v for k, v in sorted(resolved.items())},
                "convention": (
                    "minimal total squared displacement, as used for the "
                    "equivalent oxygens of the UMA control"
                ),
            }
        )

    if reasons:
        correspondence.reasons = reasons
        correspondence.checks = checks
        return correspondence

    if len(set(pairs.values())) != len(pairs) or len(pairs) != len(before_atoms):
        reasons.append(
            f"the derived map pairs {len(pairs)} of {len(before_atoms)} atoms "
            f"onto {len(set(pairs.values()))} distinct pose atoms, so it is "
            "not a bijection"
        )
        correspondence.reasons = reasons
        correspondence.checks = checks
        return correspondence

    verification = _verify_correspondence(
        pairs, before_atoms, after_atoms, before_tree, after_tree, tolerance
    )
    reasons.extend(verification["reasons"])
    correspondence.max_intra_group_deviation_angstrom = verification[
        "max_intra_group_deviation_angstrom"
    ]
    correspondence.max_bond_length_deviation_angstrom = verification[
        "max_bond_length_deviation_angstrom"
    ]
    checks["intra_group_pairs_checked"] = verification["intra_group_pairs_checked"]
    checks["rotatable_bonds_checked"] = verification["rotatable_bonds_checked"]

    correspondence.pairs = pairs
    correspondence.determined_by_signature = sorted(determined)
    correspondence.serial_order_preserved = all(
        source == target for source, target in pairs.items()
    )
    correspondence.reasons = reasons
    correspondence.established = not reasons
    correspondence.checks = checks
    if not correspondence.established:
        correspondence.pairs = {}
    return correspondence


def _verify_correspondence(
    pairs: Dict[int, int],
    before_atoms: Dict[int, Dict[str, object]],
    after_atoms: Dict[int, Dict[str, object]],
    before_tree: TorsionTree,
    after_tree: TorsionTree,
    tolerance: float,
) -> Dict[str, object]:
    """Test a derived map against the invariants it was supposed to respect.

    Two independent tests. Every distance inside a rigid group must survive
    the mapping, and every rotatable bond must still be a bond of the same
    length between the mapped atoms. A map that is wrong within a group whose
    geometry is not symmetric fails the first; a map that crosses groups fails
    the second.
    """
    reasons: List[str] = []
    worst_intra = 0.0
    intra_pairs = 0
    for serials in before_tree.groups.values():
        present = [s for s in serials if s in before_atoms]
        for first, second in itertools.combinations(present, 2):
            before_distance = float(
                np.linalg.norm(
                    np.asarray(before_atoms[first]["position"], dtype=float)
                    - np.asarray(before_atoms[second]["position"], dtype=float)
                )
            )
            after_distance = float(
                np.linalg.norm(
                    np.asarray(
                        after_atoms[pairs[first]]["position"], dtype=float
                    )
                    - np.asarray(
                        after_atoms[pairs[second]]["position"], dtype=float
                    )
                )
            )
            deviation = abs(before_distance - after_distance)
            worst_intra = max(worst_intra, deviation)
            intra_pairs += 1
    if worst_intra > tolerance:
        reasons.append(
            "a distance inside a rigid group changed by "
            f"{round(worst_intra, 4)} A under the derived correspondence, "
            f"more than the {tolerance} A tolerance. Vina's optimisation "
            "cannot change such a distance, so either the correspondence is "
            "wrong or the pose is not a pose of this ligand"
        )

    after_bond_set = {frozenset(bond) for bond in after_tree.bonds}
    worst_bond = 0.0
    bonds_checked = 0
    for first, second in before_tree.bonds:
        if first not in pairs or second not in pairs:
            reasons.append(
                f"rotatable bond {first}-{second} names an atom with no "
                "correspondence"
            )
            continue
        mapped = frozenset((pairs[first], pairs[second]))
        if mapped not in after_bond_set:
            reasons.append(
                f"rotatable bond {first}-{second} maps onto "
                f"{sorted(mapped)}, which is not a rotatable bond of the "
                "written pose; the mapping does not preserve the torsion tree"
            )
            continue
        before_length = float(
            np.linalg.norm(
                np.asarray(before_atoms[first]["position"], dtype=float)
                - np.asarray(before_atoms[second]["position"], dtype=float)
            )
        )
        after_length = float(
            np.linalg.norm(
                np.asarray(after_atoms[pairs[first]]["position"], dtype=float)
                - np.asarray(after_atoms[pairs[second]]["position"], dtype=float)
            )
        )
        worst_bond = max(worst_bond, abs(before_length - after_length))
        bonds_checked += 1
    if worst_bond > tolerance:
        reasons.append(
            f"a rotatable bond length changed by {round(worst_bond, 4)} A "
            f"under the derived correspondence, more than {tolerance} A. "
            "Rotation about a bond does not change its length"
        )

    return {
        "reasons": reasons,
        "max_intra_group_deviation_angstrom": round(worst_intra, 4),
        "max_bond_length_deviation_angstrom": (
            round(worst_bond, 4) if bonds_checked else None
        ),
        "intra_group_pairs_checked": intra_pairs,
        "rotatable_bonds_checked": bonds_checked,
    }


# ---------------------------------------------------------------------------
# B. Chemical identity and heavy-atom selection
# ---------------------------------------------------------------------------


def load_identity_recovery(
    ligand_pdbqt: Path, ligand_record: Path
) -> verify.IdentityRecovery:
    """Recover the starting ligand's atom identities, or fail closed.

    This is the PEP-001 mechanism used unchanged: Meeko's `REMARK INDEX MAP`
    joined to `atom_mapping` in the stage-3 ligand record. It is reused rather
    than reimplemented so that the identities this module reports and the
    identities the verifier checked are the same identities, established by
    the same validated code.
    """
    try:
        atoms = verify.atoms_by_serial(ligand_pdbqt)
        index_map = verify.parse_index_map(ligand_pdbqt)
        mapping = verify.load_ligand_atom_mapping(ligand_record)
    except verify.PdbqtVerificationError as error:
        raise ScoringError(str(error)) from error
    recovery = verify.recover_identities(atoms, index_map, mapping)
    if not recovery.established:
        raise ScoringError(
            "atom identities could not be established for "
            f"{ligand_pdbqt}: " + "; ".join(recovery.reasons)
        )
    return recovery


def bind_verification_record(
    record: Dict[str, object],
    ligand_pdbqt: Path,
    ligand_record: Optional[Path] = None,
    design_id: Optional[str] = None,
) -> Dict[str, object]:
    """Tie a stage-4 verification record to the file about to be scored.

    A verified record proves nothing about *this* PDBQT unless it is the
    record for this PDBQT. PEP-001's record carries no checksum of the file it
    verified, so one cannot be demanded of it. What it does carry is a set of
    explicit claims about the file, and every one of them can be re-derived
    here: the atom count, the heavy-atom count, the AutoDock atom-type census,
    the TORSDOF value, how that declaration was formed, and the number of
    BRANCH records. Each is recomputed from the file now being scored and
    required to agree, so the record is treated as a claim to be re-checked
    rather than as evidence to be trusted.

    Together these fix the molecule and its torsion state. In particular they
    separate the two variants of one design, whose whole difference is their
    torsion tree: the `spec_0574` flexible ligand declares TORSDOF 30 over 30
    BRANCH records and its backbone-rigid counterpart declares 14 over 14, so
    the record for one cannot be used to validate the other.

    The design ID and the file names are checked too. Directory paths are
    recorded but not required to match, because the same record is read on the
    machine that wrote it and on the machine that scores.

    Raises `ScoringError` naming every disagreement. What this cannot detect
    is stated in the returned record: see `limitation`.
    """
    try:
        atoms = verify.atoms_by_serial(ligand_pdbqt)
        bonds = verify.parse_torsion_tree(ligand_pdbqt)
    except verify.PdbqtVerificationError as error:
        raise ScoringError(
            f"cannot re-derive the verified quantities from {ligand_pdbqt}: "
            f"{error}"
        ) from error

    declared = record.get("torsdof")
    torsdof = declared if isinstance(declared, dict) else {}
    recorded: Dict[str, object] = {
        "atoms_in_pdbqt": record.get("atoms_in_pdbqt"),
        "heavy_atoms_in_pdbqt": record.get("heavy_atoms_in_pdbqt"),
        "atom_type_census": record.get("atom_type_census"),
        "torsdof_declared": torsdof.get("declared"),
        "torsdof_declaration": torsdof.get("declaration"),
        "branch_records": torsdof.get("branch_records"),
    }
    observed: Dict[str, object] = {
        "atoms_in_pdbqt": len(atoms),
        # The record's own rule for a heavy atom, reproduced so that the two
        # numbers are comparable at all. This module selects heavy atoms by
        # AutoDock type elsewhere, which is the better rule; here the point is
        # agreement with what stage 4 counted, not the better rule.
        "heavy_atoms_in_pdbqt": sum(
            1 for atom in atoms.values()
            if not str(atom["atom_name"]).upper().startswith("H")
        ),
        "atom_type_census": verify.atom_type_census(atoms),
        "torsdof_declared": verify.torsdof(ligand_pdbqt),
        "torsdof_declaration": verify.torsdof_status(ligand_pdbqt),
        "branch_records": len(bonds),
    }

    checks: List[Dict[str, object]] = []
    mismatches: List[str] = []
    for quantity in recorded:
        claimed, found = recorded[quantity], observed[quantity]
        if claimed is None:
            mismatches.append(
                f"the record does not state {quantity}, so it cannot be bound "
                "to this file"
            )
            agrees = False
        else:
            agrees = claimed == found
            if not agrees:
                mismatches.append(
                    f"{quantity}: the record says {claimed!r}, the file being "
                    f"scored has {found!r}"
                )
        checks.append(
            {
                "quantity": quantity,
                "recorded": claimed,
                "observed_in_scored_file": found,
                "agrees": agrees,
            }
        )

    recorded_pdbqt = record.get("pdbqt")
    if not recorded_pdbqt:
        mismatches.append(
            "the record does not name the PDBQT it verified, so it cannot be "
            "bound to this file"
        )
    elif Path(str(recorded_pdbqt)).name != ligand_pdbqt.name:
        mismatches.append(
            f"the record verified {Path(str(recorded_pdbqt)).name!r}, not "
            f"{ligand_pdbqt.name!r}"
        )
    recorded_ligand_record = record.get("ligand_record")
    if ligand_record is not None:
        if not recorded_ligand_record:
            mismatches.append(
                "the record was made without a ligand record, so the "
                "identities it checked are not the identities used here"
            )
        elif Path(str(recorded_ligand_record)).name != ligand_record.name:
            mismatches.append(
                "the record used ligand record "
                f"{Path(str(recorded_ligand_record)).name!r}, not "
                f"{ligand_record.name!r}"
            )
    if design_id is not None and record.get("design_id") != design_id:
        mismatches.append(
            f"the record is for design {record.get('design_id')!r}, not "
            f"{design_id!r}"
        )

    if mismatches:
        raise ScoringError(
            "the stage-4 verification record does not belong to the file "
            "being scored, so it is not evidence about it: "
            + "; ".join(mismatches)
            + ". Verify this exact PDBQT with verify_peptide_pdbqt.py and "
            "score against that record."
        )

    return {
        "bound": True,
        "method": (
            "every quantity the stage-4 record states about the PDBQT was "
            "re-derived from the file being scored and required to agree: "
            "atom count, heavy-atom count, AutoDock atom-type census, TORSDOF "
            "value, TORSDOF declaration status and BRANCH count, together "
            "with the design ID and the file names. The record is a claim to "
            "be re-checked, not evidence to be trusted"
        ),
        "checks": checks,
        "recorded_pdbqt_path": None if not recorded_pdbqt else str(recorded_pdbqt),
        "scored_pdbqt_path": str(ligand_pdbqt),
        "paths_identical": (
            bool(recorded_pdbqt) and str(recorded_pdbqt) == str(ligand_pdbqt)
        ),
        "limitation": (
            "The stage-4 record carries no checksum of the PDBQT it verified, "
            "so an edit that preserved the atom count, the atom-type census, "
            "the TORSDOF declaration and the BRANCH count while moving "
            "coordinates would not be caught by these checks. Pass "
            "--predicted-structure to re-establish coordinate preservation "
            "directly against the structure stage 3 wrote. Adding a "
            "pdbqt_sha256 field to the stage-4 record would close the gap at "
            "its source, but that is a change to verify_peptide_pdbqt.py and "
            "is outside this issue's permitted file scope."
        ),
    }


def recheck_coordinate_preservation(
    atoms: Dict[int, Dict[str, object]],
    recovery: verify.IdentityRecovery,
    source: Path,
) -> Dict[str, object]:
    """Re-run PEP-001's coordinate comparison on the file being scored.

    The verification record's coordinate claim was made about the file as it
    stood when stage 4 ran. This repeats the comparison here, against the
    structure stage 3 wrote, so that the claim is re-established for the file
    now in hand rather than carried over. PEP-001's own comparison function is
    used, so the two agree by construction instead of by inspection.
    """
    try:
        return verify.compare_against_source(atoms, recovery, source)
    except verify.PdbqtVerificationError as error:
        raise ScoringError(
            f"cannot compare {source} against the ligand PDBQT: {error}"
        ) from error


def read_verification_record(
    path: Path,
    ligand_pdbqt: Optional[Path] = None,
    ligand_record: Optional[Path] = None,
    design_id: Optional[str] = None,
) -> Dict[str, object]:
    """Read a stage-4 verification record, and bind it to the scored file.

    Scoring a PDBQT that failed verification would attach a score to a file
    whose chemistry was not established, so `verified: false` is a blocker and
    the schema is checked rather than assumed from the filename.

    A verified record for *another* file is the subtler failure, and the one
    `bind_verification_record` exists to prevent: given `ligand_pdbqt` the
    record's claims are re-derived from the file being scored and required to
    agree. Called without it the record is read but explicitly not bound, and
    the returned summary says so, so a record can never read as evidence about
    a file it was never checked against.
    """
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ScoringError(
            f"cannot read verification record {path}: {error}"
        ) from error
    schema = str(record.get("schema", ""))
    if not schema.startswith("mur-ligase-peptide-pdbqt-check/"):
        raise ScoringError(
            f"{path} is not a peptide PDBQT verification record "
            f"(schema {schema!r})"
        )
    if not record.get("verified"):
        raise ScoringError(
            f"{path} records an unverified ligand PDBQT: "
            f"{record.get('verification_failures')}. Fix stage 4 before "
            "scoring; a score on an unverified file is a score on chemistry "
            "that was never established."
        )
    binding: Dict[str, object]
    if ligand_pdbqt is None:
        binding = {
            "bound": False,
            "reason": (
                "no ligand PDBQT was supplied to bind the record against, so "
                "it is not established that this record describes the file "
                "that was scored"
            ),
        }
    else:
        binding = bind_verification_record(
            record, ligand_pdbqt, ligand_record, design_id
        )

    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema": schema,
        "verified": True,
        "bound_to_scored_input": binding,
        "design_id": record.get("design_id"),
        "torsdof": (record.get("torsdof") or {}).get("declared"),
        "identity_recovery_method": (
            (record.get("identity_recovery") or {}).get("method")
        ),
        "backbone_phi_psi_count": (
            (record.get("torsion_tree") or {}).get("backbone_phi_psi_count")
        ),
        "side_chain_torsion_count": (
            (record.get("torsion_tree") or {}).get("side_chain_count")
        ),
    }


def heavy_atom_serials(
    atoms: Dict[int, Dict[str, object]],
    recovery: Optional[verify.IdentityRecovery] = None,
) -> Tuple[List[int], Dict[str, object]]:
    """Select heavy atoms by AutoDock atom type, not by atom name.

    In a real Meeko ligand PDBQT the atom names are bare element letters, so a
    name-based test is wrong in both directions: it misses heavy atoms whose
    name it cannot interpret, and it can discard a heavy atom whose name
    begins with H. The AutoDock atom type is the column Vina itself reads, and
    `HD`, `HS` and `H` are its hydrogen types.

    A blank atom type is fatal rather than assumed heavy: an atom whose type
    is unknown cannot be classified, and a wrongly included hydrogen changes
    every count it enters. Where identities are available, the type's
    classification is cross-checked against the recovered element, and a
    disagreement is fatal too.
    """
    heavy: List[int] = []
    hydrogen: List[int] = []
    for serial in sorted(atoms):
        atom_type = str(atoms[serial]["atom_type"]).strip()
        if not atom_type:
            raise ScoringError(
                f"atom serial {serial} has a blank AutoDock atom type, so it "
                "cannot be classified as heavy or hydrogen. A PDBQT without "
                "atom types is not a file Vina can score."
            )
        is_hydrogen = atom_type.upper() in HYDROGEN_ATOM_TYPES
        if recovery is not None:
            identity = recovery.by_serial.get(serial)
            if identity is not None and identity.element:
                element_is_hydrogen = identity.element.upper() in ("H", "D")
                if element_is_hydrogen != is_hydrogen:
                    raise ScoringError(
                        f"atom serial {serial} has AutoDock type {atom_type!r} "
                        f"but recovered element {identity.element!r}; the atom "
                        "type and the chemical identity disagree about "
                        "whether it is a hydrogen"
                    )
        (hydrogen if is_hydrogen else heavy).append(serial)

    report: Dict[str, object] = {
        "selected_by": (
            "AutoDock atom type, cross-checked against the recovered element "
            "where identities are available"
        ),
        "hydrogen_atom_types": sorted(HYDROGEN_ATOM_TYPES),
        "heavy_atoms": len(heavy),
        "hydrogen_atoms": len(hydrogen),
        "atom_type_census": verify.atom_type_census(atoms),
    }
    if recovery is not None:
        expected = sorted(
            identity.label
            for identity in recovery.by_serial.values()
            if not identity.is_hydrogen
        )
        evaluated = sorted(
            recovery.by_serial[serial].label
            for serial in heavy
            if serial in recovery.by_serial
        )
        # A heavy atom of the prepared molecule that never reached the PDBQT
        # has no serial, so it cannot appear among the identified atoms. It
        # has to be read from the recovery's own record of what went missing,
        # or "expected" would mean no more than "found", and the check would
        # be incapable of reporting a dropped atom.
        absent = list(recovery.heavy_atoms_absent_from_pdbqt)
        report.update(
            {
                "expected_heavy_atom_identities": len(expected) + len(absent),
                "evaluated_heavy_atom_identities": len(evaluated),
                "heavy_atoms_without_identity": [
                    serial for serial in heavy if serial not in recovery.by_serial
                ],
                "expected_but_not_evaluated": sorted(
                    set(expected) - set(evaluated)
                ) + absent,
                "constructed_c_terminal_atom": CONSTRUCTED_C_TERMINAL_ATOM,
                "constructed_c_terminal_atom_evaluated": any(
                    label.endswith(f":{CONSTRUCTED_C_TERMINAL_ATOM}")
                    for label in evaluated
                ),
            }
        )
    if not heavy:
        raise ScoringError(
            "no heavy atoms were selected: every atom is typed as hydrogen, "
            "so there is nothing to measure against the box"
        )
    return heavy, report


def pose_box_containment(
    atoms: Dict[int, Dict[str, object]],
    box_record: Dict,
    recovery: Optional[verify.IdentityRecovery] = None,
) -> Dict[str, object]:
    """Check every heavy atom lies inside the box Vina will build maps for.

    An atom outside the box is not merely untidy: Vina's grid maps are only
    defined inside it, so a score computed with atoms outside is not the score
    of this pose.

    Heavy atoms are selected by atom type, and where identities are available
    the selection is required to cover **every** heavy atom of the prepared
    molecule, the constructed C-terminal OXT included. A containment check
    that quietly omitted an atom would report a pose as contained on the
    strength of the atoms it happened to look at.
    """
    serials, selection = heavy_atom_serials(atoms, recovery)
    positions = np.asarray(
        [atoms[serial]["position"] for serial in serials], dtype=float
    )
    try:
        containment = peptide_geometry.box_containment(positions, box_record)
    except peptide_geometry.GeometryError as error:
        raise ScoringError(f"box containment could not be computed: {error}") from error
    missing = selection.get("expected_but_not_evaluated") or []
    if missing:
        raise ScoringError(
            "heavy atoms of the prepared molecule were not evaluated against "
            f"the box: {missing[:10]}. Every heavy atom must be checked, "
            "including the constructed C-terminal OXT."
        )
    unidentified = selection.get("heavy_atoms_without_identity") or []
    if unidentified:
        raise ScoringError(
            f"heavy atom serial(s) {unidentified[:10]} have no recovered "
            "chemical identity, so what was measured against the box cannot "
            "be said to be the prepared molecule's heavy atoms."
        )
    containment["heavy_atom_selection"] = selection
    containment["consequence"] = (
        "Every heavy atom lies inside the box used to build the maps."
        if containment["fully_contained"]
        else (
            "Vina's affinity maps are defined only inside the box. A score "
            "computed while atoms lie outside it is not a score of this pose."
        )
    )
    return containment


# ---------------------------------------------------------------------------
# C. Structural alignment: Kabsch superposition
# ---------------------------------------------------------------------------


@dataclass
class Superposition:
    """The rigid transform that best maps `mobile` onto `reference`.

    `rotation` and `translation` satisfy `rotation @ p + translation ~= q` for
    corresponding points `p` of the mobile set and `q` of the reference set.
    `rmsd` is the residual after that transform, which is the part of the
    difference no rigid motion can account for.
    """

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    rotation_degrees: float
    rotation_axis: List[float]
    centroid_shift_angstrom: float
    atoms: int
    reflection_corrected: bool

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points @ self.rotation.T + self.translation

    def as_dict(self) -> Dict[str, object]:
        return {
            "atoms_fitted": self.atoms,
            "rotation_degrees": round(self.rotation_degrees, 4),
            "rotation_axis": [round(float(v), 4) for v in self.rotation_axis],
            "centroid_shift_angstrom": round(self.centroid_shift_angstrom, 4),
            "residual_rmsd_angstrom": round(self.rmsd, 4),
            "reflection_corrected": self.reflection_corrected,
            "method": (
                "Kabsch least-squares superposition by singular value "
                "decomposition, with the determinant corrected so the result "
                "is a proper rotation and never a reflection"
            ),
        }


def kabsch_superposition(
    mobile: np.ndarray, reference: np.ndarray
) -> Superposition:
    """Least-squares fit of one point set onto another, by Kabsch's method.

    Both arrays are (N, 3) in Angstroms and in correspondence row by row; the
    caller establishes that correspondence, because a superposition of
    mispaired atoms is meaningless however small its residual.

    The determinant of the candidate rotation is checked and corrected, so an
    improper rotation (a reflection, which would fit a mirror-image structure)
    can never be returned. The reported rotation angle is that of the fitted
    rotation about its own axis, and `centroid_shift_angstrom` is the distance
    between the two sets' centroids, which is the translation component.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ScoringError(
            "superposition needs two (N, 3) arrays of the same shape, got "
            f"{mobile.shape} and {reference.shape}"
        )
    if mobile.shape[0] < MINIMUM_SUPERPOSITION_ATOMS:
        raise ScoringError(
            f"{mobile.shape[0]} atom(s) cannot define a rigid-body fit; at "
            f"least {MINIMUM_SUPERPOSITION_ATOMS} non-collinear atoms are "
            "required"
        )
    if not (np.all(np.isfinite(mobile)) and np.all(np.isfinite(reference))):
        raise ScoringError("superposition requires finite coordinates")

    mobile_centroid = mobile.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    centred_mobile = mobile - mobile_centroid
    centred_reference = reference - reference_centroid

    covariance = centred_mobile.T @ centred_reference
    left, _, right = np.linalg.svd(covariance)
    determinant = float(np.linalg.det(right.T @ left.T))
    reflection_corrected = determinant < 0.0
    correction = np.diag([1.0, 1.0, -1.0 if reflection_corrected else 1.0])
    rotation = right.T @ correction @ left.T
    translation = reference_centroid - rotation @ mobile_centroid

    residuals = (centred_mobile @ rotation.T) - centred_reference
    rmsd = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))

    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(skew))
    axis = (skew / norm) if norm > 1e-12 else np.zeros(3, dtype=float)

    return Superposition(
        rotation=rotation,
        translation=translation,
        rmsd=rmsd,
        rotation_degrees=angle,
        rotation_axis=[float(v) for v in axis],
        centroid_shift_angstrom=float(
            np.linalg.norm(reference_centroid - mobile_centroid)
        ),
        atoms=int(mobile.shape[0]),
        reflection_corrected=reflection_corrected,
    )


# ---------------------------------------------------------------------------
# D. Displacement: absolute, rigid-body, internal and side-chain
# ---------------------------------------------------------------------------


def _statistics(values: Sequence[float]) -> Dict[str, object]:
    """RMSD, mean and maximum of a list of distances, or nulls when empty."""
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


def displacement_report(
    before_atoms: Dict[int, Dict[str, object]],
    after_atoms: Dict[int, Dict[str, object]],
    correspondence: Correspondence,
    recovery: verify.IdentityRecovery,
) -> Dict[str, object]:
    """Measure how the pose moved, separating four different things.

    Local optimisation of a backbone-rigid peptide still has six rigid-body
    degrees of freedom, so the peptide is *expected* to translate and rotate
    as a unit. Reported as one displacement, that legitimate motion is
    indistinguishable from the predicted backbone being reshaped. The four
    quantities here are therefore kept apart:

    **1. Receptor-frame displacement.** Per-atom distance from the predicted
    position to the relaxed position, in the frame the receptor and the box
    live in, split by backbone and side chain. This is what moved, in the
    pocket.

    **2. Rigid-body movement.** The translation and rotation that best carry
    the relaxed backbone onto the predicted backbone. A peptide that moved as
    a unit shows up here and nowhere else.

    **3. Internal backbone deformation.** The residual backbone RMSD after
    that superposition. This, and only this, says whether the predicted
    backbone was altered. A pure translation or a pure rotation leaves it at
    zero.

    **4. Side-chain displacement.** Reported twice, with the convention
    stated: in the receptor frame, and again after applying the
    backbone-derived transform, which measures side-chain motion in the
    peptide's own frame and so excludes whole-body motion.

    Hydrogens are excluded throughout: Vina's scoring function ignores them
    and their positions were idealised rather than predicted.
    """
    if not correspondence.established:
        return {
            "measured": False,
            "reason": (
                "atom correspondence between the starting ligand and the "
                "written pose was not established, so no displacement can be "
                "measured: " + "; ".join(correspondence.reasons)
            ),
            "correspondence": correspondence.as_dict(),
        }

    heavy_serials, selection = heavy_atom_serials(before_atoms, recovery)
    backbone_before: List[List[float]] = []
    backbone_after: List[List[float]] = []
    backbone_labels: List[str] = []
    side_before: List[List[float]] = []
    side_after: List[List[float]] = []
    side_labels: List[str] = []
    unidentified: List[int] = []

    for serial in heavy_serials:
        identity = recovery.by_serial.get(serial)
        if identity is None:
            unidentified.append(serial)
            continue
        before = list(before_atoms[serial]["position"])
        after = list(after_atoms[correspondence.pairs[serial]]["position"])
        if identity.atom_name.upper() in BACKBONE_ATOM_NAMES:
            backbone_before.append(before)
            backbone_after.append(after)
            backbone_labels.append(identity.label)
        else:
            side_before.append(before)
            side_after.append(after)
            side_labels.append(identity.label)

    def per_atom(
        labels: Sequence[str], first: Sequence[Sequence[float]],
        second: Sequence[Sequence[float]],
    ) -> List[float]:
        return [
            float(
                np.linalg.norm(
                    np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
                )
            )
            for a, b in zip(first, second)
        ]

    backbone_distances = per_atom(backbone_labels, backbone_before, backbone_after)
    side_distances = per_atom(side_labels, side_before, side_after)

    movements = [
        {
            "atom": label,
            "moved_angstrom": round(distance, 4),
            "backbone": True,
        }
        for label, distance in zip(backbone_labels, backbone_distances)
    ] + [
        {
            "atom": label,
            "moved_angstrom": round(distance, 4),
            "backbone": False,
        }
        for label, distance in zip(side_labels, side_distances)
    ]
    movements.sort(key=lambda entry: entry["moved_angstrom"], reverse=True)

    report: Dict[str, object] = {
        "measured": True,
        "correspondence": correspondence.as_dict(),
        "heavy_atom_selection": selection,
        "atoms_without_identity": unidentified,
        "receptor_frame": {
            "frame": (
                "the receptor's coordinate frame, the frame the box record "
                "and every clash distance are in; no superposition applied"
            ),
            "all_heavy_atoms": _statistics(backbone_distances + side_distances),
            "backbone": _statistics(backbone_distances),
            "side_chain": _statistics(side_distances),
            "largest_movements": movements[:15],
            "backbone_atom_names": sorted(BACKBONE_ATOM_NAMES),
        },
        "hydrogens": (
            "excluded: Vina ignores them and their positions were idealised, "
            "not predicted"
        ),
    }

    if len(backbone_before) < MINIMUM_SUPERPOSITION_ATOMS:
        report["rigid_body"] = {
            "measured": False,
            "reason": (
                f"{len(backbone_before)} backbone heavy atom(s) with "
                "recovered identities; a rigid-body fit needs at least "
                f"{MINIMUM_SUPERPOSITION_ATOMS}"
            ),
        }
        report["internal_backbone_deformation"] = {
            "measured": False,
            "reason": "no backbone superposition was possible",
        }
        report["side_chain"] = {
            "receptor_frame": _statistics(side_distances),
            "after_backbone_superposition": {
                "measured": False,
                "reason": "no backbone superposition was possible",
            },
        }
        return report

    fit = kabsch_superposition(
        np.asarray(backbone_after, dtype=float),
        np.asarray(backbone_before, dtype=float),
    )
    report["rigid_body"] = dict(
        fit.as_dict(),
        measured=True,
        fitted_on=(
            "backbone heavy atoms with recovered identities "
            f"({', '.join(sorted(BACKBONE_ATOM_NAMES))})"
        ),
        direction=(
            "the transform carrying the relaxed backbone onto the predicted "
            "backbone; its magnitude is the whole-peptide motion"
        ),
        interpretation=(
            "A translation and rotation here with a near-zero internal "
            "deformation below means the peptide moved as a unit and its "
            "predicted backbone conformation was preserved. That is a "
            "legitimate result of local optimisation with a backbone-rigid "
            "torsion tree, and must not be read as the backbone having been "
            "reshaped."
        ),
    )

    aligned_backbone = fit.apply(np.asarray(backbone_after, dtype=float))
    backbone_residuals = [
        float(np.linalg.norm(aligned - np.asarray(before, dtype=float)))
        for aligned, before in zip(aligned_backbone, backbone_before)
    ]
    worst_index = (
        int(np.argmax(backbone_residuals)) if backbone_residuals else None
    )
    report["internal_backbone_deformation"] = dict(
        _statistics(backbone_residuals),
        measured=True,
        rmsd_after_superposition_angstrom=round(fit.rmsd, 4),
        largest_residual_atom=(
            backbone_labels[worst_index] if worst_index is not None else None
        ),
        convention=(
            "residual backbone RMSD after Kabsch superposition of the relaxed "
            "backbone onto the predicted backbone, over backbone heavy atoms "
            "in validated correspondence"
        ),
        interpretation=(
            "This is the only quantity that says whether the predicted "
            "backbone conformation changed. It is invariant under translation "
            "and rotation of the whole peptide by construction."
        ),
    )

    if side_before:
        aligned_side = fit.apply(np.asarray(side_after, dtype=float))
        side_internal = [
            float(np.linalg.norm(aligned - np.asarray(before, dtype=float)))
            for aligned, before in zip(aligned_side, side_before)
        ]
        internal_side = dict(
            _statistics(side_internal),
            measured=True,
            convention=(
                "side-chain displacement after applying the transform fitted "
                "on the backbone, so whole-peptide translation and rotation "
                "are removed and what remains is side-chain motion relative "
                "to the peptide's own backbone frame"
            ),
        )
    else:
        internal_side = {
            "measured": False,
            "reason": "no side-chain heavy atoms with recovered identities",
        }
    report["side_chain"] = {
        "receptor_frame": dict(
            _statistics(side_distances),
            convention="no superposition; displacement in the receptor frame",
        ),
        "after_backbone_superposition": internal_side,
        "note": (
            "The two are different measurements of the same motion and will "
            "differ whenever the peptide moved as a unit. Neither is more "
            "correct; each answers a different question."
        ),
    }
    return report


# ---------------------------------------------------------------------------
# E. Steric clashes, before and after relaxation
# ---------------------------------------------------------------------------

# AutoDock atom types carry the element plus a role suffix: `A` is aromatic
# carbon, and a trailing `A` or `S` marks a hydrogen-bond acceptor or donor.
# The element is needed for the van der Waals radius, so the mapping is
# written out rather than inferred from the first letter, which would make
# `A` nitrogen-free carbon into element "A".
ELEMENT_FOR_ATOM_TYPE: Dict[str, str] = {
    "C": "C", "A": "C",
    "N": "N", "NA": "N", "NS": "N",
    "O": "O", "OA": "O", "OS": "O",
    "S": "S", "SA": "S",
    "P": "P", "F": "F", "CL": "CL", "BR": "BR", "I": "I",
    "H": "H", "HD": "H", "HS": "H",
}


def element_for_atom(
    atom: Dict[str, object], identity: Optional[verify.AtomIdentity]
) -> str:
    """Return an atom's element, preferring the recovered chemical identity.

    The identity comes from the stage-3 ligand record, which knows the element
    from the molecule itself. The AutoDock type is the fallback, through an
    explicit table: a van der Waals radius chosen from a guessed element would
    change every overlap it enters, so an unrecognised type is fatal.
    """
    if identity is not None and identity.element:
        return identity.element.upper()
    atom_type = str(atom["atom_type"]).strip().upper()
    if atom_type not in ELEMENT_FOR_ATOM_TYPE:
        raise ScoringError(
            f"AutoDock atom type {atom_type!r} is not in this module's "
            "element table, so its element and van der Waals radius are "
            "unknown. Add it explicitly rather than guessing."
        )
    return ELEMENT_FOR_ATOM_TYPE[atom_type]


def peptide_atom_sites(
    atoms: Dict[int, Dict[str, object]],
    serials: Sequence[int],
    recovery: verify.IdentityRecovery,
    frame: str,
    positions: Optional[Dict[int, Sequence[float]]] = None,
) -> List[peptide_contacts.AtomSite]:
    """Build identity-carrying contact sites for the peptide's heavy atoms.

    The identities are the recovered ones, not the PDBQT's own: a report that
    named every clashing atom `UNL1:C` would be useless, and naming them from
    the generic file is exactly the mistake this module exists to avoid.

    `positions`, where given, replaces the coordinates while keeping the
    identities, which is how the relaxed pose's atoms are named.
    """
    sites: List[peptide_contacts.AtomSite] = []
    for serial in serials:
        identity = recovery.by_serial.get(serial)
        if identity is None:
            raise ScoringError(
                f"heavy atom serial {serial} has no recovered identity, so a "
                "clash involving it could not be named"
            )
        coordinates = (
            positions[serial] if positions is not None
            else atoms[serial]["position"]
        )
        sites.append(
            peptide_contacts.AtomSite(
                author_chain=identity.author_chain or "pep",
                residue_name=identity.residue_name or "UNK",
                residue_seqid=int(identity.residue_seqid),
                insertion_code="",
                atom_name=identity.atom_name,
                element=element_for_atom(atoms[serial], identity),
                position=np.asarray(coordinates, dtype=float),
                is_amino_acid=True,
                frame=frame,
            )
        )
    return sites


def receptor_site_sets(
    structure_path: Path,
    component: str,
    author_chains: Optional[Sequence[str]] = None,
) -> Tuple[
    List[peptide_contacts.AtomSite],
    List[peptide_contacts.AtomSite],
    Dict[str, object],
]:
    """Split a prepared receptor into protein and one named cofactor.

    The split is by residue classification, never by chain letter: in the
    prepared ANP receptor the cofactor shares author chain A with the protein,
    so a chain-letter selection would pool them and report peptide-to-ANP
    contacts as peptide-to-protein ones. The census is returned so that the
    partition is evidence in the record rather than an assumption.
    """
    try:
        structure = peptide_geometry.read_structure(structure_path)
    except peptide_geometry.GeometryError as error:
        raise ScoringError(
            f"cannot read receptor structure {structure_path}: {error}"
        ) from error
    sites = peptide_contacts.collect_model_sites(
        structure[0], author_chains=author_chains
    )
    protein, other, census = peptide_contacts.partition_sites(sites)
    component_sites = peptide_contacts.sites_for_component(other, component)
    report = dict(
        census.as_dict(),
        structure=str(structure_path),
        structure_sha256=file_sha256(structure_path),
        component=component,
        component_atoms=len(component_sites),
        protein_atoms=len(protein),
        author_chains_selected=(
            None if author_chains is None else list(author_chains)
        ),
    )
    if not component_sites:
        report["component_warning"] = (
            f"no {component} atoms were found, so the cofactor comparison is "
            "undefined rather than clash-free"
        )
    return protein, component_sites, report


def clash_report_for_pose(
    peptide_sites: Sequence[peptide_contacts.AtomSite],
    protein_sites: Sequence[peptide_contacts.AtomSite],
    component_sites: Sequence[peptide_contacts.AtomSite],
    component: str,
    pose_description: str,
    contact_cutoff: float = peptide_contacts.DEFAULT_CONTACT_CUTOFF,
    clash_overlap_tolerance: float = (
        peptide_contacts.DEFAULT_CLASH_OVERLAP_TOLERANCE
    ),
) -> Dict[str, object]:
    """Measure one pose's clashes against protein and cofactor separately.

    The two targets are never pooled, and the overlap definition is the
    project's existing one: a pair clashes when its Bondi van der Waals
    spheres interpenetrate by more than `clash_overlap_tolerance`, with
    contacts counted below `contact_cutoff` strictly.

    Clashes involving the constructed C-terminal OXT are counted separately,
    because the project's earlier figure of 24 clashes was measured on the
    BoltzGen pose before that atom existed. Comparing a count that includes it
    against one that does not would attribute a construction decision to the
    relaxation.
    """
    protein_report = peptide_contacts.contact_report(
        peptide_sites, protein_sites, "MurC protein",
        contact_cutoff, clash_overlap_tolerance,
    )
    component_report = peptide_contacts.contact_report(
        peptide_sites, component_sites, f"{component} cofactor",
        contact_cutoff, clash_overlap_tolerance,
    )
    summary = peptide_contacts.clash_summary([protein_report, component_report])

    oxt_clashes = 0
    for report in (protein_report, component_report):
        for pair in report.get("clashing_pairs") or []:
            if pair["first"]["atom_name"].upper() == CONSTRUCTED_C_TERMINAL_ATOM:
                oxt_clashes += 1
    total = summary.get("total_clashes")
    return {
        "pose": pose_description,
        "protein": protein_report,
        "cofactor": component_report,
        "summary": summary,
        "constructed_oxt_accounting": {
            "atom": CONSTRUCTED_C_TERMINAL_ATOM,
            "clashes_involving_it": oxt_clashes,
            "total_clashes": total,
            "total_excluding_it": (
                None if total is None else int(total) - oxt_clashes
            ),
            "why": (
                "OXT is constructed at stage 3, not predicted by BoltzGen. "
                "The project's earlier count of 24 clashes (16 protein, 8 "
                "cofactor) was measured on the 55-heavy-atom refolded pose, "
                "before OXT existed, so only the excluding-it count is "
                "comparable with it. That earlier figure was also measured "
                "against a receptor on the Mac, not necessarily this one."
            ),
        },
        "thresholds": {
            "contact_cutoff_angstrom": contact_cutoff,
            "contact_inequality": "strictly less than",
            "clash_overlap_tolerance_angstrom": clash_overlap_tolerance,
            "clash_inequality": "overlap strictly greater than the tolerance",
            "overlap_definition": (
                "r_i + r_j - d, with Bondi van der Waals radii per element, "
                "as defined in peptide_contacts"
            ),
            "radii_source": "Bondi 1964, per element, hydrogens excluded",
        },
    }


def clash_comparison(
    initial: Dict[str, object], optimised: Optional[Dict[str, object]]
) -> Dict[str, object]:
    """Compare two clash reports measured with identical definitions.

    Only the difference between two measurements made the same way is
    interpretable, so the thresholds are repeated here and a mismatch is
    refused rather than differenced.
    """
    comparison: Dict[str, object] = {
        "initial": initial,
        "after_local_optimisation": optimised,
        "definitions_identical": (
            None if optimised is None
            else initial["thresholds"] == optimised["thresholds"]
        ),
    }
    if optimised is None:
        comparison["change"] = None
        comparison["note"] = (
            "No local optimisation was requested, so there is no second pose "
            "to compare. The initial clash set stands as the measurement of "
            "the predicted pose."
        )
        return comparison
    if not comparison["definitions_identical"]:
        raise ScoringError(
            "the two clash reports were measured with different thresholds, "
            "so their difference is not interpretable"
        )

    def counts(report: Dict[str, object]) -> Dict[str, object]:
        return {
            "protein": report["protein"].get("clash_count"),
            "cofactor": report["cofactor"].get("clash_count"),
            "total": report["summary"].get("total_clashes"),
            "backbone": report["summary"].get("backbone_clashes"),
            "side_chain": report["summary"].get("side_chain_clashes"),
            "excluding_constructed_oxt": report["constructed_oxt_accounting"][
                "total_excluding_it"
            ],
        }

    before, after = counts(initial), counts(optimised)
    change = {
        key: (
            None if before[key] is None or after[key] is None
            else int(after[key]) - int(before[key])
        )
        for key in before
    }
    comparison["counts_initial"] = before
    comparison["counts_optimised"] = after
    comparison["change"] = change
    comparison["interpretation"] = (
        "A reduction in clashes means Vina's local optimisation found a "
        "placement with less van der Waals overlap under this rigid "
        "receptor. It does not establish that the relaxed pose is correct, "
        "that the design binds, or that any remaining overlap is tolerable. "
        "A backbone overlap cannot be relieved by side-chain motion, so a "
        "backbone clash that persists is the expected result and not a "
        "failure of the relaxation."
    )
    return comparison


def receptor_model_consistency(
    receptor_pdbqt: Path, receptor_structure: Path
) -> Dict[str, object]:
    """Check the scored receptor and the clash-measured receptor agree.

    The receptor Vina scores against is a PDBQT; the receptor the clash
    analysis measures against is the prepared structure it was built from.
    They must be the same model, or the clash geometry belongs to a different
    receptor from the score. Atoms are paired on full identity, which the
    receptor files do carry, and only the paired atoms' maximum deviation is
    a test. Unpaired counts are expected and reported, because receptor
    preparation adds polar hydrogens to the PDBQT.
    """
    try:
        pdbqt_atoms = receptor_provenance.parse_atom_records(receptor_pdbqt)
        structure_atoms = receptor_provenance.parse_atom_records(
            receptor_structure
        )
    except receptor_provenance.ReceptorError as error:
        return {
            "checked": False,
            "reason": f"a receptor file could not be parsed: {error}",
        }
    report = receptor_provenance.compare_atom_sets(
        structure_atoms, pdbqt_atoms
    ).as_dict()
    deviation = report.get("max_deviation_angstrom")
    # An atom in the PDBQT and not in the structure is only expected when it
    # is a hydrogen preparation added. Classifying them is what makes the
    # permissible case distinguishable from a receptor holding heavy atoms the
    # clash report cannot see, which the unpaired count alone cannot be.
    extra_non_hydrogen = sorted(
        str(key)
        for key, atom in pdbqt_atoms.items()
        if key not in structure_atoms
        and not receptor_provenance.is_hydrogen(atom)
    )
    extra_hydrogens = sum(
        1
        for key, atom in pdbqt_atoms.items()
        if key not in structure_atoms and receptor_provenance.is_hydrogen(atom)
    )
    return {
        "checked": True,
        "paired_atoms": report.get("n_paired"),
        "only_in_structure": report.get("n_missing"),
        "missing_from_pdbqt_examples": (report.get("missing_atoms") or [])[:10],
        "only_in_pdbqt": report.get("n_unexpected"),
        "only_in_pdbqt_hydrogens": extra_hydrogens,
        "only_in_pdbqt_non_hydrogen": extra_non_hydrogen,
        "moved_atom_count": len(report.get("moved_atoms") or []),
        "max_deviation_angstrom": deviation,
        "max_deviation_atom": report.get("max_deviation_atom"),
        "coordinates_consistent": (
            None if deviation is None
            else bool(deviation <= RIGID_DISTANCE_TOLERANCE_ANGSTROM)
        ),
        "tolerance_angstrom": RIGID_DISTANCE_TOLERANCE_ANGSTROM,
        "note": (
            "Unpaired *hydrogens* in the PDBQT are expected: Meeko adds polar "
            "hydrogens when preparing a receptor. An unpaired heavy atom on "
            "either side is not, and neither is a coordinate disagreement "
            "among the paired atoms; both mean the clash distances and the "
            "score belong to different receptor models. "
            "receptor_clash_prerequisites() turns these into blockers."
        ),
    }


def cofactor_provenance(
    component: str,
    partition: Dict[str, object],
    expected_component_atoms: Optional[int] = None,
    manifest_path: Optional[Path] = None,
    receptor_pdbqt: Optional[Path] = None,
    receptor_structure: Optional[Path] = None,
) -> Dict[str, object]:
    """Establish that the cofactor being measured is the validated one.

    Counting the cofactor's atoms and reporting the number is not a check: a
    truncated cofactor that is truncated the same way in both receptor files
    is self-consistent, passes every comparison between them, and produces a
    clash count against a cofactor that is not the one the project validated.
    Nothing downstream can detect it. The expectation has to come from outside
    the two files.

    Two ways to supply it, and at least one is required:

    1. **`expected_component_atoms`** -- the number stated by the researcher.
       For the ANP-containing MurC model prepared in this project that is 31
       heavy atoms.
    2. **The receptor manifest** `prepare_receptor.py` wrote. It records the
       SHA-256 of both of its outputs and names the components it retained, so
       a manifest whose recorded digests match the exact two files in use
       establishes that these are the validated receptor bytes. A truncated
       cofactor changes those bytes and so cannot match. This is provenance of
       identity rather than a count, which is the stronger of the two, but it
       requires the manifest to travel with the receptor.

    Returns the verdict with its basis, and the reasons when there is none.
    """
    observed = int(partition.get("component_atoms") or 0)
    verdict: Dict[str, object] = {
        "established": False,
        "basis": None,
        "component": component,
        "observed_atoms": observed,
        "expected_atoms": expected_component_atoms,
        "reasons": [],
        "why_required": (
            "a cofactor truncated identically in both receptor files is "
            "self-consistent and would pass every check that compares them "
            "with each other, so the expectation must come from outside them"
        ),
    }
    reasons: List[str] = verdict["reasons"]  # type: ignore[assignment]

    if expected_component_atoms is not None:
        if observed == expected_component_atoms:
            verdict["established"] = True
            verdict["basis"] = (
                f"{observed} {component} atoms, matching the "
                f"{expected_component_atoms} stated with "
                "--expected-component-atoms"
            )
            return verdict
        reasons.append(
            f"the receptor holds {observed} {component} atoms, not the "
            f"{expected_component_atoms} stated with "
            "--expected-component-atoms, so the cofactor is not the model "
            "this comparison was set up for"
        )
        return verdict

    if manifest_path is None:
        reasons.append(
            f"neither --expected-component-atoms nor --receptor-manifest was "
            f"given, so it is not established that the {observed} "
            f"{component} atoms in this receptor are the validated "
            f"{component}. State the expected count (31 heavy atoms for the "
            "ANP-containing MurC model prepared in this project) or supply "
            "the manifest prepare_receptor.py wrote"
        )
        return verdict

    try:
        manifest = receptor_provenance.read_manifest(manifest_path)
    except receptor_provenance.ReceptorError as error:
        reasons.append(f"the receptor manifest is not usable: {error}")
        return verdict

    outputs = manifest.get("outputs") or {}
    retained = manifest.get("retained_components") or {}
    digests: Dict[str, object] = {}
    for label, path, field in (
        ("receptor_pdbqt", receptor_pdbqt, "receptor_pdbqt_sha256"),
        ("receptor_structure", receptor_structure, "prepared_pdb_sha256"),
    ):
        recorded = outputs.get(field)
        if path is None:
            reasons.append(f"no {label} was supplied to compare against the manifest")
            continue
        if not recorded:
            reasons.append(f"the manifest records no {field}")
            continue
        actual = file_sha256(path)
        digests[label] = {"recorded": recorded, "actual": actual}
        if recorded != actual:
            reasons.append(
                f"the manifest's {field} does not match {path}, so this is "
                "not the receptor the manifest describes"
            )
    verdict["manifest"] = {
        "path": str(manifest_path),
        "schema": manifest.get("schema"),
        "receptor_condition": manifest.get("receptor_condition"),
        "retained_components": retained,
        "digests": digests,
    }
    if component not in (retained.get("in_receptor_pdbqt") or []):
        reasons.append(
            f"the manifest does not record {component} as retained in the "
            "receptor PDBQT"
        )
    if reasons:
        return verdict
    verdict["established"] = True
    verdict["basis"] = (
        f"the receptor manifest {Path(manifest_path).name} records "
        f"{component} as retained, and its SHA-256 of both receptor files "
        f"matches the two files in use, so these are the validated receptor "
        f"bytes and the {observed} {component} atoms in them are the "
        "validated cofactor"
    )
    return verdict


def receptor_clash_prerequisites(
    consistency: Optional[Dict[str, object]],
    partition: Optional[Dict[str, object]],
    component: str,
    expected_component_atoms: Optional[int] = None,
    cofactor: Optional[Dict[str, object]] = None,
) -> List[str]:
    """What must hold before a protein/cofactor clash comparison means anything.

    The score comes from the receptor PDBQT and the clash distances from the
    prepared structure, so the two have to be the same model. Where they are
    not, the resulting clash report is not a weaker measurement but a
    measurement of something else, and reporting it next to the score with a
    caveat invites exactly the reading it would deserve least: a clash count
    that appears to describe the scored complex. Each of these is therefore a
    blocker and not a limitation.

    An atom of the clash-measured structure that is absent from the scored
    PDBQT is fatal in a particular way: nothing in the clash report would show
    its absence, because the report measures the structure. An extra
    *hydrogen* in the PDBQT is expected and permitted, since receptor
    preparation adds polar hydrogens; an extra heavy atom is not.

    A cofactor with no atoms is undefined rather than clash-free, which is the
    difference between "ANP is not clashing" and "ANP was not there".

    Returns the blockers, empty when the comparison may proceed.
    """
    blockers: List[str] = []
    if consistency is None or partition is None:
        return [
            "no receptor consistency check or partition was made, so a clash "
            "comparison cannot be founded"
        ]
    if not consistency.get("checked"):
        blockers.append(
            "the scored receptor PDBQT and the clash-measured structure could "
            f"not be compared ({consistency.get('reason')}), so it is not "
            "established that they are the same receptor model"
        )
        return blockers
    if not consistency.get("paired_atoms"):
        blockers.append(
            "no atom of the clash-measured structure was found in the scored "
            "receptor PDBQT, so the two files are not the same receptor"
        )
    if consistency.get("coordinates_consistent") is not True:
        blockers.append(
            "the scored receptor PDBQT and the clash-measured structure "
            "disagree on the coordinates of their shared atoms (maximum "
            f"{consistency.get('max_deviation_angstrom')} A at "
            f"{consistency.get('max_deviation_atom')}, tolerance "
            f"{consistency.get('tolerance_angstrom')} A), so every clash "
            "distance would belong to a different receptor from the score"
        )
    missing = consistency.get("only_in_structure") or 0
    if missing:
        blockers.append(
            f"{missing} atom(s) of the clash-measured structure are absent "
            "from the scored receptor PDBQT, so Vina scored against a "
            "receptor missing atoms the clash report measures: "
            f"{consistency.get('missing_from_pdbqt_examples')}"
        )
    extra_heavy = consistency.get("only_in_pdbqt_non_hydrogen") or []
    if extra_heavy:
        blockers.append(
            f"{len(extra_heavy)} non-hydrogen atom(s) are in the scored "
            "receptor PDBQT but not in the clash-measured structure, so the "
            "scored receptor holds atoms the clash report cannot see: "
            f"{extra_heavy[:10]}"
        )
    if not partition.get("protein_atoms"):
        blockers.append(
            "the clash-measured structure holds no protein atoms, so a "
            "peptide-to-protein clash count would be undefined rather than "
            "zero"
        )
    component_atoms = partition.get("component_atoms") or 0
    if not component_atoms:
        blockers.append(
            f"the clash-measured structure holds no {component} atoms, so a "
            f"peptide-to-{component} clash count would be undefined rather "
            "than zero. Score against the receptor the cofactor was retained "
            f"in, or pass --clash-component with the component actually "
            "present"
        )
    # The cofactor's completeness cannot be established by comparing the two
    # receptor files with each other, so an external expectation is required
    # rather than optional. `cofactor` carries it when the caller has worked
    # it out; `expected_component_atoms` is the short form for callers that
    # only have the number.
    if cofactor is None:
        cofactor = cofactor_provenance(
            component, partition, expected_component_atoms
        )
    if not cofactor.get("established"):
        blockers.extend(str(reason) for reason in cofactor.get("reasons") or [])
    return blockers



# ---------------------------------------------------------------------------
# F. Provenance, the two scoring calls, and the command line
# ---------------------------------------------------------------------------


def plan_outputs(
    out_dir: Path,
    design_id: str,
    local_optimisation: bool,
    overwrite: bool = False,
) -> Dict[str, Optional[Path]]:
    """Decide which files this run will write, and refuse to clobber any.

    The fixed-pose path writes a record and nothing else. It does not need,
    name or reserve an optimised-pose file, so an existing optimised pose from
    an earlier run cannot block a fixed-pose score, and a fixed-pose run
    cannot be mistaken for one that produced a relaxed pose.
    """
    record_path = out_dir / f"{design_id}_score.json"
    optimised_path = (
        out_dir / f"{design_id}_optimised.pdbqt" if local_optimisation else None
    )
    planned = [path for path in (record_path, optimised_path) if path is not None]
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        raise ScoringError(
            f"Refusing to overwrite {', '.join(str(p) for p in existing)}. "
            "Choose a fresh output directory, or pass --overwrite "
            "deliberately."
        )
    return {"record": record_path, "optimised_pose": optimised_path}


def score_fixed_pose(engine) -> Dict[str, object]:
    """Score the supplied coordinates with `Vina.score()`. Nothing moves.

    This is the measurement of the BoltzGen prediction. It calls `score()` and
    only `score()`: no optimisation, and no search. `dock()` is not called
    anywhere in this module, so a search cannot be substituted for a fixed-
    pose score by any code path here.
    """
    import local_minimisation

    values = engine.score()
    record = local_minimisation.score_components(engine, values)
    record.update(
        {
            "vina_call": "Vina.score()",
            "pose": "as supplied; no atom was moved before scoring",
            "search_performed": False,
            "optimisation_performed": False,
        }
    )
    return record


def locally_optimise(
    engine,
    pose_path: Path,
    max_steps: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Relax the pose with `Vina.optimize()` and write it out.

    Separate from `score_fixed_pose` by construction: the two results are
    different quantities, and the record keeps them in different fields. This
    is still not a search; `optimize()` descends from the supplied pose and
    never explores the box.
    """
    import local_minimisation

    values = (
        engine.optimize() if max_steps is None
        else engine.optimize(max_steps=max_steps)
    )
    record = local_minimisation.score_components(engine, values)
    engine.write_pose(str(pose_path), overwrite=overwrite)
    record.update(
        {
            "vina_call": "Vina.optimize()",
            "pose_file": str(pose_path),
            "pose_file_sha256": file_sha256(pose_path),
            "max_steps": max_steps,
            "search_performed": False,
            "optimisation_performed": True,
            "what_moved": (
                "the rigid-body placement and every torsion in the ligand's "
                "torsion tree; which torsions those are is recorded in the "
                "stage-4 verification record"
            ),
        }
    )
    return record




def validated_workflow_requirements(
    verification_record: Optional[Path],
    predicted_structure: Optional[Path],
    exploratory_opt_in: bool = False,
) -> Optional[str]:
    """Refuse a validated run that was not given the evidence to be one.

    Returns the refusal message, or None when the run may proceed. The
    evidence is required rather than encouraged because a score produced
    without it is indistinguishable, in the record and in the number, from one
    produced with it. `exploratory_opt_in` waives the requirement; what it
    cannot waive is the label, which `validation_status` derives from the
    evidence itself.
    """
    missing: List[str] = []
    if verification_record is None:
        missing.append(
            "--verification-record, the stage-4 PEP-001 record for this exact "
            "PDBQT"
        )
    if predicted_structure is None:
        missing.append(
            "--predicted-structure, the stage-3 structure the PDBQT was built "
            "from"
        )
    if not missing or exploratory_opt_in:
        return None
    return (
        "the validated scoring workflow requires "
        + " and ".join(missing)
        + ". Without them the score would be attached to a PDBQT whose "
        "chemistry and predicted coordinates were not established here, and "
        "would be indistinguishable from one where they were. Pass "
        "--unvalidated-exploratory to score anyway: the record is then "
        "labelled an unvalidated exploratory score and the run reports a "
        "blocking issue."
    )


def validation_status(
    verification: Optional[Dict[str, object]],
    coordinate_recheck: Optional[Dict[str, object]],
    exploratory_opt_in: bool = False,
) -> Dict[str, object]:
    """State plainly whether this score rests on established evidence.

    Two things have to hold for a score of a *predicted pose* to mean what it
    appears to mean: the PDBQT's chemistry was verified by PEP-001 **and that
    verification belongs to this file**, and the file still holds the
    coordinates stage 3 predicted. Either one missing leaves a number that
    looks identical to a validated one, which is why the verdict is a field of
    the record rather than something a reader is left to infer from which
    arguments happened to be supplied.

    `validated` reflects the evidence and not the flags: passing
    `--unvalidated-exploratory` waives the *requirement* for that evidence, it
    does not change what was established. An exploratory run therefore cannot
    produce a record that reads as a validated one, and a record is never
    labelled validated on the strength of a stage-4 record that was not bound
    to the file scored.
    """
    bound = bool(
        verification
        and (verification.get("bound_to_scored_input") or {}).get("bound")
    )
    rechecked = bool(
        coordinate_recheck and coordinate_recheck.get("coordinates_preserved")
    )
    missing: List[str] = []
    if verification is None:
        missing.append(
            "no stage-4 PEP-001 verification record was supplied, so this "
            "PDBQT's chemistry was not established here"
        )
    elif not bound:
        missing.append(
            "the stage-4 verification record was not bound to the file "
            "scored, so it is not evidence about it"
        )
    if coordinate_recheck is None:
        missing.append(
            "no predicted structure was supplied, so it was not established "
            "here that this PDBQT still holds the predicted coordinates"
        )
    elif not rechecked:
        missing.append(
            "the PDBQT's coordinates do not match the predicted structure"
        )
    validated = not missing
    return {
        "validated": validated,
        "label": (
            "VALIDATED: stage-4 verification bound to this file, coordinates "
            "re-established against the prediction"
            if validated
            else "UNVALIDATED EXPLORATORY SCORE"
        ),
        "evidence": {
            "stage_4_verification_bound_to_this_file": bound,
            "coordinates_rechecked_against_prediction": rechecked,
        },
        "missing": missing,
        "exploratory_opt_in": exploratory_opt_in,
        "meaning": (
            "A validated label says the file scored is the file PEP-001 "
            "verified and still carries the predicted coordinates. It says "
            "nothing about the score itself, which remains an approximate "
            "computational number and not evidence of binding. An unvalidated "
            "label does NOT mean PEP-001 verification failed, and must never "
            "be read as meaning it passed: it means verification was not "
            "established for this file in this run."
        ),
    }


def pose_outside_box_blocker(
    containment: Dict[str, object], pose_description: str
) -> Optional[str]:
    """A score computed where the affinity maps are undefined is not a score.

    Vina's maps exist only inside the box it was given. An atom outside it is
    not scored approximately, it is scored against nothing, so the number that
    comes back is not a score of that pose at all. For a pose the *user*
    supplied there is an opt-out, because they may want the number recorded
    with the defect stated. For a pose Vina's own optimisation produced there
    is none: the run has no business reporting success for a relaxed score it
    computed outside its own maps.
    """
    if containment.get("fully_contained"):
        return None
    return (
        f"{pose_description} has heavy atoms outside the recorded box, so its "
        "score was computed where Vina's affinity maps are not defined and is "
        "not a score of that pose. Per-axis overshoot (A): "
        f"{containment.get('per_axis_overshoot_angstrom')}. Atoms outside: "
        f"{containment.get('atoms_outside')}"
    )



def record_optimised_validity(
    optimised: Dict[str, object], containment: Dict[str, object]
) -> Optional[str]:
    """Mark an optimised score invalid when it was computed outside the box.

    The marking and the blocker are produced together, by one function the run
    and the tests both call, so that a record can never carry `score_valid`
    true while the blocker was raised, or the blocker while the score reads as
    usable.
    """
    blocker = pose_outside_box_blocker(
        containment, "the pose Vina's local optimisation produced"
    )
    optimised["score_valid"] = blocker is None
    if blocker is not None:
        optimised["validity"] = (
            "This optimised score was computed with heavy atoms outside the "
            "box and is not a score of the relaxed pose. It is retained only "
            "so the failure is visible; it must not be quoted, compared with "
            "the initial score, or treated as an improvement."
        )
    return blocker


def exit_code_for(record: Dict[str, object]) -> int:
    """0 only when the run has nothing blocking to report.

    Shared by `main` and by the tests, so that "the run cannot report overall
    success" is a property under test rather than a line of `main` nobody
    exercises without Vina installed.
    """
    return 1 if record.get("blocking_issues") else 0


def build_record(
    design_id: str,
    ligand: Path,
    receptor: Path,
    box_record_path: Path,
    centre: Sequence[float],
    size: Sequence[float],
    seed: int,
    cpu: int,
    containment: Dict[str, object],
    torsions: Optional[int],
    initial: Dict[str, object],
    optimised: Optional[Dict[str, object]],
    ligand_record: Optional[Path] = None,
    torsdof_declaration: Optional[str] = None,
    identity_recovery: Optional[verify.IdentityRecovery] = None,
    verification: Optional[Dict[str, object]] = None,
    validation: Optional[Dict[str, object]] = None,
    coordinate_recheck: Optional[Dict[str, object]] = None,
    engine_description: str = "not recorded",
    clashes: Optional[Dict[str, object]] = None,
    receptor_structure: Optional[Path] = None,
    receptor_consistency: Optional[Dict[str, object]] = None,
    limitations: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Assemble the scoring record, keeping the two results separate.

    The record distinguishes the initial prediction from the relaxed pose in
    separate fields, and carries the provenance a reader needs to know what
    was scored: input checksums, the stage-4 verification status, the method
    by which atoms were put in correspondence, the TORSDOF the file declares,
    the scoring settings, and the limitations of the whole exercise.
    """
    inputs: Dict[str, object] = {
        "ligand_pdbqt": str(ligand),
        "ligand_pdbqt_sha256": file_sha256(ligand),
        "receptor_pdbqt": str(receptor),
        "receptor_pdbqt_sha256": file_sha256(receptor),
        "box_record": str(box_record_path),
        "box_record_sha256": file_sha256(box_record_path),
    }
    if ligand_record is not None:
        inputs["ligand_record"] = str(ligand_record)
        inputs["ligand_record_sha256"] = file_sha256(ligand_record)
    if receptor_structure is not None:
        inputs["receptor_structure"] = str(receptor_structure)
        inputs["receptor_structure_sha256"] = file_sha256(receptor_structure)

    record: Dict[str, object] = {
        "schema": SCHEMA,
        "design_id": design_id,
        # First, before any number, because it says whether the numbers below
        # rest on established evidence.
        "validation": validation
        or validation_status(verification, coordinate_recheck),
        "inputs": inputs,
        "coordinate_preservation_recheck": coordinate_recheck or {
            "checked": False,
            "reason": (
                "no --predicted-structure was supplied, so the ligand PDBQT's "
                "coordinates were not compared against the predicted "
                "structure here"
            ),
        },
        "stage_4_verification": verification or {
            "supplied": False,
            "consequence": (
                "no stage-4 verification record was supplied, so this score "
                "is attached to a PDBQT whose chemistry was not confirmed here"
            ),
        },
        "atom_identity": (
            {
                "method": identity_recovery.method,
                "established": identity_recovery.established,
                "atoms_identified": len(identity_recovery.by_serial),
                "heavy_atoms_absent_from_pdbqt": (
                    identity_recovery.heavy_atoms_absent_from_pdbqt
                ),
                "hydrogens_absent_from_pdbqt": len(
                    identity_recovery.hydrogens_absent_from_pdbqt
                ),
                "source": (
                    "Meeko REMARK INDEX MAP joined to atom_mapping in the "
                    "stage-3 ligand record. PEP-001 established this on real "
                    "Meeko output, not only in synthetic tests: identity "
                    "recovery succeeded for both spec_0574 variants and the "
                    "recovered identities reproduced the predicted "
                    "coordinates of all 56 heavy atoms to 0.000 A, which a "
                    "misread index map could not do"
                ),
            }
            if identity_recovery is not None
            else {"established": False, "reason": "no ligand record supplied"}
        ),
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
            "engine": engine_description,
            "note": (
                "score() evaluates the supplied pose; optimize() relaxes it "
                "locally. dock() is never called, so no conformational search "
                "over the box is performed."
            ),
        },
        "ligand_torsions": {
            "torsdof_from_pdbqt": torsions,
            "torsdof_declaration": torsdof_declaration,
            "source": "the TORSDOF line of the ligand PDBQT",
            "caveat": (
                "For a peptide, backbone phi and psi are ordinary rotatable "
                "bonds and become torsions unless rigidified at Meeko time. "
                "A high TORSDOF here means local optimisation can reshape the "
                "predicted backbone. Even at zero backbone torsions the six "
                "rigid-body degrees of freedom remain, so the peptide can "
                "still translate and rotate as a unit."
            ),
        },
        "box_containment_before_scoring": containment,
        "initial_pose": initial,
        "after_local_optimisation": optimised,
        "steric_clashes": clashes,
        "receptor_model_consistency": receptor_consistency,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "limitations": list(limitations or []) + [
            "A Vina score is an approximate computational score in kcal/mol. "
            "It is not a measured binding free energy, not a dissociation "
            "constant, and not evidence of binding or of enzyme inhibition.",
            "Vina's validation error is roughly 2-3 kcal/mol, so a smaller "
            "difference between two numbers here does not establish a "
            "preference.",
            "A peptide of this size is outside the chemical space Vina's "
            "scoring function was fitted on, and an improvement under local "
            "optimisation is a statement about that function, not about "
            "affinity.",
            "The receptor was rigid throughout, so a clash the real complex "
            "would relieve by moving a receptor side chain cannot be relieved "
            "here.",
            "The ANP cofactor's representation is recorded in this project as "
            "chemically unvalidated, so every distance to it is a geometric "
            "measurement against an unvalidated model.",
        ],
        "interpretation": (
            "Two separate quantities: the score of the pose BoltzGen "
            "predicted, and the score of that pose after Vina relaxed it "
            "locally. Neither is an affinity. The displacement report says "
            "how far the relaxed pose is from the prediction, and separates "
            "whole-peptide motion from a change in the predicted backbone."
        ),
    }
    if VINA_AVAILABLE:  # pragma: no cover - depends on the installation
        try:
            import vina

            record["software"]["vina"] = getattr(vina, "__version__", "unknown")
        except Exception:
            record["software"]["vina"] = "unknown"
    return record


def mapped_recovery(
    recovery: verify.IdentityRecovery, correspondence: Correspondence
) -> verify.IdentityRecovery:
    """Carry the starting ligand's identities over to the written pose.

    The pose has no index map of its own, so its atoms are named through the
    established correspondence and in no other way. The derived recovery is
    marked as such, so a record cannot read as though Meeko had identified
    the relaxed pose's atoms directly.
    """
    if not correspondence.established:
        raise ScoringError(
            "identities cannot be carried to the written pose without an "
            "established correspondence"
        )
    derived = verify.IdentityRecovery(
        method=(
            "derived: identities recovered for the starting ligand from "
            "Meeko's index map and the stage-3 ligand record, then carried "
            "to the written pose through the verified rigid-group "
            "correspondence"
        ),
        established=True,
    )
    for before_serial, after_serial in correspondence.pairs.items():
        identity = recovery.by_serial.get(before_serial)
        if identity is not None:
            derived.by_serial[after_serial] = identity
    derived.hydrogens_absent_from_pdbqt = list(
        recovery.hydrogens_absent_from_pdbqt
    )
    derived.heavy_atoms_absent_from_pdbqt = list(
        recovery.heavy_atoms_absent_from_pdbqt
    )
    return derived


def run(args: argparse.Namespace) -> Dict[str, object]:  # pragma: no cover
    """Score, and optionally relax, one peptide pose. Requires Vina."""
    require_vina()

    required = [args.ligand, args.ligand_record, args.receptor, args.box_record]
    if args.verification_record is not None:
        required.append(args.verification_record)
    if args.receptor_structure is not None:
        required.append(args.receptor_structure)
    if args.predicted_structure is not None:
        required.append(args.predicted_structure)
    if args.receptor_manifest is not None:
        required.append(args.receptor_manifest)
    for path in required:
        if not path.is_file():
            raise ScoringError(f"input not found: {path}")

    # The validated workflow requires both pieces of evidence. Scoring
    # without them is available, but only as an explicit request, and the
    # record then says what was not established rather than looking the same.
    refusal = validated_workflow_requirements(
        args.verification_record,
        args.predicted_structure,
        args.unvalidated_exploratory,
    )
    if refusal is not None:
        raise ScoringError(refusal)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = plan_outputs(
        args.out_dir, args.design_id, args.local_optimisation, args.overwrite
    )

    centre, size, box_record = load_box_record(args.box_record)
    # Bound to this PDBQT, this ligand record and this design, not merely read:
    # a verified record for another file is not evidence about this one.
    verification = (
        read_verification_record(
            args.verification_record,
            args.ligand,
            args.ligand_record,
            args.design_id,
        )
        if args.verification_record is not None
        else None
    )
    recovery = load_identity_recovery(args.ligand, args.ligand_record)
    before_atoms = pose_atoms(args.ligand)
    before_tree = parse_rigid_groups(args.ligand)

    coordinate_recheck = None
    if args.predicted_structure is not None:
        coordinate_recheck = recheck_coordinate_preservation(
            before_atoms, recovery, args.predicted_structure
        )
        if not coordinate_recheck.get("coordinates_preserved"):
            raise ScoringError(
                "the ligand PDBQT no longer matches the predicted structure "
                "it was prepared from: maximum heavy-atom deviation "
                f"{coordinate_recheck.get('max_deviation_angstrom')} A at "
                f"{coordinate_recheck.get('max_deviation_atom')}, "
                f"{len(coordinate_recheck.get('missing_atoms') or [])} atom(s) "
                "absent. Either the PDBQT changed since it was verified or "
                "this is not the structure it came from; scoring it as the "
                "predicted pose would attach the score to a pose nobody "
                "predicted."
            )

    containment = pose_box_containment(before_atoms, box_record, recovery)
    if not containment["fully_contained"] and not args.allow_outside_box:
        raise ScoringError(
            "The pose has heavy atoms outside the box, so Vina's maps do not "
            "cover it and the resulting score would not be a score of this "
            "pose. Per-axis overshoot (A): "
            f"{containment['per_axis_overshoot_angstrom']}. Enlarge the box "
            "and record the new one, or pass --allow-outside-box to record "
            "the score with this limitation stated."
        )

    limitations: List[str] = []
    blocking: List[str] = []
    receptor_consistency = None
    protein_sites: List[peptide_contacts.AtomSite] = []
    component_sites: List[peptide_contacts.AtomSite] = []
    receptor_partition = None
    prerequisites: List[str] = []
    clash_analysis_refused: Optional[str] = None
    if args.receptor_structure is not None:
        receptor_consistency = receptor_model_consistency(
            args.receptor, args.receptor_structure
        )
        protein_sites, component_sites, receptor_partition = receptor_site_sets(
            args.receptor_structure,
            args.clash_component,
            args.receptor_author_chain or None,
        )
        # A clash report measured against a receptor that is not the scored
        # receptor is not a weaker measurement, it is a measurement of
        # something else, so it is refused outright rather than recorded with
        # a caveat beside the score.
        cofactor = cofactor_provenance(
            args.clash_component,
            receptor_partition,
            args.expected_component_atoms,
            args.receptor_manifest,
            args.receptor,
            args.receptor_structure,
        )
        prerequisites = receptor_clash_prerequisites(
            receptor_consistency,
            receptor_partition,
            args.clash_component,
            args.expected_component_atoms,
            cofactor,
        )
        receptor_partition["cofactor_provenance"] = cofactor
        if prerequisites:
            blocking.extend(prerequisites)
            clash_analysis_refused = (
                "no clash comparison was made: it is not established that the "
                "receptor the score comes from and the receptor the clashes "
                "would be measured against are the same model, or that the "
                f"{args.clash_component} cofactor is present in it"
            )
            protein_sites, component_sites = [], []

    heavy_serials, _ = heavy_atom_serials(before_atoms, recovery)
    initial_clashes = None
    if args.receptor_structure is not None and clash_analysis_refused is None:
        initial_clashes = clash_report_for_pose(
            peptide_atom_sites(
                before_atoms, heavy_serials, recovery,
                "receptor frame, as predicted and prepared",
            ),
            protein_sites, component_sites, args.clash_component,
            "initial prediction, as supplied to Vina",
            args.contact_cutoff, args.clash_overlap_tolerance,
        )

    engine = Vina(sf_name="vina", seed=args.seed, cpu=args.cpu)
    engine.set_receptor(str(args.receptor))
    engine.set_ligand_from_file(str(args.ligand))
    engine.compute_vina_maps(center=list(centre), box_size=list(size))

    initial = score_fixed_pose(engine)

    optimised: Optional[Dict[str, object]] = None
    optimised_clashes = None
    if args.local_optimisation:
        optimised = locally_optimise(
            engine, outputs["optimised_pose"], args.max_steps, args.overwrite
        )
        after_atoms = pose_atoms(outputs["optimised_pose"])
        after_tree = parse_rigid_groups(outputs["optimised_pose"])
        correspondence = establish_correspondence(
            before_atoms, before_tree, after_atoms, after_tree,
            resolve_symmetric_atoms=args.resolve_symmetric_atoms,
        )
        optimised["displacement"] = displacement_report(
            before_atoms, after_atoms, correspondence, recovery
        )
        if not correspondence.established:
            blocking.append(
                "atom correspondence between the starting ligand and the "
                "written pose could not be established, so no displacement, "
                "containment or clash comparison was made for the relaxed "
                "pose: " + "; ".join(correspondence.reasons)
            )
        else:
            derived = mapped_recovery(recovery, correspondence)
            optimised["box_containment_after"] = pose_box_containment(
                after_atoms, box_record, derived
            )
            # Not a limitation beside an otherwise successful run: the
            # relaxed score is not a score, so the run must not report
            # success. --allow-outside-box does not reach this case, which is
            # Vina's own output rather than a pose the user supplied.
            outside_box = record_optimised_validity(
                optimised, optimised["box_containment_after"]
            )
            if outside_box:
                blocking.append(outside_box)
                limitations.append(outside_box)
            after_heavy, _ = heavy_atom_serials(after_atoms, derived)
            if initial_clashes is not None:
                optimised_clashes = clash_report_for_pose(
                    peptide_atom_sites(
                        after_atoms, after_heavy, derived,
                        "receptor frame, after local optimisation",
                    ),
                    protein_sites, component_sites, args.clash_component,
                    "after Vina local optimisation",
                    args.contact_cutoff, args.clash_overlap_tolerance,
                )
            moved = (
                optimised["displacement"]
                .get("receptor_frame", {})
                .get("all_heavy_atoms", {})
                .get("maximum_angstrom")
            )
            if moved == 0.0:
                optimised["warning"] = (
                    "No atom moved. Either the pose was already at a local "
                    "minimum, or the installed write_pose() wrote the input "
                    "pose rather than the optimised one. Check the "
                    "installation before interpreting this."
                )

    clashes = (
        clash_comparison(initial_clashes, optimised_clashes)
        if initial_clashes is not None
        else None
    )
    if clashes is not None and receptor_partition is not None:
        clashes["receptor_partition"] = receptor_partition
    if clash_analysis_refused is not None:
        clashes = {
            "measured": False,
            "reason": clash_analysis_refused,
            "blocking_issues": prerequisites,
            "receptor_partition": receptor_partition,
        }
    if args.receptor_structure is None:
        limitations.append(
            "no --receptor-structure was supplied, so no steric-clash "
            "comparison was made; the score alone does not say whether the "
            "pose is physically admissible"
        )
    if coordinate_recheck is None:
        limitations.append(
            "no --predicted-structure was supplied, so it was not "
            "re-established here that the ligand PDBQT still holds the "
            "coordinates stage 3 predicted; the stage-4 record's claim about "
            "that was made about the file as it stood then"
        )

    validation = validation_status(
        verification, coordinate_recheck, args.unvalidated_exploratory
    )
    if not validation["validated"]:
        blocking.append(
            "UNVALIDATED EXPLORATORY SCORE: "
            + "; ".join(str(item) for item in validation["missing"])
            + ". This record must not be read as a validated score, and does "
            "not mean PEP-001 verification passed for this file"
        )
        limitations.extend(
            f"not established in this run: {item}"
            for item in validation["missing"]
        )

    record = build_record(
        args.design_id,
        args.ligand,
        args.receptor,
        args.box_record,
        centre,
        size,
        args.seed,
        args.cpu,
        containment,
        pdbqt_torsion_count(args.ligand),
        initial,
        optimised,
        ligand_record=args.ligand_record,
        torsdof_declaration=verify.torsdof_status(args.ligand),
        identity_recovery=recovery,
        verification=verification,
        validation=validation,
        coordinate_recheck=coordinate_recheck,
        engine_description=f"{type(engine).__name__} from the vina package",
        clashes=clashes,
        receptor_structure=args.receptor_structure,
        receptor_consistency=receptor_consistency,
        limitations=limitations,
    )
    record["blocking_issues"] = blocking
    record["exit_code"] = exit_code_for(record)
    outputs["record"].write_text(json.dumps(record, indent=2) + "\n")
    record["record_path"] = str(outputs["record"])
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument(
        "--ligand-record",
        type=Path,
        required=True,
        help="the stage-3 ligand record carrying atom_mapping. Required: a "
        "Meeko ligand PDBQT has no usable atom identities of its own, so "
        "without this there is no backbone/side-chain split and no named "
        "clash",
    )
    parser.add_argument(
        "--verification-record",
        type=Path,
        default=None,
        help="the stage-4 verification record for this PDBQT; an unverified "
        "ligand is refused",
    )
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument(
        "--receptor-structure",
        type=Path,
        default=None,
        help="the prepared receptor structure the PDBQT was built from, for "
        "the steric-clash comparison. Omitted, no clash analysis is done",
    )
    parser.add_argument(
        "--predicted-structure",
        type=Path,
        default=None,
        help="the stage-3 structure the ligand PDBQT was built from. Supplied, "
        "the PDBQT's heavy-atom coordinates are compared against it here, "
        "which is the one thing the stage-4 record cannot be bound to: it "
        "carries no checksum of the file it verified",
    )
    parser.add_argument(
        "--expected-component-atoms",
        type=int,
        default=None,
        help="the number of cofactor atoms the receptor must hold -- 31 for "
        "the ANP-containing MurC model prepared in this project. Required for "
        "a clash comparison unless --receptor-manifest is given, because a "
        "cofactor truncated the same way in both receptor files would "
        "otherwise pass unnoticed",
    )
    parser.add_argument(
        "--receptor-manifest",
        type=Path,
        default=None,
        help="the manifest prepare_receptor.py wrote for this receptor. Its "
        "recorded SHA-256 of both receptor files establishes that these are "
        "the validated receptor bytes, which is the alternative to stating "
        "the cofactor atom count",
    )
    parser.add_argument(
        "--unvalidated-exploratory",
        action="store_true",
        help="score without a stage-4 verification record or a predicted "
        "structure. The record is then labelled an UNVALIDATED EXPLORATORY "
        "SCORE, the run reports a blocking issue and exits 1, and nothing in "
        "it may be read as PEP-001 verification having passed",
    )
    parser.add_argument(
        "--receptor-author-chain",
        action="append",
        default=None,
        help="restrict the receptor to these author chains; repeatable. "
        "Needed where the structure holds more than one crystallographic copy",
    )
    parser.add_argument(
        "--clash-component",
        default="ANP",
        help="the retained cofactor to report separately from the protein",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=peptide_contacts.DEFAULT_CONTACT_CUTOFF,
    )
    parser.add_argument(
        "--clash-overlap-tolerance",
        type=float,
        default=peptide_contacts.DEFAULT_CLASH_OVERLAP_TOLERANCE,
    )
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
        "--resolve-symmetric-atoms",
        action="store_true",
        help="pair atoms that no PDBQT invariant can distinguish by minimal "
        "displacement, recording the convention. Without it such a pose "
        "fails closed rather than being compared on a guess. It does not "
        "extend to whole rigid groups: two equivalent sibling branches are "
        "refused either way, because pairing branches would also fix every "
        "atom beneath them",
    )
    parser.add_argument(
        "--allow-outside-box",
        action="store_true",
        help="Record a score although the *supplied* pose has atoms outside "
        "the box, with that limitation stated in the record. It does not "
        "extend to a pose Vina's own optimisation produced outside the box, "
        "which is always a blocking issue",
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
    print(f"validation: {record['validation']['label']}", file=sys.stderr)
    for issue in record["blocking_issues"]:
        print(f"blocking: {issue}", file=sys.stderr)
    return exit_code_for(record)


if __name__ == "__main__":
    raise SystemExit(main())
