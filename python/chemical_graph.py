"""Chemically valid symmetry-aware atom correspondence and in-place RMSD.

Why this module exists
----------------------
Comparing a docked pose with a reference pose by atom name alone gives the
wrong answer whenever the molecule has interchangeable atoms. A phosphate
whose two terminal oxygens are swapped is the same physical pose, but a
name-ordered RMSD would report a deviation of roughly the O-O distance,
~2.5 A, which is larger than the 2.0 A criterion this project uses. The
comparison therefore has to minimise over the molecule's symmetry.

The RMSD here is computed **in place**: no superposition of any kind is
applied. That is deliberate and is the point of the measure. A best-fit
superposition would answer "is the docked conformer shaped like the
reference?", whereas the question being asked is "is the docked ligand in the
same place, in the receptor's own frame?". A pose translated 10 A out of the
pocket must score 10 A, not 0.

What counts as symmetry
-----------------------
Symmetry is taken from the molecular graph: atoms are interchangeable when a
graph automorphism maps one to the other, preserving element and bond order.
Two refinements make that chemically rather than merely topologically correct:

**Resonance normalisation, restricted to deprotonated groups.** In a
*deprotonated* phosphate or carboxylate the terminal oxygens are equivalent by
resonance, but a connection table records one as double-bonded and the other as
single-bonded, making them topologically distinguishable. Bonds from a central
atom to its terminal oxygens are therefore given one normalised order — but
only when none of those oxygens bears a hydrogen at the modelled protonation
state. A neutral carboxylic acid's two oxygens are *not* equivalent, and
normalising them would let a wrong pose pass. The caller must supply the
modelled protonation; without it nothing is normalised.

**Element and bond-order labels.** Automorphisms must preserve element, so a
carbon is never mapped onto a nitrogen, and bond order after normalisation, so
an amide C-N is never mapped onto a C=N.

Rotatable-bond-related symmetry (a freely spinning methyl, or amide-flip
tautomerism) is **not** treated as symmetry. A methyl's hydrogens are not
compared at all, since only heavy atoms are used, and genuine conformational
differences are what the RMSD is meant to measure.

**Stereochemistry is not verified here, and cannot be.**
--------------------------------------------------------
This is a limitation worth stating plainly rather than burying. The
automorphism search works on a graph, which has no 3D information, so:

- it **preserves** stereochemistry in the sense that the permitted mappings
  exchange only atoms that are constitutionally equivalent, and in a molecule
  whose stereocentres have distinguishable substituents no mapping touches
  them. For UMA, every one of the three normalised groups is a pair of
  equivalent terminal oxygens on an atom that is not a configurationally
  stable stereocentre once deprotonated;
- it **cannot detect** that a pose is the wrong stereoisomer. If a preparation
  step inverted a sugar centre, both structures would still share one graph,
  and the RMSD would simply report the geometric consequence of the inversion
  without identifying it as a stereochemical error. Where the inverted centre
  happens to be graph-symmetric with another, an automorphism could even map
  one onto the other and understate the deviation.

The measure therefore **assumes both structures are the same stereoisomer**.
Establishing that is the preparation stage's job, by comparing CIP codes
assigned from the 3D coordinates against the dictionary's declared
configurations. A low RMSD from this module is not evidence of correct
stereochemistry.

**This implementation is not independently validated.** Its tests are written
against the same understanding that produced the code, so they demonstrate
self-consistency, not correctness. An independent check against an established
toolkit is proposed in docs/uma_control.md and should be run before any result
from this module is relied on.

Failure rather than approximation
---------------------------------
Every failure mode raises:

- an atom present in one structure and not the other;
- an element disagreement for the same atom name;
- an automorphism search that exceeds its cap.

None of them falls back to comparing a subset. An RMSD computed over whatever
atoms happened to match is not a pose-recovery measure, and would be
systematically optimistic.

Limitations
-----------
- Hydrogens are excluded throughout. Their positions are added by preparation
  software and are not experimental.
- Bond orders come from whatever connection table the caller supplies; this
  module does not perceive chemistry from coordinates.
- The automorphism group is of the graph, not of the molecule in its
  environment: a symmetry that is real for the free ligand is still enumerated
  when the ligand sits in an asymmetric pocket. That is correct for an RMSD
  between two poses of the same molecule.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Bond order written for bonds whose order has been normalised by resonance.
RESONANCE_ORDER = "RES"

# Central elements whose terminal oxygens are treated as resonance-equivalent.
RESONANCE_CENTRES = ("P", "S", "C", "N")

# A cap on the automorphism search. A molecule needing more symmetry than this
# is not quietly truncated: the search raises instead.
DEFAULT_AUTOMORPHISM_CAP = 10_000


class CorrespondenceError(ValueError):
    """Raised when two structures cannot be compared atom for atom."""


@dataclass(frozen=True)
class MolecularGraph:
    """Heavy-atom connectivity of one chemical component.

    `names` is ordered and defines the index of every atom used elsewhere in
    this module. `elements` is parallel to it. `bonds` maps an unordered index
    pair to a bond-order string; the strings only have to be used consistently,
    so CCD values (`SING`, `DOUB`, `TRIP`, `AROM`) work as they stand.
    """
    names: tuple[str, ...]
    elements: tuple[str, ...]
    bonds: dict[frozenset[int], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.names) != len(self.elements):
            raise CorrespondenceError(
                f"{len(self.names)} atom names against {len(self.elements)} elements")
        if len(set(self.names)) != len(self.names):
            duplicated = sorted({n for n in self.names if self.names.count(n) > 1})
            raise CorrespondenceError(f"duplicated atom names: {duplicated}")
        for pair in self.bonds:
            if len(pair) != 2 or any(i >= len(self.names) or i < 0 for i in pair):
                raise CorrespondenceError(f"bond {sorted(pair)} is out of range")

    @property
    def index(self) -> dict[str, int]:
        return {name: position for position, name in enumerate(self.names)}

    @property
    def adjacency(self) -> dict[int, list[tuple[int, str]]]:
        table: dict[int, list[tuple[int, str]]] = {
            position: [] for position in range(len(self.names))}
        for pair, order in self.bonds.items():
            first, second = sorted(pair)
            table[first].append((second, order))
            table[second].append((first, order))
        return table

    def degree(self, position: int) -> int:
        return sum(1 for pair in self.bonds if position in pair)


# ---------------------------------------------------------------------------
# Resonance normalisation
# ---------------------------------------------------------------------------

def normalise_resonance(graph: MolecularGraph,
                        hydrogen_bearing: Iterable[str] | None = None,
                        centres: Sequence[str] = RESONANCE_CENTRES
                        ) -> tuple[MolecularGraph, list[str]]:
    """Give a centre's terminal oxygens one bond order, where justified.

    A group qualifies only when **all** of these hold:

    1. the central atom's element is in `centres`;
    2. it carries two or more terminal (degree-one) oxygens;
    3. **none of those oxygens bears a hydrogen in the modelled protonation
       state.**

    Condition 3 is what makes this chemically rather than merely topologically
    valid, and it is why `hydrogen_bearing` is not optional in practice. A
    deprotonated phosphate or carboxylate has genuinely equivalent terminal
    oxygens, delocalised over the group. A *neutral* carboxylic acid does not:
    one oxygen carries the proton and the other is the carbonyl, they have
    different bond lengths and different hydrogen-bonding roles, and swapping
    them describes a different structure. Normalising such a group would let a
    wrong pose pass.

    `hydrogen_bearing` names the heavy atoms that still carry a hydrogen at the
    modelled state, so the caller must have decided its protonation before the
    symmetry is computed. Passing None means "protonation unknown", and then
    **nothing is normalised**: the conservative choice, since an unjustified
    equivalence is the error that matters.

    Returns the rewritten graph and a record of every group normalised and
    every group skipped with the reason, for the manifest.
    """
    adjacency = graph.adjacency
    bonds = dict(graph.bonds)
    record: list[str] = []
    if hydrogen_bearing is None:
        return MolecularGraph(graph.names, graph.elements, bonds), [
            "no protonation state supplied, so no group was treated as "
            "resonance-equivalent; symmetry is restricted to the connection "
            "table as given"
        ]
    protonated = set(hydrogen_bearing)
    for position, element in enumerate(graph.elements):
        if element.upper() not in centres:
            continue
        terminal_oxygens = [
            neighbour for neighbour, _ in adjacency[position]
            if graph.elements[neighbour].upper() == "O"
            and graph.degree(neighbour) == 1
        ]
        if len(terminal_oxygens) < 2:
            continue
        names = sorted(graph.names[n] for n in terminal_oxygens)
        still_protonated = [name for name in names if name in protonated]
        if still_protonated:
            record.append(
                f"{graph.names[position]}: terminal oxygens {names} NOT "
                f"normalised, because {still_protonated} still bear a hydrogen "
                f"at the modelled state; they are chemically inequivalent"
            )
            continue
        orders = {graph.bonds[frozenset((position, neighbour))]
                  for neighbour in terminal_oxygens}
        for neighbour in terminal_oxygens:
            bonds[frozenset((position, neighbour))] = RESONANCE_ORDER
        record.append(
            f"{graph.names[position]}: terminal oxygens {names} normalised "
            f"from {sorted(orders)} to {RESONANCE_ORDER}; all are "
            f"deprotonated at the modelled state"
        )
    return MolecularGraph(graph.names, graph.elements, bonds), record


# ---------------------------------------------------------------------------
# Automorphisms
# ---------------------------------------------------------------------------

def refine_colours(graph: MolecularGraph) -> list[int]:
    """Partition atoms into equivalence classes by iterative refinement.

    This is one-dimensional Weisfeiler-Lehman refinement: each atom starts
    coloured by its element and degree, and is then recoloured by its own
    colour plus the multiset of its neighbours' colours and bond orders, until
    the partition stops changing. Two atoms can only be exchanged by an
    automorphism if they end in the same class, so the classes prune the
    search. Refinement alone does not prove they *can* be exchanged, which is
    why the backtracking search follows.
    """
    adjacency = graph.adjacency
    colours = [hash((element.upper(), graph.degree(position)))
               for position, element in enumerate(graph.elements)]
    for _ in range(len(graph.names)):
        signatures = [
            (colours[position],
             tuple(sorted((order, colours[neighbour])
                          for neighbour, order in adjacency[position])))
            for position in range(len(graph.names))
        ]
        lookup = {signature: number for number, signature
                  in enumerate(sorted(set(signatures), key=repr))}
        updated = [lookup[signature] for signature in signatures]
        if updated == colours:
            break
        colours = updated
    return colours


def automorphisms(graph: MolecularGraph,
                  cap: int = DEFAULT_AUTOMORPHISM_CAP) -> list[tuple[int, ...]]:
    """Enumerate the graph's automorphisms as index permutations.

    Each result maps a reference atom index to the index it may be compared
    against. The identity is always included and always first.

    Raises `CorrespondenceError` if more than `cap` automorphisms exist, rather
    than returning a truncated set: a partial symmetry group would make the
    reported RMSD depend on search order.
    """
    size = len(graph.names)
    colours = refine_colours(graph)
    adjacency = graph.adjacency
    neighbours = {position: {neighbour: order
                             for neighbour, order in adjacency[position]}
                  for position in range(size)}
    candidates = {
        position: [other for other in range(size) if colours[other] == colours[position]]
        for position in range(size)
    }
    # Most-constrained atom first: the smallest candidate list prunes hardest.
    order = sorted(range(size), key=lambda position: len(candidates[position]))

    results: list[tuple[int, ...]] = []
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def consistent(source: int, image: int) -> bool:
        if graph.elements[source].upper() != graph.elements[image].upper():
            return False
        if len(neighbours[source]) != len(neighbours[image]):
            return False
        for neighbour, bond_order in neighbours[source].items():
            mapped = mapping.get(neighbour)
            if mapped is None:
                continue
            if neighbours[image].get(mapped) != bond_order:
                return False
        # An already-mapped atom must not become adjacent where it was not.
        for neighbour, mapped in mapping.items():
            if neighbour in neighbours[source]:
                continue
            if mapped in neighbours[image]:
                return False
        return True

    def search(depth: int) -> None:
        if len(results) > cap:
            raise CorrespondenceError(
                f"more than {cap} automorphisms; the symmetry group is too "
                f"large to enumerate, so the RMSD cannot be minimised over it "
                f"reliably."
            )
        if depth == size:
            results.append(tuple(mapping[position] for position in range(size)))
            return
        source = order[depth]
        for image in candidates[source]:
            if image in used or not consistent(source, image):
                continue
            mapping[source] = image
            used.add(image)
            search(depth + 1)
            used.discard(image)
            del mapping[source]

    search(0)
    if not results:
        raise CorrespondenceError("no automorphism found, not even the identity")
    identity = tuple(range(size))
    results.sort(key=lambda permutation: permutation != identity)
    return results


def symmetry_classes(graph: MolecularGraph,
                     permutations: Iterable[tuple[int, ...]]) -> list[list[str]]:
    """Group atom names into orbits under the given permutations.

    Reported in the manifest so the symmetry actually used is inspectable
    rather than implicit. Orbits of size one are omitted.
    """
    parent = list(range(len(graph.names)))

    def root(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    for permutation in permutations:
        for source, image in enumerate(permutation):
            a, b = root(source), root(image)
            if a != b:
                parent[a] = b
    orbits: dict[int, list[str]] = defaultdict(list)
    for position, name in enumerate(graph.names):
        orbits[root(position)].append(name)
    return sorted((sorted(members) for members in orbits.values() if len(members) > 1),
                  key=lambda members: members[0])


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

def paired_coordinates(graph: MolecularGraph,
                       reference: dict[str, Sequence[float]],
                       pose: dict[str, Sequence[float]]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return reference and pose coordinate arrays in the graph's atom order.

    Raises if either structure is missing an atom the graph names, or carries
    an atom the graph does not. Comparing over the intersection would quietly
    change the measure, so it is refused.
    """
    missing_reference = [name for name in graph.names if name not in reference]
    missing_pose = [name for name in graph.names if name not in pose]
    extra_reference = sorted(set(reference) - set(graph.names))
    extra_pose = sorted(set(pose) - set(graph.names))
    problems = []
    if missing_reference:
        problems.append(f"absent from the reference: {missing_reference}")
    if missing_pose:
        problems.append(f"absent from the pose: {missing_pose}")
    if extra_reference:
        problems.append(f"in the reference but not the connection table: {extra_reference}")
    if extra_pose:
        problems.append(f"in the pose but not the connection table: {extra_pose}")
    if problems:
        raise CorrespondenceError(
            "cannot pair atoms for an RMSD; " + "; ".join(problems)
            + ". The RMSD is not computed over a subset."
        )
    first = np.array([reference[name] for name in graph.names], dtype=float)
    second = np.array([pose[name] for name in graph.names], dtype=float)
    for array, label in ((first, "reference"), (second, "pose")):
        if not np.isfinite(array).all():
            raise CorrespondenceError(f"non-finite {label} coordinates")
    return first, second


def in_place_rmsd(reference: np.ndarray, pose: np.ndarray) -> float:
    """Root-mean-square deviation with no superposition, in Angstroms."""
    if reference.shape != pose.shape:
        raise CorrespondenceError(
            f"shape mismatch {reference.shape} against {pose.shape}")
    return float(np.sqrt(((reference - pose) ** 2).sum(1).mean()))


def symmetry_aware_rmsd(graph: MolecularGraph,
                        reference: dict[str, Sequence[float]],
                        pose: dict[str, Sequence[float]],
                        permutations: Sequence[tuple[int, ...]] | None = None
                        ) -> tuple[float, tuple[int, ...]]:

    """Smallest in-place RMSD over the molecule's symmetry, in Angstroms.

    Returns the RMSD and the permutation that achieved it, so the mapping used
    can be recorded. No superposition is applied: the value is the deviation
    in the frame the coordinates are already in.
    """
    if permutations is None:
        permutations = automorphisms(graph)
    first, second = paired_coordinates(graph, reference, pose)
    best, best_permutation = float("inf"), None
    for permutation in permutations:
        value = in_place_rmsd(first, second[list(permutation)])
        if value < best:
            best, best_permutation = value, permutation
    if best_permutation is None:
        raise CorrespondenceError("no permutation was evaluated")
    return best, best_permutation


def graph_from_bond_names(names: Sequence[str], elements: Sequence[str],
                          bonds: Iterable[tuple[str, str, str]]) -> MolecularGraph:
    """Build a graph from atom names, elements and (name, name, order) bonds."""
    graph_names = tuple(names)
    lookup = {name: position for position, name in enumerate(graph_names)}
    table: dict[frozenset[int], str] = {}
    for first, second, order in bonds:
        if first not in lookup or second not in lookup:
            raise CorrespondenceError(
                f"bond {first}-{second} names an atom outside the heavy-atom set")
        table[frozenset((lookup[first], lookup[second]))] = order
    return MolecularGraph(graph_names, tuple(elements), table)
