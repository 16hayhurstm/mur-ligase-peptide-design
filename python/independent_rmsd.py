"""Validate this project's symmetry-aware RMSD against RDKit, no fitting.

Why this exists
---------------
`chemical_graph.symmetry_aware_rmsd` is a custom implementation, and its own
unit tests are not independent of it: they were written from the same
understanding of the problem. Before the redocking control's numbers mean
anything, the implementation has to agree with an established toolkit on
structures whose answer is known for reasons outside both implementations.

What is compared, and what makes it independent
-----------------------------------------------
RDKit's `rdMolAlign.CalcRMS` is the comparator because it is symmetry-corrected
and applies **no superposition**, which is the quantity this project computes.
Three things keep the comparison honest:

1. **Analytic cases.** A rigid translation by a vector of length *d* moves
   every atom by exactly *d*, so the in-place RMSD is exactly *d* for any
   atom correspondence that is the identity on a rigid body. That expectation
   comes from geometry, not from either implementation, so the translation
   cases have a third, independent answer.
2. **Equivalences are verified, not assumed, at three levels.** RDKit
   perceives symmetry from the molecule it is handed, so it is handed the
   *charged* molecule the verification wrote, not the neutral dictionary
   form. Then: its perceived equivalence classes are read back out and
   compared with this project's orbits; the self-mappings its substructure
   match permits -- which is what `CalcRMS` actually minimises over -- are
   enumerated and recorded; and its own option set is read off the installed
   build. That last one matters, because
   `symmetrizeConjugatedTerminalGroups` **defaults to true** and makes
   CalcRMS treat a conjugated terminal group's atoms as interchangeable
   regardless of this project's protonation rule. Both settings are
   evaluated, so the flag's effect is measured rather than assumed.
3. **This project's automorphism list is never supplied to RDKit.** Passing
   our own permutations in as RDKit's atom map would make the check circular:
   it would then only confirm that two routines evaluate the same mapping, not
   that they agree on which mappings are chemically allowed.

The cases
---------
Generated deterministically from the verified molecule's heavy-atom
coordinates, with no randomness and no embedding:

- `unchanged`: the coordinates as verified. Expected RMSD exactly 0.
- `translation_small`, `translation_large`: rigid translations along a fixed
  axis. Expected RMSD exactly the translation length.
- `swap_<centre>`: coordinates exchanged between two terminal oxygens of one
  centre **whose oxygens are all deprotonated at the modelled state**. Such a
  swap describes the same structure, so a symmetry-aware measure must return
  0 while a naive one returns the oxygen separation.
- `negative_swap_<centre>`: coordinates exchanged between a protonated and an
  unprotonated terminal oxygen of the same centre. These are chemically
  **inequivalent** -- one bears the proton, the other does not, and they
  differ in bond length and hydrogen-bonding role -- so this swap describes a
  different structure and both implementations must return a non-zero value.
  This is the case that actually exercises the protonation rule, and it
  **only exists where some terminal oxygen is still protonated**. In a fully
  deprotonated state there is no such pair: the case is then recorded as not
  applicable, with the reason, and is not demanded. A state with nothing
  protonated is a property of the state, not a defect.
- `negative_swap_<A>_<B>`: coordinates exchanged between two heavy atoms of
  **different elements**. An automorphism preserves element, so these can
  never be equivalent at any protonation state. This case therefore applies
  to every state and guarantees that a fully deprotonated state is still
  tested in the inequivalent direction. Both implementations must return a
  non-zero value.

A negative case that applies and comes out as zero in either implementation
fails the record, as does an applicable case that was not generated. Only the
protonation-specific case can be not applicable, and only for the reason
recorded with it.

Limitations
-----------
- These are synthetic coordinate transformations of one deposited structure.
  They validate software, and are **not** docking results, not poses and not
  evidence about binding.
- Agreement shows the two implementations compute the same quantity on these
  cases. It does not prove either is correct for every molecule, and it tests
  no stereochemistry: a swap of equivalent oxygens is stereochemically inert.
- RDKit's symmetry perception and this project's resonance-normalised graph
  automorphisms can legitimately differ in what they consider equivalent. The
  orbit comparison records that; where they differ, the exchanged-atom sets
  are the thing to compare, not the numbers.
- Hydrogens are excluded throughout, as they are from the control's measure.

The comparison stage needs RDKit, so it runs where RDKit is installed. Case
generation and this project's own side need only NumPy, and are importable
without RDKit so they can be tested anywhere.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_geometry
import chemical_component
import chemical_graph
import receptor_provenance as provenance

SCHEMA = "mur-ligase-independent-rmsd/1"

# Agreement tolerance in Angstroms. Both implementations compute the same
# closed-form quantity over the same coordinates, so a real disagreement is
# gross; this leaves room only for floating-point summation order.
AGREEMENT_TOLERANCE_ANGSTROM = 1e-6

# Fixed translation lengths, in Angstroms. The small one is below any sensible
# pose criterion and the large one well above it, so an error that scales with
# displacement has somewhere to show itself.
TRANSLATION_SMALL_ANGSTROM = 0.25
TRANSLATION_LARGE_ANGSTROM = 5.0

# Fixed axis for the translations: a unit vector with three unequal non-zero
# components, so a transposed or dropped coordinate cannot cancel out.
TRANSLATION_AXIS = (2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0)

SYMMETRISED_VARIANT = "symmetrised_conjugated_terminal_groups"
UNSYMMETRISED_VARIANT = "no_conjugated_terminal_group_symmetrisation"

# The configuration the comparison is *made under*. Chosen on intent, not on
# whichever setting happens to agree: this project treats a terminal-oxygen
# group as interchangeable when every one of its oxygens is deprotonated, and
# the symmetrising setting is the only one that can return zero for such a
# justified swap, because the connection table still carries one double and
# one single bond. Only this configuration's failures block validation.
SELECTED_VARIANT = SYMMETRISED_VARIANT

SELECTION_JUSTIFICATION = (
    "CalcRMS with symmetrizeConjugatedTerminalGroups=True is selected because "
    "a justified terminal-oxygen swap must evaluate to zero, and the "
    "unsymmetrised setting cannot give that: the molfile records one double "
    "and one single bond to the group's oxygens, so without symmetrisation "
    "the toolkit treats them as distinguishable and a chemically equivalent "
    "swap would score non-zero. The selected setting is nonetheless more "
    "permissive than this project's rule, since it symmetrises a conjugated "
    "terminal group whether or not its oxygens are protonated. That gap is "
    "exactly what the protonation negative case measures, and a zero there "
    "fails the selected configuration rather than being tolerated."
)



class ComparisonError(RuntimeError):
    """Raised when the comparison cannot be set up or its inputs disagree."""


@dataclass(frozen=True)
class SyntheticCase:
    """One synthetic coordinate case and what is expected of it.

    `coordinates` maps heavy-atom name to (x, y, z) in Angstroms, in the
    receptor frame of the verified molecule. `expected_angstrom` is the value
    geometry alone requires, or None where no closed form applies.
    `equivalence_class` is "equivalent" when the transformation describes the
    same structure and "inequivalent" when it describes a different one.
    """
    label: str
    kind: str
    coordinates: dict[str, tuple[float, float, float]]
    expected_angstrom: float | None
    equivalence_class: str
    justification: str
    atoms_exchanged: tuple[str, ...] = field(default=())


def negative_case_plan(graph: chemical_graph.MolecularGraph,
                       groups: Sequence[dict[str, object]]
                       ) -> dict[str, dict[str, object]]:
    """Decide which negative cases apply at this modelled protonation state.

    Two kinds, because one of them does not always exist:

    `protonation_oxygen_swap` exchanges a protonated and an unprotonated
    terminal oxygen of one centre. It is the case that tests the protonation
    rule specifically, and it **only exists when some terminal oxygen is still
    protonated**. In a fully deprotonated state there is no such pair, and the
    case is recorded as not applicable with the reason rather than demanded
    and failed: a state with nothing protonated is not evidence of a defect.

    `element_swap` exchanges two heavy atoms of different elements. An
    automorphism preserves element, so atoms of different elements can never
    be equivalent under any protonation state. This case therefore applies at
    every state, and it is what keeps a fully deprotonated state from having
    no negative case at all.
    """
    plan: dict[str, dict[str, object]] = {}

    candidates = [group for group in groups
                  if group["protonated"] and group["deprotonated"]]
    if candidates:
        group = candidates[0]
        protonated = str(group["protonated"][0])
        unprotonated = str(group["deprotonated"][0])
        plan["protonation_oxygen_swap"] = {
            "applicable": True,
            "centre": group["centre"],
            "atoms": [protonated, unprotonated],
            "reason": (f"{protonated} still bears a hydrogen at this state "
                       f"and {unprotonated} does not, so this pair tests the "
                       f"protonation rule directly"),
        }
    else:
        partly = [group["centre"] for group in groups
                  if group["protonated"]]
        plan["protonation_oxygen_swap"] = {
            "applicable": False,
            "centre": None,
            "atoms": [],
            "reason": (
                "no centre has both a protonated and an unprotonated terminal "
                "oxygen at this modelled state, so no such pair exists to "
                "exchange. This is a property of the state, not a gap in the "
                "comparison: the element-swap case below covers the "
                "inequivalent direction here"
                + (f". Centres with every terminal oxygen protonated: "
                   f"{partly}" if partly else "")),
        }

    # Deterministic: the lowest-named carbon and the lowest-named oxygen.
    by_element: dict[str, list[str]] = {}
    for name, element in zip(graph.names, graph.elements):
        by_element.setdefault(element.upper(), []).append(name)
    first_pair = None
    for left, right in (("C", "O"), ("C", "N"), ("O", "N"), ("P", "O")):
        if by_element.get(left) and by_element.get(right):
            first_pair = (sorted(by_element[left])[0], sorted(by_element[right])[0])
            break
    if first_pair is None:
        elements = sorted(by_element)
        if len(elements) >= 2:
            first_pair = (sorted(by_element[elements[0]])[0],
                          sorted(by_element[elements[1]])[0])
    if first_pair is None:
        plan["element_swap"] = {
            "applicable": False,
            "atoms": [],
            "reason": ("the molecule has heavy atoms of only one element, so "
                       "no cross-element pair exists"),
        }
    else:
        plan["element_swap"] = {
            "applicable": True,
            "atoms": list(first_pair),
            "reason": (f"{first_pair[0]} and {first_pair[1]} are different "
                       f"elements; an automorphism preserves element, so they "
                       f"are inequivalent at every protonation state and this "
                       f"case applies to all states"),
        }
    return plan


def terminal_oxygen_groups(graph: chemical_graph.MolecularGraph,
                           protonated: Sequence[str]
                           ) -> list[dict[str, object]]:
    """Group terminal oxygens by their central atom, with protonation marked.

    Applies the same chemical rule `chemical_graph.normalise_resonance`
    documents: a centre's terminal oxygens are equivalent only when none of
    them bears a hydrogen at the modelled protonation state. The grouping is
    returned structurally, rather than as the prose record that function
    writes, because the cases are built from the memberships.
    """
    adjacency = graph.adjacency
    still_protonated = set(protonated)
    groups: list[dict[str, object]] = []
    for position, element in enumerate(graph.elements):
        if element.upper() not in chemical_graph.RESONANCE_CENTRES:
            continue
        oxygens = sorted(
            graph.names[neighbour] for neighbour, _ in adjacency[position]
            if graph.elements[neighbour].upper() == "O"
            and graph.degree(neighbour) == 1)
        if len(oxygens) < 2:
            continue
        bearing = [name for name in oxygens if name in still_protonated]
        groups.append({
            "centre": graph.names[position],
            "terminal_oxygens": oxygens,
            "protonated": bearing,
            "deprotonated": [n for n in oxygens if n not in still_protonated],
            "all_deprotonated": not bearing,
        })
    return groups


def _translated(coordinates: dict[str, tuple[float, float, float]],
                distance: float
                ) -> dict[str, tuple[float, float, float]]:
    """Rigidly translate every atom by `distance` along the fixed axis."""
    axis = TRANSLATION_AXIS
    scale = distance / (sum(component ** 2 for component in axis) ** 0.5)
    return {name: (x + axis[0] * scale, y + axis[1] * scale, z + axis[2] * scale)
            for name, (x, y, z) in coordinates.items()}


def _with_swap(coordinates: dict[str, tuple[float, float, float]],
               first: str, second: str
               ) -> dict[str, tuple[float, float, float]]:
    """Exchange the coordinates of two named atoms, leaving the rest alone."""
    swapped = dict(coordinates)
    swapped[first], swapped[second] = coordinates[second], coordinates[first]
    return swapped


def separation(coordinates: dict[str, tuple[float, float, float]],
               first: str, second: str) -> float:
    """Distance in Angstroms between two named atoms."""
    a, b = coordinates[first], coordinates[second]
    return sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5


def synthetic_cases(coordinates: dict[str, tuple[float, float, float]],
                    groups: Sequence[dict[str, object]],
                    plan: dict[str, dict[str, object]] | None = None
                    ) -> list[SyntheticCase]:
    """Build the deterministic case set for one modelled protonation state.

    `coordinates` are the verified molecule's heavy atoms. `groups` comes from
    `terminal_oxygen_groups` at the same state, so which swaps are chemically
    permitted follows from the state rather than from topology alone. `plan`
    comes from `negative_case_plan` and decides which negative cases apply;
    passing None generates the positive cases and the protonation-specific
    negative case only. The order of the returned cases is fixed, and nothing
    here is random.
    """
    if not coordinates:
        raise ComparisonError("no heavy-atom coordinates were supplied")
    cases = [
        SyntheticCase(
            label="unchanged",
            kind="identity",
            coordinates=dict(coordinates),
            expected_angstrom=0.0,
            equivalence_class="equivalent",
            justification=("the coordinates as verified; any measure must "
                           "return exactly zero"),
        ),
        SyntheticCase(
            label="translation_small",
            kind="rigid_translation",
            coordinates=_translated(coordinates, TRANSLATION_SMALL_ANGSTROM),
            expected_angstrom=TRANSLATION_SMALL_ANGSTROM,
            equivalence_class="inequivalent",
            justification=(f"a rigid translation of "
                           f"{TRANSLATION_SMALL_ANGSTROM} A moves every atom "
                           f"by that distance, so an unfitted RMSD is exactly "
                           f"it; this expectation is geometric and "
                           f"independent of both implementations"),
        ),
        SyntheticCase(
            label="translation_large",
            kind="rigid_translation",
            coordinates=_translated(coordinates, TRANSLATION_LARGE_ANGSTROM),
            expected_angstrom=TRANSLATION_LARGE_ANGSTROM,
            equivalence_class="inequivalent",
            justification=(f"as above at {TRANSLATION_LARGE_ANGSTROM} A, well "
                           f"beyond any pose criterion, so an error scaling "
                           f"with displacement would show"),
        ),
    ]

    # Positive swaps: within a centre whose terminal oxygens are all
    # deprotonated, and therefore genuinely equivalent at this state.
    for group in groups:
        oxygens = list(group["terminal_oxygens"])
        if not group["all_deprotonated"] or len(oxygens) < 2:
            continue
        first, second = oxygens[0], oxygens[1]
        cases.append(SyntheticCase(
            label=f"swap_{group['centre']}",
            kind="equivalent_oxygen_swap",
            coordinates=_with_swap(coordinates, first, second),
            expected_angstrom=0.0,
            equivalence_class="equivalent",
            justification=(
                f"{first} and {second} are terminal oxygens of "
                f"{group['centre']} and none of that centre's terminal "
                f"oxygens bears a hydrogen at this state, so they are "
                f"delocalised and interchangeable: the swap describes the "
                f"same structure and a symmetry-aware measure must return "
                f"zero, while a naive one returns their separation of "
                f"{separation(coordinates, first, second):.3f} A"),
            atoms_exchanged=(first, second),
        ))

    # Negative swaps: across the protonated/unprotonated divide at one centre.
    # These must not come out as zero, and they are the cases that test the
    # protonation rule rather than the topology.
    for group in groups:
        if group["all_deprotonated"]:
            continue
        protonated = list(group["protonated"])
        unprotonated = list(group["deprotonated"])
        if not protonated or not unprotonated:
            continue
        first, second = protonated[0], unprotonated[0]
        cases.append(SyntheticCase(
            label=f"negative_swap_{group['centre']}",
            kind="protonation_oxygen_swap",
            coordinates=_with_swap(coordinates, first, second),
            expected_angstrom=None,
            equivalence_class="inequivalent",
            justification=(
                f"{first} still bears a hydrogen at this state and {second} "
                f"does not, so these terminal oxygens of {group['centre']} "
                f"are chemically inequivalent: they differ in bond length and "
                f"hydrogen-bonding role, the swap describes a different "
                f"structure, and both implementations must return a non-zero "
                f"value. Their separation is "
                f"{separation(coordinates, first, second):.3f} A"),
            atoms_exchanged=(first, second),
        ))

    # The cross-element swap, which applies at every protonation state and so
    # guarantees a negative case even where nothing is protonated.
    element_swap = (plan or {}).get("element_swap") or {}
    if element_swap.get("applicable"):
        first, second = (str(name) for name in element_swap["atoms"])
        cases.append(SyntheticCase(
            label=f"negative_swap_{first}_{second}",
            kind="element_swap",
            coordinates=_with_swap(coordinates, first, second),
            expected_angstrom=None,
            equivalence_class="inequivalent",
            justification=(
                f"{element_swap['reason']}. The swap therefore describes a "
                f"different structure and both implementations must return a "
                f"non-zero value. Their separation is "
                f"{separation(coordinates, first, second):.3f} A"),
            atoms_exchanged=(first, second),
        ))
    return cases


NEGATIVE_KINDS = ("protonation_oxygen_swap", "element_swap")


def our_rmsd(graph: chemical_graph.MolecularGraph,
             reference: dict[str, tuple[float, float, float]],
             pose: dict[str, tuple[float, float, float]],
             permutations: Sequence[tuple[int, ...]]
             ) -> tuple[float, tuple[str, ...]]:
    """This project's symmetry-aware RMSD, and the atoms the mapping exchanged.

    Returns the value in Angstroms and the names of the atoms the minimising
    permutation moved, so a disagreement can be diagnosed by mapping rather
    than by number.
    """
    value, permutation = chemical_graph.symmetry_aware_rmsd(
        graph, reference, pose, permutations)
    exchanged = tuple(sorted(
        graph.names[position] for position, target in enumerate(permutation)
        if position != target))
    return value, exchanged


def orbits_as_sets(orbits: object) -> list[frozenset[str]]:
    """Normalise an orbit description to comparable sets of atom names."""
    collected: list[frozenset[str]] = []
    if isinstance(orbits, dict):
        values = orbits.values()
    else:
        values = orbits or []
    for entry in values:
        if isinstance(entry, str):
            collected.append(frozenset([entry]))
        else:
            collected.append(frozenset(entry))
    return sorted(collected, key=lambda group: sorted(group))


def _nontrivial(groups: Sequence[frozenset[str]]) -> list[list[str]]:
    """The orbits with more than one member, as sorted name lists."""
    return sorted((sorted(group) for group in groups if len(group) > 1))


def compare(cases: Sequence[SyntheticCase],
            reference: dict[str, tuple[float, float, float]],
            graph: chemical_graph.MolecularGraph,
            permutations: Sequence[tuple[int, ...]],
            toolkit_rmsd,
            tolerance: float = AGREEMENT_TOLERANCE_ANGSTROM,
            selected_variant: str = SELECTED_VARIANT
            ) -> list[dict[str, object]]:
    """Run both implementations over every case and record the comparison.

    `toolkit_rmsd` is called as `toolkit_rmsd(case)` and returns either a value
    in Angstroms, None when the toolkit could not evaluate that case, or a
    mapping of variant name to value when the toolkit was run under more than
    one option set. With a mapping, `selected_variant` names the configuration
    agreement is **judged** on; every other variant is recorded as a
    diagnostic, reported but not binding, because the toolkit's own options
    decide which mappings it permits and a setting that was not selected
    describes a different rule.

    Injecting the callable keeps this function testable without RDKit, and
    keeps this project's permutations out of the toolkit's hands.
    """
    rows: list[dict[str, object]] = []
    for case in cases:
        ours, exchanged = our_rmsd(graph, reference, case.coordinates,
                                   permutations)
        returned = toolkit_rmsd(case)
        if isinstance(returned, dict):
            variants = returned
            theirs = variants.get(selected_variant)
        else:
            variants = {selected_variant: returned}
            theirs = returned
        diagnostics = {name: value for name, value in variants.items()
                       if name != selected_variant}
        difference = None if theirs is None else abs(ours - theirs)
        agrees = difference is not None and difference <= tolerance
        expected = case.expected_angstrom
        matches_expected = (None if expected is None
                            else abs(ours - expected) <= tolerance)
        row = {
            "label": case.label,
            "kind": case.kind,
            "equivalence_class": case.equivalence_class,
            "atoms_exchanged_in_the_case": list(case.atoms_exchanged),
            "justification": case.justification,
            "ours_angstrom": round(ours, 9),
            "toolkit_angstrom": None if theirs is None else round(theirs, 9),
            "toolkit_variant_judged": selected_variant,
            "toolkit_angstrom_by_variant": {
                name: None if value is None else round(value, 9)
                for name, value in variants.items()},
            "diagnostic_variants": {
                name: {
                    "angstrom": None if value is None else round(value, 9),
                    "agrees_with_ours": (value is not None
                                         and abs(ours - value) <= tolerance),
                }
                for name, value in diagnostics.items()},
            "absolute_difference_angstrom": (None if difference is None
                                             else round(difference, 12)),
            "agrees": agrees,
            "analytic_expectation_angstrom": expected,
            "ours_matches_analytic_expectation": matches_expected,
            "atoms_exchanged_by_our_minimising_mapping": list(exchanged),
        }
        if case.equivalence_class == "inequivalent":
            # A non-zero value is the whole point of these cases: if the
            # measure returned zero it would have treated a different
            # structure as the same one. Every evaluated variant is checked,
            # because an option that symmetrises a group our protonation rule
            # keeps distinct would show up as a zero here and nowhere else.
            row["ours_is_nonzero"] = ours > tolerance
            row["toolkit_is_nonzero"] = (None if theirs is None
                                         else theirs > tolerance)
            # Split by configuration: a zero under the selected setting is a
            # blocking failure, because that is the rule being validated. A
            # zero under a diagnostic setting is reported, not binding.
            row["selected_treats_the_case_as_equivalent"] = (
                theirs is not None and theirs <= tolerance)
            row["diagnostic_variants_treating_the_case_as_equivalent"] = sorted(
                name for name, value in diagnostics.items()
                if value is not None and value <= tolerance)
        rows.append(row)
    return rows


def assemble_record(state: str,
                    component: str,
                    inputs: dict[str, object],
                    protonation_state: dict[str, object],
                    tools: dict[str, object],
                    groups: Sequence[dict[str, object]],
                    equivalence: dict[str, object],
                    rows: Sequence[dict[str, object]],
                    plan: dict[str, dict[str, object]] | None = None,
                    toolkit_options: dict[str, object] | None = None,
                    tolerance: float = AGREEMENT_TOLERANCE_ANGSTROM
                    ) -> dict[str, object]:
    """Build the structured comparison record the analysis stage validates.

    The record is tied to its input files by SHA-256 and to the software by
    version, and it states its own verdict so a downstream reader gates on the
    result rather than on the file existing.
    """
    plan = plan or {}
    differences = [row["absolute_difference_angstrom"] for row in rows
                   if row["absolute_difference_angstrom"] is not None]
    unevaluated = [row["label"] for row in rows
                   if row["toolkit_angstrom"] is None]
    negatives = [row for row in rows if row["kind"] in NEGATIVE_KINDS]
    analytic = [row for row in rows
                if row["analytic_expectation_angstrom"] is not None]

    all_agree = bool(rows) and all(row["agrees"] for row in rows)
    analytic_agree = all(row["ours_matches_analytic_expectation"]
                         for row in analytic) if analytic else False

    # Per negative kind: whether the state makes it applicable, whether it was
    # generated, and whether both implementations kept it non-zero. A kind that
    # does not apply at this state is recorded with its reason and does not
    # count against the verdict; a kind that applies and is missing does.
    negative_detail: dict[str, object] = {}
    missing: list[str] = []
    undistinguished: list[str] = []
    wrongly_equivalent: list[dict[str, object]] = []
    diagnostic_equivalent: list[dict[str, object]] = []
    for kind in NEGATIVE_KINDS:
        entry = dict(plan.get(kind, {}) or {})
        generated = [row for row in negatives if row["kind"] == kind]
        entry["generated"] = [row["label"] for row in generated]
        if entry.get("applicable") and not generated:
            entry["distinguished"] = False
            missing.append(kind)
        elif not generated:
            entry["distinguished"] = None
        else:
            ok = all(row.get("ours_is_nonzero") and row.get("toolkit_is_nonzero")
                     for row in generated)
            entry["distinguished"] = ok
            if not ok:
                undistinguished.extend(row["label"] for row in generated
                                       if not (row.get("ours_is_nonzero")
                                               and row.get("toolkit_is_nonzero")))
            for row in generated:
                if row.get("selected_treats_the_case_as_equivalent"):
                    wrongly_equivalent.append(
                        {"case": row["label"],
                         "configuration": row.get("toolkit_variant_judged")})
                permissive = row.get(
                    "diagnostic_variants_treating_the_case_as_equivalent") or []
                if permissive:
                    diagnostic_equivalent.append({"case": row["label"],
                                                  "variants": permissive})
        negative_detail[kind] = entry

    applicable_kinds = [kind for kind in NEGATIVE_KINDS
                        if (plan.get(kind, {}) or {}).get("applicable")]
    negative_distinguished = bool(applicable_kinds) and not missing and all(
        negative_detail[kind].get("distinguished") is not False
        for kind in applicable_kinds) and bool(negatives) and all(
        row.get("ours_is_nonzero") and row.get("toolkit_is_nonzero")
        for row in negatives)

    failures: list[str] = []
    if not rows:
        failures.append("no cases were evaluated")
    if unevaluated:
        failures.append(f"the toolkit evaluated no value for {unevaluated}")
    if not all_agree:
        disagreeing = [row["label"] for row in rows if not row["agrees"]]
        failures.append(f"the implementations disagree on {disagreeing}")
    if not analytic_agree:
        failing = [row["label"] for row in analytic
                   if not row["ours_matches_analytic_expectation"]]
        failures.append(
            f"this project's value departs from the geometric expectation on "
            f"{failing}")
    if not applicable_kinds:
        failures.append(
            "no negative case applies at this state, which should not happen: "
            "the element-swap case applies at every protonation state")
    if missing:
        failures.append(
            f"a negative case that applies at this state was not generated: "
            f"{missing}")
    if undistinguished:
        failures.append(
            f"an inequivalent swap came out as zero in at least one "
            f"implementation: {undistinguished}")
    for entry in wrongly_equivalent:
        failures.append(
            f"under the selected configuration "
            f"{entry['configuration']!r} the toolkit treated the "
            f"inequivalent case {entry['case']} as equivalent: the selected "
            f"configuration permits a mapping this project's protonation rule "
            f"forbids, so the two measures are not computing the same "
            f"quantity. Neither the tolerance nor the equivalence rule may be "
            f"changed to force agreement; this needs a decision")
    if not equivalence.get("agree"):
        failures.append(
            "the toolkit's perceived atom equivalences do not match this "
            "project's orbits, so the two measures may not be computing the "
            "same quantity")

    # Diagnostics: the configurations that were not selected. Disagreement
    # here is reported prominently and does not invalidate a selected
    # configuration that passes every required check on its own.
    diagnostic_names = sorted({
        name for row in rows
        for name in (row.get("diagnostic_variants") or {})})
    diagnostic_disagreements = [
        {"case": row["label"],
         "variant": name,
         "toolkit_angstrom": detail["angstrom"],
         "ours_angstrom": row["ours_angstrom"]}
        for row in rows
        for name, detail in (row.get("diagnostic_variants") or {}).items()
        if not detail["agrees_with_ours"]]
    diagnostics = {
        "variants": diagnostic_names,
        "status": ("reported, not binding: a diagnostic setting describes a "
                   "different equivalence rule, so its disagreement does not "
                   "invalidate the selected configuration"),
        "n_disagreements": len(diagnostic_disagreements),
        "disagreements": diagnostic_disagreements,
        "inequivalent_cases_treated_as_equivalent": diagnostic_equivalent,
    }

    # The one case where a diagnostic result does bear on the verdict: if the
    # selected configuration has failed and no evaluated setting reproduces
    # the intended equivalences either, say so plainly rather than leaving the
    # impression that some other setting would have worked.
    if failures and diagnostic_names and not diagnostic_disagreements:
        failures.append(
            f"a diagnostic configuration {diagnostic_names} did agree with "
            f"this project on every case while the selected configuration did "
            f"not; this is evidence about which rule each setting implements "
            f"and needs a decision, not a change of selection to whichever "
            f"setting agrees")
    elif failures and diagnostic_names:
        failures.append(
            f"neither the selected configuration nor the diagnostic "
            f"configurations {diagnostic_names} reproduced this project's "
            f"equivalences and values; do not widen the tolerance or change "
            f"the equivalence rule to force agreement")

    return {
        "schema": SCHEMA,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "result": "failed" if failures else "passed",
        "failure_reasons": failures,
        "content_class": ("synthetic software validation of the RMSD "
                          "implementation; these are not docking results, "
                          "not poses, and not evidence about binding"),
        "state": state,
        "component": component,
        "tools": tools,
        "inputs": inputs,
        "protonation_state": protonation_state,
        "measure": {
            "name": "symmetry-aware heavy-atom RMSD",
            "superposition": "none",
            "hydrogens": "excluded",
            "toolkit_function": "rdkit.Chem.rdMolAlign.CalcRMS",
            "toolkit_applies_no_alignment": True,
            "our_permutations_supplied_to_the_toolkit": False,
            "selected_configuration": {
                "variant": SELECTED_VARIANT,
                "symmetrize_conjugated_terminal_groups": (
                    SELECTED_VARIANT == SYMMETRISED_VARIANT),
                "selected_on": ("intent: the configuration meant to respect "
                                "this project's protonation-dependent "
                                "equivalences, chosen before the values were "
                                "seen, not the setting that happened to agree"),
                "justification": SELECTION_JUSTIFICATION,
                "binding": ("only this configuration's failures block "
                            "validation"),
            },
            "toolkit_options": toolkit_options or {
                "recorded": False,
                "note": "no toolkit option set was recorded for this run",
            },
            "note": ("this project's automorphism list is deliberately not "
                     "given to the toolkit; doing so would confirm only that "
                     "two routines evaluate the same mapping. The toolkit's "
                     "own options decide which mappings it permits, so they "
                     "are recorded above rather than assumed"),
        },
        "terminal_oxygen_groups": list(groups),
        "negative_cases": negative_detail,
        "diagnostics": diagnostics,
        "equivalence": equivalence,
        "tolerance_angstrom": tolerance,
        "cases": list(rows),
        "checks": {
            "n_cases": len(rows),
            "all_cases_agree": all_agree,
            "max_absolute_difference_angstrom": (max(differences)
                                                 if differences else None),
            "analytic_expectations_met": analytic_agree,
            "negative_case_distinguished": negative_distinguished,
            "negative_kinds_applicable": applicable_kinds,
            "negative_kinds_not_applicable": [
                kind for kind in NEGATIVE_KINDS if kind not in applicable_kinds],
            "selected_permitted_an_inequivalent_mapping": wrongly_equivalent,
            "diagnostic_permitted_an_inequivalent_mapping": diagnostic_equivalent,
            "diagnostic_disagreements": len(diagnostic_disagreements),
            "n_inequivalent_cases": len(negatives),
            "equivalences_independently_verified": bool(
                equivalence.get("agree")),
        },
        "limitations": [
            "synthetic coordinate transformations of one deposited structure: "
            "this validates software, not any docking result",
            "agreement on these cases does not prove either implementation "
            "correct for every molecule",
            "no stereochemistry is tested; a swap of equivalent oxygens is "
            "stereochemically inert",
            "the toolkit's symmetry perception and this project's "
            "resonance-normalised automorphisms can legitimately differ; "
            "where they do, compare the exchanged-atom sets, not the numbers",
            "the verdict is made under one selected toolkit configuration; "
            "the diagnostic settings describe different equivalence rules and "
            "are reported rather than averaged in",
        ],
    }


# ---------------------------------------------------------------------------
# The RDKit side. Imported lazily so this module loads without RDKit.
# ---------------------------------------------------------------------------

def _calcrms_options() -> dict[str, object]:
    """Record the toolkit's own RMSD options, read off the installed build.

    `CanonicalRankAtoms` says which atoms RDKit considers equivalent, but it
    does not say which mappings `CalcRMS` will actually enumerate: CalcRMS has
    its own options, and in particular `symmetrizeConjugatedTerminalGroups`
    defaults to true, which makes it treat a conjugated terminal group's atoms
    as interchangeable **regardless of this project's protonation rule**. That
    is precisely the behaviour that could let a protonated and an unprotonated
    oxygen be exchanged for free, so the flag is read from the installed
    signature rather than assumed, and both settings are evaluated.
    """
    import inspect
    from rdkit.Chem import rdMolAlign

    options: dict[str, object] = {"recorded": True,
                                  "function": "rdkit.Chem.rdMolAlign.CalcRMS"}
    try:
        signature = inspect.signature(rdMolAlign.CalcRMS)
        options["signature"] = str(signature)
        options["parameters"] = {
            name: ("<no default>" if parameter.default is inspect.Parameter.empty
                   else repr(parameter.default))
            for name, parameter in signature.parameters.items()}
    except (TypeError, ValueError):
        # Boost.Python builds often expose no introspectable signature.
        options["signature"] = None
        options["parameters"] = None
        options["signature_note"] = (
            "the installed build exposes no introspectable signature; the "
            "docstring is recorded instead and the flag was probed by call")
    options["docstring"] = (rdMolAlign.CalcRMS.__doc__ or "").strip()
    options["symmetrize_conjugated_terminal_groups_supported"] = (
        _supports_symmetrisation_flag())
    options["symmetrize_conjugated_terminal_groups_default"] = True
    options["default_note"] = (
        "RDKit's default symmetrises conjugated terminal groups. This "
        "project normalises a terminal-oxygen group only when every one of "
        "its oxygens is deprotonated, so the two rules can differ, and the "
        "comparison evaluates both settings rather than trusting either")
    options["variants_evaluated"] = [SYMMETRISED_VARIANT,
                                     UNSYMMETRISED_VARIANT]
    return options


def _supports_symmetrisation_flag() -> bool:
    """Whether the installed CalcRMS accepts the symmetrisation keyword."""
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    probe = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"))
    from rdkit.Chem import AllChem
    if AllChem.EmbedMolecule(probe, randomSeed=1) != 0:
        return False
    try:
        rdMolAlign.CalcRMS(probe, probe,
                           symmetrizeConjugatedTerminalGroups=False)
    except (TypeError, ValueError, RuntimeError):
        return False
    return True


def _rdkit_self_matches(molecule, maximum: int = 100000) -> dict[str, object]:
    """Independently enumerate the molecule's self-matches under RDKit.

    This answers a question `CanonicalRankAtoms` does not: which atom
    correspondences RDKit's substructure matcher considers valid between the
    molecule and itself. Each is reported as the set of atoms it moves, by
    name, so a correspondence can be compared with this project's orbits
    directly rather than inferred from equivalence classes.

    **This is not a confirmed list of the mappings `CalcRMS` uses.** CalcRMS
    is a separate routine with its own options, conjugated terminal-group
    symmetrisation among them, and nothing here has been checked against the
    installed implementation to establish that it enumerates this same set.
    Treat the output as independent evidence about RDKit's matcher, and read
    the negative cases for what the selected CalcRMS configuration actually
    permits.
    """
    names = [atom.GetPropsAsDict().get("_TriposAtomName") or str(atom.GetIdx())
             for atom in molecule.GetAtoms()]
    matches = molecule.GetSubstructMatches(
        molecule, uniquify=False, useChirality=True, maxMatches=maximum)
    exchanged: set[frozenset[str]] = set()
    for match in matches:
        moved = frozenset(str(names[index]) for index, target
                          in enumerate(match) if index != target)
        if moved:
            exchanged.add(moved)
    reached_limit = len(matches) >= maximum
    return {
        "basis": ("GetSubstructMatches(molecule, molecule, uniquify=False, "
                  "useChirality=True): self-matches enumerated independently "
                  "from RDKit's substructure matcher"),
        "relationship_to_calcrms": (
            "NOT verified to be the mappings CalcRMS uses. CalcRMS is a "
            "separate routine with its own options, including conjugated "
            "terminal-group symmetrisation, and that correspondence has not "
            "been checked against the installed implementation"),
        "n_matches": len(matches),
        "match_limit": maximum,
        "reached_match_limit": reached_limit,
        "enumeration_complete": not reached_limit,
        "limit_note": (
            "the enumeration stopped at the limit, so the sets below are a "
            "subset of the molecule's self-matches and absence from them "
            "proves nothing" if reached_limit
            else "the enumeration finished below the limit"),
        "exchanged_atom_sets": sorted((sorted(group) for group in exchanged),
                                      key=lambda group: (len(group), group)),
    }


def _rdkit_equivalence_classes(molecule) -> list[frozenset[str]]:
    """Read RDKit's own perceived symmetry classes, by atom name.

    Uses RDKit's canonical ranking with ties kept, which is its own
    perception of equivalent atoms on the molecule as given. Nothing from
    this project is passed in. Atom names come from the molfile's atom order,
    which the verification established equals the component's heavy-atom
    order.
    """
    from rdkit import Chem
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=False))
    by_rank: dict[int, set[str]] = {}
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        name = atom.GetPropsAsDict().get("_TriposAtomName") or str(index)
        by_rank.setdefault(ranks[index], set()).add(str(name))
    return sorted((frozenset(group) for group in by_rank.values()),
                  key=lambda group: sorted(group))


def _load_molecule(path: Path):
    """Read the verified hydrogenated molfile, dropping hydrogens."""
    from rdkit import Chem
    molecule = Chem.MolFromMolFile(str(path), removeHs=True, sanitize=True)
    if molecule is None:
        raise ComparisonError(
            f"RDKit could not read {path}; the comparison cannot be made "
            f"against a molecule the toolkit will not parse")
    return molecule


def _posed_copy(molecule, names: Sequence[str],
                coordinates: dict[str, tuple[float, float, float]]):
    """A copy of the molecule carrying one case's coordinates.

    The molecule's atom order is taken to be `names`, which the verification
    record establishes against the component's heavy-atom order. Only
    coordinates are changed: the connection table, the formal charges and the
    protonation are the verified molecule's, so RDKit perceives symmetry from
    the state that was decided.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    copy = Chem.Mol(molecule)
    conformer = copy.GetConformer()
    for index, name in enumerate(names):
        x, y, z = coordinates[name]
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    return copy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ligand_manifest", type=Path)
    parser.add_argument("verification_record", type=Path,
                        help="The passing record from verify_ligand_rdkit.py; "
                             "its hydrogenated molfile is the molecule "
                             "compared")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--component-dictionary", type=Path,
                        help="Defaults to the path recorded in the manifest")
    parser.add_argument("--tolerance", type=float,
                        default=AGREEMENT_TOLERANCE_ANGSTROM)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        _run(args)
    except (ComparisonError, chemical_graph.CorrespondenceError,
            chemical_component.ComponentError, provenance.ReceptorError,
            box_geometry.BoxError) as error:
        sys.exit(f"error: {error}")


def _run(args: argparse.Namespace) -> None:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolAlign

    manifest = provenance.read_manifest_like(args.ligand_manifest,
                                             "mur-ligase-ligand/1")
    verification = provenance.read_manifest_like(
        args.verification_record, "mur-ligase-ligand-verification/1")
    if verification.get("result") != "passed":
        raise ComparisonError(
            f"{args.verification_record} records result "
            f"{verification.get('result')!r}; the comparison must be made on "
            f"a verified molecule")

    # The verification named the manifest it describes by digest. The same
    # agreement rule the other stages use is applied here, so a manifest an
    # earlier check stage modified in place is still accepted, and says so.
    recorded = (verification.get("inputs", {}) or {}).get(
        "ligand_manifest_sha256")
    identity = provenance.manifest_hash_agreement(args.ligand_manifest, recorded)
    if recorded and not identity["agrees"]:
        raise ComparisonError(
            f"{args.verification_record} verified a different ligand "
            f"manifest: {identity['reason']}")
    if identity["legacy_compatibility_used"]:
        print(f"note: {identity['basis']}")

    molfile = Path((verification.get("outputs", {}) or {}).get(
        "hydrogenated_molfile", ""))
    if not molfile.is_file():
        raise ComparisonError(
            f"the verified hydrogenated molfile is missing: {molfile!s}")
    recorded_molfile = (verification.get("outputs", {}) or {}).get(
        "hydrogenated_molfile_sha256")
    actual_molfile = box_geometry.file_sha256(molfile)
    if recorded_molfile and recorded_molfile != actual_molfile:
        raise ComparisonError(
            f"{molfile} has changed since it was verified (recorded "
            f"{recorded_molfile[:12]}..., found {actual_molfile[:12]}...)")

    dictionary_path = Path(args.component_dictionary
                           or manifest["dictionary"]["path"])
    component = chemical_component.read_component(dictionary_path)
    deprotonated = set(manifest["protonation_state"]["deprotonated_atoms"])
    still_protonated = sorted(set(component.hydrogens_on) - deprotonated)
    graph, resonance = chemical_graph.normalise_resonance(
        component.graph, hydrogen_bearing=still_protonated)
    permutations = chemical_graph.automorphisms(graph)
    groups = terminal_oxygen_groups(component.graph, still_protonated)
    plan = negative_case_plan(component.graph, groups)

    molecule = _load_molecule(molfile)
    names = list(component.graph.names)
    if molecule.GetNumAtoms() != len(names):
        raise ComparisonError(
            f"{molfile} has {molecule.GetNumAtoms()} heavy atoms against the "
            f"dictionary's {len(names)}; the atom correspondence this "
            f"comparison relies on does not hold")
    conformer = molecule.GetConformer()
    reference_coordinates = {
        name: (conformer.GetAtomPosition(index).x,
               conformer.GetAtomPosition(index).y,
               conformer.GetAtomPosition(index).z)
        for index, name in enumerate(names)}

    # Equivalences are verified rather than assumed: RDKit's own perception,
    # read off the charged molecule it was handed, against our orbits.
    our_orbits = orbits_as_sets(
        chemical_graph.symmetry_classes(graph, permutations))
    their_orbits = _rdkit_equivalence_classes(molecule)
    ours_named = _nontrivial(our_orbits)
    theirs_named = _nontrivial(their_orbits)
    # RDKit's canonical ranking keys atoms by index when a molfile carries no
    # names, so compare the partitions by index where names are unavailable.
    if all(group and group[0].isdigit() for group in theirs_named):
        index_of = {name: str(position) for position, name in enumerate(names)}
        ours_named = sorted(sorted(index_of[name] for name in group)
                            for group in ours_named)
    equivalence = {
        "basis": ("RDKit CanonicalRankAtoms(breakTies=False) on the charged "
                  "molecule the verification wrote, against this project's "
                  "resonance-normalised graph automorphism orbits"),
        "our_nontrivial_orbits": ours_named,
        "toolkit_nontrivial_orbits": theirs_named,
        "agree": ours_named == theirs_named,
        "resonance_normalisation": resonance,
        "limitation": (
            "equivalence classes say which atoms the toolkit considers "
            "equivalent; they do not by themselves say which mappings CalcRMS "
            "enumerates. The independently enumerated self-matches and the "
            "toolkit's own option set are recorded under "
            "measure.toolkit_options; neither is a confirmed list of CalcRMS "
            "mappings, and the negative cases are what measure the selected "
            "configuration's effect directly"),
        "note": ("only orbits with more than one member are compared; "
                 "singletons carry no equivalence. A disagreement does not "
                 "show either side wrong, but it does mean the two measures "
                 "need not compute the same quantity"),
    }

    cases = synthetic_cases(reference_coordinates, groups, plan)

    toolkit_options = _calcrms_options()
    toolkit_options["self_matches"] = _rdkit_self_matches(molecule)
    supports_flag = toolkit_options[
        "symmetrize_conjugated_terminal_groups_supported"]

    def toolkit_rmsd(case: SyntheticCase) -> dict[str, float | None]:
        """Both option sets, so the flag's effect is measured, not assumed."""
        probe = _posed_copy(molecule, names, case.coordinates)
        values: dict[str, float | None] = {}
        try:
            values[SYMMETRISED_VARIANT] = float(
                rdMolAlign.CalcRMS(probe, molecule))
        except (RuntimeError, ValueError):
            values[SYMMETRISED_VARIANT] = None
        if supports_flag:
            try:
                values[UNSYMMETRISED_VARIANT] = float(rdMolAlign.CalcRMS(
                    probe, molecule,
                    symmetrizeConjugatedTerminalGroups=False))
            except (RuntimeError, ValueError):
                values[UNSYMMETRISED_VARIANT] = None
        return values

    rows = compare(cases, reference_coordinates, graph, permutations,
                   toolkit_rmsd, tolerance=args.tolerance,
                   primary_variant=SYMMETRISED_VARIANT)

    record = assemble_record(
        state=str(verification.get("state")),
        component=component.identifier,
        inputs={
            "ligand_manifest": str(args.ligand_manifest),
            "ligand_manifest_sha256": box_geometry.file_sha256(
                args.ligand_manifest),
            "ligand_manifest_identity": identity,
            "verification_record": str(args.verification_record),
            "verification_record_sha256": box_geometry.file_sha256(
                args.verification_record),
            "hydrogenated_molfile": str(molfile),
            "hydrogenated_molfile_sha256": actual_molfile,
            "component_dictionary": str(dictionary_path),
            "component_dictionary_sha256": box_geometry.file_sha256(
                dictionary_path),
        },
        protonation_state=manifest["protonation_state"],
        tools={"rdkit": rdBase.rdkitVersion,
               "python": sys.version.split()[0]},
        groups=groups,
        equivalence=equivalence,
        rows=rows,
        plan=plan,
        toolkit_options=toolkit_options,
        tolerance=args.tolerance,
    )
    provenance.write_manifest(args.out, record, overwrite=args.overwrite)

    print(f"molecule         : {molfile}")
    print(f"state            : {record['state']}, "
          f"charge {manifest['protonation_state']['modelled_formal_charge']}")
    print(f"equivalences     : "
          f"{'agree' if equivalence['agree'] else 'DISAGREE'} "
          f"(ours {ours_named} against toolkit {theirs_named})")
    print(f"\n{'case':28s} {'ours':>12s} {'RDKit':>12s} {'diff':>10s}  expected")
    for row in rows:
        expected = row["analytic_expectation_angstrom"]
        theirs = row["toolkit_angstrom"]
        print(f"{row['label']:28s} {row['ours_angstrom']:12.6f} "
              f"{'n/a' if theirs is None else format(theirs, '12.6f')} "
              f"{'n/a' if row['absolute_difference_angstrom'] is None else format(row['absolute_difference_angstrom'], '10.2e')}  "
              f"{'-' if expected is None else format(expected, '.6f')}")
    print(f"\nCalcRMS options  : symmetrizeConjugatedTerminalGroups "
          f"default True, flag supported: {supports_flag}")
    matches = toolkit_options["self_matches"]
    print(f"  independent self-matches: {matches['n_matches']}"
          f"{' (REACHED LIMIT)' if matches['reached_match_limit'] else ''}, "
          f"exchanged sets {matches['exchanged_atom_sets']}")
    print(f"    not verified to be the mappings CalcRMS uses")
    print(f"  selected configuration : {SELECTED_VARIANT} (binding)")
    if record["diagnostics"]["n_disagreements"]:
        print(f"  DIAGNOSTIC DISAGREEMENT: "
              f"{record['diagnostics']['n_disagreements']} case/variant "
              f"pairs; reported, not binding")
        for entry in record["diagnostics"]["disagreements"]:
            print(f"    {entry['case']} under {entry['variant']}: "
                  f"ours {entry['ours_angstrom']}, "
                  f"toolkit {entry['toolkit_angstrom']}")
    for kind, entry in record["negative_cases"].items():
        state = ("applicable" if entry.get("applicable")
                 else "NOT APPLICABLE")
        print(f"  {kind}: {state}, distinguished="
              f"{entry.get('distinguished')}")
        if not entry.get("applicable"):
            print(f"    reason: {entry.get('reason')}")
    print(f"\nresult           : {record['result']}")
    for reason in record["failure_reasons"]:
        print(f"  - {reason}")
    print(f"wrote            : {args.out}")
    print("\nThese are synthetic coordinate cases: software validation, not "
          "docking results.")


if __name__ == "__main__":
    main()
