"""Audit which named oxygen pairs may be exchanged, independently of our graph.

Why this module exists separately
--------------------------------
The redocking measure's symmetry-aware RMSD permits certain atom exchanges.
Validating the arithmetic against a toolkit is only meaningful if both sides
use the *same* set of permitted exchanges, and establishing that set is a
chemical judgement, not an arithmetic one. This module makes that judgement
from the component dictionary alone and writes down the reasoning for each
named pair, so the mapping convention can be reviewed atom by atom.

It is deliberately independent of `chemical_graph`:

- the dictionary is parsed here, directly, with Gemmi;
- connectivity and hydrogen attachment are read from
  `_chem_comp_bond`/`_chem_comp_atom` rows;
- **nothing is taken from `chemical_graph.automorphisms`, from
  `normalise_resonance`, or from any orbit or permutation that module
  produces.**

That independence is the point. A numerical comparison whose permitted
mappings came from the implementation under test would only confirm that the
implementation agrees with itself.

What an audit decides, and on what grounds
------------------------------------------
Two named oxygens of one centre may be exchanged at a modelled protonation
state only when **all** of the following hold, each recorded per pair:

1. both are oxygen;
2. both are terminal, that is bonded to exactly one heavy atom;
3. both are bonded to the *same* heavy atom, the centre;
4. **neither bears a hydrogen at the modelled state.**

Condition 4 is the one that distinguishes a chemical judgement from a
topological one. A deprotonated phosphate or carboxylate has genuinely
delocalised terminal oxygens. A group retaining a proton does not: one oxygen
carries it and the other does not, they differ in bond length and in
hydrogen-bonding role, and exchanging them describes a different structure.

Bond order is deliberately **not** a criterion. A dictionary records one
double and one single bond to a deprotonated carboxylate's oxygens, which is
a drawing convention for a delocalised group rather than a statement that the
two oxygens differ. Condition 4, not the bond order, is what settles it.

What this establishes, and what it does not
-------------------------------------------
An audit is an explicitly recorded **mapping convention**, reviewable against
the dictionary. It is not experimental evidence about the molecule, and
agreement reached under it validates arithmetic, not chemistry.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import permutations as all_permutations
from pathlib import Path
from typing import Iterable, Sequence

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "mur-ligase-oxygen-pair-audit/1"


class AuditError(RuntimeError):
    """Raised when the dictionary cannot support an audit."""


@dataclass(frozen=True)
class DictionaryChemistry:
    """Connectivity and hydrogen attachment, read straight from a CCD entry.

    `heavy_atom_names` is in dictionary order. `heavy_neighbours` maps a heavy
    atom to the heavy atoms bonded to it. `hydrogens_on` maps a heavy atom to
    the hydrogens bonded to it in the **neutral dictionary form**; the modelled
    state is applied separately, so the dictionary is never mistaken for the
    state being docked.
    """
    identifier: str
    heavy_atom_names: tuple[str, ...]
    elements: dict[str, str]
    heavy_neighbours: dict[str, tuple[str, ...]]
    hydrogens_on: dict[str, tuple[str, ...]]
    bond_orders: dict[frozenset[str], str]
    source_path: str


def read_dictionary_chemistry(path: Path | str) -> DictionaryChemistry:
    """Parse a CCD entry for connectivity and hydrogen attachment only."""
    source = Path(path)
    if not source.is_file():
        raise AuditError(f"dictionary entry not found: {source}")
    try:
        block = gemmi.cif.read(str(source)).sole_block()
    except (RuntimeError, ValueError) as error:
        raise AuditError(f"cannot read {source}: {error}") from error

    found = block.find_value("_chem_comp.id")
    identifier = gemmi.cif.as_string(found).strip() if found else ""
    if not identifier:
        raise AuditError(f"{source} has no _chem_comp.id")

    elements: dict[str, str] = {}
    heavy: list[str] = []
    atom_rows = list(block.find("_chem_comp_atom.", ["atom_id", "type_symbol"]))
    if not atom_rows:
        raise AuditError(f"{source} has no _chem_comp_atom loop")
    for row in atom_rows:
        # as_string handles CIF quoting; stripping by hand would eat the
        # prime in a name such as O3'.
        name = gemmi.cif.as_string(row[0])
        element = gemmi.cif.as_string(row[1]).upper()
        elements[name] = element
        if element != "H":
            heavy.append(name)

    heavy_set = set(heavy)
    neighbours: dict[str, list[str]] = {name: [] for name in heavy}
    hydrogens: dict[str, list[str]] = {}
    orders: dict[frozenset[str], str] = {}
    bond_rows = list(block.find("_chem_comp_bond.",
                                ["atom_id_1", "atom_id_2", "value_order"]))
    if not bond_rows:
        raise AuditError(
            f"{source} has no _chem_comp_bond loop, so connectivity is "
            f"unavailable and no audit can be made")
    for row in bond_rows:
        first = gemmi.cif.as_string(row[0])
        second = gemmi.cif.as_string(row[1])
        order = gemmi.cif.as_string(row[2]).strip()
        if first in heavy_set and second in heavy_set:
            neighbours[first].append(second)
            neighbours[second].append(first)
            orders[frozenset((first, second))] = order
            continue
        heavy_end, light = ((first, second) if first in heavy_set
                            else (second, first))
        if heavy_end in heavy_set and elements.get(light) == "H":
            hydrogens.setdefault(heavy_end, []).append(light)

    return DictionaryChemistry(
        identifier=identifier,
        heavy_atom_names=tuple(heavy),
        elements=elements,
        heavy_neighbours={name: tuple(sorted(values))
                          for name, values in neighbours.items()},
        hydrogens_on={name: tuple(sorted(values))
                      for name, values in hydrogens.items()},
        bond_orders=orders,
        source_path=str(source),
    )


def protonated_at_state(chemistry: DictionaryChemistry,
                        deprotonated: Iterable[str]) -> list[str]:
    """Heavy atoms still bearing a hydrogen at the modelled state.

    The dictionary form is the neutral molecule, so a site keeps its hydrogen
    unless the modelled state deprotonates it. `deprotonated` comes from the
    ligand manifest, which records that decision.
    """
    removed = set(deprotonated)
    return sorted(name for name in chemistry.hydrogens_on if name not in removed)


def audit_pairs(chemistry: DictionaryChemistry,
                deprotonated: Iterable[str]) -> dict[str, object]:
    """Audit every candidate oxygen pair at one modelled protonation state.

    A candidate is any pair of oxygens sharing a bonded heavy atom. Every
    candidate is reported with each condition evaluated, whether it was
    allowed or refused, so the convention can be checked pair by pair rather
    than taken on trust.
    """
    removed = set(deprotonated)
    unknown = removed - set(chemistry.hydrogens_on)
    if unknown:
        raise AuditError(
            f"the modelled state deprotonates {sorted(unknown)}, which bear "
            f"no hydrogen in {chemistry.identifier}'s neutral dictionary "
            f"form; the state and the dictionary disagree")

    still_protonated = set(protonated_at_state(chemistry, removed))
    audited: list[dict[str, object]] = []
    allowed_by_centre: dict[str, list[str]] = {}

    for centre in chemistry.heavy_atom_names:
        oxygens = sorted(
            name for name in chemistry.heavy_neighbours[centre]
            if chemistry.elements.get(name) == "O"
            and len(chemistry.heavy_neighbours[name]) == 1)
        if len(oxygens) < 2:
            continue
        for first, second in sorted(
                (a, b) for index, a in enumerate(oxygens)
                for b in oxygens[index + 1:]):
            protonated_members = sorted(
                name for name in (first, second) if name in still_protonated)
            conditions = {
                "both_are_oxygen": True,
                "both_are_terminal": True,
                "share_one_centre": centre,
                "neither_bears_a_hydrogen_at_this_state": not protonated_members,
            }
            allowed = not protonated_members
            if allowed:
                allowed_by_centre.setdefault(centre, [])
                for name in (first, second):
                    if name not in allowed_by_centre[centre]:
                        allowed_by_centre[centre].append(name)
            audited.append({
                "pair": [first, second],
                "centre": centre,
                "centre_element": chemistry.elements.get(centre),
                "bond_orders_in_the_dictionary": [
                    chemistry.bond_orders.get(frozenset((centre, first))),
                    chemistry.bond_orders.get(frozenset((centre, second)))],
                "hydrogens_in_the_neutral_dictionary": {
                    first: list(chemistry.hydrogens_on.get(first, ())),
                    second: list(chemistry.hydrogens_on.get(second, ()))},
                "still_protonated_at_this_state": protonated_members,
                "conditions": conditions,
                "exchange_allowed": allowed,
                "reason": (
                    f"both are terminal oxygens of {centre} and neither bears "
                    f"a hydrogen at this state, so the group is delocalised "
                    f"and the exchange describes the same structure"
                    if allowed else
                    f"{protonated_members} still bear a hydrogen at this "
                    f"state, so these oxygens are chemically inequivalent: "
                    f"one carries the proton and the other does not, and the "
                    f"exchange describes a different structure"),
                "bond_order_note": (
                    "the dictionary's bond orders are not a criterion; a "
                    "delocalised group is conventionally drawn with one "
                    "double and one single bond"),
            })

    return {
        "schema": SCHEMA,
        "component": chemistry.identifier,
        "dictionary": chemistry.source_path,
        "independence": (
            "connectivity and hydrogen attachment parsed directly from the "
            "dictionary in this module; nothing is taken from "
            "chemical_graph, its orbits or its automorphisms"),
        "modelled_state": {
            "deprotonated_atoms": sorted(removed),
            "sites_still_protonated": sorted(still_protonated),
        },
        "criteria": [
            "both atoms are oxygen",
            "both are terminal, bonded to exactly one heavy atom",
            "both are bonded to the same heavy atom",
            "neither bears a hydrogen at the modelled state",
        ],
        "audited_pairs": audited,
        "allowed_pairs": [entry["pair"] for entry in audited
                          if entry["exchange_allowed"]],
        "refused_pairs": [entry["pair"] for entry in audited
                          if not entry["exchange_allowed"]],
        "exchangeable_groups": {centre: sorted(names) for centre, names
                                in sorted(allowed_by_centre.items())},
        "status": ("an explicitly recorded mapping convention, reviewable "
                   "against the dictionary; not experimental evidence about "
                   "the molecule"),
    }


def allowed_mappings(audit: dict[str, object],
                     heavy_atom_names: Sequence[str]
                     ) -> list[tuple[int, ...]]:
    """Build the full permitted mapping set from an audit, as index tuples.

    Within each audited exchangeable group every ordering of that group's
    oxygens is permitted, and groups are independent, so the full set is the
    product of the groups' symmetric groups. Constructed here from the audit
    alone: no permutation is read from `chemical_graph`.

    Returned in a fixed order with the identity first, over the index space of
    `heavy_atom_names`.
    """
    index = {name: position for position, name in enumerate(heavy_atom_names)}
    groups = [sorted(names) for _, names
              in sorted((audit.get("exchangeable_groups") or {}).items())]
    identity = tuple(range(len(heavy_atom_names)))
    mappings = [identity]
    for group in groups:
        positions = [index[name] for name in group]
        extended: list[tuple[int, ...]] = []
        for mapping in mappings:
            for ordering in all_permutations(positions):
                candidate = list(mapping)
                for source, target in zip(positions, ordering):
                    candidate[source] = mapping[target]
                extended.append(tuple(candidate))
        mappings = extended
    # De-duplicate while keeping the identity first and the order stable.
    seen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []
    for mapping in mappings:
        if mapping not in seen:
            seen.add(mapping)
            ordered.append(mapping)
    return ordered


def exchanged_atom_sets(mappings: Iterable[tuple[int, ...]],
                        heavy_atom_names: Sequence[str]
                        ) -> list[list[str]]:
    """The atoms each mapping moves, by name, for comparison and recording."""
    collected: set[frozenset[str]] = set()
    for mapping in mappings:
        moved = frozenset(heavy_atom_names[position] for position, target
                          in enumerate(mapping) if position != target)
        if moved:
            collected.add(moved)
    return sorted((sorted(group) for group in collected),
                  key=lambda group: (len(group), group))
