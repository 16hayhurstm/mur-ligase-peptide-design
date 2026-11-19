"""Read authoritative chemistry from a PDB Chemical Component Dictionary entry.

The dictionary is the reference for what a ligand *is*: its atom names,
elements, connectivity, bond orders and declared stereochemistry. It is not
the reference for what to dock. Those are different things and this module
keeps them apart:

**The dictionary structure** is the component as deposited in the CCD, which
for a ligand like UMA or ANP is the **neutral** form, with every acid
protonated and `_chem_comp.pdbx_formal_charge` of 0.

**The modelled protonation state** is a separate, deliberate choice about the
pH the docking is meant to represent. It has to be stated, justified and
recorded; it cannot be read off the dictionary, and inheriting the
dictionary's neutral form by default would silently dock a fully protonated
diphosphate at physiological pH.

`ProtonationState` therefore names the atoms to deprotonate and carries the
reason, and the resulting formal charge is derived rather than asserted.

Limitations
-----------
- Connectivity and stereochemistry are read as deposited; nothing here checks
  the dictionary against the literature.
- Declared stereo configurations include centres that are not stereogenic in
  solution. Both phosphorus atoms of UMA carry an `R` label in the CCD, but a
  phosphate bearing two equivalent oxygens is not a stable stereocentre; such
  labels are reported, not enforced.
- This module reads a dictionary entry and compares it with coordinates. It
  does not perceive chemistry from coordinates, and it does not assign atom
  types or partial charges, which are the preparation tool's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi

import chemical_graph


class ComponentError(ValueError):
    """Raised when a dictionary entry is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class IonisableSite:
    """A heavy atom that bears an acidic hydrogen in the neutral form."""
    atom: str
    element: str
    hydrogens: tuple[str, ...]
    attached_to: str
    attached_element: str
    group: str


@dataclass
class ChemicalComponent:
    """One CCD entry, with the heavy-atom graph and the polar-hydrogen map."""
    identifier: str
    name: str
    formula: str
    dictionary_formal_charge: int
    graph: chemical_graph.MolecularGraph
    elements: dict[str, str]
    hydrogens_on: dict[str, tuple[str, ...]]
    stereocentres: dict[str, str]
    aromatic_atoms: tuple[str, ...]
    n_hydrogens: int
    source_path: str

    @property
    def heavy_atom_names(self) -> tuple[str, ...]:
        return self.graph.names

    def ionisable_sites(self) -> list[IonisableSite]:
        """Oxygens bearing a hydrogen in the neutral form, with their context.

        Only oxygen acids are reported. Nitrogen-bound hydrogens in amides and
        in the uracil ring are not acidic at any pH this project models, and
        offering them as candidates would invite a wrong choice.
        """
        adjacency = self.graph.adjacency
        index = self.graph.index
        sites = []
        for atom, hydrogens in sorted(self.hydrogens_on.items()):
            if self.elements.get(atom, "").upper() != "O":
                continue
            position = index[atom]
            heavy_neighbours = [self.graph.names[neighbour]
                                for neighbour, _ in adjacency[position]]
            if not heavy_neighbours:
                continue
            attached = heavy_neighbours[0]
            attached_element = self.elements[attached].upper()
            terminal_oxygens = [
                self.graph.names[n] for n, _ in adjacency[index[attached]]
                if self.elements[self.graph.names[n]].upper() == "O"
                and self.graph.degree(n) == 1
            ]
            if attached_element == "P":
                group = "phosphate/phosphonate hydroxyl"
            elif attached_element == "C" and len(terminal_oxygens) >= 2:
                group = "carboxylic acid hydroxyl"
            elif attached_element == "C":
                group = "alcohol hydroxyl"
            else:
                group = f"{attached_element}-bound hydroxyl"
            sites.append(IonisableSite(
                atom=atom, element="O", hydrogens=tuple(hydrogens),
                attached_to=attached, attached_element=attached_element,
                group=group))
        return sites


def read_component(path: Path | str) -> ChemicalComponent:
    """Read a CCD entry (as fetched from files.rcsb.org/ligands/download)."""
    source = Path(path)
    if not source.is_file():
        raise ComponentError(f"dictionary entry not found: {source}")
    try:
        document = gemmi.cif.read(str(source))
        block = document.sole_block()
    except (RuntimeError, ValueError) as error:
        raise ComponentError(f"cannot read {source}: {error}") from error

    def value(tag: str, default: str = "") -> str:
        found = block.find_value(tag)
        # gemmi.cif.as_string removes CIF quoting correctly; hand-rolled
        # stripping would eat the prime in an atom name such as C2'.
        return gemmi.cif.as_string(found).strip() if found else default

    identifier = value("_chem_comp.id")
    if not identifier:
        raise ComponentError(f"{source} has no _chem_comp.id")

    elements: dict[str, str] = {}
    hydrogens_on: dict[str, list[str]] = {}
    stereocentres: dict[str, str] = {}
    aromatic: list[str] = []
    heavy_names: list[str] = []
    hydrogen_names: list[str] = []
    rows = list(block.find("_chem_comp_atom.",
                           ["atom_id", "type_symbol", "charge",
                            "pdbx_aromatic_flag", "pdbx_stereo_config"]))
    if not rows:
        raise ComponentError(f"{source} has no _chem_comp_atom loop")
    for row in rows:
        atom = gemmi.cif.as_string(row[0])
        element = gemmi.cif.as_string(row[1]).upper()
        elements[atom] = element
        if element == "H":
            hydrogen_names.append(atom)
            continue
        heavy_names.append(atom)
        if gemmi.cif.as_string(row[3]).strip() == "Y":
            aromatic.append(atom)
        configuration = gemmi.cif.as_string(row[4]).strip()
        if configuration not in (".", "?", "N", ""):
            stereocentres[atom] = configuration

    heavy_set = set(heavy_names)
    bonds: list[tuple[str, str, str]] = []
    bond_rows = list(block.find("_chem_comp_bond.",
                                ["atom_id_1", "atom_id_2", "value_order"]))
    if not bond_rows:
        raise ComponentError(
            f"{source} has no _chem_comp_bond loop, so connectivity is unavailable")
    for row in bond_rows:
        first = gemmi.cif.as_string(row[0])
        second = gemmi.cif.as_string(row[1])
        order = gemmi.cif.as_string(row[2]).strip()
        if first in heavy_set and second in heavy_set:
            bonds.append((first, second, order))
            continue
        heavy, light = ((first, second) if first in heavy_set else (second, first))
        if heavy in heavy_set and elements.get(light) == "H":
            hydrogens_on.setdefault(heavy, []).append(light)

    graph = chemical_graph.graph_from_bond_names(
        heavy_names, [elements[name] for name in heavy_names], bonds)
    try:
        charge = int(value("_chem_comp.pdbx_formal_charge", "0"))
    except ValueError:
        charge = 0
    return ChemicalComponent(
        identifier=identifier,
        name=value("_chem_comp.name"),
        formula=value("_chem_comp.formula"),
        dictionary_formal_charge=charge,
        graph=graph,
        elements=elements,
        hydrogens_on={atom: tuple(sorted(values))
                      for atom, values in hydrogens_on.items()},
        stereocentres=stereocentres,
        aromatic_atoms=tuple(aromatic),
        n_hydrogens=len(hydrogen_names),
        source_path=str(source),
    )


@dataclass
class ProtonationState:
    """A deliberate, justified protonation choice at a stated pH.

    `deprotonated` names the heavy atoms whose acidic hydrogen is removed
    relative to the dictionary's neutral form. `justification` is required:
    this project's rules forbid a scientific assumption being supplied by a
    default, and the modelled charge of a polyanion is not a detail.
    """
    component: str
    ph: float
    deprotonated: tuple[str, ...]
    justification: str
    dictionary_formal_charge: int

    @property
    def formal_charge(self) -> int:
        """Charge derived from the dictionary's charge and the deprotonations."""
        return self.dictionary_formal_charge - len(self.deprotonated)

    def as_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "modelled_ph": self.ph,
            "dictionary_formal_charge": self.dictionary_formal_charge,
            "deprotonated_atoms": list(self.deprotonated),
            "modelled_formal_charge": self.formal_charge,
            "justification": self.justification,
            "note": ("the dictionary structure is the neutral component; this "
                     "protonation state is a modelling choice applied to it"),
        }


def validated_protonation(component: ChemicalComponent, deprotonate: list[str],
                          ph: float, justification: str) -> ProtonationState:
    """Build a protonation state, checking every named atom can lose a proton.

    Raises if an atom does not exist, carries no hydrogen in the neutral form,
    or is not an oxygen acid, and if no justification is given. Deprotonating
    an atom that has no acidic hydrogen would change the molecule's formula
    without changing its charge in the way intended.
    """
    if not justification.strip():
        raise ComponentError(
            "a protonation state must carry a justification; the modelled "
            "charge is a scientific choice and is not allowed a silent default."
        )
    sites = {site.atom: site for site in component.ionisable_sites()}
    problems = []
    for atom in deprotonate:
        if atom not in component.elements:
            problems.append(f"{atom} is not in {component.identifier}")
        elif atom not in component.hydrogens_on:
            problems.append(f"{atom} bears no hydrogen in the neutral form")
        elif atom not in sites:
            problems.append(
                f"{atom} is not an oxygen acid "
                f"(element {component.elements[atom]})")
    if problems:
        raise ComponentError("; ".join(problems))
    if len(set(deprotonate)) != len(deprotonate):
        raise ComponentError(f"repeated atoms in {deprotonate}")
    return ProtonationState(
        component=component.identifier, ph=ph,
        deprotonated=tuple(deprotonate), justification=justification.strip(),
        dictionary_formal_charge=component.dictionary_formal_charge)


def compare_with_coordinates(component: ChemicalComponent,
                             observed: dict[str, str]) -> dict[str, object]:
    """Check observed atoms against the dictionary's heavy-atom set.

    `observed` maps atom name to element symbol, as read from a structure.
    Returns a report; the caller decides what is fatal. Names and elements are
    compared, never positions: this establishes chemical identity, not pose.
    """
    expected = set(component.heavy_atom_names)
    present = set(observed)
    mismatched = sorted(
        f"{name}: dictionary {component.elements[name]}, observed {observed[name]}"
        for name in expected & present
        if component.elements[name].upper() != observed[name].upper()
    )
    return {
        "component": component.identifier,
        "dictionary_heavy_atoms": len(expected),
        "observed_atoms": len(present),
        "missing_from_observed": sorted(expected - present),
        "not_in_dictionary": sorted(present - expected),
        "element_mismatches": mismatched,
        "identical_atom_set": not (expected ^ present) and not mismatched,
    }
