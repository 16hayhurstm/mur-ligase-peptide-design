"""Identity-carrying contact and steric-overlap analysis for a placed peptide.

This module answers one question about a peptide that has already been placed
in a receptor frame: which atoms are too close to which, and to what. It is
deliberately separate from `peptide_geometry`, which reports distances between
bare coordinate arrays. A coordinate array cannot say that a clash belongs to
one arginine side chain, and that is exactly the distinction a relaxation
decision turns on.

Why residue class, and never the chain letter
---------------------------------------------
In the prepared ANP-containing MurC receptor the cofactor is stored as

    HETATM 3549  PG  ANP A 603 ...

that is, under **the same author chain letter as the protein**. Selecting "the
atoms of chain A" therefore yields protein *and* cofactor, and a peptide-to-
protein contact count built that way silently includes peptide-to-ANP
contacts. For `spec_0574` the difference is measurable: 4 peptide-to-protein
pairs below 2.0 A, 1 peptide-to-ANP pair, and 5 if the two are pooled.

Every selection here is therefore made by **residue classification**, using
Gemmi's chemical-component table to ask whether a residue is an amino acid,
and the partition census is reported in the output so that a pooled selection
cannot pass unnoticed. The chain letter is recorded as provenance, never used
as a discriminator.

Contacts, overlaps and what neither establishes
-----------------------------------------------
Two quantities are reported, because they answer different questions.

A **contact** is a heavy-atom pair separated by less than a stated cutoff. It
is a geometric proximity and nothing else: it is not an interaction, not a
hydrogen bond and not evidence of binding.

A **steric overlap** is `r_i + r_j - d`, the amount by which two atoms'
van der Waals spheres interpenetrate. This is the quantity relevant to whether
a pose is physically admissible, and it uses element-dependent radii, so a
carbon-carbon approach and a nitrogen-oxygen approach at the same distance are
judged differently.

Hydrogen bonds are not excluded by sleight of hand
--------------------------------------------------
A nitrogen and an oxygen in a hydrogen bond sit about 2.8-3.0 A apart, while
their Bondi radii sum to 3.07 A. A hydrogen bond is therefore a small formal
overlap, around 0.1-0.3 A, and a naive overlap test would report every
hydrogen bond in a structure as a clash.

Two things are done about this, and neither is a silent exemption. First
`clash_overlap_tolerance` defaults to 0.4 A, above the hydrogen-bonding range
and below any real steric conflict. Second, every reported pair carries a
`potential_hydrogen_bond_pair` flag, set when both atoms are nitrogen, oxygen
or sulphur, so a reader can see which flagged pairs might be explained that
way rather than having them removed from the record. Geometry alone cannot
distinguish a hydrogen bond from a clash between the same two elements: that
needs hydrogen positions and angles, which are not modelled here.

Hydrogens
---------
Hydrogen atoms are excluded by default. BoltzGen refolded output carries no
hydrogens, and the deposited receptor carries none either, so a hydrogen-
inclusive comparison would be asymmetric and meaningless. `include_hydrogen`
exists for a structure where hydrogens are present on both sides.

Limitations
-----------
- Van der Waals radii are a single number per element. They ignore
  hybridisation, charge and bonding environment.
- No symmetry copies are generated, and no periodic images are considered.
- A clash is reported, never repaired. Nothing here moves an atom.
- The ANP representation in the receptor is recorded in this project as
  chemically unvalidated. Distances to it are geometric measurements against
  that unvalidated model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gemmi
import numpy as np

# Bondi van der Waals radii in Angstroms, A. Bondi, J. Phys. Chem. 68 (1964)
# 441-451, with the hydrogen value 1.20 A as given there. These are the radii
# used for the steric-overlap test. They are tabulated per element and so
# carry no information about hybridisation, formal charge or bonding
# environment; an aromatic carbon and an aliphatic carbon get the same radius.
VAN_DER_WAALS_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
}

# Elements treated as capable of hydrogen bonding for the purpose of flagging a
# reported overlap as possibly a hydrogen bond rather than a steric conflict.
HYDROGEN_BONDING_ELEMENTS = frozenset({"N", "O", "S"})

# Default overlap, in Angstroms, above which a pair is called a clash. Chosen
# to sit above the formal overlap implied by a hydrogen bond (about 0.1-0.3 A
# for N/O pairs) and below any genuine steric conflict.
DEFAULT_CLASH_OVERLAP_TOLERANCE = 0.4

# Default contact cutoff in Angstroms, strictly less-than, matching the
# convention used by `map_pocket.py` and `peptide_geometry.py`.
DEFAULT_CONTACT_CUTOFF = 4.0


class ContactError(ValueError):
    """Raised when a contact analysis cannot be carried out as specified."""


@dataclass
class AtomSite:
    """One heavy atom with the identity needed to name it in a report.

    `position` is a length-3 array in Angstroms, in whatever frame the caller
    supplied. Where a transform has been applied, `frame` records that, so a
    record cannot be read as being in the original frame.
    """

    author_chain: str
    residue_name: str
    residue_seqid: int
    insertion_code: str
    atom_name: str
    element: str
    position: np.ndarray
    is_amino_acid: bool
    frame: str = "as read"

    @property
    def label(self) -> str:
        """A compact human-readable identity, e.g. ``A/HIS348/NE2``."""
        icode = self.insertion_code.strip()
        return (
            f"{self.author_chain}/{self.residue_name}"
            f"{self.residue_seqid}{icode}/{self.atom_name}"
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "author_chain": self.author_chain,
            "residue_name": self.residue_name,
            "residue_seqid": self.residue_seqid,
            "insertion_code": self.insertion_code.strip(),
            "atom_name": self.atom_name,
            "element": self.element,
            "is_amino_acid": self.is_amino_acid,
            "frame": self.frame,
        }


@dataclass
class ResiduePartition:
    """The amino-acid / non-amino-acid census of a selection of atoms.

    This exists so that a conflated selection is visible in the output rather
    than inferred. `non_amino_acid_components` names what was found, so a
    retained cofactor, an ion or a water appears by name.
    """

    amino_acid_atoms: int = 0
    non_amino_acid_atoms: int = 0
    non_amino_acid_components: Dict[str, int] = field(default_factory=dict)
    author_chains: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "amino_acid_atoms": self.amino_acid_atoms,
            "non_amino_acid_atoms": self.non_amino_acid_atoms,
            "non_amino_acid_components": dict(
                sorted(self.non_amino_acid_components.items())
            ),
            "author_chains": sorted(self.author_chains),
            "note": (
                "Selection is by residue classification, not by chain letter. "
                "In the prepared receptor ANP shares author chain A with the "
                "protein, so a chain-letter selection would pool them."
            ),
        }


def is_amino_acid_residue(residue_name: str) -> bool:
    """Report whether a residue name is a tabulated amino acid.

    This is the single discriminator used to separate protein from everything
    else. An unrecognised name is *not* an amino acid, so an unknown component
    is grouped with the cofactors and appears by name in the partition census
    rather than being counted as protein.
    """
    info = gemmi.find_tabulated_residue(residue_name)
    return bool(info is not None and info.is_amino_acid())


def is_hydrogen(element: str) -> bool:
    return element.upper() in ("H", "D")


def radius_for_element(element: str) -> float:
    """Return the Bondi van der Waals radius for an element symbol.

    An element with no tabulated radius is fatal rather than defaulted: a
    guessed radius would silently change every overlap it takes part in, and
    the manganese ions this project omits are exactly the kind of atom whose
    effective radius is not a settled single number.
    """
    key = element.strip().upper()
    if not key:
        raise ContactError("Atom has an empty element symbol")
    if key not in VAN_DER_WAALS_RADII:
        raise ContactError(
            f"No van der Waals radius tabulated for element {element!r}. "
            "Add it explicitly with a cited source rather than defaulting."
        )
    return VAN_DER_WAALS_RADII[key]


def collect_sites(
    chain: gemmi.Chain,
    transform: Optional[gemmi.Transform] = None,
    include_hydrogen: bool = False,
) -> List[AtomSite]:
    """Build identity-carrying atom sites for one chain.

    `transform`, where given, is applied to every position, and the resulting
    sites are labelled as being in the transformed frame. The transform is
    applied uniformly to the whole chain: this function cannot be used to fit
    part of a chain independently.
    """
    frame = "as read" if transform is None else "transformed to reference frame"
    sites: List[AtomSite] = []
    for residue in chain:
        classified = is_amino_acid_residue(residue.name)
        for atom in residue:
            element = atom.element.name
            if not include_hydrogen and is_hydrogen(element):
                continue
            position = transform.apply(atom.pos) if transform is not None else atom.pos
            sites.append(
                AtomSite(
                    author_chain=chain.name,
                    residue_name=residue.name,
                    residue_seqid=residue.seqid.num,
                    insertion_code=residue.seqid.icode,
                    atom_name=atom.name,
                    element=element,
                    position=np.asarray(
                        [position.x, position.y, position.z], dtype=float
                    ),
                    is_amino_acid=classified,
                    frame=frame,
                )
            )
    return sites


def collect_model_sites(
    model: gemmi.Model,
    author_chains: Optional[Sequence[str]] = None,
    transform: Optional[gemmi.Transform] = None,
    include_hydrogen: bool = False,
) -> List[AtomSite]:
    """Collect sites across a model, optionally restricted to author chains.

    `author_chains` selects which crystallographic copy is under assessment.
    It is not a protein/ligand discriminator and must not be used as one; see
    `partition_sites`.
    """
    wanted = None if author_chains is None else {str(c) for c in author_chains}
    sites: List[AtomSite] = []
    for chain in model:
        if wanted is not None and chain.name not in wanted:
            continue
        sites.extend(collect_sites(chain, transform, include_hydrogen))
    return sites


def partition_sites(
    sites: Sequence[AtomSite],
) -> Tuple[List[AtomSite], List[AtomSite], ResiduePartition]:
    """Split sites into amino-acid and non-amino-acid sets, with a census.

    Returns `(protein_sites, non_protein_sites, census)`. The census is the
    evidence that the split happened and what it found, and is meant to be
    written into any record built from these sets.
    """
    protein: List[AtomSite] = []
    other: List[AtomSite] = []
    census = ResiduePartition()
    chains = set()
    for site in sites:
        chains.add(site.author_chain)
        if site.is_amino_acid:
            protein.append(site)
            census.amino_acid_atoms += 1
        else:
            other.append(site)
            census.non_amino_acid_atoms += 1
            census.non_amino_acid_components[site.residue_name] = (
                census.non_amino_acid_components.get(site.residue_name, 0) + 1
            )
    census.author_chains = sorted(chains)
    return protein, other, census


def sites_for_component(
    sites: Sequence[AtomSite], component: str
) -> List[AtomSite]:
    """Select the sites of one named chemical component, e.g. ``ANP``."""
    return [s for s in sites if s.residue_name == component]


def positions_of(sites: Sequence[AtomSite]) -> np.ndarray:
    """Return an (N, 3) coordinate array for a list of sites."""
    if not sites:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([s.position for s in sites], dtype=float)


def _pair_distances(
    first: Sequence[AtomSite], second: Sequence[AtomSite]
) -> np.ndarray:
    a = positions_of(first)
    b = positions_of(second)
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 0), dtype=float)
    deltas = a[:, None, :] - b[None, :, :]
    return np.sqrt((deltas * deltas).sum(axis=-1))


def describe_pair(
    first: AtomSite, second: AtomSite, distance: float
) -> Dict[str, object]:
    """Describe one atom pair, with radii, overlap and a hydrogen-bond flag."""
    radius_first = radius_for_element(first.element)
    radius_second = radius_for_element(second.element)
    overlap = radius_first + radius_second - distance
    polar = (
        first.element.upper() in HYDROGEN_BONDING_ELEMENTS
        and second.element.upper() in HYDROGEN_BONDING_ELEMENTS
    )
    return {
        "first": first.as_dict(),
        "second": second.as_dict(),
        "distance_angstrom": round(float(distance), 4),
        "radius_sum_angstrom": round(radius_first + radius_second, 4),
        "overlap_angstrom": round(float(overlap), 4),
        "potential_hydrogen_bond_pair": polar,
    }


def contact_report(
    peptide_sites: Sequence[AtomSite],
    target_sites: Sequence[AtomSite],
    target_description: str,
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
    clash_overlap_tolerance: float = DEFAULT_CLASH_OVERLAP_TOLERANCE,
    maximum_reported_pairs: int = 50,
) -> Dict[str, object]:
    """Compare a peptide against one target atom set and report what is close.

    `target_description` is carried into the record so a reader can tell a
    peptide-to-protein report from a peptide-to-ANP report without inferring
    it from the atom counts.

    Contacts use `contact_cutoff` strictly less-than. Clashes use the van der
    Waals overlap exceeding `clash_overlap_tolerance`, also strictly greater-
    than. Both sets of pairs are reported, worst overlap first, truncated at
    `maximum_reported_pairs` with the truncation stated.

    An empty target set yields a record whose distances are `None` rather than
    zero, because no distance is defined; this is not a pass.
    """
    if contact_cutoff <= 0:
        raise ContactError("contact_cutoff must be positive")
    if clash_overlap_tolerance < 0:
        raise ContactError("clash_overlap_tolerance must not be negative")

    record: Dict[str, object] = {
        "target": target_description,
        "peptide_heavy_atoms": len(peptide_sites),
        "target_heavy_atoms": len(target_sites),
        "contact_cutoff_angstrom": contact_cutoff,
        "clash_overlap_tolerance_angstrom": clash_overlap_tolerance,
        "radii_source": "Bondi 1964 van der Waals radii, per element",
    }
    distances = _pair_distances(peptide_sites, target_sites)
    if distances.size == 0:
        record.update(
            {
                "minimum_distance_angstrom": None,
                "closest_pair": None,
                "contacts_below_cutoff": None,
                "clashing_pairs": None,
                "clash_count": None,
                "worst_overlap_angstrom": None,
                "clashing_peptide_residues": None,
                "note": (
                    "One of the atom sets is empty, so no distance is defined. "
                    "This is an undefined comparison, not an absence of clashes."
                ),
            }
        )
        return record

    flat_index = int(np.argmin(distances))
    i, j = np.unravel_index(flat_index, distances.shape)
    minimum = float(distances[i, j])

    contact_indices = np.argwhere(distances < contact_cutoff)
    clashes: List[Dict[str, object]] = []
    for pi, pj in contact_indices:
        pair = describe_pair(
            peptide_sites[int(pi)], target_sites[int(pj)], float(distances[pi, pj])
        )
        if pair["overlap_angstrom"] > clash_overlap_tolerance:
            clashes.append(pair)
    clashes.sort(key=lambda p: p["overlap_angstrom"], reverse=True)

    residue_counts: Dict[str, int] = {}
    for pair in clashes:
        first = pair["first"]
        key = (
            f"{first['residue_name']}{first['residue_seqid']}"
            f"{first['insertion_code']}"
        )
        residue_counts[key] = residue_counts.get(key, 0) + 1

    record.update(
        {
            "minimum_distance_angstrom": round(minimum, 4),
            "closest_pair": describe_pair(
                peptide_sites[int(i)], target_sites[int(j)], minimum
            ),
            "contacts_below_cutoff": int(len(contact_indices)),
            "clash_count": len(clashes),
            "worst_overlap_angstrom": (
                clashes[0]["overlap_angstrom"] if clashes else None
            ),
            "clashing_pairs": clashes[:maximum_reported_pairs],
            "clashing_pairs_truncated": len(clashes) > maximum_reported_pairs,
            "clashing_peptide_residues": dict(sorted(residue_counts.items())),
        }
    )
    return record


def clash_summary(reports: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate several contact reports into one decision-shaped summary.

    The aggregate deliberately does **not** collapse the targets together: it
    reports the per-target clash counts alongside the total, so the
    protein/cofactor distinction that this module exists to preserve survives
    into the summary.

    `peptide_residues_involved` is the union across targets.

    The side-chain / backbone split is reported with counts and worst overlaps
    rather than as a single boolean, because the remedy depends on the
    magnitude as well as the location. A clash set that is entirely on side
    chains is a candidate for a side-chain-only restrained relaxation, which
    leaves the predicted backbone untouched. A single modest backbone overlap
    alongside many side-chain ones is a different situation from a backbone
    buried in the receptor, and a boolean cannot tell them apart.

    `potential_hydrogen_bond_pairs` counts flagged pairs where both atoms can
    hydrogen bond. These are the pairs most likely to be real interactions
    rather than steric conflicts, and they are counted, not removed.
    """
    per_target: Dict[str, object] = {}
    total = 0
    undefined: List[str] = []
    residues: Dict[str, int] = {}
    backbone_atoms = {"N", "CA", "C", "O", "OXT"}
    backbone_pairs: List[Dict[str, object]] = []
    side_chain_count = 0
    polar_count = 0
    worst_overall = None
    for report in reports:
        name = str(report["target"])
        count = report.get("clash_count")
        per_target[name] = count
        if count is None:
            undefined.append(name)
            continue
        total += int(count)
        for pair in report.get("clashing_pairs") or []:
            first = pair["first"]
            key = (
                f"{first['residue_name']}{first['residue_seqid']}"
                f"{first['insertion_code']}"
            )
            residues[key] = residues.get(key, 0) + 1
            overlap = float(pair["overlap_angstrom"])
            if worst_overall is None or overlap > worst_overall:
                worst_overall = overlap
            if pair["potential_hydrogen_bond_pair"]:
                polar_count += 1
            if first["atom_name"] in backbone_atoms:
                backbone_pairs.append(
                    {
                        "target": name,
                        "peptide_atom": first["label"],
                        "target_atom": pair["second"]["label"],
                        "distance_angstrom": pair["distance_angstrom"],
                        "overlap_angstrom": pair["overlap_angstrom"],
                    }
                )
            else:
                side_chain_count += 1
    backbone_pairs.sort(key=lambda p: p["overlap_angstrom"], reverse=True)
    return {
        "clashes_by_target": per_target,
        "total_clashes": total,
        "undefined_comparisons": undefined,
        "peptide_residues_involved": dict(sorted(residues.items())),
        "side_chain_clashes": side_chain_count,
        "backbone_clashes": len(backbone_pairs),
        "backbone_clashing_pairs": backbone_pairs,
        "worst_overlap_angstrom": worst_overall,
        "worst_backbone_overlap_angstrom": (
            backbone_pairs[0]["overlap_angstrom"] if backbone_pairs else None
        ),
        "peptide_backbone_involved": bool(backbone_pairs),
        "side_chain_only": bool(total > 0 and not backbone_pairs),
        "potential_hydrogen_bond_pairs": polar_count,
        "interpretation": (
            "Clash counts are geometric, not energetic. A side-chain-only "
            "clash set is a candidate for a restrained relaxation that holds "
            "the predicted backbone fixed. A clash on peptide backbone atoms "
            "cannot be resolved that way: relieving it means moving the "
            "predicted backbone, which changes the hypothesis under test, so "
            "its magnitude is reported for an explicit decision rather than "
            "being absorbed into a pass or a fail."
        ),
    }
