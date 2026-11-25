"""Tests for identity-carrying contact and steric-overlap analysis.

Three kinds of test, kept separate on purpose:

1. Pure geometry on hand-placed atoms, where the expected distance and overlap
   are arithmetic a reader can check without running anything.
2. Gemmi-backed synthetic structures, which test the selection and partition
   logic that the real receptor's chain layout makes hazardous.
3. Tests against the real prepared MurC receptor, which are the only ones that
   establish the protein/cofactor conflation is actually handled. They skip
   when the file is absent so the suite still runs on a bare checkout.

Where a number is asserted it was derived independently: Bondi radii are
quoted per element, and every overlap assertion states the arithmetic.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import gemmi  # noqa: E402
import numpy as np  # noqa: E402

import peptide_contacts as pc  # noqa: E402

REAL_RECEPTOR = (
    REPOSITORY_ROOT
    / "project_private"
    / "epicac_receptor_validation"
    / "receptors"
    / "murc_anp_nomn_try1_prepared.pdb"
)


def site(
    atom_name: str,
    element: str,
    position,
    residue_name: str = "ALA",
    seqid: int = 1,
    chain: str = "A",
    is_amino_acid: bool = True,
) -> pc.AtomSite:
    """Build one atom site directly, so a test states its own geometry."""
    return pc.AtomSite(
        author_chain=chain,
        residue_name=residue_name,
        residue_seqid=seqid,
        insertion_code="",
        atom_name=atom_name,
        element=element,
        position=np.asarray(position, dtype=float),
        is_amino_acid=is_amino_acid,
    )


def chain_with_residues(name: str, residues) -> gemmi.Chain:
    """Build a Gemmi chain from (residue_name, seqid, [(atom, element, xyz)])."""
    chain = gemmi.Chain(name)
    for residue_name, seqid, atoms in residues:
        residue = gemmi.Residue()
        residue.name = residue_name
        residue.seqid = gemmi.SeqId(seqid, " ")
        for atom_name, element, position in atoms:
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(element)
            atom.pos = gemmi.Position(*position)
            residue.add_atom(atom)
        chain.add_residue(residue)
    return chain


class RadiusTests(unittest.TestCase):
    def test_bondi_radii_match_the_published_values(self):
        # A. Bondi, J. Phys. Chem. 68 (1964) 441-451.
        self.assertEqual(pc.radius_for_element("C"), 1.70)
        self.assertEqual(pc.radius_for_element("N"), 1.55)
        self.assertEqual(pc.radius_for_element("O"), 1.52)
        self.assertEqual(pc.radius_for_element("P"), 1.80)
        self.assertEqual(pc.radius_for_element("S"), 1.80)

    def test_element_symbol_case_does_not_matter(self):
        self.assertEqual(pc.radius_for_element("c"), pc.radius_for_element("C"))
        self.assertEqual(pc.radius_for_element(" Cl "), 1.75)

    def test_an_untabulated_element_is_fatal_rather_than_defaulted(self):
        # Manganese is the live case: this project omits the MurC manganese
        # ions, and a guessed radius would silently enter every overlap.
        with self.assertRaises(pc.ContactError) as raised:
            pc.radius_for_element("MN")
        self.assertIn("MN", str(raised.exception))

    def test_an_empty_element_symbol_is_fatal(self):
        with self.assertRaises(pc.ContactError):
            pc.radius_for_element("   ")


class OverlapArithmeticTests(unittest.TestCase):
    def test_overlap_is_radius_sum_minus_distance(self):
        # C at origin, N at 2.00 A. Radii 1.70 + 1.55 = 3.25; overlap 1.25.
        first = site("CZ", "C", (0.0, 0.0, 0.0))
        second = site("NE2", "N", (2.0, 0.0, 0.0), residue_name="HIS", seqid=348)
        pair = pc.describe_pair(first, second, 2.0)
        self.assertEqual(pair["radius_sum_angstrom"], 3.25)
        self.assertEqual(pair["overlap_angstrom"], 1.25)
        self.assertEqual(pair["distance_angstrom"], 2.0)

    def test_atoms_further_apart_than_their_radii_have_negative_overlap(self):
        # Two carbons 4.00 A apart: 1.70 + 1.70 - 4.00 = -0.60.
        pair = pc.describe_pair(
            site("CB", "C", (0.0, 0.0, 0.0)),
            site("CB", "C", (4.0, 0.0, 0.0)),
            4.0,
        )
        self.assertEqual(pair["overlap_angstrom"], -0.60)

    def test_nitrogen_oxygen_pair_is_flagged_as_hydrogen_bond_capable(self):
        pair = pc.describe_pair(
            site("NH2", "N", (0.0, 0.0, 0.0)),
            site("OE2", "O", (2.9, 0.0, 0.0), residue_name="GLU"),
            2.9,
        )
        self.assertTrue(pair["potential_hydrogen_bond_pair"])
        # 1.55 + 1.52 - 2.90 = 0.17: a hydrogen bond is a small formal overlap.
        self.assertEqual(pair["overlap_angstrom"], 0.17)

    def test_carbon_pair_is_not_flagged_as_hydrogen_bond_capable(self):
        pair = pc.describe_pair(
            site("CZ", "C", (0.0, 0.0, 0.0)),
            site("CE1", "C", (3.0, 0.0, 0.0)),
            3.0,
        )
        self.assertFalse(pair["potential_hydrogen_bond_pair"])

    def test_sulphur_counts_as_hydrogen_bond_capable(self):
        pair = pc.describe_pair(
            site("SD", "S", (0.0, 0.0, 0.0), residue_name="MET"),
            site("OG", "O", (3.3, 0.0, 0.0), residue_name="SER"),
            3.3,
        )
        self.assertTrue(pair["potential_hydrogen_bond_pair"])


class ContactReportTests(unittest.TestCase):
    def test_a_normal_hydrogen_bond_is_not_called_a_clash(self):
        # N to O at 2.90 A gives overlap 0.17, below the 0.4 A tolerance, so it
        # is a contact but not a clash. This is the behaviour that stops the
        # test reporting every hydrogen bond in the structure.
        peptide = [site("NH2", "N", (0.0, 0.0, 0.0), residue_name="ARG", seqid=2)]
        target = [site("OE2", "O", (2.9, 0.0, 0.0), residue_name="GLU", seqid=352)]
        report = pc.contact_report(peptide, target, "test target")
        self.assertEqual(report["contacts_below_cutoff"], 1)
        self.assertEqual(report["clash_count"], 0)
        self.assertIsNone(report["worst_overlap_angstrom"])

    def test_a_severe_interpenetration_is_called_a_clash(self):
        peptide = [site("CZ", "C", (0.0, 0.0, 0.0), residue_name="ARG", seqid=2)]
        target = [site("NE2", "N", (1.527, 0.0, 0.0), residue_name="HIS", seqid=348)]
        report = pc.contact_report(peptide, target, "test target")
        self.assertEqual(report["clash_count"], 1)
        # 1.70 + 1.55 - 1.527 = 1.723.
        self.assertEqual(report["worst_overlap_angstrom"], 1.723)
        self.assertEqual(report["clashing_peptide_residues"], {"ARG2": 1})

    def test_the_closest_pair_is_reported_with_full_identity(self):
        peptide = [
            site("CA", "C", (0.0, 0.0, 0.0), residue_name="GLY", seqid=5),
            site("CZ", "C", (10.0, 0.0, 0.0), residue_name="ARG", seqid=2),
        ]
        target = [site("NE2", "N", (1.6, 0.0, 0.0), residue_name="HIS", seqid=348)]
        report = pc.contact_report(peptide, target, "test target")
        self.assertEqual(report["minimum_distance_angstrom"], 1.6)
        self.assertEqual(report["closest_pair"]["first"]["label"], "A/GLY5/CA")
        self.assertEqual(report["closest_pair"]["second"]["label"], "A/HIS348/NE2")

    def test_clashing_pairs_are_ordered_worst_overlap_first(self):
        peptide = [
            site("CZ", "C", (0.0, 0.0, 0.0), residue_name="ARG", seqid=2),
            site("NE", "N", (0.0, 5.0, 0.0), residue_name="ARG", seqid=2),
        ]
        target = [
            site("NE2", "N", (2.6, 0.0, 0.0), residue_name="HIS", seqid=348),
            site("CE1", "C", (0.0, 6.5, 0.0), residue_name="HIS", seqid=348),
        ]
        report = pc.contact_report(peptide, target, "test target")
        overlaps = [p["overlap_angstrom"] for p in report["clashing_pairs"]]
        self.assertEqual(overlaps, sorted(overlaps, reverse=True))

    def test_the_tolerance_changes_what_counts_as_a_clash(self):
        # N to O at 2.90 A: overlap 0.17. Clash at tolerance 0.1, not at 0.4.
        peptide = [site("NH2", "N", (0.0, 0.0, 0.0), residue_name="ARG", seqid=2)]
        target = [site("OE2", "O", (2.9, 0.0, 0.0), residue_name="GLU", seqid=352)]
        strict = pc.contact_report(
            peptide, target, "t", clash_overlap_tolerance=0.1
        )
        lenient = pc.contact_report(
            peptide, target, "t", clash_overlap_tolerance=0.4
        )
        self.assertEqual(strict["clash_count"], 1)
        self.assertEqual(lenient["clash_count"], 0)

    def test_an_atom_beyond_the_contact_cutoff_is_not_examined(self):
        peptide = [site("CZ", "C", (0.0, 0.0, 0.0))]
        target = [site("CE1", "C", (50.0, 0.0, 0.0))]
        report = pc.contact_report(peptide, target, "t")
        self.assertEqual(report["contacts_below_cutoff"], 0)
        self.assertEqual(report["clash_count"], 0)
        self.assertEqual(report["minimum_distance_angstrom"], 50.0)

    def test_an_empty_target_gives_undefined_distances_not_zero(self):
        report = pc.contact_report([site("CA", "C", (0, 0, 0))], [], "nothing")
        self.assertIsNone(report["minimum_distance_angstrom"])
        self.assertIsNone(report["clash_count"])
        self.assertIn("undefined comparison", report["note"])

    def test_an_empty_peptide_also_gives_undefined_distances(self):
        report = pc.contact_report([], [site("CA", "C", (0, 0, 0))], "something")
        self.assertIsNone(report["minimum_distance_angstrom"])

    def test_the_target_description_is_carried_into_the_record(self):
        report = pc.contact_report(
            [site("CA", "C", (0, 0, 0))],
            [site("CA", "C", (3, 0, 0))],
            "ANP cofactor, author chain A",
        )
        self.assertEqual(report["target"], "ANP cofactor, author chain A")

    def test_a_negative_tolerance_is_rejected(self):
        with self.assertRaises(pc.ContactError):
            pc.contact_report([], [], "t", clash_overlap_tolerance=-1.0)

    def test_a_non_positive_cutoff_is_rejected(self):
        with self.assertRaises(pc.ContactError):
            pc.contact_report([], [], "t", contact_cutoff=0.0)

    def test_truncation_is_stated_when_pairs_are_dropped(self):
        peptide = [
            site("C%d" % i, "C", (0.0, 0.0, float(i) * 0.01)) for i in range(4)
        ]
        target = [
            site("N%d" % i, "N", (1.5, 0.0, float(i) * 0.01)) for i in range(4)
        ]
        report = pc.contact_report(
            peptide, target, "t", maximum_reported_pairs=3
        )
        self.assertEqual(report["clash_count"], 16)
        self.assertEqual(len(report["clashing_pairs"]), 3)
        self.assertTrue(report["clashing_pairs_truncated"])


class PartitionTests(unittest.TestCase):
    """The protein/cofactor split, which must never use the chain letter."""

    def test_anp_is_not_classified_as_an_amino_acid(self):
        self.assertFalse(pc.is_amino_acid_residue("ANP"))
        self.assertFalse(pc.is_amino_acid_residue("UMA"))
        self.assertFalse(pc.is_amino_acid_residue("HOH"))

    def test_standard_residues_are_classified_as_amino_acids(self):
        for name in ("ALA", "ARG", "GLY", "HIS", "PHE", "SER", "THR", "MET"):
            self.assertTrue(pc.is_amino_acid_residue(name), name)

    def test_an_unknown_component_is_grouped_with_the_cofactors_by_name(self):
        # An unrecognised residue must not be counted as protein; it must show
        # up by name so a reader can see what entered the comparison.
        chain = chain_with_residues(
            "A",
            [
                ("ALA", 1, [("CA", "C", (0, 0, 0))]),
                ("ZZZ", 2, [("X1", "C", (5, 0, 0))]),
            ],
        )
        protein, other, census = pc.partition_sites(pc.collect_sites(chain))
        self.assertEqual(len(protein), 1)
        self.assertEqual(len(other), 1)
        self.assertEqual(census.non_amino_acid_components, {"ZZZ": 1})

    def test_protein_and_cofactor_sharing_one_chain_letter_are_separated(self):
        # This is the real receptor's layout in miniature: ANP under chain A
        # alongside the protein. A chain-letter selection would pool them.
        chain = chain_with_residues(
            "A",
            [
                ("ALA", 1, [("CA", "C", (0, 0, 0)), ("CB", "C", (1.5, 0, 0))]),
                ("HIS", 348, [("NE2", "N", (3, 0, 0))]),
                ("ANP", 603, [("PG", "P", (6, 0, 0)), ("O2G", "O", (7, 0, 0))]),
            ],
        )
        protein, other, census = pc.partition_sites(pc.collect_sites(chain))
        self.assertEqual(len(protein), 3)
        self.assertEqual(len(other), 2)
        self.assertEqual(census.amino_acid_atoms, 3)
        self.assertEqual(census.non_amino_acid_atoms, 2)
        self.assertEqual(census.non_amino_acid_components, {"ANP": 2})
        # Both sets carry the same chain letter, which is why it cannot be the
        # discriminator.
        self.assertEqual({s.author_chain for s in protein + other}, {"A"})

    def test_sites_for_component_selects_by_residue_name(self):
        chain = chain_with_residues(
            "A",
            [
                ("ANP", 603, [("PG", "P", (0, 0, 0))]),
                ("UMA", 602, [("C1", "C", (5, 0, 0))]),
            ],
        )
        _, other, _ = pc.partition_sites(pc.collect_sites(chain))
        self.assertEqual(len(pc.sites_for_component(other, "ANP")), 1)
        self.assertEqual(len(pc.sites_for_component(other, "UMA")), 1)
        self.assertEqual(len(pc.sites_for_component(other, "MN")), 0)

    def test_hydrogens_are_excluded_by_default_and_included_on_request(self):
        chain = chain_with_residues(
            "A",
            [("ALA", 1, [("CA", "C", (0, 0, 0)), ("HA", "H", (1, 0, 0))])],
        )
        self.assertEqual(len(pc.collect_sites(chain)), 1)
        self.assertEqual(len(pc.collect_sites(chain, include_hydrogen=True)), 2)

    def test_collect_model_sites_restricts_to_the_named_author_chains(self):
        model = gemmi.Model("1")
        model.add_chain(chain_with_residues("A", [("ALA", 1, [("CA", "C", (0, 0, 0))])]))
        model.add_chain(chain_with_residues("B", [("ALA", 1, [("CA", "C", (9, 0, 0))])]))
        both = pc.collect_model_sites(model)
        just_a = pc.collect_model_sites(model, author_chains=["A"])
        self.assertEqual(len(both), 2)
        self.assertEqual(len(just_a), 1)
        self.assertEqual(just_a[0].author_chain, "A")

    def test_a_transform_is_applied_and_recorded_in_the_frame_field(self):
        chain = chain_with_residues("A", [("ALA", 1, [("CA", "C", (1, 2, 3))])])
        transform = gemmi.Transform()
        transform.vec.x = 10.0
        moved = pc.collect_sites(chain, transform=transform)
        np.testing.assert_allclose(moved[0].position, [11.0, 2.0, 3.0])
        self.assertIn("transformed", moved[0].frame)
        unmoved = pc.collect_sites(chain)
        self.assertEqual(unmoved[0].frame, "as read")


class ClashSummaryTests(unittest.TestCase):
    def test_a_side_chain_only_clash_set_is_identified_as_such(self):
        peptide = [site("CZ", "C", (0.0, 0.0, 0.0), residue_name="ARG", seqid=2)]
        target = [site("NE2", "N", (1.6, 0.0, 0.0), residue_name="HIS", seqid=348)]
        report = pc.contact_report(peptide, target, "protein")
        summary = pc.clash_summary([report])
        self.assertTrue(summary["side_chain_only"])
        self.assertFalse(summary["peptide_backbone_involved"])
        self.assertEqual(summary["side_chain_clashes"], 1)
        self.assertEqual(summary["backbone_clashes"], 0)
        self.assertIsNone(summary["worst_backbone_overlap_angstrom"])

    def test_a_backbone_clash_defeats_the_side_chain_only_verdict(self):
        # A glycine has no side chain beyond CA, so any glycine clash is a
        # backbone clash by construction.
        peptide = [site("O", "O", (0.0, 0.0, 0.0), residue_name="GLY", seqid=5)]
        target = [site("CA", "C", (2.0, 0.0, 0.0), residue_name="GLY", seqid=28)]
        summary = pc.clash_summary([pc.contact_report(peptide, target, "protein")])
        self.assertFalse(summary["side_chain_only"])
        self.assertTrue(summary["peptide_backbone_involved"])
        self.assertEqual(summary["backbone_clashes"], 1)
        # 1.52 + 1.70 - 2.00 = 1.22.
        self.assertEqual(summary["worst_backbone_overlap_angstrom"], 1.22)

    def test_backbone_and_side_chain_clashes_are_counted_separately(self):
        peptide = [
            site("O", "O", (0.0, 0.0, 0.0), residue_name="GLY", seqid=5),
            site("CZ", "C", (0.0, 10.0, 0.0), residue_name="ARG", seqid=2),
        ]
        target = [
            site("CA", "C", (2.4, 0.0, 0.0), residue_name="GLY", seqid=28),
            site("NE2", "N", (1.6, 10.0, 0.0), residue_name="HIS", seqid=348),
        ]
        summary = pc.clash_summary([pc.contact_report(peptide, target, "protein")])
        self.assertEqual(summary["backbone_clashes"], 1)
        self.assertEqual(summary["side_chain_clashes"], 1)
        self.assertEqual(summary["total_clashes"], 2)
        self.assertEqual(
            summary["peptide_residues_involved"], {"ARG2": 1, "GLY5": 1}
        )

    def test_per_target_counts_are_not_collapsed_together(self):
        peptide = [site("CD", "C", (0.0, 0.0, 0.0), residue_name="ARG", seqid=2)]
        protein = pc.contact_report(
            peptide,
            [site("NE2", "N", (1.6, 0.0, 0.0), residue_name="HIS", seqid=348)],
            "MurC protein",
        )
        cofactor = pc.contact_report(
            peptide,
            [
                site(
                    "O2G", "O", (1.58, 0.0, 0.0), residue_name="ANP",
                    seqid=603, is_amino_acid=False,
                )
            ],
            "ANP cofactor",
        )
        summary = pc.clash_summary([protein, cofactor])
        self.assertEqual(
            summary["clashes_by_target"], {"MurC protein": 1, "ANP cofactor": 1}
        )
        self.assertEqual(summary["total_clashes"], 2)

    def test_an_undefined_comparison_is_named_and_not_counted_as_zero(self):
        defined = pc.contact_report(
            [site("CZ", "C", (0.0, 0.0, 0.0))],
            [site("NE2", "N", (1.6, 0.0, 0.0))],
            "protein",
        )
        undefined = pc.contact_report(
            [site("CZ", "C", (0.0, 0.0, 0.0))], [], "ANP cofactor"
        )
        summary = pc.clash_summary([defined, undefined])
        self.assertEqual(summary["undefined_comparisons"], ["ANP cofactor"])
        self.assertEqual(summary["total_clashes"], 1)

    def test_hydrogen_bond_capable_clashes_are_counted_not_removed(self):
        peptide = [site("NH2", "N", (0.0, 0.0, 0.0), residue_name="ARG", seqid=2)]
        target = [site("OE2", "O", (2.5, 0.0, 0.0), residue_name="GLU", seqid=352)]
        summary = pc.clash_summary([pc.contact_report(peptide, target, "protein")])
        self.assertEqual(summary["total_clashes"], 1)
        self.assertEqual(summary["potential_hydrogen_bond_pairs"], 1)


@unittest.skipUnless(REAL_RECEPTOR.is_file(), "prepared receptor not present")
class RealReceptorTests(unittest.TestCase):
    """The conflation hazard, measured on the actual prepared receptor.

    These are the tests that matter: the synthetic cases show the partition
    logic works, but only the real file establishes that this receptor's
    layout is the hazardous one.
    """

    @classmethod
    def setUpClass(cls):
        structure = gemmi.read_structure(str(REAL_RECEPTOR))
        structure.setup_entities()
        cls.chain_a = [c for c in structure[0] if c.name == "A"][0]
        cls.sites = pc.collect_sites(cls.chain_a)
        cls.protein, cls.other, cls.census = pc.partition_sites(cls.sites)

    def test_the_receptor_stores_anp_under_the_protein_chain_letter(self):
        # The premise of this whole module. If this ever fails, the receptor
        # layout changed and the conflation warning needs revisiting.
        residue_names = {r.name for r in self.chain_a}
        self.assertIn("ANP", residue_names)
        self.assertIn("HIS", residue_names)

    def test_the_partition_separates_3547_protein_atoms_from_31_anp_atoms(self):
        # Independently cross-checked: the receptor manifest records 3578
        # heavy atoms in total, and 3547 + 31 = 3578.
        self.assertEqual(len(self.protein), 3547)
        self.assertEqual(len(self.other), 31)
        self.assertEqual(self.census.non_amino_acid_components, {"ANP": 31})
        self.assertEqual(len(self.protein) + len(self.other), 3578)

    def test_no_anp_atom_reaches_the_protein_set(self):
        self.assertNotIn("ANP", {s.residue_name for s in self.protein})

    def test_a_chain_letter_selection_would_pool_protein_and_cofactor(self):
        # The defect this module exists to prevent, demonstrated: selecting
        # every atom of chain A yields protein and cofactor together.
        self.assertEqual(len(self.sites), 3578)
        self.assertEqual(len({s.author_chain for s in self.sites}), 1)

    def test_every_receptor_element_has_a_tabulated_radius(self):
        # If the receptor ever retains manganese, this fails rather than
        # letting a guessed radius into the overlap arithmetic.
        for atom_site in self.sites:
            pc.radius_for_element(atom_site.element)


if __name__ == "__main__":
    unittest.main()
