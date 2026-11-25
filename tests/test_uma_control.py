"""Tests for ligand chemistry, symmetry-aware RMSD and redocking analysis.

Scope, and what these tests do not establish
--------------------------------------------
Software correctness only. Fixtures are small hand-written molecules whose
symmetry and RMSD can be worked out on paper, plus a minimal mmCIF and a
minimal component-dictionary entry.

Three kinds of check, deliberately separated:

- **pure**: the molecular graph, resonance normalisation, automorphisms,
  in-place RMSD, pose-file parsing and the recovery logic. Nothing installed
  is needed beyond NumPy.
- **Gemmi-backed**: dictionary reading, reference extraction and ligand
  building against real files. Skipped if Gemmi is absent.
- **no real-tool checks at all**: there is no stub for `mk_prepare_ligand.py`
  here. Hydrogen placement, atom typing and partial charges are produced by a
  chemistry toolkit, and nothing in this file exercises or imitates one. The
  `check` stage is tested against a hand-written PDBQT fixture, which proves
  the verification logic and says nothing about the preparation.

No test here establishes that a pose is correct, that a protonation state is
right, or anything biological.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import chemical_graph as graph_module  # noqa: E402

LIGAND_REFERENCE = ROOT / "python" / "ligand_reference.py"
PREPARE_LIGAND = ROOT / "python" / "prepare_ligand.py"
ANALYSE = ROOT / "python" / "analyse_redocking.py"

try:
    import gemmi  # noqa: F401
    import chemical_component  # noqa: F401
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False


def acetate_graph(resonance_split: bool = True) -> graph_module.MolecularGraph:
    """CH3-C(=O)-O(H): a carboxylate with two terminal oxygens.

    With `resonance_split` the two oxygens carry different bond orders, as a
    connection table records them; normalisation should then make them
    equivalent.
    """
    order = "DOUB" if resonance_split else "SING"
    return graph_module.graph_from_bond_names(
        ["C1", "C2", "O1", "O2"], ["C", "C", "O", "O"],
        [("C1", "C2", "SING"), ("C2", "O1", order), ("C2", "O2", "SING")])


class GraphTests(unittest.TestCase):
    def test_duplicate_atom_names_are_refused(self):
        with self.assertRaises(graph_module.CorrespondenceError):
            graph_module.MolecularGraph(("C1", "C1"), ("C", "C"), {})

    def test_mismatched_elements_length_is_refused(self):
        with self.assertRaises(graph_module.CorrespondenceError):
            graph_module.MolecularGraph(("C1", "C2"), ("C",), {})

    def test_bond_naming_an_unknown_atom_is_refused(self):
        with self.assertRaises(graph_module.CorrespondenceError):
            graph_module.graph_from_bond_names(
                ["C1"], ["C"], [("C1", "O9", "SING")])


class ResonanceTests(unittest.TestCase):
    def test_terminal_oxygens_are_normalised_and_recorded(self):
        normalised, record = graph_module.normalise_resonance(
            acetate_graph(), hydrogen_bearing=[])
        self.assertEqual(len(record), 1)
        self.assertIn("O1", record[0])
        self.assertIn("O2", record[0])
        orders = {normalised.bonds[frozenset((1, 2))],
                  normalised.bonds[frozenset((1, 3))]}
        self.assertEqual(orders, {graph_module.RESONANCE_ORDER})

    def test_normalisation_creates_the_symmetry(self):
        raw = acetate_graph()
        self.assertEqual(len(graph_module.automorphisms(raw)), 1)
        normalised, _ = graph_module.normalise_resonance(raw, hydrogen_bearing=[])
        permutations = graph_module.automorphisms(normalised)
        self.assertEqual(len(permutations), 2)
        self.assertEqual(graph_module.symmetry_classes(normalised, permutations),
                         [["O1", "O2"]])

    def test_a_single_terminal_oxygen_is_not_normalised(self):
        """An ester or ketone oxygen has no partner to be equivalent to."""
        single = graph_module.graph_from_bond_names(
            ["C1", "C2", "O1"], ["C", "C", "O"],
            [("C1", "C2", "SING"), ("C2", "O1", "DOUB")])
        _, record = graph_module.normalise_resonance(single, hydrogen_bearing=[])
        self.assertEqual(record, [])

    def test_a_bridging_oxygen_is_not_normalised(self):
        """Only degree-one oxygens are terminal; a bridge is not."""
        bridged = graph_module.graph_from_bond_names(
            ["P1", "O1", "O2", "C1"], ["P", "O", "O", "C"],
            [("P1", "O1", "DOUB"), ("P1", "O2", "SING"), ("O2", "C1", "SING")])
        _, record = graph_module.normalise_resonance(bridged, hydrogen_bearing=[])
        self.assertEqual(record, [])


class ResonanceRestrictionTests(unittest.TestCase):
    """Negative tests: equivalence must be refused where it is not justified."""

    def test_a_neutral_carboxylic_acid_is_not_normalised(self):
        """O-H and C=O are chemically different; swapping them is a real change."""
        acid = acetate_graph()
        normalised, record = graph_module.normalise_resonance(
            acid, hydrogen_bearing=["O2"])
        self.assertEqual(len(graph_module.automorphisms(normalised)), 1)
        self.assertTrue(any("NOT normalised" in line for line in record))
        self.assertTrue(any("still bear a hydrogen" in line for line in record))

    def test_the_deprotonated_form_is_normalised(self):
        normalised, record = graph_module.normalise_resonance(
            acetate_graph(), hydrogen_bearing=[])
        self.assertEqual(len(graph_module.automorphisms(normalised)), 2)
        self.assertTrue(any("all are\ndeprotonated" in line.replace(" ", " ")
                            or "deprotonated at the modelled state" in line
                            for line in record))

    def test_unknown_protonation_normalises_nothing(self):
        """The conservative default: no equivalence without evidence for it."""
        normalised, record = graph_module.normalise_resonance(
            acetate_graph(), hydrogen_bearing=None)
        self.assertEqual(len(graph_module.automorphisms(normalised)), 1)
        self.assertIn("no protonation state supplied", record[0])

    def test_a_non_equivalent_oxygen_swap_cannot_receive_a_false_pass(self):
        """A carboxylic acid whose O-H and C=O are exchanged must score the shift.

        This is the case the old, protonation-blind normalisation would have
        waved through: the two oxygens are 2.0 A apart, the exchange is a real
        chemical difference, and the RMSD must register it.
        """
        acid = acetate_graph()
        reference = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                     "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0)}
        swapped = dict(reference)
        swapped["O1"], swapped["O2"] = reference["O2"], reference["O1"]

        protonated, _ = graph_module.normalise_resonance(
            acid, hydrogen_bearing=["O2"])
        value, permutation = graph_module.symmetry_aware_rmsd(
            protonated, reference, swapped,
            graph_module.automorphisms(protonated))
        self.assertAlmostEqual(value, (2 * 2.0 ** 2 / 4) ** 0.5)
        self.assertEqual(permutation, tuple(range(4)))
        self.assertGreater(value, 0.9)

        # The same swap on the deprotonated form is a relabelling, not a change.
        deprotonated, _ = graph_module.normalise_resonance(
            acid, hydrogen_bearing=[])
        value, _ = graph_module.symmetry_aware_rmsd(
            deprotonated, reference, swapped,
            graph_module.automorphisms(deprotonated))
        self.assertAlmostEqual(value, 0.0)

    def test_a_stereochemical_inversion_is_not_waved_through(self):
        """An inverted centre must not map onto itself for free.

        A tetrahedral carbon with four distinguishable substituents has no
        automorphism, so inverting it produces a real deviation rather than a
        symmetry-absorbed zero. This test fixes that behaviour; it does not
        make the measure a stereochemistry check, which it cannot be.
        """
        centre = graph_module.graph_from_bond_names(
            ["C1", "N1", "O1", "S1", "C2"], ["C", "N", "O", "S", "C"],
            [("C1", "N1", "SING"), ("C1", "O1", "SING"),
             ("C1", "S1", "SING"), ("C1", "C2", "SING")])
        permutations = graph_module.automorphisms(centre)
        self.assertEqual(len(permutations), 1,
                         "four distinguishable substituents admit no symmetry")
        reference = {"C1": (0.0, 0.0, 0.0), "N1": (1.0, 1.0, 1.0),
                     "O1": (-1.0, -1.0, 1.0), "S1": (-1.0, 1.0, -1.0),
                     "C2": (1.0, -1.0, -1.0)}
        # Reflecting through z inverts the centre's configuration.
        inverted = {name: (x, y, -z) for name, (x, y, z) in reference.items()}
        value, _ = graph_module.symmetry_aware_rmsd(
            centre, reference, inverted, permutations)
        self.assertGreater(value, 0.8)

    def test_two_graph_symmetric_centres_show_the_documented_blind_spot(self):
        """Where symmetry does exist, the measure can understate an inversion.

        This is recorded as a test because it is a real limitation, not a bug
        to be fixed inside an RMSD: the graph has no 3D information. It is why
        stereochemistry must be verified at the preparation stage.
        """
        symmetric = graph_module.graph_from_bond_names(
            ["C0", "C1", "C2", "O1", "O2"], ["C", "C", "C", "O", "O"],
            [("C0", "C1", "SING"), ("C0", "C2", "SING"),
             ("C1", "O1", "SING"), ("C2", "O2", "SING")])
        permutations = graph_module.automorphisms(symmetric)
        self.assertGreater(len(permutations), 1,
                           "the two arms are constitutionally equivalent")
        reference = {"C0": (0.0, 0.0, 0.0), "C1": (1.5, 0.0, 0.0),
                     "C2": (-1.5, 0.0, 0.0), "O1": (2.2, 1.0, 0.0),
                     "O2": (-2.2, 1.0, 0.0)}
        exchanged = {"C0": (0.0, 0.0, 0.0), "C1": (-1.5, 0.0, 0.0),
                     "C2": (1.5, 0.0, 0.0), "O1": (-2.2, 1.0, 0.0),
                     "O2": (2.2, 1.0, 0.0)}
        value, permutation = graph_module.symmetry_aware_rmsd(
            symmetric, reference, exchanged, permutations)
        self.assertAlmostEqual(value, 0.0)
        self.assertNotEqual(permutation, tuple(range(5)))


class AutomorphismTests(unittest.TestCase):
    def test_an_asymmetric_molecule_has_only_the_identity(self):
        asymmetric = graph_module.graph_from_bond_names(
            ["C1", "C2", "N1", "O1"], ["C", "C", "N", "O"],
            [("C1", "C2", "SING"), ("C2", "N1", "SING"), ("N1", "O1", "SING")])
        permutations = graph_module.automorphisms(asymmetric)
        self.assertEqual(permutations, [(0, 1, 2, 3)])

    def test_elements_are_never_exchanged(self):
        """A carbon and a nitrogen in the same position are not equivalent."""
        pair = graph_module.graph_from_bond_names(
            ["C1", "C2", "N1"], ["C", "C", "N"],
            [("C1", "C2", "SING"), ("C1", "N1", "SING")])
        self.assertEqual(len(graph_module.automorphisms(pair)), 1)

    def test_three_distinguishable_groups_give_eight_automorphisms(self):
        """Independent two-fold swaps multiply: 2 x 2 x 2.

        The three centres are different elements, so no automorphism exchanges
        one group for another and only the local oxygen swaps remain. This is
        the situation in UMA, whose two phosphates and one carboxylate sit in
        chemically distinct environments.
        """
        names, elements, bonds = ["C0"], ["C"], []
        for group, element in enumerate(("P", "S", "C")):
            centre = f"X{group}"
            names += [centre, f"O{group}a", f"O{group}b"]
            elements += [element, "O", "O"]
            bonds += [("C0", centre, "SING"),
                      (centre, f"O{group}a", "DOUB"),
                      (centre, f"O{group}b", "SING")]
        built = graph_module.graph_from_bond_names(names, elements, bonds)
        normalised, record = graph_module.normalise_resonance(
            built, hydrogen_bearing=[])
        self.assertEqual(len(record), 3)
        permutations = graph_module.automorphisms(normalised)
        self.assertEqual(len(permutations), 8)
        self.assertEqual(len(graph_module.symmetry_classes(normalised, permutations)),
                         3)

    def test_equivalent_groups_are_also_exchanged_with_each_other(self):
        """Three identical groups on one centre give 3! x 2^3 = 48.

        Whole-group exchange is as real a symmetry as a local oxygen swap, and
        is found too. Recorded as a test so the distinction from the previous
        case is deliberate rather than accidental.
        """
        names, elements, bonds = ["C0"], ["C"], []
        for group in range(3):
            centre = f"P{group}"
            names += [centre, f"O{group}a", f"O{group}b"]
            elements += ["P", "O", "O"]
            bonds += [("C0", centre, "SING"),
                      (centre, f"O{group}a", "DOUB"),
                      (centre, f"O{group}b", "SING")]
        normalised, _ = graph_module.normalise_resonance(
            graph_module.graph_from_bond_names(names, elements, bonds),
            hydrogen_bearing=[])
        self.assertEqual(len(graph_module.automorphisms(normalised)), 48)

    def test_the_identity_comes_first(self):
        normalised, _ = graph_module.normalise_resonance(
            acetate_graph(), hydrogen_bearing=[])
        permutations = graph_module.automorphisms(normalised)
        self.assertEqual(permutations[0], tuple(range(len(normalised.names))))

    def test_the_cap_raises_rather_than_truncating(self):
        normalised, _ = graph_module.normalise_resonance(
            acetate_graph(), hydrogen_bearing=[])
        with self.assertRaises(graph_module.CorrespondenceError):
            graph_module.automorphisms(normalised, cap=0)


class RmsdTests(unittest.TestCase):
    def setUp(self):
        self.graph, _ = graph_module.normalise_resonance(
            acetate_graph(), hydrogen_bearing=[])
        self.permutations = graph_module.automorphisms(self.graph)
        self.reference = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                          "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0)}

    def test_identical_poses_give_zero(self):
        value, permutation = graph_module.symmetry_aware_rmsd(
            self.graph, self.reference, self.reference, self.permutations)
        self.assertEqual(value, 0.0)
        self.assertEqual(permutation, tuple(range(4)))

    def test_swapped_equivalent_oxygens_give_zero(self):
        swapped = dict(self.reference)
        swapped["O1"], swapped["O2"] = self.reference["O2"], self.reference["O1"]
        value, permutation = graph_module.symmetry_aware_rmsd(
            self.graph, self.reference, swapped, self.permutations)
        self.assertAlmostEqual(value, 0.0)
        self.assertNotEqual(permutation, tuple(range(4)))

    def test_the_naive_comparison_would_have_been_wrong(self):
        """Without symmetry the same pose scores the O-O separation."""
        swapped = dict(self.reference)
        swapped["O1"], swapped["O2"] = self.reference["O2"], self.reference["O1"]
        first, second = graph_module.paired_coordinates(
            self.graph, self.reference, swapped)
        self.assertAlmostEqual(graph_module.in_place_rmsd(first, second),
                               (2 * 2.0 ** 2 / 4) ** 0.5)

    def test_no_superposition_is_applied(self):
        """A rigid translation must appear in full, not be fitted away."""
        for shift in (0.5, 3.0, 10.0):
            moved = {name: (x + shift, y, z)
                     for name, (x, y, z) in self.reference.items()}
            value, _ = graph_module.symmetry_aware_rmsd(
                self.graph, self.reference, moved, self.permutations)
            self.assertAlmostEqual(value, shift)

    def test_a_rotated_pose_is_not_fitted_away(self):
        rotated = {name: (-x, -y, z)
                   for name, (x, y, z) in self.reference.items()}
        value, _ = graph_module.symmetry_aware_rmsd(
            self.graph, self.reference, rotated, self.permutations)
        self.assertGreater(value, 1.0)

    def test_a_missing_atom_raises_rather_than_comparing_a_subset(self):
        partial = {name: position for name, position in self.reference.items()
                   if name != "O2"}
        with self.assertRaises(graph_module.CorrespondenceError) as caught:
            graph_module.symmetry_aware_rmsd(
                self.graph, self.reference, partial, self.permutations)
        self.assertIn("not computed over a subset", str(caught.exception))

    def test_an_extra_atom_raises(self):
        extra = dict(self.reference, O3=(9.0, 9.0, 9.0))
        with self.assertRaises(graph_module.CorrespondenceError):
            graph_module.symmetry_aware_rmsd(
                self.graph, self.reference, extra, self.permutations)

    def test_non_finite_coordinates_raise(self):
        broken = dict(self.reference, O2=(float("nan"), 0.0, 0.0))
        with self.assertRaises(graph_module.CorrespondenceError):
            graph_module.symmetry_aware_rmsd(
                self.graph, self.reference, broken, self.permutations)

    def test_rmsd_is_the_root_mean_square_not_the_maximum(self):
        """One atom moved 2 A out of four gives 1.0 A, not 2.0 A."""
        moved = dict(self.reference)
        moved["C1"] = (0.0, 2.0, 0.0)
        value, _ = graph_module.symmetry_aware_rmsd(
            self.graph, self.reference, moved, self.permutations)
        self.assertAlmostEqual(value, (4.0 / 4) ** 0.5)


# ---------------------------------------------------------------------------
# Pose files and the recovery logic
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "python"))
import analyse_redocking  # noqa: E402


def pose_block(rank: int, score: float, atoms: dict[str, tuple[float, float, float]],
               include_hydrogen: bool = False) -> str:
    lines = [f"MODEL {rank}",
             f"REMARK VINA RESULT:  {score:8.3f}      0.000      0.000"]
    for serial, (name, (x, y, z)) in enumerate(atoms.items(), start=1):
        element = name[0]
        lines.append(f"ATOM  {serial:5d} {name:<4} LIG A 601    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  0.00  0.00    +0.000 {element:<2}")
    if include_hydrogen:
        lines.append(f"ATOM  {len(atoms)+1:5d} HO1  LIG A 601    "
                     f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  0.00  0.00    +0.200 HD")
    lines.append("ENDMDL")
    return "\n".join(lines)


class PoseFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.atoms = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                      "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0)}

    def test_models_scores_and_order_are_read(self):
        path = self.root / "poses.pdbqt"
        path.write_text("\n".join([
            pose_block(1, -9.5, self.atoms),
            pose_block(2, -8.1, self.atoms)]) + "\n")
        poses = analyse_redocking.parse_pose_file(path)
        self.assertEqual(len(poses), 2)
        self.assertAlmostEqual(poses[0]["score"], -9.5)
        self.assertAlmostEqual(poses[1]["score"], -8.1)
        self.assertEqual(set(poses[0]["coordinates"]), set(self.atoms))

    def test_hydrogens_are_excluded_from_poses(self):
        path = self.root / "poses.pdbqt"
        path.write_text(pose_block(1, -9.0, self.atoms, include_hydrogen=True) + "\n")
        poses = analyse_redocking.parse_pose_file(path)
        self.assertEqual(set(poses[0]["coordinates"]), set(self.atoms))

    def test_an_empty_file_raises(self):
        path = self.root / "poses.pdbqt"
        path.write_text("REMARK nothing\n")
        with self.assertRaises(graph_module.CorrespondenceError):
            analyse_redocking.parse_pose_file(path)

    def test_a_repeated_atom_within_one_pose_raises(self):
        path = self.root / "poses.pdbqt"
        text = pose_block(1, -9.0, self.atoms).splitlines()
        text.insert(3, text[2])
        path.write_text("\n".join(text) + "\n")
        with self.assertRaises(graph_module.CorrespondenceError):
            analyse_redocking.parse_pose_file(path)

    def test_missing_pose_directory_raises(self):
        with self.assertRaises(graph_module.CorrespondenceError):
            analyse_redocking.collect_poses(self.root, "uma")


class RecoveryLogicTests(unittest.TestCase):
    """The per-seed and overall recovery outcomes, on hand-built RMSD values."""

    def setUp(self):
        self.graph, _ = graph_module.normalise_resonance(
            acetate_graph(), hydrogen_bearing=[])
        self.permutations = graph_module.automorphisms(self.graph)
        self.reference = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                          "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0)}

    def poses(self, plan):
        """plan: {seed: [(rank, shift, score)]}; shift becomes the RMSD."""
        poses = []
        for seed, entries in plan.items():
            for rank, shift, score in entries:
                moved = {name: (x + shift, y, z)
                         for name, (x, y, z) in self.reference.items()}
                poses.append(analyse_redocking.Pose(
                    seed=seed, rank=rank, score=score, coordinates=moved,
                    source=f"seed_{seed}"))
        return poses

    def analyse(self, plan, criterion=2.0):
        return analyse_redocking.analyse(
            self.reference, self.graph, self.permutations,
            self.poses(plan), criterion)

    def test_consistent_recovery_needs_every_seed(self):
        result = self.analyse({1: [(1, 0.5, -9.0)], 42: [(1, 1.0, -9.1)],
                               137: [(1, 1.9, -8.8)]})
        self.assertTrue(result["summary"]["consistent_recovery"])
        self.assertEqual(result["summary"]["outcome"], "consistent recovery")
        self.assertEqual(result["summary"]["n_seeds_top_ranked_meets_criterion"], 3)

    def test_one_failing_seed_is_a_mixed_outcome(self):
        result = self.analyse({1: [(1, 0.5, -9.0)], 42: [(1, 1.0, -9.1)],
                               137: [(1, 6.0, -8.8)]})
        self.assertFalse(result["summary"]["consistent_recovery"])
        self.assertEqual(result["summary"]["outcome"], "mixed outcome")
        self.assertEqual(result["summary"]["n_seeds_top_ranked_meets_criterion"], 2)

    def test_a_pose_just_over_the_criterion_fails_and_is_reported_continuously(self):
        result = self.analyse({1: [(1, 2.001, -9.0)]})
        seed = result["per_seed"][0]
        self.assertEqual(seed["top_ranked"]["meets_criterion"], "no")
        self.assertAlmostEqual(seed["top_ranked"]["rmsd_angstrom"], 2.001, places=3)

    def test_a_pose_exactly_at_the_criterion_passes(self):
        result = self.analyse({1: [(1, 2.0, -9.0)]})
        self.assertEqual(result["per_seed"][0]["top_ranked"]["meets_criterion"],
                         "yes")

    def test_top_ranked_and_closest_are_reported_separately(self):
        """A scoring failure: the search found a good pose but ranked it third."""
        result = self.analyse({1: [(1, 7.0, -9.5), (2, 4.0, -9.0),
                                   (3, 0.4, -8.2)]})
        seed = result["per_seed"][0]
        self.assertEqual(seed["top_ranked"]["rank"], 1)
        self.assertAlmostEqual(seed["top_ranked"]["rmsd_angstrom"], 7.0)
        self.assertEqual(seed["top_ranked"]["meets_criterion"], "no")
        self.assertEqual(seed["closest_to_reference"]["rank"], 3)
        self.assertAlmostEqual(seed["closest_to_reference"]["rmsd_angstrom"], 0.4)
        self.assertEqual(seed["closest_to_reference"]["meets_criterion"], "yes")
        self.assertIn("failure to rank it first", result["summary"]["outcome"])
        self.assertIn("recovery among retained poses",
                      result["summary"]["outcome"])

    def test_no_recovery_at_all(self):
        result = self.analyse({1: [(1, 9.0, -9.0)], 42: [(1, 8.0, -9.0)]})
        self.assertEqual(result["summary"]["outcome"], "no recovery")

    def test_scores_and_ranks_accompany_every_pose(self):
        result = self.analyse({1: [(1, 0.5, -9.0), (2, 3.0, -8.0)]})
        for row in result["per_pose"]:
            self.assertIn("score_kcal_per_mol", row)
            self.assertIn("rank", row)
            self.assertIn("symmetry_aware_rmsd_angstrom", row)

    def test_fewer_poses_in_one_seed_is_handled(self):
        """Vina may retain different numbers of poses per seed."""
        result = self.analyse({1: [(1, 0.5, -9.0), (2, 3.0, -8.0)],
                               42: [(1, 0.6, -9.1)]})
        counts = {entry["seed"]: entry["n_poses_retained"]
                  for entry in result["per_seed"]}
        self.assertEqual(counts, {1: 2, 42: 1})


# ---------------------------------------------------------------------------
# Dictionary, reference and ligand building. Real Gemmi; no chemistry toolkit.
# ---------------------------------------------------------------------------

MINIMAL_CCD = """data_LIG
_chem_comp.id LIG
_chem_comp.name "TEST ACID"
_chem_comp.formula "C2 H4 O2"
_chem_comp.pdbx_formal_charge 0
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_stereo_config
LIG C1 C 0 N N
LIG C2 C 0 N R
LIG O1 O 0 N N
LIG O2 O 0 N N
LIG "O3'" O 0 N N
LIG H1 H 0 N N
LIG HO2 H 0 N N
LIG "HO3'" H 0 N N
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
LIG C1 C2 SING N N
LIG C2 O1 DOUB N N
LIG C2 O2 SING N N
LIG C1 "O3'" SING N N
LIG C1 H1 SING N N
LIG O2 HO2 SING N N
LIG "O3'" "HO3'" SING N N
#
"""

MINIMAL_CIF = """data_fix
_entry.id FIX
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM   1 N N  . GLY A 1 1 ? 0.000 0.000 0.000 1.00 10.00 1 A 1
ATOM   2 C CA . GLY A 1 1 ? 1.000 0.000 0.000 1.00 10.00 1 A 1
ATOM   3 C C  . GLY A 1 1 ? 2.000 0.000 0.000 1.00 10.00 1 A 1
ATOM   4 O O  . GLY A 1 1 ? 3.000 0.000 0.000 1.00 10.00 1 A 1
HETATM 5 C C1 . LIG B 2 . ? 0.000 0.000 0.000 1.00 10.00 601 A 1
HETATM 6 C C2 . LIG B 2 . ? 1.500 0.000 0.000 1.00 10.00 601 A 1
HETATM 7 O O1 . LIG B 2 . ? 2.200 1.000 0.000 1.00 10.00 601 A 1
HETATM 8 O O2 . LIG B 2 . ? 2.200 -1.000 0.000 1.00 10.00 601 A 1
HETATM 9 O "O3'" . LIG B 2 . ? -0.700 1.200 0.000 1.00 10.00 601 A 1
HETATM 10 C C1 . LIG C 2 . ? 50.000 0.000 0.000 1.00 10.00 602 B 1
HETATM 11 C C2 . LIG C 2 . ? 51.500 0.000 0.000 1.00 10.00 602 B 1
HETATM 12 O O1 . LIG C 2 . ? 52.200 1.000 0.000 1.00 10.00 602 B 1
HETATM 13 O O2 . LIG C 2 . ? 52.200 -1.000 0.000 1.00 10.00 602 B 1
HETATM 14 O "O3'" . LIG C 2 . ? 49.300 1.200 0.000 1.00 10.00 602 B 1
"""


@unittest.skipUnless(GEMMI_AVAILABLE, "these stages require Gemmi")
class ReceptorConditionInterpretationTests(unittest.TestCase):
    """The interpretation text must follow the recorded receptor condition.

    These are wording checks, deliberately. The defect they guard against was
    a factual claim - "the receptor has neither ANP nor manganese" - printed
    unconditionally, including for a campaign whose own record said ANP was
    retained. What matters is therefore which condition each branch describes,
    not the prose style, so each assertion names the substantive claim it
    requires or forbids.
    """

    def test_apo_states_anp_is_absent(self):
        limits = analyse_redocking.receptor_condition_limits("apo")
        joined = " ".join(limits)
        self.assertIn("'apo'", joined)
        self.assertIn("ANP is absent", joined)
        self.assertNotIn("ANP is retained", joined)

    def test_anp_present_never_claims_the_receptor_lacks_anp(self):
        limits = analyse_redocking.receptor_condition_limits("anp-present")
        joined = " ".join(limits)
        self.assertIn("ANP is retained", joined)
        self.assertNotIn("ANP is absent", joined)
        self.assertNotIn("neither ANP nor manganese", joined)

    def test_anp_present_records_that_anp_chemistry_is_unvalidated(self):
        # Retaining ANP makes Meeko's perceived typing and charges part of the
        # result, so the limitation has to travel with the condition.
        joined = " ".join(
            analyse_redocking.receptor_condition_limits("anp-present"))
        self.assertIn("Meeko", joined)
        self.assertIn("not been independently validated", joined)

    def test_both_conditions_state_the_shared_removals(self):
        for condition in ("apo", "anp-present"):
            joined = " ".join(
                analyse_redocking.receptor_condition_limits(condition))
            with self.subTest(condition=condition):
                self.assertIn("manganese", joined)
                self.assertIn("397 crystallographic waters", joined)
                self.assertIn("not validation against the native state",
                              joined)

    def test_neither_condition_is_called_a_replicate_of_the_other(self):
        for condition in ("apo", "anp-present"):
            joined = " ".join(
                analyse_redocking.receptor_condition_limits(condition))
            with self.subTest(condition=condition):
                self.assertIn("not a", joined)
                self.assertIn("replicate", joined)

    def test_an_unrecorded_condition_describes_no_composition(self):
        for condition in (None, "", "something-else"):
            limits = analyse_redocking.receptor_condition_limits(condition)
            joined = " ".join(limits)
            with self.subTest(condition=condition):
                self.assertIn("states no receptor condition", joined)
                self.assertNotIn("ANP is absent", joined)
                self.assertNotIn("ANP is retained", joined)

    def test_no_unconditional_claim_about_the_receptor_remains(self):
        """Guard the defect itself: a composition claim outside the branches.

        This reads the module's source rather than its behaviour, because the
        defect was one sentence asserted on every path, and the wording is
        what made it wrong. The phrase is now spoken only by the apo branch,
        which the tests above check, so its reappearance anywhere in the
        module is the regression.
        """
        source = Path(analyse_redocking.__file__).read_text()
        self.assertNotIn("neither ANP nor manganese", source)

    def test_the_closing_sentence_follows_the_condition_too(self):
        apo = analyse_redocking.receptor_condition_sentence("apo")
        anp = analyse_redocking.receptor_condition_sentence("anp-present")
        unknown = analyse_redocking.receptor_condition_sentence(None)
        self.assertIn("no ANP", apo)
        self.assertIn("ANP retained", anp)
        self.assertNotIn("no ANP", anp)
        self.assertIn("no receptor condition", unknown)
        # Whatever the receptor held, the control is silent on affinity.
        for sentence in (apo, anp, unknown):
            self.assertIn("peptide", sentence)
            self.assertIn("affinity", sentence)


class DictionaryAndReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ccd = self.root / "LIG.cif"
        self.ccd.write_text(MINIMAL_CCD)
        self.structure = self.root / "fixture.cif"
        self.structure.write_text(MINIMAL_CIF)

    def run_script(self, script, *args, success=True):
        finished = subprocess.run(
            [sys.executable, str(script), *[str(a) for a in args]],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(finished.returncode == 0, success,
                         finished.stdout + finished.stderr)
        return finished

    def test_primed_atom_names_survive_dictionary_reading(self):
        """A name such as O3' must not lose its prime to quote stripping."""
        component = chemical_component.read_component(self.ccd)
        self.assertIn("O3'", component.heavy_atom_names)
        self.assertEqual(len(component.heavy_atom_names), 5)
        self.assertEqual(component.n_hydrogens, 3)

    def test_ionisable_sites_are_oxygen_acids_only(self):
        component = chemical_component.read_component(self.ccd)
        sites = {site.atom: site.group for site in component.ionisable_sites()}
        self.assertEqual(set(sites), {"O2", "O3'"})
        self.assertEqual(sites["O2"], "carboxylic acid hydroxyl")
        self.assertEqual(sites["O3'"], "alcohol hydroxyl")

    def test_protonation_state_derives_the_charge_and_needs_a_reason(self):
        component = chemical_component.read_component(self.ccd)
        state = chemical_component.validated_protonation(
            component, ["O2"], 7.4, "carboxylic acid, pKa well below pH 7.4")
        self.assertEqual(state.formal_charge, -1)
        self.assertEqual(state.as_dict()["dictionary_formal_charge"], 0)
        with self.assertRaises(chemical_component.ComponentError):
            chemical_component.validated_protonation(component, ["O2"], 7.4, "  ")

    def test_deprotonating_an_impossible_atom_is_refused(self):
        component = chemical_component.read_component(self.ccd)
        for atoms in (["O1"], ["C1"], ["O9"]):
            with self.assertRaises(chemical_component.ComponentError):
                chemical_component.validated_protonation(
                    component, atoms, 7.4, "test")

    def test_reference_extraction_preserves_coordinates_exactly(self):
        prefix = self.root / "lig"
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG", prefix,
                        "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        record = json.loads((self.root / "lig.reference.json").read_text())
        self.assertEqual(record["selection"]["ligand_author_chain_id"], "A")
        self.assertEqual(record["selection"]["n_heavy_atoms"], 5)
        self.assertEqual(record["outputs"]["max_deviation_from_deposited_angstrom"],
                         0.0)
        self.assertEqual(
            record["atom_mapping"]["O1"]["coordinates_angstrom"], [2.2, 1.0, 0.0])
        self.assertTrue(record["chemical_identity"]["identical_atom_set"])
        # The reference assumes no protonation state, so no group is
        # treated as resonance-equivalent and no orbit is reported here.
        self.assertEqual(record["symmetry"]["orbits"], [])
        self.assertIn("no protonation state",
                      record["symmetry"]["resonance_normalisation"][0])

    def test_the_second_copy_is_selectable_and_different(self):
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "copyC", "--ligand-canonical-chain", "C",
                        "--component-dictionary", self.ccd)
        record = json.loads((self.root / "copyC.reference.json").read_text())
        self.assertEqual(record["selection"]["ligand_author_chain_id"], "B")
        self.assertEqual(record["atom_mapping"]["C1"]["coordinates_angstrom"][0],
                         50.0)

    def test_an_existing_reference_is_never_replaced(self):
        prefix = self.root / "lig"
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG", prefix,
                        "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        before = (self.root / "lig_reference.pdb").read_bytes()
        finished = self.run_script(
            LIGAND_REFERENCE, self.structure, "LIG", prefix,
            "--ligand-canonical-chain", "B",
            "--component-dictionary", self.ccd, success=False)
        self.assertIn("never replaced", finished.stderr)
        self.assertEqual((self.root / "lig_reference.pdb").read_bytes(), before)

    def test_an_absent_component_is_refused(self):
        finished = self.run_script(
            LIGAND_REFERENCE, self.structure, "XXX", self.root / "none",
            "--ligand-canonical-chain", "B", success=False)
        self.assertIn("no XXX residue", finished.stderr)

    def test_building_the_ligand_records_the_protonation_decision(self):
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        self.run_script(PREPARE_LIGAND, "build",
                        self.root / "lig.reference.json", self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "pKa well below pH 7.4")
        manifest = json.loads((self.root / "built.ligand.json").read_text())
        self.assertEqual(manifest["protonation_state"]["modelled_formal_charge"], -1)
        self.assertEqual(manifest["protonation_state"]["deprotonated_atoms"], ["O2"])
        self.assertIn("pKa", manifest["protonation_state"]["justification"])
        self.assertEqual(manifest["dictionary"]["formal_charge"], 0)
        self.assertEqual(manifest["retained_polar_hydrogen_sites"], ["O3'"])
        self.assertFalse(manifest["preparation"]["performed"])

    def test_the_molfile_carries_bond_orders_and_charges(self):
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        self.run_script(PREPARE_LIGAND, "build",
                        self.root / "lig.reference.json", self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "test")
        lines = (self.root / "built_ligand.mol").read_text().splitlines()
        counts = lines[3]
        self.assertEqual(int(counts[0:3]), 5)
        self.assertEqual(int(counts[3:6]), 4)
        self.assertTrue(any(line.startswith("M  CHG") for line in lines))
        self.assertEqual(lines[-1], "M  END")
        # No hydrogens are written: placing them is the toolkit's job.
        self.assertEqual(sum(1 for line in lines[4:9] if line[31:34].strip() == "H"),
                         0)

    def test_the_build_refuses_a_dictionary_for_another_component(self):
        other = self.root / "OTH.cif"
        other.write_text(MINIMAL_CCD.replace("LIG", "OTH"))
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        finished = self.run_script(
            PREPARE_LIGAND, "build", self.root / "lig.reference.json",
            self.root / "built", "--component-dictionary", other,
            "--ph", "7.4", "--deprotonate", "O2",
            "--protonation-justification", "test", success=False)
        self.assertIn("dictionary is for", finished.stderr)

    def test_the_check_stage_catches_a_moved_heavy_atom(self):
        """Preparation must not move the heavy atoms."""
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        self.run_script(PREPARE_LIGAND, "build",
                        self.root / "lig.reference.json", self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "test")
        moved = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                 "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0),
                 "O3'": (-0.7, 1.7, 0.0)}
        pdbqt = self.root / "moved.pdbqt"
        pdbqt.write_text(self.pdbqt_text(moved, charge_total=-1.0))
        finished = self.run_script(
            PREPARE_LIGAND, "check", self.root / "built.ligand.json", pdbqt,
            "--out", self.root / "prep.json", "--overwrite",
            success=False)
        self.assertIn("must not move the heavy atoms", finished.stderr)

    def test_the_check_stage_catches_a_charge_that_contradicts_the_decision(self):
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        self.run_script(PREPARE_LIGAND, "build",
                        self.root / "lig.reference.json", self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "test")
        reference = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                     "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0),
                     "O3'": (-0.7, 1.2, 0.0)}
        pdbqt = self.root / "neutral.pdbqt"
        pdbqt.write_text(self.pdbqt_text(reference, charge_total=0.0))
        finished = self.run_script(
            PREPARE_LIGAND, "check", self.root / "built.ligand.json", pdbqt,
            "--out", self.root / "prep.json", "--overwrite",
            success=False)
        self.assertIn("did not apply the protonation state", finished.stderr)

    def test_the_check_stage_records_torsdof_and_types(self):
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        self.run_script(PREPARE_LIGAND, "build",
                        self.root / "lig.reference.json", self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "test")
        reference = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                     "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0),
                     "O3'": (-0.7, 1.2, 0.0)}
        pdbqt = self.root / "good.pdbqt"
        pdbqt.write_text(self.pdbqt_text(reference, charge_total=-1.0,
                                         torsdof=3, branches=2,
                                         hydrogen=True))
        self.run_script(PREPARE_LIGAND, "check", self.root / "built.ligand.json",
                        pdbqt, "--out", self.root / "built.preparation.json",
                        "--charge-model", "gasteiger",
                        "--preparation-command", "mk_prepare_ligand.py -i x.sdf")
        record = json.loads(
            (self.root / "built.preparation.json").read_text())
        self.assertEqual(record["schema"], "mur-ligase-ligand-preparation/1")
        self.assertEqual(record["result"], "passed")
        preparation = record["preparation"]
        self.assertTrue(preparation["performed"])
        self.assertEqual(preparation["torsdof"], 3)
        self.assertEqual(preparation["n_branch_records"], 2)
        self.assertEqual(preparation["n_hydrogens"], 1)
        self.assertEqual(preparation["n_heavy_atoms"], 5)
        self.assertTrue(preparation["heavy_atom_coordinates"]["preserved"])
        self.assertEqual(preparation["partial_charge_model"], "gasteiger")
        self.assertIn("OA", preparation["autodock_type_census"])

    def test_the_check_stage_refuses_duplicate_atom_names(self):
        """A repeated name must fail, not be collapsed into a matching set.

        Two records called O1 would reduce to one dictionary entry, the
        heavy-atom name set would still equal the reference's, and the
        comparison would pass while one record was silently discarded.
        """
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        self.run_script(PREPARE_LIGAND, "build",
                        self.root / "lig.reference.json", self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "test")
        reference = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                     "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0),
                     "O3'": (-0.7, 1.2, 0.0)}
        text = self.pdbqt_text(reference, charge_total=-1.0).splitlines()
        # Duplicate the O1 record with different coordinates.
        duplicate = [line for line in text if line[12:16].strip() == "O1"][0]
        text.insert(text.index(duplicate) + 1,
                    duplicate[:30] + f"{9.999:8.3f}" + duplicate[38:])
        pdbqt = self.root / "duplicate.pdbqt"
        pdbqt.write_text("\n".join(text) + "\n")
        finished = self.run_script(
            PREPARE_LIGAND, "check", self.root / "built.ligand.json", pdbqt,
            "--out", self.root / "prep.json", "--overwrite",
            success=False)
        self.assertIn("repeats atom names", finished.stderr)
        self.assertIn("O1", finished.stderr)

    def test_the_reference_refuses_duplicate_atom_names(self):
        duplicated = self.root / "duplicated.cif"
        lines = MINIMAL_CIF.splitlines()
        # Inside the same residue block, so the one-copy guard does not fire
        # first and the duplicate name itself is what is caught.
        original = next(line for line in lines
                        if " O1 . LIG B " in line)
        lines.insert(lines.index(original) + 1,
                     'HETATM 15 O O1 . LIG B 2 . ? 9.000 9.000 9.000 '
                     '1.00 10.00 601 A 1')
        duplicated.write_text("\n".join(lines) + "\n")
        finished = self.run_script(
            LIGAND_REFERENCE, duplicated, "LIG", self.root / "dup",
            "--ligand-canonical-chain", "B", success=False)
        self.assertIn("duplicated atom name", finished.stderr)

    def test_the_check_stage_refuses_a_missing_heavy_atom(self):
        self.run_script(LIGAND_REFERENCE, self.structure, "LIG",
                        self.root / "lig", "--ligand-canonical-chain", "B",
                        "--component-dictionary", self.ccd)
        self.run_script(PREPARE_LIGAND, "build",
                        self.root / "lig.reference.json", self.root / "built",
                        "--component-dictionary", self.ccd,
                        "--ph", "7.4", "--deprotonate", "O2",
                        "--protonation-justification", "test")
        partial = {"C1": (0.0, 0.0, 0.0), "C2": (1.5, 0.0, 0.0),
                   "O1": (2.2, 1.0, 0.0), "O2": (2.2, -1.0, 0.0)}
        pdbqt = self.root / "partial.pdbqt"
        pdbqt.write_text(self.pdbqt_text(partial, charge_total=-1.0))
        finished = self.run_script(
            PREPARE_LIGAND, "check", self.root / "built.ligand.json", pdbqt,
            "--out", self.root / "prep.json", "--overwrite",
            success=False)
        self.assertIn("not made over a subset", finished.stderr)

    @staticmethod
    def pdbqt_text(atoms, charge_total, torsdof=0, branches=0, hydrogen=False):
        """A hand-written ligand PDBQT fixture. No tool produced this."""
        types = {"C": "C", "O": "OA", "N": "N", "P": "P"}
        per_atom = charge_total / len(atoms)
        lines = ["ROOT"]
        for serial, (name, (x, y, z)) in enumerate(atoms.items(), start=1):
            element = name[0]
            lines.append(
                f"ATOM  {serial:5d} {name:<4} LIG A 601    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  0.00  0.00   "
                f"{per_atom:+7.3f} {types.get(element, element):<2}")
        if hydrogen:
            lines.append(
                f"ATOM  {len(atoms)+1:5d} HO3  LIG A 601    "
                f"{0.000:8.3f}{0.000:8.3f}{0.000:8.3f}  0.00  0.00   "
                f"{0.0:+7.3f} HD")
        lines.append("ENDROOT")
        for branch in range(branches):
            lines.append(f"BRANCH   {branch + 1}   {branch + 2}")
            lines.append(f"ENDBRANCH   {branch + 1}   {branch + 2}")
        lines.append(f"TORSDOF {torsdof}")
        return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
