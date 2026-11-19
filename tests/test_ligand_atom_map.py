"""Tests for the PDBQT-serial to reference-name mapping.

Everything here runs without a chemistry toolkit: the mapping is built from
text and geometry, and the whole point is that it is validated rather than
assumed. No test asserts anything chemical.

The fixtures imitate what Meeko 0.7.1 actually wrote for UMA: a PDBQT whose
atom-name column holds the **element**, so names repeat and cannot key a
correspondence, plus a `REMARK INDEX MAP` relating it to the input file. The
real index-map semantics were established from that output and are re-derived
by the code on every run; `IndexMapDirectionTests` covers that derivation.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import box_geometry  # noqa: E402
import ligand_atom_map as atom_map  # noqa: E402

# Four heavy atoms and two hydrogens, in the reference (dictionary) order.
REFERENCE_NAMES = ("C1", "O1", "O2", "N1")
REFERENCE_ATOMS = {
    "C1": (0.000, 0.000, 0.000),
    "O1": (1.500, 0.000, 0.000),
    "O2": (0.000, 1.500, 0.000),
    "N1": (0.000, 0.000, 1.500),
}
REFERENCE_ELEMENTS = {"C1": "C", "O1": "O", "O2": "O", "N1": "N"}
# Four hydrogens at input indices 5-8. The real UMA input held 82 atoms
# against 56 in the PDBQT, so input indices ran above the serial range and the
# index-map direction was unambiguous. These fixtures reproduce that: a mapped
# hydrogen with a high input index is what makes the direction decidable.
HYDROGEN_POSITIONS = {
    5: (2.000, 0.500, 0.000),
    6: (0.500, 2.000, 0.000),
    7: (0.500, 0.500, 2.000),
    8: (-1.000, -1.000, 0.000),
}
# The polar hydrogen Meeko keeps, by input index. The other three are merged.
RETAINED_HYDROGEN = 7


def write_molfile(path: Path, order=REFERENCE_NAMES,
                  hydrogens=(5, 6, 7, 8)):
    """A V2000 molfile: heavy atoms in `order`, then hydrogens appended."""
    atoms = [(REFERENCE_ELEMENTS[name], REFERENCE_ATOMS[name])
             for name in order]
    atoms += [("H", HYDROGEN_POSITIONS[index]) for index in hydrogens]
    lines = ["fixture", "  murligpd" + "0" * 10 + "3D", "comment",
             f"{len(atoms):>3}{0:>3}  0  0  0  0  0  0  0  0999 V2000"]
    for element, (x, y, z) in atoms:
        lines.append(f"{x:>10.4f}{y:>10.4f}{z:>10.4f} {element:<3} 0  0  0  0"
                     f"  0  0  0  0  0  0  0")
    lines.append("M  END")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_pdbqt(path: Path, serial_to_input, order=REFERENCE_NAMES,
                index_map=None, reverse_map=False, charge=0.0):
    """A ligand PDBQT whose name column holds the element, as Meeko writes.

    `serial_to_input` maps 1-based PDBQT serial to 1-based input atom index.
    An input index above the heavy-atom count is written as a hydrogen.
    `index_map` overrides the REMARK pairs, so a test can write a map that
    disagrees with the atoms.
    """
    lines = []
    pairs = (index_map if index_map is not None
             else [(index, serial) for serial, index
                   in sorted(serial_to_input.items())])
    if reverse_map:
        pairs = [(second, first) for first, second in pairs]
    flat = [str(value) for pair in pairs for value in pair]
    for start in range(0, len(flat), 24):
        lines.append("REMARK INDEX MAP " + " ".join(flat[start:start + 24]))
    lines.append("ROOT")
    for serial, index in sorted(serial_to_input.items()):
        if index > len(order):
            element, position, kind = "H", HYDROGEN_POSITIONS[index], "HD"
        else:
            name = order[index - 1]
            element = REFERENCE_ELEMENTS[name]
            position = REFERENCE_ATOMS[name]
            kind = {"C": "C", "O": "OA", "N": "N"}[element]
        x, y, z = position
        lines.append(
            f"ATOM  {serial:5d} {element:<4} UNL     1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {charge:+6.3f} {kind:<2}")
    lines.append("ENDROOT")
    path.write_text("\n".join(lines) + "\n")
    return path


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_pdbqt_atoms_are_read_with_their_serials(self):
        path = write_pdbqt(self.root / "l.pdbqt",
                           {1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN})
        atoms = atom_map.read_pdbqt_atoms(path)
        self.assertEqual([atom.serial for atom in atoms], [1, 2, 3, 4, 5])
        self.assertEqual([atom.name for atom in atoms],
                         ["C", "O", "O", "N", "H"])
        self.assertEqual(len({atom.name for atom in atoms}), 4,
                         "the fixture must reproduce the repeated-name problem")

    def test_repeated_serials_are_refused(self):
        path = write_pdbqt(self.root / "l.pdbqt", {1: 1, 2: 2})
        text = path.read_text().splitlines()
        atom_lines = [line for line in text if line.startswith("ATOM")]
        text.insert(text.index(atom_lines[0]) + 1, atom_lines[0])
        path.write_text("\n".join(text) + "\n")
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.read_pdbqt_atoms(path)
        self.assertIn("repeats atom serials", str(caught.exception))

    def test_a_missing_index_map_is_refused_with_the_remedy(self):
        path = write_pdbqt(self.root / "l.pdbqt",
                           {1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN})
        path.write_text("\n".join(
            line for line in path.read_text().splitlines()
            if not line.startswith("REMARK INDEX MAP")) + "\n")
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.parse_index_map(path)
        self.assertIn("--add_index_map", str(caught.exception))

    def test_an_odd_number_of_integers_is_refused(self):
        path = self.root / "l.pdbqt"
        path.write_text("REMARK INDEX MAP 1 2 3\nATOM      1  C   UNL     1 "
                        "     0.000   0.000   0.000  1.00  0.00    +0.000 C\n")
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.parse_index_map(path)
        self.assertIn("whole number of pairs", str(caught.exception))

    def test_a_repeated_block_is_collapsed_but_a_contradiction_is_not(self):
        """Meeko repeats the map once per model; that is not a conflict."""
        path = self.root / "l.pdbqt"
        path.write_text("REMARK INDEX MAP 1 1 2 2\n"
                        "REMARK INDEX MAP 1 1 2 2\n")
        self.assertEqual(atom_map.parse_index_map(path), [(1, 1), (2, 2)])
        path.write_text("REMARK INDEX MAP 1 1\nREMARK INDEX MAP 1 2\n")
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.parse_index_map(path)
        self.assertIn("maps 1 to both", str(caught.exception))

    def test_molfile_atoms_are_read_one_based_in_file_order(self):
        path = write_molfile(self.root / "m.mol")
        atoms = atom_map.read_molfile_atoms(path)
        self.assertEqual([atom.index for atom in atoms],
                         [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual([atom.element for atom in atoms],
                         ["C", "O", "O", "N", "H", "H", "H", "H"])
        self.assertEqual(atoms[0].position, (0.0, 0.0, 0.0))
        self.assertEqual(sum(1 for atom in atoms if atom.is_hydrogen), 4)


class IndexMapDirectionTests(unittest.TestCase):
    """The direction is derived from the file, never assumed."""

    def test_serials_in_the_second_column_are_recognised(self):
        pairs = [(22, 1), (23, 2), (24, 3), (21, 4)]
        self.assertEqual(atom_map.index_map_direction(pairs, 4), "second")

    def test_serials_in_the_first_column_are_recognised(self):
        pairs = [(1, 22), (2, 23), (3, 24), (4, 21)]
        self.assertEqual(atom_map.index_map_direction(pairs, 4), "first")

    def test_identical_readings_make_the_direction_immaterial(self):
        """The identity map reads the same either way, so it is accepted."""
        pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
        self.assertEqual(atom_map.index_map_direction(pairs, 4), "either")

    def test_an_involution_also_reads_the_same_either_way(self):
        pairs = [(1, 2), (2, 1), (3, 4), (4, 3)]
        self.assertEqual(atom_map.index_map_direction(pairs, 4), "either")

    def test_a_genuinely_ambiguous_map_is_refused_rather_than_guessed(self):
        """A non-involution permutation of 1..n reads differently each way."""
        pairs = [(1, 2), (2, 3), (3, 1)]
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.index_map_direction(pairs, 3)
        self.assertIn("the two readings disagree", str(caught.exception))
        self.assertIn("Refusing", str(caught.exception))

    def test_a_map_matching_neither_column_is_refused(self):
        pairs = [(10, 20), (11, 21)]
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.index_map_direction(pairs, 4)
        self.assertIn("do not correspond", str(caught.exception))

    def test_a_pair_count_that_differs_from_the_atom_count_is_refused(self):
        pairs = [(1, 1), (2, 2)]
        with self.assertRaises(atom_map.AtomMapError):
            atom_map.index_map_direction(pairs, 4)


class BuildAtomMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.molfile = write_molfile(self.root / "withH.mol")

    def build(self, serial_to_input, with_hydrogen=True, **kwargs):
        """Build a map, adding the retained polar hydrogen by default.

        The hydrogen is what gives the input column a value above the serial
        range, so the index-map direction is decidable - exactly as in the
        real UMA case, where 82 input atoms produced a 56-atom PDBQT.
        """
        serials = dict(serial_to_input)
        if with_hydrogen and "index_map" not in kwargs:
            serials[max(serials) + 1] = RETAINED_HYDROGEN
        pdbqt = write_pdbqt(self.root / "lig.pdbqt", serials, **kwargs)
        return atom_map.build_atom_map(
            pdbqt, self.molfile, REFERENCE_NAMES, REFERENCE_ATOMS,
            REFERENCE_ELEMENTS)

    def test_identity_order_maps_every_heavy_atom(self):
        record = self.build({1: 1, 2: 2, 3: 3, 4: 4})
        self.assertEqual(record["serial_to_reference_name"],
                         {"1": "C1", "2": "O1", "3": "O2", "4": "N1"})
        self.assertEqual(record["counts"]["heavy_atoms_mapped"], 4)
        self.assertTrue(record["checks"]["bijective_over_reference_heavy_atoms"])
        self.assertEqual(record["checks"]["max_coordinate_deviation_angstrom"],
                         0.0)

    def test_reordered_atoms_map_correctly(self):
        """Meeko reorders atoms into a torsion tree; the map must follow it.

        For the real UMA preparation the first PDBQT atom was input atom 22,
        the beta phosphorus, because Meeko rooted the tree at the diphosphate.
        """
        record = self.build({1: 4, 2: 3, 3: 1, 4: 2})
        self.assertEqual(record["serial_to_reference_name"],
                         {"1": "N1", "2": "O2", "3": "C1", "4": "O1"})
        self.assertEqual(record["reference_name_to_serial"],
                         {"C1": 3, "N1": 1, "O1": 4, "O2": 2})

    def test_the_reverse_direction_is_handled(self):
        record = self.build({1: 4, 2: 3, 3: 1, 4: 2}, reverse_map=True)
        self.assertEqual(record["serial_to_reference_name"]["1"], "N1")
        self.assertIn("PDBQT serial, input atom index",
                      record["index_map"]["direction"])

    def test_merged_hydrogens_are_counted_not_treated_as_failures(self):
        """Meeko merges non-polar hydrogens; those inputs appear in no pair."""
        record = self.build({1: 1, 2: 2, 3: 3, 4: 4})
        counts = record["counts"]
        self.assertEqual(counts["input_hydrogens"], 4)
        self.assertEqual(counts["hydrogen_serials_mapped"], 1)
        self.assertEqual(counts["input_hydrogens_merged"], 3)
        self.assertEqual(record["merged_hydrogens"]["input_indices"],
                         [5, 6, 8])
        self.assertEqual(counts["heavy_atoms_mapped"], 4)
        self.assertEqual(counts["pdbqt_atoms"], 5)

    def test_a_missing_heavy_atom_entry_is_refused(self):
        with self.assertRaises(atom_map.AtomMapError) as caught:
            self.build({1: 1, 2: 2, 3: 3})
        message = str(caught.exception)
        self.assertTrue("not reached by any PDBQT serial" in message
                        or "do not correspond" in message, message)

    def test_a_duplicate_mapping_entry_is_refused(self):
        """Two serials claiming one reference atom is not a bijection."""
        with self.assertRaises(atom_map.AtomMapError) as caught:
            self.build({1: 1, 2: 1, 3: 3, 4: 4, 5: RETAINED_HYDROGEN},
                       index_map=[(1, 1), (1, 2), (3, 3), (4, 4),
                                  (RETAINED_HYDROGEN, 5)])
        message = str(caught.exception)
        self.assertTrue("claimed by serials" in message
                        or "maps 1 to both" in message
                        or "do not correspond" in message, message)

    def test_a_map_naming_an_atom_outside_the_input_is_refused(self):
        with self.assertRaises(atom_map.AtomMapError) as caught:
            self.build({1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN},
                       index_map=[(1, 1), (2, 2), (3, 3), (99, 4),
                                  (RETAINED_HYDROGEN, 5)])
        self.assertIn("not in", str(caught.exception))

    def test_a_map_from_the_wrong_input_is_refused(self):
        """The molfile must be the reference's hydrogenated form.

        Here the input molfile holds the same elements in a different order,
        so position 1 is no longer C1. The coordinate and element checks on
        that link catch it before the index map is composed.
        """
        other = write_molfile(self.root / "other.mol",
                              order=("N1", "O2", "O1", "C1"))
        pdbqt = write_pdbqt(self.root / "lig.pdbqt",
                            {1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN})
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.build_atom_map(pdbqt, other, REFERENCE_NAMES,
                                    REFERENCE_ATOMS, REFERENCE_ELEMENTS)
        message = str(caught.exception)
        self.assertIn("not the hydrogenated form of this reference", message)

    def test_a_heavy_atom_count_mismatch_is_refused(self):
        short = write_molfile(self.root / "short.mol",
                              order=("C1", "O1", "O2"), hydrogens=(5, 6, 7, 8))
        pdbqt = write_pdbqt(self.root / "lig.pdbqt",
                            {1: 1, 2: 2, 3: 3, 4: RETAINED_HYDROGEN},
                            order=("C1", "O1", "O2"))
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.build_atom_map(pdbqt, short, REFERENCE_NAMES,
                                    REFERENCE_ATOMS, REFERENCE_ELEMENTS)
        self.assertIn("heavy atoms but the reference names",
                      str(caught.exception))

    def test_moved_coordinates_are_refused(self):
        moved = dict(REFERENCE_ATOMS, O2=(0.0, 1.6, 0.0))
        pdbqt = write_pdbqt(self.root / "lig.pdbqt",
                            {1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN})
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.build_atom_map(pdbqt, self.molfile, REFERENCE_NAMES,
                                    moved, REFERENCE_ELEMENTS)
        self.assertIn("not the hydrogenated form", str(caught.exception))


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.molfile = write_molfile(self.root / "withH.mol")
        self.pdbqt = write_pdbqt(self.root / "lig.pdbqt",
                                 {1: 4, 2: 3, 3: 1, 4: 2,
                                  5: RETAINED_HYDROGEN})
        self.record = atom_map.build_atom_map(
            self.pdbqt, self.molfile, REFERENCE_NAMES, REFERENCE_ATOMS,
            REFERENCE_ELEMENTS)
        self.path = self.root / "lig.atom_map.json"
        self.path.write_text(json.dumps(self.record, indent=2) + "\n")

    def test_the_record_round_trips(self):
        back = atom_map.read_atom_map(self.path)
        self.assertEqual(atom_map.name_by_serial(back),
                         {1: "N1", 2: "O2", 3: "C1", 4: "O1"})

    def test_the_record_is_tied_to_its_pdbqt_by_hash(self):
        atom_map.read_atom_map(self.path, expected_pdbqt=self.pdbqt)
        other = write_pdbqt(self.root / "other.pdbqt",
                            {1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN})
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.read_atom_map(self.path, expected_pdbqt=other)
        self.assertIn("maps a different PDBQT", str(caught.exception))

    def test_a_wrong_schema_is_refused(self):
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"schema": "something-else/1"}))
        with self.assertRaises(Exception):
            atom_map.read_atom_map(bad)

    def test_an_empty_mapping_is_refused(self):
        with self.assertRaises(atom_map.AtomMapError) as caught:
            atom_map.name_by_serial({"serial_to_reference_name": {}})
        self.assertIn("holds no mapping", str(caught.exception))

    def test_the_record_names_its_inputs_by_hash(self):
        self.assertEqual(self.record["inputs"]["prepared_pdbqt_sha256"],
                         box_geometry.file_sha256(self.pdbqt))
        self.assertEqual(self.record["inputs"]["input_molfile_sha256"],
                         box_geometry.file_sha256(self.molfile))


import analyse_redocking  # noqa: E402


class PoseLinkageTests(unittest.TestCase):
    """Poses link to the prepared ligand through the campaign record.

    A docked pose file is Vina's output: it has its own hash and does not
    carry the prepared ligand's. Requiring it to would be wrong, so the link
    runs through the docking run's recorded ligand inputs instead.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared_hash = "a" * 64
        self.atom_map_record = {"inputs": {"prepared_pdbqt_sha256":
                                           self.prepared_hash}}

    def linkage(self, run_record):
        return analyse_redocking.pose_linkage(
            self.atom_map_record, run_record, self.root / "run_record.json")

    def test_a_matching_campaign_input_establishes_the_link(self):
        result = self.linkage({"ligand_inputs": [
            {"path": "ligands/uma.pdbqt", "sha256": self.prepared_hash}]})
        self.assertTrue(result["established"])
        self.assertEqual(result["matched_campaign_input"], "ligands/uma.pdbqt")
        self.assertIn("not compared against the input", result["note"])

    def test_the_link_is_found_among_several_inputs(self):
        result = self.linkage({"ligand_inputs": [
            {"path": "ligands/other.pdbqt", "sha256": "b" * 64},
            {"path": "ligands/uma.pdbqt", "sha256": self.prepared_hash}]})
        self.assertTrue(result["established"])

    def test_a_campaign_listing_another_ligand_does_not_establish_it(self):
        result = self.linkage({"ligand_inputs": [
            {"path": "ligands/someone_else.pdbqt", "sha256": "f" * 64}]})
        self.assertFalse(result["established"])
        self.assertIn("not among this run's recorded ligand inputs",
                      result["reason"])
        self.assertEqual(result["campaign_ligand_inputs"],
                         ["ligands/someone_else.pdbqt"])

    def test_no_run_record_does_not_establish_it(self):
        result = self.linkage({})
        self.assertFalse(result["established"])
        self.assertIn("no run record", result["reason"])

    def test_an_atom_map_without_a_hash_does_not_establish_it(self):
        self.atom_map_record = {"inputs": {}}
        result = self.linkage({"ligand_inputs": [
            {"path": "x", "sha256": self.prepared_hash}]})
        self.assertFalse(result["established"])
        self.assertIn("records no prepared-ligand hash", result["reason"])

    def test_an_absent_ligand_inputs_list_does_not_establish_it(self):
        result = self.linkage({"seeds": [1, 42, 137]})
        self.assertFalse(result["established"])

    def test_a_pose_file_is_not_required_to_carry_the_input_hash(self):
        """The design point: the map is read without an expected-PDBQT check.

        `read_atom_map` enforces the hash only when a caller names the file it
        should describe, which the preparation check does and the analysis
        deliberately does not.
        """
        molfile = write_molfile(self.root / "withH.mol")
        prepared = write_pdbqt(self.root / "prepared.pdbqt",
                               {1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN})
        record = atom_map.build_atom_map(
            prepared, molfile, REFERENCE_NAMES, REFERENCE_ATOMS,
            REFERENCE_ELEMENTS)
        path = self.root / "m.json"
        path.write_text(json.dumps(record, indent=2) + "\n")

        # Read without naming a file: accepted, as the analysis needs.
        self.assertEqual(len(atom_map.name_by_serial(
            atom_map.read_atom_map(path))), 4)
        # Read while naming a *different* file: refused, as preparation needs.
        pose = write_pdbqt(self.root / "pose.pdbqt",
                           {1: 1, 2: 2, 3: 3, 4: 4, 5: RETAINED_HYDROGEN},
                           charge=-0.1)
        self.assertNotEqual(box_geometry.file_sha256(pose),
                            box_geometry.file_sha256(prepared))
        with self.assertRaises(atom_map.AtomMapError):
            atom_map.read_atom_map(path, expected_pdbqt=pose)


if __name__ == "__main__":
    unittest.main()
