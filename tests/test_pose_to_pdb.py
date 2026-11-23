"""Tests for PDBQT-to-PDB conversion for viewing.

The scientific risk is silent substitution: a conversion that moved an atom,
renamed one, dropped one, or let a viewer guess the connectivity would produce
a picture that looks right and compares the wrong things. These tests assert
that coordinates are copied exactly, names come from the atom map, elements
come from the dictionary rather than the AutoDock type column, and the
dictionary's bonds are written explicitly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chemical_component  # noqa: E402
import pose_to_pdb  # noqa: E402
from test_local_minimisation import DiagnosticFixture  # noqa: E402


class ConversionTests(DiagnosticFixture):

    def setUp(self):
        super().setUp()
        self.component = chemical_component.read_component(self.ccd)
        self.output = self.root / "view" / "pose.pdb"

    def convert(self, **overrides):
        values = dict(pose_path=self.ligand, atom_map_path=self.atom_map,
                      dictionary_path=self.ccd, destination=self.output,
                      residue="LIG", chain="L", residue_number=1)
        values.update(overrides)
        return pose_to_pdb.convert(**values)

    def written_atoms(self, path=None):
        atoms = {}
        for line in Path(path or self.output).read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                atoms[line[12:16].strip()] = {
                    "position": (float(line[30:38]), float(line[38:46]),
                                 float(line[46:54])),
                    "element": line[76:78].strip(),
                    "residue": line[17:21].strip(),
                    "chain": line[21],
                    "serial": int(line[6:11]),
                }
        return atoms

    def test_every_coordinate_is_copied_exactly(self):
        record = self.convert()
        atoms = self.written_atoms()
        self.assertEqual(len(atoms), len(self.coordinates))
        for name, expected in self.coordinates.items():
            for got, want in zip(atoms[name]["position"], expected):
                self.assertAlmostEqual(got, want, places=3)
        self.assertLess(record["max_coordinate_deviation_angstrom"], 1e-6)

    def test_names_come_from_the_atom_map_not_the_pdbqt_name_column(self):
        """The PDBQT name column is element-like and must not be trusted."""
        self.convert()
        atoms = self.written_atoms()
        self.assertEqual(set(atoms), set(self.component.heavy_atom_names))

    def test_elements_come_from_the_dictionary(self):
        self.convert()
        atoms = self.written_atoms()
        for name, entry in atoms.items():
            self.assertEqual(entry["element"],
                             self.component.elements[name],
                             f"{name} must take its element from the CCD")

    def test_the_dictionary_bonds_are_written_as_conect_records(self):
        self.convert()
        text = Path(self.output).read_text()
        conect = [line for line in text.splitlines()
                  if line.startswith("CONECT")]
        self.assertEqual(len(conect), len(self.component.graph.bonds))
        # Every bond must join two serials that exist in the file.
        serials = {entry["serial"] for entry in self.written_atoms().values()}
        for line in conect:
            self.assertIn(int(line[6:11]), serials)
            self.assertIn(int(line[11:16]), serials)

    def test_atom_order_follows_the_dictionary_so_poses_are_comparable(self):
        self.convert()
        order = [name for name, _ in sorted(
            self.written_atoms().items(), key=lambda item: item[1]["serial"])]
        self.assertEqual(tuple(order), self.component.heavy_atom_names)

    def test_no_hydrogens_are_written(self):
        self.convert()
        for entry in self.written_atoms().values():
            self.assertNotEqual(entry["element"], "H")

    def test_the_residue_and_chain_are_what_was_requested(self):
        self.convert(residue="UMA", chain="Z")
        for entry in self.written_atoms().values():
            self.assertEqual(entry["residue"], "UMA")
            self.assertEqual(entry["chain"], "Z")

    def test_a_pose_missing_a_heavy_atom_is_refused(self):
        kept = [line for line in self.ligand.read_text().splitlines()
                if not (line.startswith("ATOM") and " O1 " in line)]
        truncated = self.root / "short.pdbqt"
        truncated.write_text("\n".join(kept) + "\n")
        with self.assertRaises(Exception) as caught:
            self.convert(pose_path=truncated)
        self.assertRegex(str(caught.exception), "O1|heavy atom|atom map")

    def test_a_multi_pose_file_is_refused(self):
        single = self.ligand.read_text()
        multi = self.root / "multi.pdbqt"
        multi.write_text(f"MODEL 1\n{single}ENDMDL\nMODEL 2\n{single}ENDMDL\n")
        with self.assertRaises(pose_to_pdb.ConversionError) as caught:
            self.convert(pose_path=multi)
        self.assertIn("extract one first", str(caught.exception))

    def test_an_existing_output_is_not_overwritten_by_default(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("earlier structure\n")
        with self.assertRaises(pose_to_pdb.ConversionError):
            self.convert()
        self.assertEqual(Path(self.output).read_text(), "earlier structure\n")
        self.convert(overwrite=True)
        self.assertIn("CONECT", Path(self.output).read_text())

    def test_the_record_states_that_no_superposition_was_applied(self):
        record = self.convert()
        self.assertIn("no alignment", record["coordinate_frame"])
        self.assertEqual(record["perception"],
                         "none; names from the atom map, bonds from the "
                         "dictionary")
        self.assertIn("coordinates copied unchanged",
                      Path(self.output).read_text())


if __name__ == "__main__":
    unittest.main()
