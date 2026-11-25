"""Tests for verifying a Meeko-produced peptide PDBQT.

Meeko is not installed here, so these tests build PDBQT files by hand. That is
the right way round in any case: the point of the checker is to catch a Meeko
output that is wrong, and a wrong output is easier to construct than to
provoke.

The case that matters most is the silent one. If
`--rigidify_bonds_smarts` fails to match on the installed Meeko build, the
PDBQT is perfectly valid and carries every backbone torsion, so the only
symptom is a TORSDOF count nobody checked. `test_a_silently_unrigidified_tree_is_caught`
is that scenario.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import verify_peptide_pdbqt as vpp  # noqa: E402

# Two glycines: serials 1-4 are residue 1 (N, CA, C, O), 5-8 residue 2.
# Coordinates are arbitrary but consistent between files.
DIPEPTIDE = [
    (1, "N", "GLY", 1, "N", 0.0, 0.0, 0.0),
    (2, "CA", "GLY", 1, "C", 1.5, 0.0, 0.0),
    (3, "C", "GLY", 1, "C", 2.5, 1.0, 0.0),
    (4, "O", "GLY", 1, "OA", 2.5, 2.2, 0.0),
    (5, "N", "GLY", 2, "N", 3.8, 0.8, 0.0),
    (6, "CA", "GLY", 2, "C", 4.9, 1.7, 0.0),
    (7, "C", "GLY", 2, "C", 6.2, 1.0, 0.0),
    (8, "OXT", "GLY", 2, "OA", 7.0, 1.6, 0.0),
]

# A serine side chain to give a genuine side-chain torsion: CA-CB.
SERINE_EXTRA = [
    (9, "CB", "SER", 3, "C", 5.0, 3.2, 0.0),
    (10, "OG", "SER", 3, "OA", 5.6, 3.9, 0.0),
    (11, "CA", "SER", 3, "C", 4.2, 4.0, 0.0),
]


def atom_line(serial, name, resname, resnum, atom_type, x, y, z, charge=0.0):
    """One PDBQT ATOM record, with the columns in the right places.

    Record 1-6, serial 7-11, atom name 13-16, altloc 17, residue name 18-20,
    chain 22, residue number 23-26, coordinates 31-54, partial charge 71-76,
    atom type 78-79.
    """
    name_field = name if len(name) == 4 else f" {name:<3}"
    return (
        f"ATOM  {serial:>5} {name_field}{'':1}{resname:>3} "
        f"{'P':1}{resnum:>4}{'':1}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00    "
        f"{charge:>6.3f} {atom_type:<2}"
    )


def build_pdbqt(atoms, branches, torsdof=None, as_pdb=False) -> str:
    """Assemble a PDBQT (or a plain PDB when `as_pdb`) from atoms and branches."""
    lines = []
    if not as_pdb:
        lines.append("ROOT")
    for atom in atoms:
        serial, name, resname, resnum, atom_type, x, y, z = atom[:8]
        if as_pdb:
            lines.append(
                atom_line(serial, name, resname, resnum, "", x, y, z).rstrip()
            )
        else:
            lines.append(atom_line(serial, name, resname, resnum, atom_type, x, y, z))
    if not as_pdb:
        lines.append("ENDROOT")
        for first, second in branches:
            lines.append(f"BRANCH {first:>3} {second:>3}")
            lines.append(f"ENDBRANCH {first:>3} {second:>3}")
        if torsdof is not None:
            lines.append(f"TORSDOF {torsdof}")
    return "\n".join(lines) + "\n"


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


class AtomReadingTests(unittest.TestCase):
    def test_atoms_are_read_with_serial_name_type_and_charge(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=0)
            )
            atoms = vpp.atoms_by_serial(path)
            self.assertEqual(len(atoms), 8)
            self.assertEqual(atoms[1]["atom_name"], "N")
            self.assertEqual(atoms[1]["residue_name"], "GLY")
            self.assertEqual(atoms[4]["atom_type"], "OA")
            self.assertEqual(atoms[8]["label"], "GLY2:OXT")

    def test_a_duplicated_serial_is_fatal(self):
        # The torsion tree refers to atoms by serial, so a duplicate makes the
        # tree ambiguous rather than merely untidy.
        duplicated = list(DIPEPTIDE) + [
            (1, "CB", "GLY", 1, "C", 9.0, 9.0, 9.0)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt", build_pdbqt(duplicated, [])
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.atoms_by_serial(path)
            self.assertIn("appears twice", str(raised.exception))

    def test_a_file_with_no_atoms_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", "ROOT\nENDROOT\nTORSDOF 0\n")
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.atoms_by_serial(path)
            self.assertIn("no atom records", str(raised.exception))

    def test_a_missing_file_is_fatal(self):
        with self.assertRaises(vpp.PdbqtVerificationError):
            vpp.atoms_by_serial(Path("/nonexistent/l.pdbqt"))


class TorsionTreeTests(unittest.TestCase):
    def test_branch_records_are_read_as_bonds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary),
                "l.pdbqt",
                build_pdbqt(DIPEPTIDE, [(1, 2), (2, 3)], torsdof=2),
            )
            self.assertEqual(vpp.parse_torsion_tree(path), [(1, 2), (2, 3)])

    def test_a_tree_with_no_branches_is_a_rigid_ligand(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=0)
            )
            self.assertEqual(vpp.parse_torsion_tree(path), [])

    def test_a_malformed_branch_record_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(DIPEPTIDE, []) + "BRANCH 1\n",
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.parse_torsion_tree(path)
            self.assertIn("malformed BRANCH", str(raised.exception))

    def test_non_integer_branch_serials_are_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(DIPEPTIDE, []) + "BRANCH one two\n",
            )
            with self.assertRaises(vpp.PdbqtVerificationError):
                vpp.parse_torsion_tree(path)

    def test_torsdof_is_read_and_absence_gives_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with_count = write(
                directory, "a.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=11)
            )
            without = write(directory, "b.pdbqt", build_pdbqt(DIPEPTIDE, []))
            self.assertEqual(vpp.torsdof(with_count), 11)
            self.assertIsNone(vpp.torsdof(without))


class ClassificationTests(unittest.TestCase):
    def atoms(self, extra=()):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(list(DIPEPTIDE) + list(extra), []),
            )
            return vpp.atoms_by_serial(path)

    def test_phi_is_recognised_from_n_ca_within_a_residue(self):
        result = vpp.classify_torsions([(1, 2)], self.atoms())
        self.assertEqual(result["backbone_phi_psi_count"], 1)
        self.assertEqual(result["backbone_phi_psi"][0]["torsion"], "phi")
        self.assertFalse(result["backbone_is_rigid"])

    def test_psi_is_recognised_from_ca_c_within_a_residue(self):
        result = vpp.classify_torsions([(2, 3)], self.atoms())
        self.assertEqual(result["backbone_phi_psi"][0]["torsion"], "psi")

    def test_omega_is_recognised_from_c_n_across_residues(self):
        # Serial 3 is GLY1 C, serial 5 is GLY2 N: the amide bond.
        result = vpp.classify_torsions([(3, 5)], self.atoms())
        self.assertEqual(result["amide_omega_count"], 1)
        self.assertEqual(result["amide_omega"][0]["torsion"], "omega")

    def test_a_side_chain_bond_is_classified_as_such(self):
        result = vpp.classify_torsions([(11, 9)], self.atoms(SERINE_EXTRA))
        self.assertEqual(result["side_chain_count"], 1)
        self.assertEqual(result["backbone_phi_psi_count"], 0)

    def test_a_fully_rigid_backbone_is_reported(self):
        result = vpp.classify_torsions([(11, 9), (9, 10)], self.atoms(SERINE_EXTRA))
        self.assertTrue(result["backbone_is_rigid"])
        self.assertEqual(result["side_chain_count"], 2)

    def test_an_unresolvable_serial_is_reported_not_dropped(self):
        # Dropping it would understate the backbone torsion count, which is
        # the number the rigidification check turns on.
        result = vpp.classify_torsions([(1, 999)], self.atoms())
        self.assertEqual(len(result["unresolved"]), 1)
        self.assertIn("999", result["unresolved"][0])
        self.assertFalse(result["backbone_is_rigid"])

    def test_a_ca_c_pair_in_different_residues_is_not_called_psi(self):
        # Serial 2 is GLY1 CA, serial 7 is GLY2 C. Same names as psi, but
        # across residues, so it must not be classified as a backbone torsion.
        result = vpp.classify_torsions([(2, 7)], self.atoms())
        self.assertEqual(result["backbone_phi_psi_count"], 0)
        self.assertEqual(result["side_chain_count"], 1)

    def test_bond_labels_name_both_atoms(self):
        result = vpp.classify_torsions([(1, 2)], self.atoms())
        self.assertEqual(result["backbone_phi_psi"][0]["bond"], "GLY1:N - GLY1:CA")


class CensusTests(unittest.TestCase):
    def test_atom_types_are_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", build_pdbqt(DIPEPTIDE, []))
            census = vpp.atom_type_census(vpp.atoms_by_serial(path))
            self.assertEqual(census, {"C": 4, "N": 2, "OA": 2})

    def test_named_atoms_report_their_type_and_charge(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", build_pdbqt(DIPEPTIDE, []))
            typing = vpp.types_for_atom_names(
                vpp.atoms_by_serial(path), ["OXT"]
            )
            self.assertIn("GLY2:OXT", typing)
            self.assertEqual(typing["GLY2:OXT"]["atom_type"], "OA")


class VerificationTests(unittest.TestCase):
    def test_a_silently_unrigidified_tree_is_caught(self):
        # The scenario the checker exists for: a valid PDBQT that still
        # carries every backbone torsion because the SMARTS did not match.
        backbone_torsions = [(1, 2), (2, 3), (5, 6), (6, 7)]
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(DIPEPTIDE, backbone_torsions, torsdof=4),
            )
            record = vpp.verify(path, None, "d", expect_backbone_rigid=True)
            self.assertFalse(record["verified"])
            self.assertEqual(record["torsion_tree"]["backbone_phi_psi_count"], 4)
            failure = record["verification_failures"][0]
            self.assertIn("did not match on this Meeko build", failure)

    def test_the_same_tree_passes_when_rigidity_was_not_expected(self):
        # Without the expectation the backbone torsions are reported but not a
        # failure: a flexible tree is a legitimate thing to prepare.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(DIPEPTIDE, [(1, 2), (2, 3)], torsdof=2),
            )
            record = vpp.verify(path, None, "d", expect_backbone_rigid=False)
            self.assertTrue(record["verified"])
            self.assertEqual(record["torsion_tree"]["backbone_phi_psi_count"], 2)

    def test_a_rotatable_amide_always_fails(self):
        # No expectation flag needed: an omega torsion means the molecule was
        # not treated as a peptide at all.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(DIPEPTIDE, [(3, 5)], torsdof=1),
            )
            record = vpp.verify(path, None, "d")
            self.assertFalse(record["verified"])
            self.assertIn("not treated as a peptide", record["verification_failures"][0])

    def test_torsdof_disagreeing_with_the_branch_count_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(DIPEPTIDE, [(11, 9)], torsdof=7),
            )
            record = vpp.verify(path, None, "d")
            self.assertFalse(record["torsdof"]["agrees_with_branch_count"])
            self.assertTrue(
                any("disagrees" in f for f in record["verification_failures"])
            )

    def test_coordinate_preservation_passes_when_nothing_moved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = write(
                directory, "src.pdb", build_pdbqt(DIPEPTIDE, [], as_pdb=True)
            )
            path = write(
                directory, "l.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=0)
            )
            record = vpp.verify(path, source, "d")
            preservation = record["coordinate_preservation"]
            self.assertTrue(preservation["coordinates_preserved"])
            self.assertEqual(preservation["source_heavy_atoms"], 8)
            self.assertEqual(preservation["pdbqt_heavy_atoms"], 8)
            self.assertTrue(record["verified"])

    def test_a_moved_atom_is_caught_and_named(self):
        moved = [
            atom if atom[1] != "CA" else
            (atom[0], atom[1], atom[2], atom[3], atom[4], atom[5], atom[6], atom[7] + 1.0)
            for atom in DIPEPTIDE
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = write(
                directory, "src.pdb", build_pdbqt(DIPEPTIDE, [], as_pdb=True)
            )
            path = write(directory, "l.pdbqt", build_pdbqt(moved, [], torsdof=0))
            record = vpp.verify(path, source, "d")
            self.assertFalse(record["verified"])
            self.assertFalse(
                record["coordinate_preservation"]["coordinates_preserved"]
            )
            self.assertTrue(
                any("not preserved" in f for f in record["verification_failures"])
            )

    def test_a_dropped_heavy_atom_is_caught(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = write(
                directory, "src.pdb", build_pdbqt(DIPEPTIDE, [], as_pdb=True)
            )
            path = write(
                directory, "l.pdbqt", build_pdbqt(DIPEPTIDE[:-1], [], torsdof=0)
            )
            record = vpp.verify(path, source, "d")
            self.assertFalse(record["verified"])
            self.assertTrue(
                any("absent from the PDBQT" in f
                    for f in record["verification_failures"])
            )

    def test_omitting_the_source_leaves_preservation_unestablished(self):
        # It must not read as a pass: not checking is different from checking
        # and finding nothing wrong.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=0)
            )
            record = vpp.verify(path, None, "d")
            self.assertFalse(record["coordinate_preservation"]["checked"])
            self.assertIn(
                "not established",
                record["coordinate_preservation"]["reason"],
            )

    def test_the_record_warns_against_rename_atoms(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=0)
            )
            record = vpp.verify(path, None, "d")
            self.assertIn(
                "--rename_atoms",
                record["expectations"]["rename_atoms_must_not_be_used"],
            )

    def test_a_clean_backbone_rigid_ligand_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            atoms = list(DIPEPTIDE) + list(SERINE_EXTRA)
            source = write(directory, "src.pdb", build_pdbqt(atoms, [], as_pdb=True))
            path = write(
                directory, "l.pdbqt",
                build_pdbqt(atoms, [(11, 9), (9, 10)], torsdof=2),
            )
            record = vpp.verify(path, source, "d", expect_backbone_rigid=True)
            self.assertTrue(record["verified"])
            self.assertEqual(record["verification_failures"], [])
            self.assertTrue(record["torsion_tree"]["backbone_is_rigid"])
            self.assertEqual(record["torsion_tree"]["side_chain_count"], 2)


class CommandLineTests(unittest.TestCase):
    def test_a_verified_ligand_exits_zero_and_writes_its_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = write(
                directory, "l.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=0)
            )
            out = directory / "check.json"
            code = vpp.main(
                [str(path), "--design-id", "d", "--json-out", str(out)]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["design_id"], "d")

    def test_a_failing_ligand_exits_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                build_pdbqt(DIPEPTIDE, [(3, 5)], torsdof=1),
            )
            self.assertEqual(vpp.main([str(path), "--design-id", "d"]), 1)

    def test_an_unreadable_pdbqt_exits_two(self):
        self.assertEqual(
            vpp.main(["/nonexistent/l.pdbqt", "--design-id", "d"]), 2
        )

    def test_an_existing_output_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = write(
                directory, "l.pdbqt", build_pdbqt(DIPEPTIDE, [], torsdof=0)
            )
            out = write(directory, "check.json", "{}")
            with self.assertRaises(SystemExit):
                vpp.main(
                    [str(path), "--design-id", "d", "--json-out", str(out)]
                )


if __name__ == "__main__":
    unittest.main()
