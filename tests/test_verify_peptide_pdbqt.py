"""Tests for verifying a Meeko-produced peptide PDBQT.

Meeko is not installed here, so these tests build PDBQT files by hand. That is
the right way round in any case: the point of the checker is to catch an
output that is wrong, and a wrong output is easier to construct than to
provoke.

**These files are synthetic.** They were written to match the Meeko v0.7.1
source reading of `REMARK INDEX MAP`, not copied from a real output, and no
real Meeko PDBQT was available. A pass here is evidence that the checker
behaves as specified; it is **not** validation of the real Epicac files, and
the format itself remains unconfirmed against one.

Two fixtures matter:

`meeko_style_pdbqt` reproduces the observed Epicac failure: residue `UNL`,
residue number 1, blank chain, and atom names that are bare element letters
repeated many times, so identities like `:1 UNL C` occur several times over.
Nothing can be keyed on them.

`ligand_record` reproduces the `atom_mapping` that stage 3 writes, which is
what gives the index map's input indices their chemical meaning.
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

# The original dipeptide, as stage 3 knows it: GLY1 (N, CA, C, O) then
# GLY2 (N, CA, C, OXT). Coordinates are arbitrary but consistent throughout.
ORIGINAL_ATOMS = [
    ("GLY", 1, "N", "N", 0.0, 0.0, 0.0),
    ("GLY", 1, "CA", "C", 1.5, 0.0, 0.0),
    ("GLY", 1, "C", "C", 2.5, 1.0, 0.0),
    ("GLY", 1, "O", "O", 2.5, 2.2, 0.0),
    ("GLY", 2, "N", "N", 3.8, 0.8, 0.0),
    ("GLY", 2, "CA", "C", 4.9, 1.7, 0.0),
    ("GLY", 2, "C", "C", 6.2, 1.0, 0.0),
    ("GLY", 2, "OXT", "O", 7.0, 1.6, 0.0),
]

# Two non-polar hydrogens, which Meeko merges, so they appear in the ligand
# record but never in the PDBQT or its index map.
MERGED_HYDROGENS = [("GLY", 1, "HA2", "H"), ("GLY", 1, "HA3", "H")]

# A serine side chain, for a genuine side-chain torsion (CA-CB and CB-OG).
SERINE_ATOMS = [
    ("SER", 3, "N", "N", 7.5, 3.0, 0.0),
    ("SER", 3, "CA", "C", 8.5, 3.6, 0.0),
    ("SER", 3, "CB", "C", 9.4, 4.4, 0.0),
    ("SER", 3, "OG", "O", 10.2, 5.1, 0.0),
]

# AutoDock types, by element, as Meeko would assign for these atoms.
TYPE_BY_ELEMENT = {"N": "N", "C": "C", "O": "OA", "H": "HD"}


def atom_line(serial, name, resname, resnum, chain, atom_type, x, y, z,
              charge=0.0, pdb=False):
    """One ATOM record with the columns in the right places.

    Record 1-6, serial 7-11, atom name 13-16, altloc 17, residue name 18-20,
    chain 22, residue number 23-26, coordinates 31-54, partial charge 71-76,
    atom type 78-79.
    """
    name_field = name if len(name) == 4 else f" {name:<3}"
    line = (
        f"ATOM  {serial:>5} {name_field}{'':1}{resname:>3} "
        f"{chain:1}{resnum:>4}{'':1}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
    )
    if pdb:
        return line
    return line + f"    {charge:>6.3f} {atom_type:<2}"


def source_pdb(atoms) -> str:
    """The source structure, with its real residue and atom names."""
    lines = [
        atom_line(i, name, resname, resnum, "P", "", x, y, z, pdb=True)
        for i, (resname, resnum, name, _element, x, y, z)
        in enumerate(atoms, start=1)
    ]
    return "\n".join(lines) + "\n"


def index_map_lines(pairs) -> list:
    """Render `REMARK INDEX MAP` pairs the way Meeko's writer does.

    Each pair is appended as " %d %d" while the line stays strictly under 79
    characters, then a new line restarts with the full prefix. Reproduced from
    `break_long_remark_lines` so the wrapping is exercised rather than assumed.
    """
    remarks = [vpp.INDEX_MAP_PREFIX]
    for input_index, serial in pairs:
        piece = f" {input_index} {serial}"
        if len(remarks[-1]) + len(piece) < 79:
            remarks[-1] += piece
        else:
            remarks.append(vpp.INDEX_MAP_PREFIX + piece)
    return remarks


def meeko_style_pdbqt(atoms, branches, torsdof=None, index_pairs=None,
                      generic=True, coordinates=None) -> str:
    """Build a PDBQT as Meeko writes one for a ligand read from an SDF.

    With `generic`, every atom gets residue `UNL`, residue number 1, a blank
    chain and a bare element letter for a name, which is what the real
    `spec_0574` output carries and why nothing can be keyed on it.

    `index_pairs` are (input index, PDBQT serial) pairs, both 1-based.
    `coordinates`, when given, replaces the coordinates by serial, for testing
    that a moved atom is caught.
    """
    lines = []
    if index_pairs is not None:
        lines.extend(index_map_lines(index_pairs))
    lines.append("ROOT")
    for serial, (resname, resnum, name, element, x, y, z) in enumerate(
        atoms, start=1
    ):
        if coordinates and serial in coordinates:
            x, y, z = coordinates[serial]
        lines.append(
            atom_line(
                serial,
                element if generic else name,
                "UNL" if generic else resname,
                1 if generic else resnum,
                "" if generic else "P",
                TYPE_BY_ELEMENT[element],
                x, y, z,
            )
        )
    lines.append("ENDROOT")
    for first, second in branches:
        lines.append(f"BRANCH {first:>3} {second:>3}")
        lines.append(f"ENDBRANCH {first:>3} {second:>3}")
    if torsdof is not None:
        lines.append(f"TORSDOF {torsdof}")
    return "\n".join(lines) + "\n"


def ligand_record(atoms, hydrogens=()) -> str:
    """A stage-3 ligand record carrying `atom_mapping`."""
    mapping = []
    for index, (resname, resnum, name, element, *_rest) in enumerate(atoms):
        mapping.append(
            {
                "molecule_index_zero_based": index,
                "element": element,
                "author_chain": "P",
                "residue_name": resname,
                "residue_seqid": resnum,
                "atom_name": name,
            }
        )
    for offset, (resname, resnum, name, element) in enumerate(hydrogens):
        mapping.append(
            {
                "molecule_index_zero_based": len(atoms) + offset,
                "element": element,
                "author_chain": "P",
                "residue_name": resname,
                "residue_seqid": resnum,
                "atom_name": name,
            }
        )
    return json.dumps({"design_id": "d", "atom_mapping": mapping}, indent=2)


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


def identity_pairs(count: int):
    """Index map pairs where input index i maps to serial i."""
    return [(i, i) for i in range(1, count + 1)]


class IndexMapParsingTests(unittest.TestCase):
    def test_pairs_are_read_as_serial_to_input_index(self):
        # Meeko writes (input index, PDBQT serial); the parser keys by serial.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0,
                    index_pairs=[(1, 5), (2, 6)],
                ),
            )
            self.assertEqual(vpp.parse_index_map(path), {5: 1, 6: 2})

    def test_pairs_wrapped_over_several_remark_lines_are_all_read(self):
        # The wrapping is greedy below 79 characters with the prefix repeated,
        # so a parser that read only the first line would lose most of the map.
        pairs = [(i, i) for i in range(1, 41)]
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0, index_pairs=pairs
                ),
            )
            lines = [
                line for line in path.read_text().splitlines()
                if line.startswith(vpp.INDEX_MAP_PREFIX)
            ]
            self.assertGreater(len(lines), 1, "fixture must exercise wrapping")
            self.assertTrue(all(len(line) < 79 for line in lines))
            self.assertEqual(len(vpp.parse_index_map(path)), 40)

    def test_a_file_with_no_index_map_gives_an_empty_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0),
            )
            self.assertEqual(vpp.parse_index_map(path), {})

    def test_an_odd_number_of_values_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0)
                + f"{vpp.INDEX_MAP_PREFIX} 1 1 2\n",
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.parse_index_map(path)
            self.assertIn("odd", str(raised.exception))

    def test_a_repeated_serial_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0,
                    index_pairs=[(1, 3), (2, 3)],
                ),
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.parse_index_map(path)
            self.assertIn("serial 3 twice", str(raised.exception))

    def test_a_repeated_input_index_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0,
                    index_pairs=[(4, 1), (4, 2)],
                ),
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.parse_index_map(path)
            self.assertIn("input index 4 twice", str(raised.exception))

    def test_a_non_integer_value_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0)
                + f"{vpp.INDEX_MAP_PREFIX} 1 one\n",
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.parse_index_map(path)
            self.assertIn("non-integer", str(raised.exception))

    def test_a_zero_index_is_rejected_as_not_one_based(self):
        # Both numbers are 1-based in Meeko v0.7.1. A zero means either the
        # format changed or something has already been decremented.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0)
                + f"{vpp.INDEX_MAP_PREFIX} 0 1\n",
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.parse_index_map(path)
            self.assertIn("1-based", str(raised.exception))


class LigandRecordTests(unittest.TestCase):
    def test_atom_mapping_is_keyed_by_zero_based_molecule_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            mapping = vpp.load_ligand_atom_mapping(path)
            self.assertEqual(len(mapping), 8)
            self.assertEqual(mapping[0].label, "GLY1:N")
            self.assertEqual(mapping[7].label, "GLY2:OXT")

    def test_a_record_without_atom_mapping_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "lig.json", json.dumps({"a": 1}))
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.load_ligand_atom_mapping(path)
            self.assertIn("atom_mapping", str(raised.exception))

    def test_invalid_json_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "lig.json", "{not json")
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.load_ligand_atom_mapping(path)
            self.assertIn("not valid JSON", str(raised.exception))

    def test_a_duplicated_molecule_index_is_fatal(self):
        record = json.loads(ligand_record(ORIGINAL_ATOMS))
        record["atom_mapping"][1]["molecule_index_zero_based"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "lig.json", json.dumps(record))
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.load_ligand_atom_mapping(path)
            self.assertIn("appears twice", str(raised.exception))

    def test_a_heavy_atom_without_a_name_is_fatal(self):
        record = json.loads(ligand_record(ORIGINAL_ATOMS))
        del record["atom_mapping"][2]["atom_name"]
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "lig.json", json.dumps(record))
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.load_ligand_atom_mapping(path)
            self.assertIn("cannot be identified", str(raised.exception))

    def test_a_hydrogen_without_a_name_is_tolerated(self):
        # Hydrogens added after the PDB stage carry no PDB identity. Meeko
        # merges them, so they never need identifying.
        record = json.loads(ligand_record(ORIGINAL_ATOMS))
        record["atom_mapping"].append(
            {"molecule_index_zero_based": 8, "element": "H"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "lig.json", json.dumps(record))
            mapping = vpp.load_ligand_atom_mapping(path)
            self.assertTrue(mapping[8].is_hydrogen)

    def test_a_missing_record_file_is_fatal(self):
        with self.assertRaises(vpp.PdbqtVerificationError):
            vpp.load_ligand_atom_mapping(Path("/nonexistent/lig.json"))


class IdentityRecoveryTests(unittest.TestCase):
    """The core of PEP-001: recovering identity a real PDBQT does not carry."""

    def build(self, directory, index_pairs=None, atoms=None, hydrogens=(),
              branches=(), torsdof=0, coordinates=None):
        atoms = ORIGINAL_ATOMS if atoms is None else atoms
        if index_pairs is None:
            index_pairs = identity_pairs(len(atoms))
        pdbqt = write(
            directory, "l.pdbqt",
            meeko_style_pdbqt(
                atoms, list(branches), torsdof=torsdof,
                index_pairs=index_pairs, coordinates=coordinates,
            ),
        )
        record = write(directory, "lig.json", ligand_record(atoms, hydrogens))
        return pdbqt, record

    def test_a_complete_index_map_reconstructs_the_original_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record = self.build(Path(temporary))
            atoms = vpp.atoms_by_serial(pdbqt)
            recovery = vpp.recover_identities(
                atoms,
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertTrue(recovery.established, recovery.reasons)
            labels = [recovery.by_serial[s].label for s in sorted(atoms)]
            self.assertEqual(
                labels,
                ["GLY1:N", "GLY1:CA", "GLY1:C", "GLY1:O",
                 "GLY2:N", "GLY2:CA", "GLY2:C", "GLY2:OXT"],
            )

    def test_the_pdbqt_identities_really_are_useless_without_the_map(self):
        # Guards the premise: if this fixture ever stopped being generic and
        # repeated, the rest of these tests would prove nothing.
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, _ = self.build(Path(temporary))
            atoms = vpp.atoms_by_serial(pdbqt)
            self.assertEqual({str(a["residue_name"]) for a in atoms.values()},
                             {"UNL"})
            names = [str(a["atom_name"]) for a in atoms.values()]
            self.assertGreater(len(names), len(set(names)))

    def test_the_mapping_is_actually_used_rather_than_assumed_to_be_identity(self):
        # Reverse the correspondence: input index i maps to serial 9 - i. If
        # the code quietly assumed serial == index, every identity would come
        # back wrong, and an identity map could never reveal that.
        reversed_pairs = [(i, 9 - i) for i in range(1, 9)]
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record = self.build(
                Path(temporary), index_pairs=reversed_pairs
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertTrue(recovery.established, recovery.reasons)
            self.assertEqual(recovery.by_serial[8].label, "GLY1:N")
            self.assertEqual(recovery.by_serial[1].label, "GLY2:OXT")

    def test_merged_hydrogens_absent_from_the_map_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record = self.build(
                Path(temporary), hydrogens=MERGED_HYDROGENS
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertTrue(recovery.established, recovery.reasons)
            self.assertEqual(
                recovery.hydrogens_absent_from_pdbqt, ["GLY1:HA2", "GLY1:HA3"]
            )
            self.assertEqual(recovery.heavy_atoms_absent_from_pdbqt, [])

    def test_a_heavy_atom_absent_from_the_map_fails_closed(self):
        # Eight atoms in the molecule, seven in the map: the eighth cannot be
        # identified, so it cannot be checked.
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record = self.build(
                Path(temporary), index_pairs=identity_pairs(7)
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertFalse(recovery.established)
            self.assertEqual(
                recovery.heavy_atoms_absent_from_pdbqt, ["GLY2:OXT"]
            )

    def test_a_serial_with_no_map_entry_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record = self.build(
                Path(temporary),
                index_pairs=[(i, i) for i in range(1, 8)] + [(8, 99)],
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertFalse(recovery.established)
            self.assertTrue(
                any("serial 8 has no entry" in r for r in recovery.reasons)
            )

    def test_an_input_index_outside_the_record_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record = self.build(
                Path(temporary),
                index_pairs=[(i, i) for i in range(1, 8)] + [(77, 8)],
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertFalse(recovery.established)
            self.assertTrue(
                any("absent from the ligand record" in r
                    for r in recovery.reasons)
            )

    def test_no_index_map_at_all_fails_closed_with_the_remedy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0),
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt), vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertFalse(recovery.established)
            self.assertIn("--add_index_map", recovery.reasons[0])


class DirectNameModeTests(unittest.TestCase):
    """The fallback, which must refuse exactly when it cannot be trusted."""

    def test_generic_residue_names_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0),
            )
            recovery = vpp.identities_from_pdbqt_names(
                vpp.atoms_by_serial(path)
            )
            self.assertFalse(recovery.established)
            self.assertTrue(
                any("generic residue name" in r for r in recovery.reasons)
            )
            self.assertTrue(
                any("--ligand-record" in r for r in recovery.reasons)
            )

    def test_repeated_identities_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0),
            )
            recovery = vpp.identities_from_pdbqt_names(
                vpp.atoms_by_serial(path)
            )
            self.assertTrue(
                any("repeats atom identities" in r for r in recovery.reasons)
            )

    def test_real_unique_names_are_accepted(self):
        # A PDBQT that genuinely carries identities, as the earlier synthetic
        # fixtures did, still verifies without a ligand record.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0, generic=False
                ),
            )
            recovery = vpp.identities_from_pdbqt_names(
                vpp.atoms_by_serial(path)
            )
            self.assertTrue(recovery.established, recovery.reasons)
            self.assertEqual(recovery.by_serial[1].label, "GLY1:N")


class AtomReadingTests(unittest.TestCase):
    def test_atoms_are_read_with_serial_name_type_charge_and_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0, generic=False
                ),
            )
            atoms = vpp.atoms_by_serial(path)
            self.assertEqual(len(atoms), 8)
            self.assertEqual(atoms[1]["atom_name"], "N")
            self.assertEqual(atoms[1]["residue_name"], "GLY")
            self.assertEqual(atoms[4]["atom_type"], "OA")
            self.assertEqual(atoms[8]["position"], (7.0, 1.6, 0.0))

    def test_a_duplicated_serial_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0).replace(
                    "ATOM      2", "ATOM      1", 1
                ),
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


class TorsionTreeReadingTests(unittest.TestCase):
    def test_branch_records_are_read_as_bonds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [(1, 2), (2, 3)], torsdof=2),
            )
            self.assertEqual(vpp.parse_torsion_tree(path), [(1, 2), (2, 3)])

    def test_a_tree_with_no_branches_is_a_rigid_ligand(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0),
            )
            self.assertEqual(vpp.parse_torsion_tree(path), [])

    def test_a_malformed_branch_record_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, []) + "BRANCH 1\n",
            )
            with self.assertRaises(vpp.PdbqtVerificationError) as raised:
                vpp.parse_torsion_tree(path)
            self.assertIn("malformed BRANCH", str(raised.exception))

    def test_non_integer_branch_serials_are_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, []) + "BRANCH one two\n",
            )
            with self.assertRaises(vpp.PdbqtVerificationError):
                vpp.parse_torsion_tree(path)

    def test_torsdof_is_read_and_absence_gives_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with_count = write(
                directory, "a.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=11),
            )
            without = write(
                directory, "b.pdbqt", meeko_style_pdbqt(ORIGINAL_ATOMS, [])
            )
            self.assertEqual(vpp.torsdof(with_count), 11)
            self.assertIsNone(vpp.torsdof(without))


class ClassificationTests(unittest.TestCase):
    """Torsions classified from recovered identities, not PDBQT names."""

    def recovery(self, atoms=None):
        atoms = ORIGINAL_ATOMS if atoms is None else atoms
        by_serial = {
            serial: vpp.AtomIdentity(
                residue_name=resname, residue_seqid=resnum,
                atom_name=name, element=element,
            )
            for serial, (resname, resnum, name, element, *_r)
            in enumerate(atoms, start=1)
        }
        return vpp.IdentityRecovery(
            by_serial=by_serial, method="test", established=True
        )

    def test_phi_is_recognised_from_n_ca_within_a_residue(self):
        result = vpp.classify_torsions([(1, 2)], self.recovery())
        self.assertEqual(result["backbone_phi_psi_count"], 1)
        self.assertEqual(result["backbone_phi_psi"][0]["torsion"], "phi")
        self.assertFalse(result["backbone_is_rigid"])

    def test_psi_is_recognised_from_ca_c_within_a_residue(self):
        result = vpp.classify_torsions([(2, 3)], self.recovery())
        self.assertEqual(result["backbone_phi_psi"][0]["torsion"], "psi")

    def test_omega_is_recognised_from_c_n_across_residues(self):
        # Serial 3 is GLY1 C, serial 5 is GLY2 N: the amide bond.
        result = vpp.classify_torsions([(3, 5)], self.recovery())
        self.assertEqual(result["amide_omega_count"], 1)
        self.assertEqual(result["amide_omega"][0]["torsion"], "omega")

    def test_a_side_chain_bond_is_classified_as_such(self):
        atoms = ORIGINAL_ATOMS + SERINE_ATOMS
        # Serials 10 and 11 are SER3 CA and SER3 CB.
        result = vpp.classify_torsions([(10, 11)], self.recovery(atoms))
        self.assertEqual(result["side_chain_count"], 1)
        self.assertEqual(result["backbone_phi_psi_count"], 0)

    def test_a_fully_rigid_backbone_is_reported(self):
        atoms = ORIGINAL_ATOMS + SERINE_ATOMS
        result = vpp.classify_torsions([(10, 11), (11, 12)], self.recovery(atoms))
        self.assertTrue(result["backbone_is_rigid"])
        self.assertEqual(result["side_chain_count"], 2)

    def test_an_unresolvable_serial_is_reported_not_dropped(self):
        # Dropping it would understate the backbone torsion count, which is
        # the number the rigidification check turns on.
        result = vpp.classify_torsions([(1, 999)], self.recovery())
        self.assertEqual(len(result["unresolved"]), 1)
        self.assertIn("999", result["unresolved"][0])
        self.assertFalse(result["backbone_is_rigid"])

    def test_a_ca_c_pair_in_different_residues_is_not_called_psi(self):
        # Serial 2 is GLY1 CA, serial 7 is GLY2 C. Same names as psi, but
        # across residues, so it must not be a backbone torsion.
        result = vpp.classify_torsions([(2, 7)], self.recovery())
        self.assertEqual(result["backbone_phi_psi_count"], 0)
        self.assertEqual(result["side_chain_count"], 1)

    def test_bond_labels_name_both_recovered_atoms(self):
        result = vpp.classify_torsions([(1, 2)], self.recovery())
        self.assertEqual(
            result["backbone_phi_psi"][0]["bond"], "GLY1:N - GLY1:CA"
        )


class CensusTests(unittest.TestCase):
    def test_atom_types_are_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0),
            )
            census = vpp.atom_type_census(vpp.atoms_by_serial(path))
            self.assertEqual(census, {"C": 4, "N": 2, "OA": 2})

    def test_named_atoms_report_their_type_and_charge_via_recovery(self):
        # The name comes from the recovery, so this works although the PDBQT
        # itself calls the atom "O" in residue UNL.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0,
                    index_pairs=identity_pairs(8),
                ),
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            atoms = vpp.atoms_by_serial(pdbqt)
            recovery = vpp.recover_identities(
                atoms, vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            typing = vpp.types_for_atom_names(atoms, recovery, ["OXT"])
            self.assertIn("GLY2:OXT", typing)
            self.assertEqual(typing["GLY2:OXT"]["atom_type"], "OA")
            self.assertEqual(typing["GLY2:OXT"]["serial"], 8)


class EndToEndVerificationTests(unittest.TestCase):
    def build(self, directory, branches=(), torsdof=0, index_pairs=None,
              atoms=None, coordinates=None, hydrogens=()):
        atoms = ORIGINAL_ATOMS if atoms is None else atoms
        if index_pairs is None:
            index_pairs = identity_pairs(len(atoms))
        pdbqt = write(
            directory, "l.pdbqt",
            meeko_style_pdbqt(
                atoms, list(branches), torsdof=torsdof,
                index_pairs=index_pairs, coordinates=coordinates,
            ),
        )
        record = write(directory, "lig.json", ligand_record(atoms, hydrogens))
        source = write(directory, "src.pdb", source_pdb(atoms))
        return pdbqt, record, source

    def test_a_generic_meeko_pdbqt_verifies_with_the_ligand_record(self):
        # The headline PEP-001 case: the file that previously produced no
        # record at all now verifies end to end.
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(Path(temporary))
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertTrue(result["verified"], result["verification_failures"])
            self.assertTrue(result["identity_recovery"]["established"])
            self.assertTrue(
                result["coordinate_preservation"]["coordinates_preserved"]
            )
            self.assertEqual(
                result["coordinate_preservation"]["heavy_atoms_compared"], 8
            )

    def test_the_same_file_fails_closed_without_the_ligand_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, _record, source = self.build(Path(temporary))
            result = vpp.verify(pdbqt, source, "d")
            self.assertFalse(result["verified"])
            self.assertFalse(result["identity_recovery"]["established"])
            self.assertFalse(result["coordinate_preservation"]["checked"])
            self.assertIn(
                "identities could not be established",
                result["verification_failures"][0],
            )

    def test_a_deliberate_coordinate_change_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(
                Path(temporary), coordinates={2: (1.5, 0.0, 1.0)}
            )
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertFalse(result["verified"])
            preservation = result["coordinate_preservation"]
            self.assertFalse(preservation["coordinates_preserved"])
            self.assertEqual(preservation["moved_atom_count"], 1)
            self.assertEqual(preservation["moved_atoms"][0]["atom"], "GLY1:CA")
            self.assertAlmostEqual(
                preservation["moved_atoms"][0]["moved_angstrom"], 1.0, places=3
            )

    def test_a_dropped_heavy_atom_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS[:-1], [], torsdof=0,
                    index_pairs=identity_pairs(7),
                ),
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            source = write(directory, "src.pdb", source_pdb(ORIGINAL_ATOMS))
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertFalse(result["verified"])
            # The atom is absent from the molecule's map, so recovery already
            # fails closed and names the atom.
            self.assertEqual(
                result["identity_recovery"]["heavy_atoms_absent_from_pdbqt"],
                ["GLY2:OXT"],
            )

    def test_merged_hydrogens_do_not_break_a_clean_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(
                Path(temporary), hydrogens=MERGED_HYDROGENS
            )
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertTrue(result["verified"], result["verification_failures"])
            self.assertEqual(
                result["identity_recovery"]["hydrogens_absent_from_pdbqt"],
                ["GLY1:HA2", "GLY1:HA3"],
            )

    def test_a_silently_unrigidified_tree_is_caught(self):
        # The scenario the checker exists for: a valid PDBQT that still
        # carries every backbone torsion because the SMARTS did not match.
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(
                Path(temporary),
                branches=[(1, 2), (2, 3), (5, 6), (6, 7)],
                torsdof=4,
            )
            result = vpp.verify(
                pdbqt, source, "d", ligand_record=record,
                expect_backbone_rigid=True,
            )
            self.assertFalse(result["verified"])
            self.assertEqual(
                result["torsion_tree"]["backbone_phi_psi_count"], 4
            )
            self.assertTrue(
                any("did not match on this Meeko build" in f
                    for f in result["verification_failures"])
            )

    def test_the_same_tree_passes_when_rigidity_was_not_expected(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(
                Path(temporary), branches=[(1, 2), (2, 3)], torsdof=2
            )
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertTrue(result["verified"], result["verification_failures"])
            self.assertEqual(
                result["torsion_tree"]["backbone_phi_psi_count"], 2
            )

    def test_a_rotatable_amide_always_fails(self):
        # No expectation flag needed: an omega torsion means the molecule was
        # not treated as a peptide at all.
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(
                Path(temporary), branches=[(3, 5)], torsdof=1
            )
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("not treated as a peptide" in f
                    for f in result["verification_failures"])
            )

    def test_torsdof_disagreeing_with_the_branch_count_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(
                Path(temporary), branches=[(1, 2)], torsdof=7
            )
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertFalse(result["torsdof"]["agrees_with_branch_count"])
            self.assertTrue(
                any("disagrees" in f for f in result["verification_failures"])
            )

    def test_omitting_the_source_leaves_preservation_unestablished(self):
        # Not checking is different from checking and finding nothing wrong.
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, _source = self.build(Path(temporary))
            result = vpp.verify(pdbqt, None, "d", ligand_record=record)
            self.assertFalse(result["coordinate_preservation"]["checked"])
            self.assertIn(
                "no --source given",
                result["coordinate_preservation"]["reason"],
            )

    def test_the_record_states_the_index_map_format_is_unconfirmed(self):
        # The honesty requirement: a synthetic pass must not read as
        # validation of the real files.
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(Path(temporary))
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertIn("NOT yet confirmed", result["index_map_format"])
            self.assertIn("v0.7.1", result["index_map_format"])
            self.assertIn("not validation of a real", result["interpretation"])

    def test_the_schema_records_the_second_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdbqt, record, source = self.build(Path(temporary))
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertEqual(result["schema"], "mur-ligase-peptide-pdbqt-check/2")


class CommandLineTests(unittest.TestCase):
    def build(self, directory):
        pdbqt = write(
            directory, "l.pdbqt",
            meeko_style_pdbqt(
                ORIGINAL_ATOMS, [], torsdof=0,
                index_pairs=identity_pairs(8),
            ),
        )
        record = write(directory, "lig.json", ligand_record(ORIGINAL_ATOMS))
        source = write(directory, "src.pdb", source_pdb(ORIGINAL_ATOMS))
        return pdbqt, record, source

    def test_a_verified_ligand_exits_zero_and_writes_its_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt, record, source = self.build(directory)
            out = directory / "check.json"
            code = vpp.main([
                str(pdbqt), "--design-id", "d",
                "--source", str(source), "--ligand-record", str(record),
                "--json-out", str(out),
            ])
            self.assertEqual(code, 0)
            written = json.loads(out.read_text())
            self.assertEqual(written["design_id"], "d")
            self.assertTrue(written["verified"])

    def test_a_generic_pdbqt_without_the_record_exits_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt, _record, source = self.build(directory)
            code = vpp.main(
                [str(pdbqt), "--design-id", "d", "--source", str(source)]
            )
            self.assertEqual(code, 1)

    def test_an_unreadable_pdbqt_exits_two(self):
        self.assertEqual(
            vpp.main(["/nonexistent/l.pdbqt", "--design-id", "d"]), 2
        )

    def test_a_malformed_index_map_exits_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(ORIGINAL_ATOMS, [], torsdof=0)
                + f"{vpp.INDEX_MAP_PREFIX} 1 1 2\n",
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            self.assertEqual(
                vpp.main([
                    str(pdbqt), "--design-id", "d",
                    "--ligand-record", str(record),
                ]),
                2,
            )

    def test_an_existing_output_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt, _record, _source = self.build(directory)
            out = write(directory, "check.json", "{}")
            with self.assertRaises(SystemExit):
                vpp.main([
                    str(pdbqt), "--design-id", "d", "--json-out", str(out)
                ])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Regression tests for the three fail-closed gaps found in independent review
# of the first PEP-001 patch, 26 November 2026. Each one passes verification
# under that patch and must not.
# ---------------------------------------------------------------------------

# A three-atom source, for the duplicate-identity case. Kept small so the
# arithmetic of the false positive is readable.
TINY_SOURCE = [
    ("GLY", 1, "N", "N", 0.0, 0.0, 0.0),
    ("GLY", 1, "CA", "C", 1.5, 0.0, 0.0),
    ("GLY", 1, "O", "O", 2.5, 1.0, 0.0),
]


class FailClosedRegressionTests(unittest.TestCase):
    """The three gaps, as tests that fail before the fix."""

    def test_an_index_map_serial_with_no_atom_record_fails_closed(self):
        # Gap 1. The map names serial 9, which does not exist among the atom
        # records. Iterating the atoms alone never looks at it, so the bogus
        # entry passes unnoticed.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], torsdof=0,
                    index_pairs=identity_pairs(8) + [(9, 9)],
                ),
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertFalse(recovery.established)
            self.assertTrue(
                any("9" in reason and "no atom record" in reason
                    for reason in recovery.reasons),
                recovery.reasons,
            )

    def test_an_extra_map_serial_cannot_mask_a_dropped_heavy_atom(self):
        # Gap 1, the way it actually bites. The PDBQT is missing GLY2:OXT, but
        # the map still carries a pair for it pointing at a serial that does
        # not exist. Because the "which molecule atoms are mapped" test read
        # the map rather than the atoms, the dropped atom looked accounted for.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS[:-1], [], torsdof=0,
                    index_pairs=identity_pairs(8),
                ),
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertFalse(recovery.established)

    def test_two_molecule_indices_resolving_to_one_identity_fail_closed(self):
        # Gap 2. Molecule indices 1 and 2 both claim to be GLY1:CA.
        record_data = {
            "design_id": "d",
            "atom_mapping": [
                {"molecule_index_zero_based": 0, "element": "N",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "N"},
                {"molecule_index_zero_based": 1, "element": "C",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "CA"},
                {"molecule_index_zero_based": 2, "element": "C",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "CA"},
                {"molecule_index_zero_based": 3, "element": "O",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "O"},
            ],
        }
        four_atoms = [
            ("GLY", 1, "N", "N", 0.0, 0.0, 0.0),
            ("GLY", 1, "CA", "C", 1.5, 0.0, 0.0),
            ("GLY", 1, "CA", "C", 1.5, 0.0, 0.0),
            ("GLY", 1, "O", "O", 2.5, 1.0, 0.0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    four_atoms, [], torsdof=0, index_pairs=identity_pairs(4)
                ),
            )
            record = write(
                directory, "lig.json", json.dumps(record_data)
            )
            recovery = vpp.recover_identities(
                vpp.atoms_by_serial(pdbqt),
                vpp.parse_index_map(pdbqt),
                vpp.load_ligand_atom_mapping(record),
            )
            self.assertFalse(recovery.established)
            self.assertTrue(
                any("GLY1:CA" in reason for reason in recovery.reasons),
                recovery.reasons,
            )

    def test_a_duplicate_identity_end_to_end_is_not_a_false_positive(self):
        # Gap 2, end to end, and the reason it matters. Four PDBQT atoms, a
        # three-atom source, and a record in which two molecule indices claim
        # GLY1:CA. Both compare cleanly against the one source CA, nothing is
        # left unpaired, and the old code reported verified=True while one
        # PDBQT atom had never been checked against its own source atom.
        # Note heavy_atoms_compared (4) exceeding source_heavy_atoms (3): that
        # arithmetic is the signature of the false positive.
        record_data = {
            "design_id": "d",
            "atom_mapping": [
                {"molecule_index_zero_based": 0, "element": "N",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "N"},
                {"molecule_index_zero_based": 1, "element": "C",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "CA"},
                {"molecule_index_zero_based": 2, "element": "C",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "CA"},
                {"molecule_index_zero_based": 3, "element": "O",
                 "author_chain": "P", "residue_name": "GLY",
                 "residue_seqid": 1, "atom_name": "O"},
            ],
        }
        pdbqt_atoms = [
            ("GLY", 1, "N", "N", 0.0, 0.0, 0.0),
            ("GLY", 1, "CA", "C", 1.5, 0.0, 0.0),
            ("GLY", 1, "CA", "C", 1.5, 0.0, 0.0),
            ("GLY", 1, "O", "O", 2.5, 1.0, 0.0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    pdbqt_atoms, [], torsdof=0,
                    index_pairs=identity_pairs(4),
                ),
            )
            record = write(directory, "lig.json", json.dumps(record_data))
            source = write(directory, "src.pdb", source_pdb(TINY_SOURCE))
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertFalse(
                result["verified"],
                "duplicate recovered identities must not verify",
            )

    def test_a_missing_torsdof_is_a_verification_failure(self):
        # Gap 3. The consistency result was None rather than False when the
        # declaration was absent, and the check tested `is False`, so a file
        # with no TORSDOF line passed.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], index_pairs=identity_pairs(8)
                ),
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            source = write(directory, "src.pdb", source_pdb(ORIGINAL_ATOMS))
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("TORSDOF" in failure
                    for failure in result["verification_failures"]),
                result["verification_failures"],
            )

    def test_a_malformed_torsdof_is_a_verification_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pdbqt = write(
                directory, "l.pdbqt",
                meeko_style_pdbqt(
                    ORIGINAL_ATOMS, [], index_pairs=identity_pairs(8)
                ) + "TORSDOF many\n",
            )
            record = write(
                directory, "lig.json", ligand_record(ORIGINAL_ATOMS)
            )
            source = write(directory, "src.pdb", source_pdb(ORIGINAL_ATOMS))
            result = vpp.verify(pdbqt, source, "d", ligand_record=record)
            self.assertFalse(result["verified"])
            self.assertTrue(
                any("TORSDOF" in failure
                    for failure in result["verification_failures"]),
                result["verification_failures"],
            )
