"""Tests for receptor provenance checks and preparation stop rules.

Scope, and what these tests do not establish
--------------------------------------------
Software correctness only. Fixtures are small hand-written PDB and PDBQT
files whose expected answers are derivable by reading them.

Three kinds of check are deliberately separated:

- **pure** tests of atom parsing, correspondence, pocket census and manifest
  handling, which need nothing installed;
- **Gemmi-backed** tests of `prepare_receptor.py`'s cleaning stage, run
  against a small real structure fixture, skipped if Gemmi is absent;
- **stub-Meeko** tests, which put a fake `mk_prepare_receptor.py` on PATH to
  exercise the subprocess, output and correspondence checks. The stub performs
  no chemistry: no atom typing, no protonation, no charges. These tests say
  nothing about whether real Meeko produces a correct receptor.

Nothing here validates atom typing, partial charges, protonation,
stereochemistry or any biological property. Those require the real tools and
belong on the machine that has them.
"""
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import box_geometry  # noqa: E402
import receptor_provenance as provenance  # noqa: E402

PREPARE = ROOT / "python" / "prepare_receptor.py"

try:
    import gemmi  # noqa: F401
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False


def pdb_line(serial, name, resname, chain, resseq, x, y, z,
             element=None, altloc=" ", icode=" ", record="ATOM  "):
    """Build one PDB ATOM/HETATM line at the correct column positions."""
    element = element or name.lstrip("0123456789")[:1]
    return (f"{record}{serial:>5} {name:<4}{altloc}{resname:>3} {chain}"
            f"{resseq:>4}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            f"          {element:>2}")


# A two-residue fixture: glycine 25 complete (N, CA, C, O) and alanine 26
# complete (N, CA, C, O, CB), plus one hydrogen that must be ignored by the
# heavy-atom checks.
FIXTURE_PDB = "\n".join([
    pdb_line(1, "N", "GLY", "A", 25, 1.0, 1.0, 1.0),
    pdb_line(2, "CA", "GLY", "A", 25, 2.0, 1.0, 1.0),
    pdb_line(3, "C", "GLY", "A", 25, 3.0, 1.0, 1.0),
    pdb_line(4, "O", "GLY", "A", 25, 4.0, 1.0, 1.0),
    pdb_line(5, "N", "ALA", "A", 26, 1.0, 2.0, 1.0),
    pdb_line(6, "CA", "ALA", "A", 26, 2.0, 2.0, 1.0),
    pdb_line(7, "C", "ALA", "A", 26, 3.0, 2.0, 1.0),
    pdb_line(8, "O", "ALA", "A", 26, 4.0, 2.0, 1.0),
    pdb_line(9, "CB", "ALA", "A", 26, 2.0, 3.0, 1.0),
    pdb_line(10, "HA", "ALA", "A", 26, 2.5, 2.5, 1.5, element="H"),
]) + "\n"


class AtomParsingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pdb = self.root / "fixture.pdb"
        self.pdb.write_text(FIXTURE_PDB)

    def test_atoms_are_read_with_full_identity(self):
        records = provenance.parse_atom_records(self.pdb)
        self.assertEqual(len(records), 10)
        key = provenance.AtomKey("A", 26, "", "ALA", "CB", "")
        self.assertIn(key, records)
        self.assertEqual(records[key].position, (2.0, 3.0, 1.0))
        self.assertEqual(records[key].element, "C")

    def test_hydrogens_are_identified_and_excluded(self):
        records = provenance.parse_atom_records(self.pdb)
        self.assertEqual(len(provenance.heavy_atoms(records)), 9)

    def test_pdbqt_columns_are_read_the_same_way(self):
        """PDBQT keeps columns 1-54; its trailing type field is not an element."""
        pdbqt = self.root / "fixture.pdbqt"
        pdbqt.write_text("\n".join(
            line[:54] + "  0.000 A " for line in FIXTURE_PDB.splitlines()) + "\n")
        records = provenance.parse_atom_records(pdbqt)
        key = provenance.AtomKey("A", 26, "", "ALA", "CB", "")
        self.assertEqual(records[key].position, (2.0, 3.0, 1.0))
        # "A" is an AutoDock aromatic-carbon type, not an element symbol, so
        # the element must come from the atom name instead.
        self.assertEqual(records[key].element, "C")

    def test_a_file_without_atoms_is_rejected(self):
        empty = self.root / "empty.pdb"
        empty.write_text("REMARK nothing here\n")
        with self.assertRaises(provenance.ReceptorError):
            provenance.parse_atom_records(empty)

    def test_a_duplicated_atom_identity_is_rejected(self):
        doubled = self.root / "doubled.pdb"
        doubled.write_text(FIXTURE_PDB + FIXTURE_PDB.splitlines()[0] + "\n")
        with self.assertRaises(provenance.ReceptorError):
            provenance.parse_atom_records(doubled)

    def test_a_malformed_coordinate_is_rejected(self):
        broken = self.root / "broken.pdb"
        lines = FIXTURE_PDB.splitlines()
        lines[0] = lines[0][:30] + "   abc  " + lines[0][38:]
        broken.write_text("\n".join(lines) + "\n")
        with self.assertRaises(provenance.ReceptorError):
            provenance.parse_atom_records(broken)

    def test_insertion_codes_distinguish_residues(self):
        with_codes = self.root / "icode.pdb"
        with_codes.write_text(
            pdb_line(1, "CA", "GLY", "A", 25, 1.0, 1.0, 1.0) + "\n"
            + pdb_line(2, "CA", "GLY", "A", 25, 9.0, 9.0, 9.0, icode="A") + "\n")
        records = provenance.parse_atom_records(with_codes)
        self.assertEqual(len(records), 2)
        self.assertEqual({key.insertion_code for key in records}, {"", "A"})


class CorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "a.pdb").write_text(FIXTURE_PDB)
        self.source = provenance.heavy_atoms(
            provenance.parse_atom_records(self.root / "a.pdb"))

    def prepared_from(self, lines):
        path = self.root / "b.pdb"
        path.write_text("\n".join(lines) + "\n")
        return provenance.heavy_atoms(provenance.parse_atom_records(path))

    def test_identical_files_report_no_deviation(self):
        report = provenance.compare_atom_sets(self.source, self.source)
        self.assertEqual(report.n_paired, 9)
        self.assertEqual(report.n_missing, 0)
        self.assertEqual(report.max_deviation_angstrom, 0.0)
        self.assertTrue(report.coordinates_preserved)

    def test_rounding_sized_differences_are_within_tolerance(self):
        """0.001 A per axis is representation rounding, not movement."""
        lines = [line for line in FIXTURE_PDB.splitlines()]
        lines[8] = pdb_line(9, "CB", "ALA", "A", 26, 2.001, 3.001, 1.001)
        report = provenance.compare_atom_sets(self.source,
                                              self.prepared_from(lines))
        self.assertLess(report.max_deviation_angstrom,
                        provenance.COORDINATE_TOLERANCE_ANGSTROM)
        self.assertTrue(report.coordinates_preserved)

    def test_a_single_moved_atom_is_caught_despite_a_tiny_global_rmsd(self):
        """One atom moved 2 A among nine must not be averaged away."""
        lines = [line for line in FIXTURE_PDB.splitlines()]
        lines[8] = pdb_line(9, "CB", "ALA", "A", 26, 4.0, 3.0, 1.0)
        report = provenance.compare_atom_sets(self.source,
                                              self.prepared_from(lines))
        self.assertFalse(report.coordinates_preserved)
        self.assertAlmostEqual(report.max_deviation_angstrom, 2.0)
        self.assertIn("A:26 ALA CB", report.max_deviation_atom)
        self.assertEqual(len(report.moved_atoms), 1)

    def test_pocket_subset_is_reported_separately(self):
        lines = [line for line in FIXTURE_PDB.splitlines()]
        lines[8] = pdb_line(9, "CB", "ALA", "A", 26, 4.0, 3.0, 1.0)
        pocket = provenance.pocket_atom_keys(self.source, "A", [25])
        report = provenance.compare_atom_sets(
            self.source, self.prepared_from(lines),
            subset_keys=pocket, subset_name="pocket heavy atoms")
        # Residue 25 is untouched, so the pocket subset is clean while the
        # whole-receptor comparison is not.
        self.assertEqual(report.subset_n_paired, 4)
        self.assertEqual(report.subset_max_deviation_angstrom, 0.0)
        self.assertFalse(report.coordinates_preserved)

    def test_a_missing_atom_is_counted_not_measured(self):
        lines = FIXTURE_PDB.splitlines()[:-2]
        report = provenance.compare_atom_sets(self.source,
                                              self.prepared_from(lines))
        self.assertEqual(report.n_missing, 1)
        self.assertFalse(report.complete)
        self.assertIn("A:26 ALA CB", report.missing_atoms[0])

    def test_agreed_transformation_follows_a_renamed_atom(self):
        """MSE SE becomes MET SD: paired as a transformation, not a loss."""
        source = self.prepared_from([
            pdb_line(1, "CA", "MSE", "A", 31, 1.0, 1.0, 1.0),
            pdb_line(2, "SE", "MSE", "A", 31, 2.0, 1.0, 1.0, element="SE"),
        ])
        prepared = self.prepared_from([
            pdb_line(1, "CA", "MET", "A", 31, 1.0, 1.0, 1.0),
            pdb_line(2, "SD", "MET", "A", 31, 2.0, 1.0, 1.0, element="S"),
        ])
        report = provenance.compare_atom_sets(
            source, prepared,
            residue_transformations={("A", 31, ""): ("MSE", "MET")})
        self.assertEqual(report.n_paired, 2)
        self.assertEqual(report.n_missing, 0)
        self.assertEqual(report.max_deviation_angstrom, 0.0)
        self.assertEqual(len(report.applied_transformations), 1)
        self.assertIn("selenium replaced by sulphur",
                      report.applied_transformations[0])

    def test_an_unrecorded_transformation_is_refused(self):
        with self.assertRaises(provenance.ReceptorError):
            provenance.compare_atom_sets(
                self.source, self.source,
                residue_transformations={("A", 26, ""): ("ALA", "GLY")})

    def test_deliberately_removed_atoms_are_not_losses(self):
        lines = FIXTURE_PDB.splitlines()[:-2]
        removed = {provenance.AtomKey("A", 26, "", "ALA", "CB", "")}
        report = provenance.compare_atom_sets(
            self.source, self.prepared_from(lines),
            deliberately_removed=removed)
        self.assertEqual(report.n_missing, 0)
        self.assertEqual(report.n_paired, 8)


class PocketCensusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "a.pdb").write_text(FIXTURE_PDB)
        self.records = provenance.parse_atom_records(self.root / "a.pdb")

    def test_complete_residues_pass(self):
        census = provenance.pocket_census(self.records, "A", [25, 26])
        self.assertEqual(census["n_found"], 2)
        self.assertEqual(census["residues_absent"], [])
        self.assertEqual(census["residues_incomplete"], [])
        self.assertEqual(census["total_heavy_atoms"], 9)

    def test_an_incomplete_side_chain_is_flagged(self):
        """Alanine without its CB has four heavy atoms, not five."""
        path = self.root / "b.pdb"
        path.write_text("\n".join(FIXTURE_PDB.splitlines()[:-2]) + "\n")
        census = provenance.pocket_census(
            provenance.parse_atom_records(path), "A", [25, 26])
        self.assertEqual(len(census["residues_incomplete"]), 1)
        self.assertIn("4/5", census["residues_incomplete"][0])

    def test_a_requested_residue_that_is_absent_is_reported(self):
        census = provenance.pocket_census(self.records, "A", [25, 26, 99])
        self.assertEqual(census["residues_absent"], [99])

    def test_the_wrong_chain_finds_nothing(self):
        census = provenance.pocket_census(self.records, "B", [25, 26])
        self.assertEqual(census["n_found"], 0)
        self.assertEqual(census["residues_absent"], [25, 26])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.structure = self.root / "source.cif"
        self.structure.write_text("data_test\n")

    def manifest(self, **overrides):
        base = dict(
            condition="apo",
            selections={"protein_author_chain_id": "A",
                        "pocket_author_residue_ids": ["25", "26"],
                        "pocket_cutoff_angstrom": 5.0},
            source={"path": str(self.structure),
                    "sha256": box_geometry.file_sha256(self.structure)},
            outputs={"receptor_pdbqt_sha256": "a" * 64},
            transformations={}, removed={}, retained={},
            checks={"fixture": True}, commands=[], tools={},
            unresolved=["test"],
        )
        base.update(overrides)
        return provenance.build_manifest(**base)

    def test_round_trip_preserves_the_compared_fields(self):
        path = provenance.write_manifest(self.root / "m.json", self.manifest())
        back = provenance.read_manifest(path)
        self.assertEqual(back["receptor_condition"], "apo")
        self.assertEqual(back["selections"]["protein_author_chain_id"], "A")
        self.assertEqual(back["unresolved_decisions"], ["test"])

    def test_an_existing_manifest_is_not_overwritten_silently(self):
        path = self.root / "m.json"
        provenance.write_manifest(path, self.manifest())
        with self.assertRaises(provenance.ReceptorError):
            provenance.write_manifest(path, self.manifest())
        provenance.write_manifest(path, self.manifest(condition="anp-present"),
                                  overwrite=True)
        self.assertEqual(
            provenance.read_manifest(path)["receptor_condition"], "anp-present")

    def test_a_wrong_schema_is_refused(self):
        path = self.root / "m.json"
        manifest = self.manifest()
        manifest["schema"] = "something-else/1"
        provenance.write_manifest(path, manifest)
        with self.assertRaises(provenance.ReceptorError):
            provenance.read_manifest(path)

    def test_a_truncated_manifest_is_refused(self):
        path = self.root / "m.json"
        manifest = self.manifest()
        del manifest["checks"]
        provenance.write_manifest(path, manifest)
        with self.assertRaises(provenance.ReceptorError):
            provenance.read_manifest(path)

    def test_a_missing_tool_is_recorded_as_missing(self):
        report = provenance.tool_version("a-tool-that-does-not-exist-12345")
        self.assertIsNone(report["path"])
        self.assertIsNone(report["version"])
        self.assertIn("not found", str(report["note"]))


class BoxAgreementTests(unittest.TestCase):
    """The box/receptor agreement check `dock.py` runs before docking."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.structure = self.root / "source.cif"
        self.structure.write_text("data_test\n")
        self.box = box_geometry.build_record(
            centre=(0, 0, 0), size=(24, 24, 24), centre_mode="bounds-midpoint",
            sizing={"mode": "explicit-size"},
            selection={"protein_author_chain_id": "A",
                       "pocket_author_residue_ids": ["25", "26"],
                       "pocket_cutoff_angstrom": 5.0},
            source_structure_path=self.structure)
        self.manifest = provenance.build_manifest(
            condition="apo",
            selections={"protein_author_chain_id": "A",
                        "pocket_author_residue_ids": ["25", "26"],
                        "pocket_cutoff_angstrom": 5.0},
            source={"path": str(self.structure),
                    "sha256": box_geometry.file_sha256(self.structure)},
            outputs={"receptor_pdbqt_sha256": "b" * 64},
            transformations={}, removed={}, retained={}, checks={},
            commands=[], tools={}, unresolved=[])

    def test_matching_box_and_manifest_agree(self):
        self.assertEqual(
            provenance.box_and_receptor_disagreements(
                self.box, self.manifest, receptor_sha256="b" * 64), [])

    def test_a_different_source_hash_disagrees(self):
        self.manifest["source_structure"]["sha256"] = "c" * 64
        problems = provenance.box_and_receptor_disagreements(
            self.box, self.manifest)
        self.assertTrue(any("different source structures" in p for p in problems))

    def test_a_missing_hash_is_reported_rather_than_assumed_equal(self):
        self.manifest["source_structure"]["sha256"] = None
        problems = provenance.box_and_receptor_disagreements(
            self.box, self.manifest)
        self.assertTrue(any("cannot be compared" in p for p in problems))

    def test_a_different_chain_disagrees(self):
        self.manifest["selections"]["protein_author_chain_id"] = "B"
        problems = provenance.box_and_receptor_disagreements(
            self.box, self.manifest)
        self.assertTrue(any("different protein copies" in p for p in problems))

    def test_a_different_cutoff_disagrees(self):
        self.manifest["selections"]["pocket_cutoff_angstrom"] = 4.0
        problems = provenance.box_and_receptor_disagreements(
            self.box, self.manifest)
        self.assertTrue(any("different pocket cutoffs" in p for p in problems))

    def test_a_different_receptor_file_disagrees(self):
        problems = provenance.box_and_receptor_disagreements(
            self.box, self.manifest, receptor_sha256="d" * 64)
        self.assertTrue(any("not the one the manifest describes" in p
                            for p in problems))


# ---------------------------------------------------------------------------
# prepare_receptor.py end to end. Gemmi is real; Meeko is a stub.
# ---------------------------------------------------------------------------

# A minimal mmCIF holding two protein copies, a UMA, an ANP, a manganese and a
# water, so chain selection, condition handling and component removal can all
# be exercised. Coordinates are chosen for arithmetic, not chemistry.
def fixture_cif() -> str:
    header = """data_fixture
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
"""
    rows, serial = [], 0

    def row(element, atom, comp, canonical, seq, x, y, z, auth_seq, auth_chain,
            group="ATOM", label_seq="."):
        nonlocal serial
        serial += 1
        rows.append(f"{group} {serial} {element} {atom} . {comp} {canonical} 1 "
                    f"{label_seq} ? {x:.3f} {y:.3f} {z:.3f} 1.00 10.00 "
                    f"{auth_seq} {auth_chain} 1")

    # Chain A: glycine 25 and alanine 26, both complete; MSE 31 complete.
    for index, (comp, atoms) in enumerate(
            [("GLY", [("N", "N"), ("C", "CA"), ("C", "C"), ("O", "O")]),
             ("ALA", [("N", "N"), ("C", "CA"), ("C", "C"), ("O", "O"),
                      ("C", "CB")])], start=1):
        for offset, (element, atom) in enumerate(atoms):
            row(element, atom, comp, "A", index, 1.0 + offset, float(index), 1.0,
                24 + index, "A", label_seq=str(index))
    mse_atoms = [("N", "N"), ("C", "CA"), ("C", "C"), ("O", "O"), ("C", "CB"),
                 ("C", "CG"), ("SE", "SE"), ("C", "CE")]
    for offset, (element, atom) in enumerate(mse_atoms):
        row(element, atom, "MSE", "A", 3, 1.0 + offset, 3.0, 1.0, 31, "A",
            group="HETATM", label_seq="3")
    # Chain B: a second copy, far away, to be discarded by selection.
    for offset, (element, atom) in enumerate(
            [("N", "N"), ("C", "CA"), ("C", "C"), ("O", "O")]):
        row(element, atom, "GLY", "B", 1, 90.0 + offset, 90.0, 90.0, 25, "B",
            label_seq="1")
    # Heteroatoms on author chain A.
    row("P", "PG", "ANP", "F", 1, 5.0, 5.0, 5.0, 603, "A", group="HETATM")
    row("N", "N3B", "ANP", "F", 1, 6.0, 5.0, 5.0, 603, "A", group="HETATM")
    row("C", "C1", "UMA", "E", 1, 2.0, 2.0, 2.0, 602, "A", group="HETATM")
    row("MN", "MN", "MN", "C", 1, 7.0, 7.0, 7.0, 701, "A", group="HETATM")
    row("O", "O", "HOH", "K", 1, 8.0, 8.0, 8.0, 801, "A", group="HETATM")
    return header + "\n".join(rows) + "\n"


MEEKO_STUB = textwrap.dedent('''
    #!/usr/bin/env python3
    """Stub standing in for Meeko's mk_prepare_receptor.py.

    Copies PDB columns 1-54 into a PDBQT-shaped file and appends a charge and
    a type field. It performs NO chemistry: no atom typing, no protonation, no
    charge calculation, no template matching. Its only purpose is to exercise
    prepare_receptor.py's subprocess, output and correspondence checks.

    Test switches: --drop-residue (by author number, repeatable),
    --shift-atom (move one atom name by 0.5 A), --fail (exit non-zero),
    --write-nothing (exit 0 having written no output).
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--read_pdb")
    parser.add_argument("-o")
    parser.add_argument("-p", action="store_true")
    parser.add_argument("--default_altloc")
    parser.add_argument("--drop-residue", action="append", default=[])
    parser.add_argument("--shift-atom")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--write-nothing", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.fail:
        print("stub failure: template matching failed", file=sys.stderr)
        sys.exit(3)
    if args.write_nothing:
        print("stub wrote nothing")
        sys.exit(0)

    lines = []
    for line in Path(args.read_pdb).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        number = line[22:26].strip()
        if number in args.drop_residue:
            print(f"Template matching failed for A:{number}")
            continue
        body = line[:54]
        if args.shift_atom and line[12:16].strip() == args.shift_atom:
            body = f"{line[:30]}{float(line[30:38]) + 0.5:8.3f}{line[38:54]}"
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        lines.append(f"{body}  0.000 {element:<2}")
    Path(f"{args.o}_rigid.pdbqt").write_text("\\n".join(lines) + "\\n")
    print(f"Files written: {args.o}_rigid.pdbqt")
''').lstrip()


@unittest.skipUnless(GEMMI_AVAILABLE, "prepare_receptor.py requires Gemmi")
class PrepareReceptorTests(unittest.TestCase):
    """Real Gemmi cleaning; Meeko stubbed, so no chemistry is validated."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.structure = self.root / "fixture.cif"
        self.structure.write_text(fixture_cif())
        self.bin = self.root / "bin"
        self.bin.mkdir()
        stub = self.bin / "mk_prepare_receptor.py"
        stub.write_text(MEEKO_STUB)
        stub.chmod(0o755)
        self.box_path = self.root / "box.json"
        box_geometry.write_record(self.box_path, box_geometry.build_record(
            centre=(2.0, 2.0, 2.0), size=(24.0, 24.0, 24.0),
            centre_mode="bounds-midpoint", sizing={"mode": "explicit-size"},
            selection={"protein_author_chain_id": "A",
                       "pocket_author_residue_ids": ["25", "26", "31"],
                       "pocket_cutoff_angstrom": 5.0},
            source_structure_path=self.structure))

    def prepare(self, *args, success=True, with_stub=True):
        path = f"{self.bin}:/usr/bin:/bin" if with_stub else "/usr/bin:/bin"
        finished = subprocess.run(
            [sys.executable, str(PREPARE), *[str(value) for value in args]],
            capture_output=True, text=True,
            env={"PATH": path, "PYTHONPATH": str(ROOT / "python")})
        self.assertEqual(finished.returncode == 0, success,
                         finished.stdout + finished.stderr)
        return finished

    def test_apo_preparation_records_every_decision(self):
        prefix = self.root / "murc_apo"
        self.prepare(self.structure, prefix, "--protein-author-chain", "A",
                     "--condition", "apo", "--box-record", self.box_path)
        manifest = provenance.read_manifest(
            prefix.with_name("murc_apo.manifest.json"))
        self.assertEqual(manifest["receptor_condition"], "apo")
        removed = manifest["removed_components"]["components_in_selected_chain"]
        self.assertEqual({"UMA": 1, "ANP": 1, "MN": 1, "HOH": 1},
                         {name: removed[name] for name in
                          ("UMA", "ANP", "MN", "HOH")})
        self.assertIn("B", manifest["removed_components"]["other_chains_discarded"])
        self.assertEqual(
            manifest["residue_transformations"]["MSE_to_MET"]["count"], 1)
        self.assertEqual(
            manifest["residue_transformations"]["MSE_to_MET"]["residues"],
            ["A:31"])
        self.assertEqual(manifest["retained_components"]["by_component"], {})
        self.assertTrue(manifest["unresolved_decisions"])

    def test_the_source_structure_is_not_modified(self):
        before = box_geometry.file_sha256(self.structure)
        self.prepare(self.structure, self.root / "murc", "--protein-author-chain",
                     "A", "--condition", "apo", "--box-record", self.box_path)
        self.assertEqual(box_geometry.file_sha256(self.structure), before)

    def test_coordinates_and_the_pocket_survive_to_the_pdbqt(self):
        prefix = self.root / "murc"
        self.prepare(self.structure, prefix, "--protein-author-chain", "A",
                     "--condition", "apo", "--box-record", self.box_path)
        manifest = provenance.read_manifest(
            prefix.with_name("murc.manifest.json"))
        for stage in ("source_to_prepared", "prepared_to_receptor"):
            check = manifest["checks"][stage]
            self.assertTrue(check["coordinates_preserved"], stage)
            self.assertEqual(check["n_missing"], 0, stage)
            self.assertEqual(check["subset"]["n_missing"], 0, stage)
        pocket = manifest["checks"]["pocket_in_receptor_pdbqt"]
        self.assertEqual(pocket["n_found"], 3)
        self.assertEqual(pocket["residues_incomplete"], [])

    def test_the_mse_to_met_transformation_is_followed_not_counted_as_loss(self):
        prefix = self.root / "murc"
        self.prepare(self.structure, prefix, "--protein-author-chain", "A",
                     "--condition", "apo", "--box-record", self.box_path)
        receptor = prefix.with_name("murc_rigid.pdbqt")
        names = {key.residue_name for key
                 in provenance.parse_atom_records(receptor)}
        self.assertIn("MET", names)
        self.assertNotIn("MSE", names)
        manifest = provenance.read_manifest(
            prefix.with_name("murc.manifest.json"))
        applied = manifest["checks"]["source_to_prepared"]["applied_transformations"]
        self.assertTrue(any("selenium replaced by sulphur" in item
                            for item in applied))

    def test_anp_present_condition_retains_anp(self):
        prefix = self.root / "murc_anp"
        self.prepare(self.structure, prefix, "--protein-author-chain", "A",
                     "--condition", "anp-present", "--box-record", self.box_path)
        manifest = provenance.read_manifest(
            prefix.with_name("murc_anp.manifest.json"))
        self.assertEqual(manifest["retained_components"]["by_component"],
                         {"ANP": 1})
        self.assertIn("ANP", manifest["retained_components"]["in_receptor_pdbqt"])
        removed = manifest["removed_components"]["components_in_selected_chain"]
        self.assertNotIn("ANP", removed)
        self.assertEqual(removed["UMA"], 1)
        self.assertEqual(removed["MN"], 1)

    def test_uma_is_absent_from_every_condition(self):
        for condition in ("apo", "anp-present"):
            prefix = self.root / f"murc_{condition}"
            self.prepare(self.structure, prefix, "--protein-author-chain", "A",
                         "--condition", condition, "--box-record", self.box_path)
            receptor = prefix.with_name(f"{prefix.name}_rigid.pdbqt")
            names = {key.residue_name
                     for key in provenance.parse_atom_records(receptor)}
            self.assertNotIn("UMA", names, condition)

    def test_anp_present_fails_when_anp_is_not_typed(self):
        """ANP retention must not be faked when Meeko does not type it."""
        prefix = self.root / "murc_anp"
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "anp-present", "--box-record", self.box_path,
            "--meeko-argument=--drop-residue", "--meeko-argument=603",
            success=False)
        self.assertIn("Meeko did not type them", finished.stderr)
        self.assertIn("Do not append the records by hand", finished.stderr)
        self.assertFalse(prefix.with_name("murc_anp.manifest.json").exists())

    def test_a_dropped_pocket_residue_stops_the_run_and_marks_outputs(self):
        prefix = self.root / "murc"
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", self.box_path,
            "--meeko-argument=--drop-residue", "--meeko-argument=26",
            success=False)
        self.assertIn("pocket heavy atoms", finished.stderr)
        self.assertFalse(prefix.with_name("murc.manifest.json").exists())
        self.assertFalse(prefix.with_name("murc_rigid.pdbqt").exists())
        self.assertTrue(prefix.with_name("murc_rigid.pdbqt.FAILED").is_file())
        self.assertTrue(prefix.with_name("murc_prepared.pdb.FAILED").is_file())

    def test_a_moved_atom_stops_the_run(self):
        prefix = self.root / "murc"
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", self.box_path,
            "--meeko-argument=--shift-atom", "--meeko-argument=CB",
            success=False)
        self.assertIn("coordinates changed", finished.stderr)
        self.assertFalse(prefix.with_name("murc.manifest.json").exists())

    def test_a_non_zero_meeko_exit_stops_the_run(self):
        prefix = self.root / "murc"
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", self.box_path,
            "--meeko-argument=--fail", success=False)
        self.assertIn("exited 3", finished.stderr)
        self.assertFalse(prefix.with_name("murc.manifest.json").exists())

    def test_meeko_writing_no_output_stops_the_run(self):
        prefix = self.root / "murc"
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", self.box_path,
            "--meeko-argument=--write-nothing", success=False)
        self.assertIn("no receptor PDBQT was written", finished.stderr)
        self.assertFalse(prefix.with_name("murc.manifest.json").exists())

    def test_a_missing_meeko_is_a_blocker_not_a_pass(self):
        prefix = self.root / "murc"
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", self.box_path,
            success=False, with_stub=False)
        self.assertIn("not found on PATH", finished.stderr)

    def test_a_pocket_residue_absent_from_the_structure_stops_the_run(self):
        other_box = self.root / "box_wrong.json"
        box_geometry.write_record(other_box, box_geometry.build_record(
            centre=(2.0, 2.0, 2.0), size=(24.0, 24.0, 24.0),
            centre_mode="bounds-midpoint", sizing={"mode": "explicit-size"},
            selection={"protein_author_chain_id": "A",
                       "pocket_author_residue_ids": ["25", "999"],
                       "pocket_cutoff_angstrom": 5.0},
            source_structure_path=self.structure))
        finished = self.prepare(
            self.structure, self.root / "murc", "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", other_box, success=False)
        self.assertIn("absent from the source structure", finished.stderr)

    def test_the_box_chain_and_the_requested_chain_must_agree(self):
        finished = self.prepare(
            self.structure, self.root / "murc", "--protein-author-chain", "B",
            "--condition", "apo", "--box-record", self.box_path, success=False)
        self.assertIn("same protein copy", finished.stderr)

    def test_an_unknown_chain_is_refused(self):
        finished = self.prepare(
            self.structure, self.root / "murc", "--protein-author-chain", "Z",
            "--condition", "apo", "--pocket-residues", "25", success=False)
        self.assertIn("no recognised protein residues", finished.stderr)

    def test_existing_outputs_are_never_replaced(self):
        prefix = self.root / "murc"
        prefix.with_name("murc_prepared.pdb").write_text("earlier work\n")
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", self.box_path, success=False)
        self.assertIn("already exists", finished.stderr)
        self.assertEqual(prefix.with_name("murc_prepared.pdb").read_text(),
                         "earlier work\n")

    def test_skip_meeko_writes_no_manifest(self):
        prefix = self.root / "murc"
        finished = self.prepare(
            self.structure, prefix, "--protein-author-chain", "A",
            "--condition", "apo", "--box-record", self.box_path, "--skip-meeko")
        self.assertIn("No receptor was produced", finished.stdout)
        self.assertTrue(prefix.with_name("murc_prepared.pdb").is_file())
        self.assertFalse(prefix.with_name("murc.manifest.json").exists())

    def test_the_manifest_ties_the_receptor_to_the_box(self):
        """The whole point: dock.py can then confirm they belong together."""
        prefix = self.root / "murc"
        self.prepare(self.structure, prefix, "--protein-author-chain", "A",
                     "--condition", "apo", "--box-record", self.box_path)
        manifest = provenance.read_manifest(
            prefix.with_name("murc.manifest.json"))
        box = box_geometry.read_record(self.box_path)
        receptor = prefix.with_name("murc_rigid.pdbqt")
        self.assertEqual(
            provenance.box_and_receptor_disagreements(
                box, manifest,
                receptor_sha256=box_geometry.file_sha256(receptor)),
            [])


if __name__ == "__main__":
    unittest.main()
