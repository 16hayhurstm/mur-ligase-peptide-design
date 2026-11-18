"""Tests for docking-box geometry, box records and per-seed docking outputs.

Scope and limits
----------------
These are software-correctness tests. Fixtures are small, hand-chosen point
sets whose expected answers are derivable by hand, not restatements of the
implementation.

Nothing here validates molecular structures, receptor chemistry, pose quality
or docking scores. The docking campaign tests run against a **stub Vina**
injected on `PYTHONPATH`: they check how `dock.py` organises seeds, poses and
failure records, and deliberately tell us nothing about AutoDock Vina itself.
Tests requiring real Vina are not included and must be run on the machine that
has it.
"""
import csv
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import box_geometry  # noqa: E402
import receptor_provenance  # noqa: E402

DOCK = ROOT / "python" / "dock.py"
MAP_POCKET = ROOT / "python" / "map_pocket.py"
BOX_TO_BILD = ROOT / "python" / "box_to_bild.py"

try:
    import gemmi  # noqa: F401
    GEMMI_AVAILABLE = True
except ImportError:
    GEMMI_AVAILABLE = False

# A deliberately asymmetric ligand: extents are exactly 10, 4 and 2 A, and the
# atom mean is displaced from the midpoint of the bounds because four of the
# five atoms sit at the low-x end.
ASYMMETRIC = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 4.0, 2.0],
    [1.0, 0.0, 0.0],
    [0.0, 4.0, 2.0],
    [10.0, 2.0, 1.0],
])
ASYMMETRIC_MIDPOINT = np.array([5.0, 2.0, 1.0])
ASYMMETRIC_EXTENT = np.array([10.0, 4.0, 2.0])


def minimal_record(centre=(5.0, 2.0, 1.0), size=(20.0, 14.0, 12.0),
                   author_chain="A", source_structure_path=None,
                   pocket=("25", "27"), cutoff=5.0, **overrides):
    """Build a valid record for transfer tests, without touching a structure."""
    record = box_geometry.build_record(
        centre=centre, size=size, centre_mode="bounds-midpoint",
        sizing={"mode": "explicit-size"},
        selection={"description": "test ligand",
                   "protein_author_chain_id": author_chain,
                   "pocket_author_residue_ids": list(pocket),
                   "pocket_cutoff_angstrom": cutoff},
        source_structure_path=source_structure_path,
    )
    record.update(overrides)
    return record


def minimal_manifest(*, source_path, receptor_path, author_chain="A",
                     pocket=("25", "27"), cutoff=5.0, condition="apo"):
    """Build a receptor manifest matching a box record, for provenance tests.

    Only the fields `dock.py` compares are populated. This is a fixture, not
    the output of a real preparation: no structure was prepared.
    """
    return receptor_provenance.build_manifest(
        condition=condition,
        selections={"protein_author_chain_id": author_chain,
                    "protein_canonical_chain_id": author_chain,
                    "pocket_author_residue_ids": list(pocket),
                    "pocket_cutoff_angstrom": cutoff},
        source={"path": str(source_path),
                "sha256": box_geometry.file_sha256(source_path),
                "model": 1, "coordinate_frame": "as read"},
        outputs={"receptor_pdbqt": str(receptor_path),
                 "receptor_pdbqt_sha256": box_geometry.file_sha256(receptor_path)},
        transformations={}, removed={}, retained={},
        checks={"fixture": "no preparation was performed"},
        commands=[], tools={},
        unresolved=["this is a test fixture, not a prepared receptor"],
    )


class CentringTests(unittest.TestCase):
    def test_bounds_midpoint_is_independent_of_atom_density(self):
        centre = box_geometry.centre_from_points(ASYMMETRIC, "bounds-midpoint")
        np.testing.assert_allclose(centre, ASYMMETRIC_MIDPOINT)

    def test_atom_mean_is_displaced_towards_the_dense_end(self):
        centre = box_geometry.centre_from_points(ASYMMETRIC, "ligand-atom-mean")
        # Mean x of the five atoms is 12/5 = 2.4, i.e. 2.6 A below the midpoint.
        self.assertAlmostEqual(float(centre[0]), 2.4)
        self.assertAlmostEqual(float(ASYMMETRIC_MIDPOINT[0] - centre[0]), 2.6)

    def test_midpoint_centring_gives_equal_clearance_on_opposite_faces(self):
        centre = box_geometry.centre_from_points(ASYMMETRIC, "bounds-midpoint")
        size = box_geometry.dimensions_from_padding(ASYMMETRIC, [5.0, 5.0, 5.0])
        low, high = box_geometry.face_clearances(centre, size, ASYMMETRIC)
        np.testing.assert_allclose(low, high)
        self.assertGreaterEqual(min(low.min(), high.min()), 5.0)

    def test_atom_mean_centring_breaks_the_padding_floor(self):
        """The historical centring takes from one face what it adds to the other."""
        centre = box_geometry.centre_from_points(ASYMMETRIC, "ligand-atom-mean")
        size = box_geometry.dimensions_from_padding(ASYMMETRIC, [5.0, 5.0, 5.0])
        low, high = box_geometry.face_clearances(centre, size, ASYMMETRIC)
        self.assertAlmostEqual(min(low.min(), high.min()), 5.0 - 2.6)

    def test_unknown_centring_mode_is_rejected(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.centre_from_points(ASYMMETRIC, "centroid-of-mass")


class DimensionTests(unittest.TestCase):
    def test_independent_axes_follow_extent_plus_two_padding(self):
        size = box_geometry.dimensions_from_padding(ASYMMETRIC, [5.0, 5.0, 5.0])
        np.testing.assert_allclose(size, ASYMMETRIC_EXTENT + 10.0)
        self.assertEqual(len(set(size.tolist())), 3)

    def test_per_axis_padding_is_applied_per_axis(self):
        size = box_geometry.dimensions_from_padding(ASYMMETRIC, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(size, [12.0, 8.0, 8.0])

    def test_padding_is_a_floor_not_a_nominal_value(self):
        """A non-integral extent must round up, never down.

        The real UMA selection has a 14.28 A extent; with 5 A per face the box
        must be at least 24.28 A. Rounding to nearest would give 24 and quietly
        remove 0.28 A of the requested clearance.
        """
        ligand = np.array([[0.0, 0.0, 0.0], [14.28, 1.0, 1.0]])
        size = box_geometry.dimensions_from_padding(ligand, [5.0, 5.0, 5.0])
        self.assertGreaterEqual(float(size[0]), 24.28)
        self.assertAlmostEqual(float(size[0]), 24.3)
        centre = box_geometry.centre_from_points(ligand, "bounds-midpoint")
        low, high = box_geometry.face_clearances(centre, size, ligand)
        self.assertGreaterEqual(min(low.min(), high.min()), 5.0)

    def test_exact_tie_rounds_up(self):
        """14.50 + 10 = 24.50 must not become 24, as round-half-even would."""
        ligand = np.array([[0.0, 0.0, 0.0], [14.50, 1.0, 1.0]])
        size = box_geometry.dimensions_from_padding(ligand, [5.0, 5.0, 5.0])
        self.assertAlmostEqual(float(size[0]), 24.5)

    def test_zero_padding_just_contains_the_bounds(self):
        size = box_geometry.dimensions_from_padding(ASYMMETRIC, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(size, ASYMMETRIC_EXTENT)


class InvalidInputTests(unittest.TestCase):
    def test_non_finite_coordinates_raise(self):
        points = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]])
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.centre_from_points(points, "bounds-midpoint")

    def test_empty_selection_raises(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.extent_from_points(np.zeros((0, 3)))

    def test_zero_and_negative_dimensions_are_rejected(self):
        for size in ([0.0, 10.0, 10.0], [-1.0, 10.0, 10.0]):
            with self.assertRaises(box_geometry.BoxError):
                box_geometry.validated_dimensions(size)

    def test_non_finite_dimension_is_rejected(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.validated_dimensions([np.inf, 10.0, 10.0])

    def test_absurd_dimension_is_rejected(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.validated_dimensions([5000.0, 10.0, 10.0])

    def test_negative_padding_is_rejected(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.dimensions_from_padding(ASYMMETRIC, [-1.0, 5.0, 5.0])

    def test_wrong_number_of_dimensions_is_rejected(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.validated_dimensions([24.0, 24.0])


class MembershipTests(unittest.TestCase):
    def setUp(self):
        self.centre = np.array([0.0, 0.0, 0.0])
        self.size = np.array([10.0, 10.0, 10.0])

    def test_atom_on_the_face_counts_as_inside(self):
        on_face = np.array([[5.0, 0.0, 0.0]])
        self.assertTrue(bool(box_geometry.points_inside(
            self.centre, self.size, on_face)[0]))
        self.assertEqual(float(box_geometry.distance_to_box(
            self.centre, self.size, on_face)[0]), 0.0)

    def test_distance_to_box_is_the_perpendicular_gap_on_one_axis(self):
        point = np.array([[7.0, 0.0, 0.0]])
        self.assertAlmostEqual(
            float(box_geometry.distance_to_box(self.centre, self.size, point)[0]),
            2.0)

    def test_distance_to_box_is_the_corner_distance_past_two_axes(self):
        point = np.array([[8.0, 9.0, 0.0]])
        self.assertAlmostEqual(
            float(box_geometry.distance_to_box(self.centre, self.size, point)[0]),
            5.0)  # sqrt(3^2 + 4^2)


class RecordTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_round_trip_preserves_every_field_used_downstream(self):
        record = minimal_record(centre=(25.438, -6.267, 48.014),
                                size=(23.6, 19.1, 24.3))
        path = box_geometry.write_record(self.root / "box.json", record)
        back = box_geometry.read_record(path)
        np.testing.assert_allclose(box_geometry.record_centre(back),
                                   [25.438, -6.267, 48.014])
        np.testing.assert_allclose(box_geometry.record_size(back),
                                   [23.6, 19.1, 24.3])
        self.assertEqual(back["units"], "angstrom")
        self.assertEqual(back["centre_mode"], "bounds-midpoint")
        self.assertEqual(back["selection"]["protein_author_chain_id"], "A")
        self.assertEqual(back["selection"]["pocket_cutoff_angstrom"], 5.0)

    def test_existing_record_is_not_overwritten_silently(self):
        path = self.root / "box.json"
        box_geometry.write_record(path, minimal_record())
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.write_record(path, minimal_record())
        box_geometry.write_record(path, minimal_record(size=(30.0, 30.0, 30.0)),
                                  overwrite=True)
        np.testing.assert_allclose(
            box_geometry.record_size(box_geometry.read_record(path)), 30.0)

    def test_wrong_schema_is_refused(self):
        path = self.root / "box.json"
        box_geometry.write_record(path, minimal_record(schema="some-other-box/9"))
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.read_record(path)

    def test_non_angstrom_units_are_refused(self):
        path = self.root / "box.json"
        box_geometry.write_record(path, minimal_record(units="nanometre"))
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.read_record(path)

    def test_corrupt_record_is_refused(self):
        path = self.root / "box.json"
        path.write_text("{not json")
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.read_record(path)

    def test_invalid_size_in_record_is_refused_on_read(self):
        path = self.root / "box.json"
        box_geometry.write_record(path, minimal_record(size_angstrom=[24.0, 0.0, 24.0]))
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.read_record(path)

    def test_two_dimensions_in_record_is_refused_on_read(self):
        path = self.root / "box.json"
        box_geometry.write_record(path, minimal_record(size_angstrom=[24.0, 24.0]))
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.read_record(path)


class CentringAndSizingCombinationTests(unittest.TestCase):
    def test_padding_requires_bounds_midpoint(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.validated_centre_and_sizing(
                "ligand-atom-mean", "padding-per-face")

    def test_permitted_combinations_are_accepted(self):
        for centre_mode, sizing_mode in (("bounds-midpoint", "padding-per-face"),
                                         ("bounds-midpoint", "explicit-size"),
                                         ("ligand-atom-mean", "explicit-size")):
            box_geometry.validated_centre_and_sizing(centre_mode, sizing_mode)

    def test_unknown_modes_are_rejected(self):
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.validated_centre_and_sizing("centroid", "explicit-size")
        with self.assertRaises(box_geometry.BoxError):
            box_geometry.validated_centre_and_sizing("bounds-midpoint", "guess")


# An mmCIF fixture whose ligand has the same deliberately asymmetric atom
# distribution as ASYMMETRIC above: extents exactly 10, 4 and 2 A, bounds
# midpoint at (5, 2, 1), and four of five atoms at the low-x end so the atom
# mean sits at x = 2.4. Two glycine/alanine residues stand in for a protein
# chain so the pocket selection has something to find. Not a molecular test:
# the coordinates are chosen for arithmetic, not chemistry.
ASYMMETRIC_CIF = """data_asym
_entry.id ASYM
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
ATOM   1  N N   . GLY A 1 1 ? 2.000 1.000 1.000 1.00 10.00 1 A 1
ATOM   2  C CA  . GLY A 1 1 ? 3.000 1.000 1.000 1.00 10.00 1 A 1
ATOM   3  C C   . GLY A 1 1 ? 4.000 1.000 1.000 1.00 10.00 1 A 1
ATOM   4  O O   . GLY A 1 1 ? 5.000 1.000 1.000 1.00 10.00 1 A 1
ATOM   5  N N   . ALA A 1 2 ? 2.000 3.000 1.000 1.00 10.00 2 A 1
ATOM   6  C CA  . ALA A 1 2 ? 3.000 3.000 1.000 1.00 10.00 2 A 1
ATOM   7  C C   . ALA A 1 2 ? 4.000 3.000 1.000 1.00 10.00 2 A 1
ATOM   8  O O   . ALA A 1 2 ? 5.000 3.000 1.000 1.00 10.00 2 A 1
HETATM 9  C C1  . LIG B 2 . ? 0.000 0.000 0.000 1.00 10.00 90 A 1
HETATM 10 C C2  . LIG B 2 . ? 1.000 0.000 0.000 1.00 10.00 90 A 1
HETATM 11 C C3  . LIG B 2 . ? 0.000 4.000 2.000 1.00 10.00 90 A 1
HETATM 12 C C4  . LIG B 2 . ? 1.000 4.000 2.000 1.00 10.00 90 A 1
HETATM 13 C C5  . LIG B 2 . ? 10.000 2.000 1.000 1.00 10.00 90 A 1
"""


@unittest.skipUnless(GEMMI_AVAILABLE, "map_pocket.py requires Gemmi")
class MapPocketBoxRecordTests(unittest.TestCase):
    """CLI regression tests for the recorded box, against an asymmetric ligand."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.structure = self.root / "asym.cif"
        self.structure.write_text(ASYMMETRIC_CIF)

    def map_pocket(self, *args, success=True):
        finished = subprocess.run(
            [sys.executable, str(MAP_POCKET), str(self.structure), "LIG", "5.0",
             "--protein-chain", "A", "--ligand-chain", "B",
             *[str(value) for value in args]],
            capture_output=True, text=True)
        self.assertEqual(finished.returncode == 0, success,
                         finished.stdout + finished.stderr)
        return finished

    def test_padding_is_delivered_on_all_six_faces(self):
        record_path = self.root / "pad5.json"
        self.map_pocket("--box-padding", "5", "--box-record", record_path)
        record = box_geometry.read_record(record_path)
        np.testing.assert_allclose(box_geometry.record_centre(record), [5.0, 2.0, 1.0])
        np.testing.assert_allclose(box_geometry.record_size(record),
                                   [20.0, 14.0, 12.0])
        # Ligand bounds are known by construction, so the clearance is checkable
        # without recomputing it the way the implementation does.
        bounds = np.array([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0]])
        low, high = box_geometry.face_clearances(
            box_geometry.record_centre(record),
            box_geometry.record_size(record), bounds)
        self.assertGreaterEqual(min(low.min(), high.min()), 5.0)

    def test_atom_mean_centring_with_padding_is_refused(self):
        record_path = self.root / "mean_pad.json"
        finished = self.map_pocket("--box-padding", "5",
                                   "--box-centre", "ligand-atom-mean",
                                   "--box-record", record_path, success=False)
        self.assertIn("cannot honour a per-face padding", finished.stderr)
        self.assertFalse(record_path.exists(),
                         "a refused combination must not leave a record behind")

    def test_atom_mean_centring_with_explicit_size_is_allowed_and_asymmetric(self):
        """Explicit sizes promise no clearance, so atom-mean stays available."""
        record_path = self.root / "mean_size.json"
        self.map_pocket("--box-size", "20", "--box-centre", "ligand-atom-mean",
                        "--box-record", record_path)
        record = box_geometry.read_record(record_path)
        centre = box_geometry.record_centre(record)
        self.assertAlmostEqual(float(centre[0]), 2.4)
        bounds = np.array([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0]])
        low, high = box_geometry.face_clearances(
            centre, box_geometry.record_size(record), bounds)
        # The displacement this box carries is exactly what padding mode must
        # not be allowed to hide: 2.4 A on one face against 7.6 A opposite.
        self.assertAlmostEqual(float(min(low.min(), high.min())), 2.4)

    def test_per_axis_padding_reaches_the_record(self):
        record_path = self.root / "pad_axes.json"
        self.map_pocket("--box-padding", "1,2,3", "--box-record", record_path)
        record = box_geometry.read_record(record_path)
        np.testing.assert_allclose(box_geometry.record_size(record),
                                   [12.0, 8.0, 8.0])
        self.assertEqual(record["sizing"]["mode"], "padding-per-face")
        self.assertEqual(record["sizing"]["padding_per_face_angstrom"],
                         [1.0, 2.0, 3.0])


class BildTests(unittest.TestCase):
    def test_bild_has_twelve_edges_and_spans_the_box(self):
        record = minimal_record(centre=(0.0, 0.0, 0.0), size=(10.0, 20.0, 30.0))
        lines = box_geometry.bild_lines(record)
        cylinders = [line for line in lines if line.startswith(".cylinder")]
        self.assertEqual(len(cylinders), 12)
        box_line = [line for line in lines if line.startswith(".box")]
        self.assertEqual(
            box_line[0].split()[1:],
            ["-5.000", "-10.000", "-15.000", "5.000", "10.000", "15.000"])

    def test_chimerax_script_paths_are_absolute_and_quoted(self):
        """A path containing a space must survive into the .cxc intact."""
        sys.path.insert(0, str(ROOT / "python"))
        import box_to_bild

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "a directory with spaces"
            root.mkdir()
            bild = root / "box 24.bild"
            bild.write_text(".box 0 0 0 1 1 1\n")
            structure = root / "1P3D copy.cif"
            structure.write_text("data_x\n")
            lines = box_to_bild.chimerax_script(
                str(structure), bild, minimal_record())
            opens = [line for line in lines if line.startswith("open ")]
            self.assertEqual(len(opens), 2)
            for line in opens:
                argument = line[len("open "):]
                self.assertTrue(argument.startswith('"') and argument.endswith('"'),
                                f"path not quoted: {line}")
                self.assertTrue(Path(argument.strip('"')).is_absolute(),
                                f"path not absolute: {line}")
            self.assertIn(f'open "{structure.resolve()}"', lines)
            self.assertIn(f'open "{bild.resolve()}"', lines)

    def test_chimerax_script_written_for_a_path_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out dir"
            root.mkdir()
            record_path = root / "box record.json"
            box_geometry.write_record(record_path, minimal_record())
            finished = subprocess.run(
                [sys.executable, str(BOX_TO_BILD), str(record_path),
                 str(root / "box 24.bild"),
                 "--chimerax-script", str(root / "view box.cxc"),
                 "--structure", str(root / "structure file.cif")],
                capture_output=True, text=True)
            self.assertEqual(finished.returncode, 0,
                             finished.stdout + finished.stderr)
            script = (root / "view box.cxc").read_text()
            self.assertIn(f'open "{(root / "structure file.cif").resolve()}"', script)
            self.assertIn(f'open "{(root / "box 24.bild").resolve()}"', script)

    def test_bild_writer_refuses_an_invalid_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.json"
            bad.write_text(json.dumps({"schema": "wrong"}))
            finished = subprocess.run(
                [sys.executable, str(BOX_TO_BILD), str(bad), str(root / "out.bild")],
                capture_output=True, text=True)
            self.assertNotEqual(finished.returncode, 0)
            self.assertFalse((root / "out.bild").exists())


# ---------------------------------------------------------------------------
# dock.py: argument and output handling. Vina is stubbed, never executed.
# ---------------------------------------------------------------------------

STUB_VINA = textwrap.dedent('''
    """Stub replacing the `vina` package so dock.py's bookkeeping is testable.

    Scores are deterministic functions of the ligand name and the seed, so a
    test can tell per-seed results apart. Ligand names select misbehaviours the
    real bindings could show: `explodes` raises, `empty` returns no poses,
    `nanscore` returns a non-finite score, `shortresult` returns fewer poses
    than asked for, and `nopose` writes no pose file. This stub performs no
    docking and says nothing about AutoDock Vina's behaviour.

    Two environment variables add failures without editing this file:
    STUB_FAIL_ON="<ligand>:<seed>,<seed>" raises for those seeds only, and
    STUB_INTERRUPT_ON="<ligand>" raises SystemExit, standing in for a run
    killed partway through.
    """
    import os
    from pathlib import Path

    __version__ = "0.0.0-stub"


    def _fail_seeds(name):
        setting = os.environ.get("STUB_FAIL_ON", "")
        ligand, _, seeds = setting.partition(":")
        if not seeds or ligand != name:
            return set()
        return {int(value) for value in seeds.split(",") if value}


    class Vina:
        def __init__(self, sf_name="vina", seed=0, cpu=1):
            self.seed = seed
            self.ligand = None

        def set_receptor(self, path):
            self.receptor = path

        def set_ligand_from_file(self, path):
            self.ligand = Path(path)
            name = self.ligand.stem
            if name == os.environ.get("STUB_INTERRUPT_ON"):
                raise SystemExit("stub interruption")
            if self.seed in _fail_seeds(name):
                raise RuntimeError(f"stub failure on seed {self.seed}")
            if name == "explodes":
                raise RuntimeError("stub failure for a deliberately bad ligand")

        def compute_vina_maps(self, center, box_size):
            self.center, self.box_size = center, box_size

        def dock(self, exhaustiveness=8, n_poses=9):
            self.n_poses = n_poses

        def energies(self, n_poses=9):
            name = self.ligand.stem
            if name == "empty":
                return []
            if name == "shortresult":
                n_poses = max(1, n_poses - 4)
            base = -5.0 - len(name) - self.seed / 1000.0
            scores = [[base + index * 0.1] for index in range(n_poses)]
            if name == "nanscore":
                scores[0] = [float("nan")]
            return scores

        def write_poses(self, path, n_poses=9, overwrite=False):
            if self.ligand.stem == "nopose":
                return
            target = Path(path)
            if target.exists() and not overwrite:
                raise RuntimeError(f"{target} exists")
            if self.ligand.stem == "emptypose":
                target.write_text("")
                return
            target.write_text(
                f"REMARK stub pose, seed {self.seed}, ligand {self.ligand.stem}\\n")
''')

RECEPTOR_PDBQT = "\n".join(
    f"ATOM  {index:5d}  C   ALA A{index:4d}    "
    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00     0.000 C"
    for index, (x, y, z) in enumerate(
        [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0),
         (0.0, 0.0, 10.0), (10.0, 10.0, 10.0)], start=1)
)


class DockCampaignTests(unittest.TestCase):
    """Mocked docking: a stub Vina on PYTHONPATH, no real docking performed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "stub").mkdir()
        (self.root / "stub" / "vina.py").write_text(STUB_VINA)
        self.receptor = self.root / "murc.pdbqt"
        self.receptor.write_text(RECEPTOR_PDBQT + "\n")
        self.ligands = self.root / "pep"
        self.ligands.mkdir()
        for name in ("pepA", "pepBB"):
            (self.ligands / f"{name}.pdbqt").write_text("REMARK stub ligand\n")
        self.source = self.root / "source.cif"
        self.source.write_text("data_test\n# stand-in source structure\n")
        self.record_path = self.root / "box.json"
        box_geometry.write_record(
            self.record_path,
            minimal_record(centre=(5.0, 5.0, 5.0), size=(20.0, 20.0, 20.0),
                           source_structure_path=self.source))
        self.manifest_path = self.root / "receptor.manifest.json"
        receptor_provenance.write_manifest(
            self.manifest_path,
            minimal_manifest(source_path=self.source,
                             receptor_path=self.receptor))

    def dock(self, *args, success=True, env=None, manifest=True):
        """Run dock.py, supplying the matching receptor manifest by default."""
        environment = {"PYTHONPATH": str(self.root / "stub"),
                       "PATH": "/usr/bin:/bin", **(env or {})}
        arguments = [str(value) for value in args]
        if manifest and "--receptor-manifest" not in arguments:
            arguments += ["--receptor-manifest", str(self.manifest_path)]
        finished = subprocess.run(
            [sys.executable, str(DOCK), *arguments],
            capture_output=True, text=True, env=environment)
        self.assertEqual(finished.returncode == 0, success,
                         finished.stdout + finished.stderr)
        return finished

    def rewrite_box(self, **overrides):
        """Replace the box record, keeping the fields the tests rely on."""
        box_geometry.write_record(
            self.record_path,
            minimal_record(**{"centre": (5.0, 5.0, 5.0),
                             "size": (20.0, 20.0, 20.0),
                             "source_structure_path": self.source,
                             **overrides}),
            overwrite=True)

    def test_every_seed_gets_its_own_poses_and_scores(self):
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1,42,137")
        for seed in (1, 42, 137):
            for name in ("pepA", "pepBB"):
                self.assertTrue(
                    (out / "poses" / f"seed_{seed}" / f"{name}_docked.pdbqt").is_file(),
                    f"missing poses for {name} seed {seed}")
        with (out / "per_seed_scores.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["seed"] for row in rows}, {"1", "42", "137"})
        # 2 ligands x 3 seeds x 9 poses, every pose recorded.
        self.assertEqual(len(rows), 54)
        best = {(row["ligand"], row["seed"]): row["score_kcal_per_mol"]
                for row in rows if row["pose_rank"] == "1"}
        self.assertEqual(len(set(best.values())), 6,
                         "per-seed scores were collapsed or overwritten")

    def test_summary_records_the_seeds_actually_completed(self):
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "7,11")
        with (out / "dock_results.csv").open() as handle:
            rows = {row["ligand"]: row for row in csv.DictReader(handle)}
        self.assertEqual(rows["pepA"]["seeds_completed"], "7 11")
        self.assertEqual(rows["pepA"]["n_seeds_completed"], "2")
        self.assertEqual(len(rows["pepA"]["best_score_per_seed"].split()), 2)

    def test_box_record_reaches_vina_without_transcription(self):
        out = self.root / "dock_run"
        self.rewrite_box(size=(21.3, 17.5, 24.3))
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path)
        run_record = json.loads((out / "run_record.json").read_text())
        self.assertEqual(run_record["box_size_angstrom"], [21.3, 17.5, 24.3])
        self.assertEqual(run_record["box_centre_angstrom"], [5.0, 5.0, 5.0])
        self.assertEqual(run_record["seeds"], [1, 42, 137])
        self.assertEqual(run_record["exhaustiveness"], 32)
        # The record travels with the results, so the box used is recoverable.
        np.testing.assert_allclose(
            box_geometry.record_size(
                box_geometry.read_record(out / "box_record.json")),
            [21.3, 17.5, 24.3])

    def results(self, out):
        with (out / "dock_results.csv").open() as handle:
            return {row["ligand"]: row for row in csv.DictReader(handle)}

    def failures(self, out):
        with (out / "failures.csv").open() as handle:
            return list(csv.DictReader(handle))

    def test_a_failing_ligand_is_recorded_and_exits_non_zero(self):
        (self.ligands / "explodes.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1,42",
                  success=False)
        self.assertEqual([row["ligand"] for row in self.failures(out)],
                         ["explodes"] * 2)
        row = self.results(out)["explodes"]
        self.assertEqual(row["result"], "failed")
        self.assertEqual(row["best_score_mean"], "")
        self.assertEqual(row["seeds_failed"], "1 42")
        # The ligands that did work are still summarised and still scored.
        self.assertEqual(self.results(out)["pepA"]["result"], "complete")

    def test_empty_result_is_a_failure_not_a_score(self):
        (self.ligands / "empty.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1", success=False)
        self.assertIn("no poses",
                      " ".join(row["error"] for row in self.failures(out)))
        self.assertEqual(self.results(out)["empty"]["result"], "failed")

    def test_non_finite_score_is_a_failure_not_a_score(self):
        (self.ligands / "nanscore.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1", success=False)
        self.assertIn("non-finite",
                      " ".join(row["error"] for row in self.failures(out)))
        self.assertEqual(self.results(out)["nanscore"]["result"], "failed")
        with (out / "per_seed_scores.csv").open() as handle:
            scored = {row["ligand"] for row in csv.DictReader(handle)}
        self.assertNotIn("nanscore", scored,
                         "a non-finite score reached the per-seed table")

    def test_fewer_poses_than_requested_is_a_valid_result(self):
        """Vina documents n_poses as a maximum, not a guarantee.

        The returned count is limited by `energy_range` and by the 1 A
        redundancy filter, so a smaller pose set means the search found fewer
        distinct modes worth reporting. Treating that as a failed docking
        would discard legitimate results.
        """
        (self.ligands / "shortresult.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1",
                  "--n-poses", "9")
        self.assertEqual(self.results(out)["shortresult"]["result"], "complete")
        self.assertEqual(self.failures(out), [])
        with (out / "per_seed_scores.csv").open() as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row["ligand"] == "shortresult"]
        # The stub returns five poses for this ligand against nine requested.
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["n_poses_returned"], "5")
        self.assertEqual(rows[0]["n_poses_requested"], "9")

    def test_an_empty_pose_set_is_still_a_failure(self):
        """Fewer poses is legitimate; none at all is not."""
        (self.ligands / "empty.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1", success=False)
        self.assertEqual(self.results(out)["empty"]["result"], "failed")

    def test_missing_and_empty_pose_files_are_failures(self):
        for name in ("nopose", "emptypose"):
            (self.ligands / f"{name}.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1", success=False)
        reasons = " ".join(row["error"] for row in self.failures(out))
        self.assertIn("no pose file was written", reasons)
        self.assertIn("is empty", reasons)
        for name in ("nopose", "emptypose"):
            self.assertEqual(self.results(out)[name]["result"], "failed")

    def test_partial_ligand_is_marked_and_spread_not_reported_from_one_seed(self):
        """A ligand with one surviving seed must not show an SD of 0.00."""
        (self.ligands / "flaky.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        finished = self.dock(self.receptor, self.ligands, out,
                             "--box-record", self.record_path,
                             "--seeds", "1,42,137", success=False,
                             env={"STUB_FAIL_ON": "flaky:42,137"})
        row = self.results(out)["flaky"]
        self.assertEqual(row["result"], "partial")
        self.assertEqual(row["n_seeds_completed"], "1")
        self.assertEqual(row["seeds_failed"], "42 137")
        self.assertEqual(row["spread_estimable"], "no")
        self.assertEqual(row["best_score_sd"], "")
        self.assertEqual(row["best_score_seed_range"], "")
        self.assertNotEqual(row["best_score_mean"], "")
        self.assertIn("PARTIAL", finished.stdout)
        self.assertIn("incomplete", finished.stdout)

    def test_sd_over_several_seeds_uses_the_sample_estimate(self):
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1,42,137")
        row = self.results(out)["pepA"]
        scores = [float(value) for value in row["best_score_per_seed"].split()]
        self.assertEqual(row["spread_estimable"], "yes")
        self.assertAlmostEqual(float(row["best_score_sd"]),
                               round(float(np.std(scores, ddof=1)), 3), places=3)

    def test_outputs_cannot_land_in_the_ligand_directory(self):
        for destination in (self.ligands, self.ligands / "docked"):
            finished = self.dock(self.receptor, self.ligands, destination,
                                 "--box-record", self.record_path, success=False)
            self.assertIn("ligand directory", finished.stderr + finished.stdout)

    def test_previous_docked_output_among_the_inputs_stops_the_run(self):
        (self.ligands / "pepA_docked.pdbqt").write_text("REMARK stale output\n")
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path, success=False)
        self.assertIn("docked output", finished.stderr + finished.stdout)

    def test_non_empty_output_directory_is_always_refused(self):
        """Directory reuse is refused outright; there is no override flag."""
        out = self.root / "dock_run"
        out.mkdir()
        (out / "dock_results.csv").write_text("stale\n")
        finished = self.dock(self.receptor, self.ligands, out,
                             "--box-record", self.record_path, success=False)
        self.assertIn("already exists and is not empty",
                      finished.stdout + finished.stderr)
        self.assertEqual((out / "dock_results.csv").read_text(), "stale\n",
                         "a refused run must not touch the existing directory")
        self.dock(self.receptor, self.ligands, out, "--box-record",
                  self.record_path, "--overwrite", success=False)

    def test_an_empty_output_directory_is_accepted(self):
        out = self.root / "dock_run"
        out.mkdir()
        self.dock(self.receptor, self.ligands, out, "--box-record",
                  self.record_path, "--seeds", "1")

    def test_run_record_is_written_before_docking_and_carries_hashes(self):
        """An interrupted run must still leave the run record behind."""
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1,42")
        run_record = json.loads((out / "run_record.json").read_text())
        self.assertEqual(run_record["status"], "completed")
        self.assertEqual(run_record["n_requested_dockings"], 4)
        self.assertEqual(run_record["vina_version"], "0.0.0-stub")
        self.assertEqual(run_record["receptor_sha256"],
                         box_geometry.file_sha256(self.receptor))
        self.assertEqual(
            {Path(entry["path"]).name: entry["sha256"]
             for entry in run_record["ligand_inputs"]},
            {path.name: box_geometry.file_sha256(path)
             for path in sorted(self.ligands.glob("*.pdbqt"))})
        self.assertEqual(run_record["box_record_sha256"],
                         box_geometry.file_sha256(self.record_path))
        self.assertIn("NOT verified", run_record["receptor_check_scope"])

    def test_results_survive_an_interruption_mid_campaign(self):
        """Tables written per ligand, so a crash keeps what finished."""
        (self.ligands / "zz_stops_here.pdbqt").write_text("REMARK stub ligand\n")
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--seeds", "1", success=False,
                  env={"STUB_INTERRUPT_ON": "zz_stops_here"})
        # pepA and pepBB sort before zz_stops_here, so both completed first.
        results = self.results(out)
        self.assertEqual(set(results), {"pepA", "pepBB"})
        self.assertEqual(results["pepA"]["result"], "complete")
        with (out / "per_seed_scores.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["ligand"] for row in rows}, {"pepA", "pepBB"})
        self.assertEqual(json.loads((out / "run_record.json").read_text())["status"],
                         "running", "the run record should still show it never finished")

    def test_box_centred_away_from_the_receptor_is_refused(self):
        """The ~99 A failure mode: a box from the other protein copy."""
        self.rewrite_box(centre=(-20.75, 80.05, 64.93), size=(24.0, 24.0, 24.0))
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path, success=False)
        self.assertIn("outside the coordinate range", finished.stderr + finished.stdout)

    def test_receptor_from_the_wrong_author_chain_is_refused(self):
        self.rewrite_box(author_chain="B")
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path, success=False)
        self.assertIn("author protein chain", finished.stderr + finished.stdout)

    def test_docking_without_a_receptor_manifest_is_refused(self):
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path,
                             success=False, manifest=False)
        self.assertIn("provenance cannot be verified",
                      finished.stdout + finished.stderr)

    def test_unverified_receptor_is_allowed_only_deliberately_and_recorded(self):
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out, "--box-record",
                  self.record_path, "--allow-unverified-receptor",
                  "--seeds", "1", manifest=False)
        run_record = json.loads((out / "run_record.json").read_text())
        self.assertFalse(run_record["receptor_provenance_verified"])
        self.assertIsNone(run_record["receptor_condition"])
        self.assertTrue(any("NOT verified" in note
                            for note in run_record["receptor_checks"]))

    def test_manifest_from_a_different_structure_is_refused(self):
        other = self.root / "other.cif"
        other.write_text("data_other\n# a different structure\n")
        receptor_provenance.write_manifest(
            self.manifest_path,
            minimal_manifest(source_path=other, receptor_path=self.receptor),
            overwrite=True)
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path, success=False)
        self.assertIn("different source structures",
                      finished.stdout + finished.stderr)

    def test_manifest_from_a_different_protein_copy_is_refused(self):
        receptor_provenance.write_manifest(
            self.manifest_path,
            minimal_manifest(source_path=self.source, receptor_path=self.receptor,
                             author_chain="B"),
            overwrite=True)
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path, success=False)
        self.assertIn("different protein copies",
                      finished.stdout + finished.stderr)

    def test_manifest_describing_another_receptor_file_is_refused(self):
        other_receptor = self.root / "other.pdbqt"
        other_receptor.write_text(RECEPTOR_PDBQT + "\nREMARK different\n")
        receptor_provenance.write_manifest(
            self.manifest_path,
            minimal_manifest(source_path=self.source,
                             receptor_path=other_receptor),
            overwrite=True)
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path, success=False)
        self.assertIn("not the one the manifest describes",
                      finished.stdout + finished.stderr)

    def test_a_different_pocket_list_is_refused(self):
        self.rewrite_box(pocket=("25", "27", "31"))
        finished = self.dock(self.receptor, self.ligands, self.root / "dock_run",
                             "--box-record", self.record_path, success=False)
        self.assertIn("different pocket residue lists",
                      finished.stdout + finished.stderr)

    def test_verified_receptor_condition_reaches_the_run_record(self):
        receptor_provenance.write_manifest(
            self.manifest_path,
            minimal_manifest(source_path=self.source, receptor_path=self.receptor,
                             condition="anp-present"),
            overwrite=True)
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out, "--box-record",
                  self.record_path, "--seeds", "1")
        run_record = json.loads((out / "run_record.json").read_text())
        self.assertTrue(run_record["receptor_provenance_verified"])
        self.assertEqual(run_record["receptor_condition"], "anp-present")
        self.assertEqual(run_record["receptor_unresolved_decisions"],
                         ["this is a test fixture, not a prepared receptor"])

    def test_repeated_seeds_are_refused(self):
        self.dock(self.receptor, self.ligands, self.root / "dock_run",
                  "--box-record", self.record_path, "--seeds", "1,1,42",
                  success=False)

    def test_check_only_validates_without_docking(self):
        out = self.root / "dock_run"
        self.dock(self.receptor, self.ligands, out,
                  "--box-record", self.record_path, "--check-only")
        self.assertFalse((out / "poses").exists())
        self.assertFalse((out / "dock_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
