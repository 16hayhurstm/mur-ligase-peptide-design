"""Tests for the Vina peptide scoring and displacement stage.

Vina itself is never run: these tests cover the parts that can be wrong
without Vina noticing, which is most of the stage. Three groups of them carry
most of the weight.

**Atom correspondence.** A real Meeko ligand PDBQT has generic `UNL` residues
and repeated element-letter atom names, and the pose Vina writes carries no
index map at all. The fixtures here reproduce exactly that, and the
correspondence tests check that the module derives the pairing from invariants
rather than from serial numbers: one of them renumbers and reorders every atom
of the pose, which code assuming `serial == serial` would get entirely wrong
while reporting a clean result.

**Rigid-body against internal motion.** A peptide with a backbone-rigid
torsion tree still has six rigid-body degrees of freedom, so it is expected to
translate and rotate. The tests apply a pure translation, a pure rotation, a
genuine backbone deformation and a side-chain-only change, and require the
report to tell them apart.

**Fail-closed behaviour.** A comparison that cannot be founded must produce a
record saying so, not a pass and not an exception.

The test molecule is a glycyl-serine dipeptide with a constructed C-terminal
OXT, eleven heavy atoms and one polar hydrogen, in a four-level torsion tree:

    root          N1, CA1
    root.1        C1, O1, N2        (the rigid amide group)
    root.1.1      CA2
    root.1.1.1    C2, O2, OXT2      (the carboxylate)
    root.1.1.2    CB2, OG2, HG2     (the serine side chain)

Its two carboxylate oxygens sit at deliberately unequal distances from C2,
1.22 A and 1.31 A, so their invariant signatures differ and the correspondence
is determined. `SYMMETRIC_CARBOXYLATE` makes those distances equal, which is
the genuinely ambiguous case, and is used for the fail-closed tests. Both
situations occur in real output: OXT is constructed at a fixed 1.25 A while
the carbonyl oxygen comes from BoltzGen's prediction, so whether they are
distinguishable is a property of the particular file and not something to
assume either way.

**No score, and no optimisation result, is produced anywhere in this file.**
The two tests that drive the scoring functions pass a recording stub that
returns fixed numbers and asserts on *which Vina methods were called*; the
numbers are not scores of anything, and the record such a stub produces names
the stub in its `engine` field rather than Vina.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import peptide_contacts  # noqa: E402
import score_peptide_pose as sps  # noqa: E402
import verify_peptide_pdbqt as vpp  # noqa: E402

BOX_30A = REPOSITORY_ROOT / "examples" / "box_murc_uma_chainE_30A.json"
BOX_CENTRE, BOX_SIZE, BOX_RECORD = sps.load_box_record(BOX_30A)

# serial: (PDB atom name, residue name, residue number, element, AutoDock
# type, partial charge, local coordinates). The local coordinates are shifted
# to the box centre by `positions()` so that every pose is inside the recorded
# 30 A box unless a test deliberately moves an atom out of it.
ATOM_SPECS = {
    1:  ("N",   "GLY", 1, "N", "N",  -0.350, (0.000, 0.000, 0.000)),
    2:  ("CA",  "GLY", 1, "C", "C",   0.100, (1.458, 0.000, 0.000)),
    3:  ("C",   "GLY", 1, "C", "C",   0.220, (2.010, 1.420, 0.100)),
    4:  ("O",   "GLY", 1, "O", "OA", -0.270, (1.330, 2.420, 0.350)),
    5:  ("N",   "SER", 2, "N", "N",  -0.340, (3.340, 1.540, 0.050)),
    6:  ("CA",  "SER", 2, "C", "C",   0.110, (4.100, 2.760, 0.200)),
    7:  ("C",   "SER", 2, "C", "C",   0.230, (5.590, 2.540, 0.300)),
    8:  ("O",   "SER", 2, "O", "OA", -0.280, (6.190, 1.488, 0.153)),
    9:  ("OXT", "SER", 2, "O", "OA", -0.290, (6.263, 3.621, 0.606)),
    10: ("CB",  "SER", 2, "C", "C",   0.120, (3.700, 3.700, 1.350)),
    11: ("OG",  "SER", 2, "O", "OA", -0.390, (4.300, 4.980, 1.300)),
    12: ("HG",  "SER", 2, "H", "HD",  0.210, (4.100, 5.400, 2.100)),
}

# The carboxylate with both C-O distances set to exactly 1.25 A, which makes
# O2 and OXT2 indistinguishable under every invariant a PDBQT carries: their
# signatures are then {1.25 to C2, 2.19 to each other} apiece.
SYMMETRIC_CARBOXYLATE = {8: (6.205, 1.462, 0.149), 9: (6.213, 3.584, 0.592)}

# The carboxylate as the real `spec_0574` ligand will carry it: the carbonyl
# oxygen at the 1.2211 A BoltzGen predicted, and OXT at the 1.25 A stage 3
# constructs it at. Measured on the real aligned pose; the two are therefore
# separable, but only by 0.029 A.
REAL_CARBOXYLATE = {8: (6.190, 1.487, 0.152), 9: (6.213, 3.584, 0.592)}

# The same pair 0.010 A apart, which is too close to pair atoms on.
BORDERLINE_CARBOXYLATE = {8: (6.200, 1.471, 0.150), 9: (6.213, 3.584, 0.592)}

# Hydrogens Meeko merges: present in the stage-3 ligand record, absent from
# the PDBQT and from its index map.
MERGED_HYDROGENS = [("HA2", "GLY", 1, "H"), ("HA3", "GLY", 1, "H")]

# The torsion tree, as (group path, atom serials, [(bond, child), ...]).
TREE = (
    "root", [1, 2],
    [
        ((2, 3), ("root.1", [3, 4, 5], [
            ((5, 6), ("root.1.1", [6], [
                ((6, 7), ("root.1.1.1", [7, 8, 9], [])),
                ((6, 10), ("root.1.1.2", [10, 11, 12], [])),
            ])),
        ])),
    ],
)

# Which atoms move when a subtree is rotated about the bond above it.
SUBTREE_SERIALS = {
    "root.1": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "root.1.1": [6, 7, 8, 9, 10, 11, 12],
    "root.1.1.1": [7, 8, 9],
    "root.1.1.2": [10, 11, 12],
}

BACKBONE_SERIALS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDE_CHAIN_SERIALS = [10, 11]
HEAVY_SERIALS = BACKBONE_SERIALS + SIDE_CHAIN_SERIALS
TORSDOF = 4


def positions(overrides=None) -> dict:
    """Local coordinates shifted to the recorded box centre, as a dict."""
    local = {s: np.asarray(spec[6], dtype=float) for s, spec in ATOM_SPECS.items()}
    if overrides:
        for serial, coordinates in overrides.items():
            local[serial] = np.asarray(coordinates, dtype=float)
    centroid = np.mean([local[s] for s in sorted(local)], axis=0)
    offset = np.asarray(BOX_CENTRE, dtype=float) - centroid
    return {serial: local[serial] + offset for serial in local}


def atom_line(serial, name, residue_name, residue_number, chain, atom_type,
              position, charge=0.0):
    """One ATOM record with the columns in the places PDBQT puts them.

    Record 1-6, serial 7-11, atom name 13-16, altloc 17, residue name 18-20,
    chain 22, residue number 23-26, coordinates 31-54, partial charge 71-76,
    AutoDock atom type 78-79.
    """
    name_field = name if len(name) == 4 else f" {name:<3}"
    x, y, z = (float(v) for v in position)
    return (
        f"ATOM  {serial:>5} {name_field}{'':1}{residue_name:>3} "
        f"{chain:1}{residue_number:>4}{'':1}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
        f"    {charge:>6.3f} {atom_type:<2}"
    )


def index_map_lines(pairs):
    """Render `REMARK INDEX MAP` pairs the way Meeko's writer wraps them."""
    remarks = [vpp.INDEX_MAP_PREFIX]
    for input_index, serial in pairs:
        piece = f" {input_index} {serial}"
        if len(remarks[-1]) + len(piece) < 79:
            remarks[-1] += piece
        else:
            remarks.append(vpp.INDEX_MAP_PREFIX + piece)
    return remarks


def ligand_pdbqt(coordinates=None, renumber=None, index_map=True,
                 torsdof=TORSDOF, omit=(), types=None, generic=True,
                 reverse_within_groups=False, extra_atoms=(), names=None,
                 swap_siblings=False):
    """Build a PDBQT as Meeko writes one for a ligand read from an SDF.

    With `generic`, every atom gets residue `UNL`, residue number 1, a blank
    chain and a bare element letter for a name, which is what the real
    `spec_0574` output carries and why nothing can be keyed on it.

    `renumber` maps original serial to written serial, and `reverse_within_
    groups` writes each group's atoms in the opposite order: together they
    reproduce a pose whose serials and file order bear no relation to the
    starting file's. `swap_siblings` writes each group's child branches in the
    opposite order, which is the one thing about a written pose that nothing
    establishes Vina preserves.
    """
    coordinates = coordinates or positions()
    renumber = renumber or {}
    types = types or {}
    names = names or {}
    lines = []
    if index_map:
        lines.extend(
            index_map_lines(
                [
                    (serial, renumber.get(serial, serial))
                    for serial in sorted(ATOM_SPECS)
                    if serial not in omit
                ]
            )
        )

    def atom(serial):
        name, residue_name, residue_number, element, atom_type, charge, _ = (
            ATOM_SPECS[serial]
        )
        return atom_line(
            renumber.get(serial, serial),
            names.get(serial, element if generic else name),
            "UNL" if generic else residue_name,
            1 if generic else residue_number,
            "" if generic else "P",
            types.get(serial, atom_type),
            coordinates[serial],
            charge,
        )

    def emit(group):
        _path, serials, children = group
        order = list(reversed(serials)) if reverse_within_groups else serials
        for serial in order:
            if serial not in omit:
                lines.append(atom(serial))
        for (first, second), child in (
            list(reversed(children)) if swap_siblings else children
        ):
            a, b = renumber.get(first, first), renumber.get(second, second)
            lines.append(f"BRANCH {a:>3} {b:>3}")
            emit(child)
            lines.append(f"ENDBRANCH {a:>3} {b:>3}")

    path, root_serials, children = TREE
    lines.append("ROOT")
    order = list(reversed(root_serials)) if reverse_within_groups else root_serials
    for serial in order:
        if serial not in omit:
            lines.append(atom(serial))
    lines.extend(extra_atoms)
    lines.append("ENDROOT")
    for (first, second), child in (
        list(reversed(children)) if swap_siblings else children
    ):
        a, b = renumber.get(first, first), renumber.get(second, second)
        lines.append(f"BRANCH {a:>3} {b:>3}")
        emit(child)
        lines.append(f"ENDBRANCH {a:>3} {b:>3}")
    if torsdof is not None:
        lines.append(f"TORSDOF {torsdof}")
    return "\n".join(lines) + "\n"


def ligand_record(omit=(), hydrogens=MERGED_HYDROGENS):
    """A stage-3 ligand record carrying `atom_mapping`, 0-based."""
    mapping = []
    for serial in sorted(ATOM_SPECS):
        if serial in omit:
            continue
        name, residue_name, residue_number, element, _type, _charge, _xyz = (
            ATOM_SPECS[serial]
        )
        mapping.append(
            {
                "molecule_index_zero_based": serial - 1,
                "element": element,
                "author_chain": "P",
                "residue_name": residue_name,
                "residue_seqid": residue_number,
                "atom_name": name,
            }
        )
    for offset, (name, residue_name, residue_number, element) in enumerate(
        hydrogens
    ):
        mapping.append(
            {
                "molecule_index_zero_based": len(ATOM_SPECS) + offset,
                "element": element,
                "author_chain": "P",
                "residue_name": residue_name,
                "residue_seqid": residue_number,
                "atom_name": name,
            }
        )
    return json.dumps({"design_id": "spec_test", "atom_mapping": mapping}, indent=2)


def verification_record(verified=True, schema="mur-ligase-peptide-pdbqt-check/2",
                        pdbqt="ligand.pdbqt", ligand_record_name="ligand.json",
                        design_id="spec_test", overrides=None):
    """A stage-4 verification record as `verify_peptide_pdbqt.py` writes one.

    The quantities a score can be bound to -- the atom count, the heavy-atom
    count, the atom-type census, the TORSDOF declaration and the BRANCH count
    -- are computed from the fixture rather than typed, so a change to the
    test molecule cannot leave a record that silently describes something
    else. `overrides` replaces top-level fields, which is how a record for a
    different ligand or a different torsion state is built.

    The heavy-atom rule is the record's own: an atom whose *written* name does
    not begin with H. On a generic Meeko PDBQT that name is the element
    letter, which is what the fixture writes.
    """
    census = {}
    for spec in ATOM_SPECS.values():
        census[spec[4]] = census.get(spec[4], 0) + 1
    record = {
        "schema": schema,
        "design_id": design_id,
        "pdbqt": pdbqt,
        "ligand_record": ligand_record_name,
        "atoms_in_pdbqt": len(ATOM_SPECS),
        "heavy_atoms_in_pdbqt": sum(
            1 for spec in ATOM_SPECS.values()
            if not spec[3].upper().startswith("H")
        ),
        "atom_type_census": census,
        "verified": verified,
        "verification_failures": [] if verified else ["a check failed"],
        "torsdof": {
            "declared": TORSDOF,
            "declaration": "present",
            "branch_records": TORSDOF,
        },
        "identity_recovery": {"method": "index map", "established": True},
        "torsion_tree": {
            "backbone_phi_psi_count": 2, "side_chain_count": 2,
        },
    }
    if overrides:
        record.update(overrides)
    return json.dumps(record)


def peptide_source_pdb(coordinates=None):
    """The stage-3 structure the ligand PDBQT was built from, heavy atoms only.

    Written with the real residue names, numbers and atom names, which is what
    the coordinate comparison joins on. The PDBQT's own generic names are
    never used for the join, so they cannot help it.
    """
    coordinates = coordinates or positions()
    lines = []
    for serial, spec in sorted(ATOM_SPECS.items()):
        name, residue_name, residue_number, element = spec[0], spec[1], spec[2], spec[3]
        if element.upper().startswith("H"):
            continue
        lines.append(
            receptor_atom_line(
                serial, name, residue_name, residue_number, element,
                coordinates[serial],
            )
        )
    return "\n".join(lines) + "\nEND\n"


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


def rotation_matrix(axis, degrees):
    """A right-handed rotation matrix about `axis` by `degrees`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(degrees)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def rigidly_moved(coordinates, translation=(0.0, 0.0, 0.0), degrees=0.0,
                  axis=(0.0, 0.0, 1.0)):
    """Apply one rigid-body motion to every atom: rotation then translation.

    The rotation is about the set's own centroid, so it is a pure rotation
    with no translation hidden inside it.
    """
    serials = sorted(coordinates)
    stack = np.asarray([coordinates[s] for s in serials], dtype=float)
    centroid = stack.mean(axis=0)
    rotation = rotation_matrix(axis, degrees)
    moved = (stack - centroid) @ rotation.T + centroid + np.asarray(
        translation, dtype=float
    )
    return {serial: moved[i] for i, serial in enumerate(serials)}


def torsion_rotated(coordinates, group, degrees):
    """Rotate one rigid group's subtree about the bond that carries it.

    This is the only motion a torsion change makes, so it preserves every
    distance inside every rigid group and every rotatable bond length, exactly
    as Vina's local optimisation does.
    """
    bonds = {
        "root.1": (2, 3), "root.1.1": (5, 6),
        "root.1.1.1": (6, 7), "root.1.1.2": (6, 10),
    }
    first, second = bonds[group]
    moved = dict(coordinates)
    origin = np.asarray(coordinates[second], dtype=float)
    axis = origin - np.asarray(coordinates[first], dtype=float)
    rotation = rotation_matrix(axis, degrees)
    for serial in SUBTREE_SERIALS[group]:
        point = np.asarray(coordinates[serial], dtype=float) - origin
        moved[serial] = point @ rotation.T + origin
    return moved


def rounded(coordinates):
    """Coordinates as a PDBQT stores them, to three decimal places."""
    return {
        serial: np.round(np.asarray(value, dtype=float), 3)
        for serial, value in coordinates.items()
    }


def prepared(directory: Path, after_coordinates=None, **kwargs):
    """Write a starting ligand, its record, and optionally a written pose.

    Returns `(ligand_path, record_path, pose_path_or_None)`. The pose carries
    no index map, as a pose Vina writes does not, so nothing in the
    correspondence can quietly depend on one.
    """
    ligand = write(directory, "ligand.pdbqt", ligand_pdbqt())
    record = write(directory, "ligand.json", ligand_record())
    pose = None
    if after_coordinates is not None:
        pose = write(
            directory,
            "pose.pdbqt",
            ligand_pdbqt(
                coordinates=rounded(after_coordinates), index_map=False, **kwargs
            ),
        )
    return ligand, record, pose


class BoxRecordTests(unittest.TestCase):
    """Retained from the first version of this stage: the box is read, not typed."""

    def test_the_recorded_30_angstrom_box_loads(self):
        centre, size, record = sps.load_box_record(BOX_30A)
        self.assertEqual(size, [30.0, 30.0, 30.0])
        self.assertEqual(len(centre), 3)
        self.assertEqual(record["schema"], "mur-ligase-docking-box/1")

    def test_a_box_without_a_centre_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "box.json",
                json.dumps({"size_angstrom": [10, 10, 10]}),
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.load_box_record(path)
            self.assertIn("centre_angstrom", str(raised.exception))

    def test_a_zero_dimension_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "box.json",
                json.dumps(
                    {"centre_angstrom": [0, 0, 0], "size_angstrom": [10, 0, 10]}
                ),
            )
            with self.assertRaises(sps.ScoringError):
                sps.load_box_record(path)

    def test_a_two_component_centre_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "box.json",
                json.dumps(
                    {"centre_angstrom": [0, 0], "size_angstrom": [10, 10, 10]}
                ),
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.load_box_record(path)
            self.assertIn("three values", str(raised.exception))

    def test_an_unreadable_box_record_is_refused(self):
        with self.assertRaises(sps.ScoringError) as raised:
            sps.load_box_record(Path("/nonexistent/box.json"))
        self.assertIn("cannot read", str(raised.exception))


class TorsionCountTests(unittest.TestCase):
    def test_the_torsdof_line_is_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", ligand_pdbqt(torsdof=30))
            self.assertEqual(sps.pdbqt_torsion_count(path), 30)

    def test_a_file_without_torsdof_gives_none_rather_than_zero(self):
        # Zero would be a claim that the ligand is rigid, which is different
        # from not knowing. The declaration status says which case it is.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", ligand_pdbqt(torsdof=None))
            self.assertIsNone(sps.pdbqt_torsion_count(path))
            self.assertEqual(vpp.torsdof_status(path), "absent")

    def test_a_malformed_torsdof_gives_none_and_is_reported_as_malformed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "l.pdbqt",
                ligand_pdbqt(torsdof=None) + "TORSDOF many\n",
            )
            self.assertIsNone(sps.pdbqt_torsion_count(path))
            self.assertEqual(vpp.torsdof_status(path), "malformed")


class RigidGroupParsingTests(unittest.TestCase):
    """The torsion tree is the frame the correspondence is built in."""

    def test_every_atom_is_placed_in_a_rigid_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", ligand_pdbqt())
            tree = sps.parse_rigid_groups(path)
            self.assertEqual(
                {path_: sorted(s) for path_, s in tree.groups.items()},
                {
                    "root": [1, 2],
                    "root.1": [3, 4, 5],
                    "root.1.1": [6],
                    "root.1.1.1": [7, 8, 9],
                    "root.1.1.2": [10, 11, 12],
                },
            )
            self.assertEqual(len(tree.group_of_serial), len(ATOM_SPECS))

    def test_a_root_level_branch_after_endroot_still_belongs_to_the_root(self):
        # The PDBQT layout puts the root's own branches after ENDROOT. A
        # parser that popped the root at ENDROOT would call that an error, or
        # would give the branch the wrong parent.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", ligand_pdbqt())
            tree = sps.parse_rigid_groups(path)
            self.assertIn(("root", "root.1"), tree.bond_groups)
            self.assertEqual(tree.bonds[0], (2, 3))

    def test_nesting_depth_is_recorded_in_the_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", ligand_pdbqt())
            tree = sps.parse_rigid_groups(path)
            self.assertEqual(tree.group_of_serial[11], "root.1.1.2")
            self.assertEqual(
                tree.bond_groups,
                [
                    ("root", "root.1"),
                    ("root.1", "root.1.1"),
                    ("root.1.1", "root.1.1.1"),
                    ("root.1.1", "root.1.1.2"),
                ],
            )

    def test_an_atom_outside_every_group_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            stray = atom_line(
                99, "C", "UNL", 1, "", "C", positions()[1], 0.0
            )
            path = write(
                Path(temporary), "l.pdbqt", ligand_pdbqt() + stray + "\n"
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.parse_rigid_groups(path)
            self.assertIn("belongs to no rigid group", str(raised.exception))

    def test_an_unclosed_branch_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            text = ligand_pdbqt().replace("ENDBRANCH   6   7\n", "", 1)
            path = write(Path(temporary), "l.pdbqt", text)
            with self.assertRaises(sps.ScoringError) as raised:
                sps.parse_rigid_groups(path)
            self.assertIn("never closed", str(raised.exception))

    def test_a_malformed_branch_record_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            text = ligand_pdbqt().replace("BRANCH   6   7", "BRANCH   6")
            path = write(Path(temporary), "l.pdbqt", text)
            with self.assertRaises(sps.ScoringError) as raised:
                sps.parse_rigid_groups(path)
            self.assertIn("malformed BRANCH", str(raised.exception))

    def test_a_file_with_no_root_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            text = ligand_pdbqt().replace("ROOT\n", "", 1).replace("ENDROOT\n", "", 1)
            path = write(Path(temporary), "l.pdbqt", text)
            with self.assertRaises(sps.ScoringError) as raised:
                sps.parse_rigid_groups(path)
            self.assertIn("ROOT", str(raised.exception))


class GenericPdbqtTests(unittest.TestCase):
    """The file the real stage produces, and what can and cannot be read from it."""

    def test_the_fixture_really_is_generic_with_repeated_identities(self):
        # If this ever stopped being true, every other test in this file would
        # be exercising a file easier than the real one.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", ligand_pdbqt())
            atoms = sps.pose_atoms(path)
            self.assertEqual({a["residue_name"] for a in atoms.values()}, {"UNL"})
            self.assertEqual({a["residue_seq"] for a in atoms.values()}, {1})
            names = [a["atom_name"] for a in atoms.values()]
            self.assertGreater(len(names), len(set(names)))
            self.assertEqual(sorted(set(names)), ["C", "H", "N", "O"])

    def test_the_old_identity_based_reader_would_have_aborted_on_it(self):
        # The defect this task exists to repair: reading the pose by full atom
        # identity raises on the duplicated identities, before Vina is even
        # constructed. The module's own reader keys on serial instead.
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "l.pdbqt", ligand_pdbqt())
            import receptor_provenance

            with self.assertRaises(receptor_provenance.ReceptorError) as raised:
                receptor_provenance.parse_atom_records(path)
            self.assertIn("duplicated atom identity", str(raised.exception))
            self.assertEqual(len(sps.pose_atoms(path)), len(ATOM_SPECS))

    def test_identities_come_from_the_ligand_record_and_the_index_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record, _ = prepared(directory)
            recovery = sps.load_identity_recovery(ligand, record)
            self.assertTrue(recovery.established)
            self.assertEqual(recovery.by_serial[9].label, "SER2:OXT")
            self.assertEqual(recovery.by_serial[1].label, "GLY1:N")
            self.assertEqual(len(recovery.hydrogens_absent_from_pdbqt), 2)
            self.assertEqual(recovery.heavy_atoms_absent_from_pdbqt, [])

    def test_a_missing_ligand_record_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(directory, "l.pdbqt", ligand_pdbqt())
            with self.assertRaises(sps.ScoringError) as raised:
                sps.load_identity_recovery(ligand, directory / "absent.json")
            self.assertIn("not found", str(raised.exception))

    def test_a_pdbqt_without_an_index_map_fails_closed_with_the_remedy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(
                directory, "l.pdbqt", ligand_pdbqt(index_map=False)
            )
            record = write(directory, "l.json", ligand_record())
            with self.assertRaises(sps.ScoringError) as raised:
                sps.load_identity_recovery(ligand, record)
            self.assertIn("--add_index_map", str(raised.exception))


class CorrespondenceTests(unittest.TestCase):
    """Which pose atom is which starting atom, and how that is established."""

    def correspondence(self, directory, after, **kwargs):
        resolve = kwargs.pop("resolve_symmetric_atoms", False)
        ligand, _record, pose = prepared(directory, after_coordinates=after, **kwargs)
        return sps.establish_correspondence(
            sps.pose_atoms(ligand),
            sps.parse_rigid_groups(ligand),
            sps.pose_atoms(pose),
            sps.parse_rigid_groups(pose),
            resolve_symmetric_atoms=resolve,
        )

    def test_an_unmoved_pose_corresponds_atom_for_atom(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(Path(temporary), positions())
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(len(found.pairs), len(ATOM_SPECS))
            self.assertEqual(found.max_intra_group_deviation_angstrom, 0.0)

    def test_renumbered_and_reordered_atoms_are_still_paired_correctly(self):
        # The acceptance test that matters most. Every serial is replaced by
        # 13 - serial and every group's atoms are written in reverse order, so
        # a module that assumed Vina preserved serials or file order would
        # pair every atom with the wrong one and report a large spurious
        # displacement while claiming success.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            renumber = {serial: 13 - serial for serial in ATOM_SPECS}
            found = self.correspondence(
                directory, positions(),
                renumber=renumber, reverse_within_groups=True,
            )
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(found.pairs, renumber)
            self.assertFalse(found.serial_order_preserved)
            self.assertEqual(found.max_intra_group_deviation_angstrom, 0.0)

    def test_whether_serials_survived_is_reported_and_not_assumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            kept = self.correspondence(directory, positions())
            self.assertTrue(kept.serial_order_preserved)
            swapped = self.correspondence(
                directory, positions(), renumber={8: 9, 9: 8},
            )
            self.assertTrue(swapped.established, swapped.reasons)
            self.assertFalse(swapped.serial_order_preserved)
            self.assertEqual(swapped.pairs[8], 9)
            self.assertEqual(swapped.pairs[9], 8)

    def test_a_rigid_body_move_preserves_the_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(
                Path(temporary),
                rigidly_moved(positions(), translation=(1.5, -0.5, 0.25), degrees=20.0),
            )
            self.assertTrue(found.established, found.reasons)
            self.assertLess(found.max_intra_group_deviation_angstrom, 0.01)

    def test_a_torsion_change_preserves_the_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(
                Path(temporary), torsion_rotated(positions(), "root.1.1.1", 40.0)
            )
            self.assertTrue(found.established, found.reasons)
            self.assertLess(found.max_bond_length_deviation_angstrom, 0.01)

    def test_a_pose_with_a_missing_atom_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(Path(temporary), positions(), omit=(11,))
            self.assertFalse(found.established)
            self.assertTrue(
                any("atom count" in reason for reason in found.reasons),
                found.reasons,
            )
            self.assertEqual(found.pairs, {})

    def test_a_pose_with_an_unexpected_extra_atom_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            extra = atom_line(
                13, "C", "UNL", 1, "", "C", positions()[2] + 0.5, 0.1
            )
            found = self.correspondence(
                Path(temporary), positions(), extra_atoms=(extra,)
            )
            self.assertFalse(found.established)
            self.assertEqual(found.pairs, {})

    def test_a_duplicated_serial_in_the_pose_is_refused_when_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            duplicate = atom_line(
                7, "C", "UNL", 1, "", "C", positions()[7], 0.23
            )
            pose = write(
                directory, "pose.pdbqt",
                ligand_pdbqt(index_map=False, extra_atoms=(duplicate,)),
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.pose_atoms(pose)
            self.assertIn("appears twice", str(raised.exception))

    def test_a_pose_whose_atom_types_differ_fails_closed(self):
        # A different molecule, or the same molecule prepared differently.
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(
                Path(temporary), positions(), types={11: "NA"}
            )
            self.assertFalse(found.established)
            self.assertEqual(found.pairs, {})

    def test_a_pose_whose_torsion_tree_differs_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(directory, "l.pdbqt", ligand_pdbqt())
            flattened = ligand_pdbqt(index_map=False).replace(
                "BRANCH   6  10", "REMARK was a branch"
            ).replace("ENDBRANCH   6  10", "REMARK was an endbranch")
            pose = write(directory, "pose.pdbqt", flattened)
            found = sps.establish_correspondence(
                sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
                sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
            )
            self.assertFalse(found.established)
            self.assertTrue(
                any("torsion trees differ" in r or "bond" in r
                    for r in found.reasons),
                found.reasons,
            )

    def test_a_distorted_rigid_group_fails_closed(self):
        # Vina cannot change a distance inside a rigid group. A pose in which
        # one did is either not this ligand or not correctly paired, and
        # either way the comparison must not be made.
        with tempfile.TemporaryDirectory() as temporary:
            moved = positions()
            moved[4] = moved[4] + np.asarray([0.6, 0.0, 0.0])
            found = self.correspondence(Path(temporary), moved)
            self.assertFalse(found.established)
            self.assertTrue(
                any("rigid group" in reason for reason in found.reasons),
                found.reasons,
            )

    def test_the_method_is_recorded_and_does_not_claim_to_use_serials(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(Path(temporary), positions())
            method = found.as_dict()["method"]
            self.assertIn("rigid group", method)
            self.assertIn("never used as evidence", method)
            self.assertIn(
                "intra_group_pairs_checked", found.as_dict()["checks"]
            )


class AmbiguousCorrespondenceTests(unittest.TestCase):
    """Atoms no PDBQT invariant can tell apart are reported, not guessed."""

    def files(self, directory, after_transform):
        before = positions(SYMMETRIC_CARBOXYLATE)
        ligand = write(
            directory, "l.pdbqt", ligand_pdbqt(coordinates=rounded(before))
        )
        pose = write(
            directory, "pose.pdbqt",
            ligand_pdbqt(
                coordinates=rounded(after_transform(before)), index_map=False
            ),
        )
        return ligand, pose

    def correspondence(self, directory, after_transform, resolve=False):
        ligand, pose = self.files(directory, after_transform)
        return sps.establish_correspondence(
            sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
            sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
            resolve_symmetric_atoms=resolve,
        )

    def test_a_symmetric_carboxylate_is_not_established_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(
                Path(temporary), lambda p: rigidly_moved(p, translation=(1.0, 0, 0))
            )
            self.assertFalse(found.established)
            self.assertEqual(found.pairs, {})
            reason = " ".join(found.reasons)
            self.assertIn("cannot be told apart", reason)
            self.assertIn("serial 8", reason)
            self.assertIn("serial 9", reason)
            self.assertIn("--resolve-symmetric-atoms", reason)

    def test_the_two_oxygens_really_are_indistinguishable_in_that_fixture(self):
        # Without this, the test above might be passing for some other reason.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, _pose = self.files(directory, lambda p: p)
            atoms = sps.pose_atoms(ligand)
            tree = sps.parse_rigid_groups(ligand)
            signatures = sps.intra_group_signatures(atoms, tree)
            self.assertEqual(
                atoms[8]["atom_type"], atoms[9]["atom_type"]
            )
            self.assertLess(
                float(np.max(np.abs(signatures[8] - signatures[9]))),
                sps.RIGID_DISTANCE_TOLERANCE_ANGSTROM,
            )

    def test_the_asymmetric_carboxylate_is_distinguishable(self):
        # The same two atoms, 1.22 A and 1.31 A from C2 rather than equal, are
        # separated by the invariant signature with no convention needed.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, _record, pose = prepared(
                directory, after_coordinates=positions()
            )
            found = sps.establish_correspondence(
                sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
                sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
            )
            self.assertTrue(found.established, found.reasons)
            self.assertIn(8, found.determined_by_signature)
            self.assertIn(9, found.determined_by_signature)

    def test_the_convention_pairs_them_by_minimal_displacement_and_records_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(
                Path(temporary),
                lambda p: rigidly_moved(p, translation=(0.4, 0, 0)),
                resolve=True,
            )
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(len(found.symmetry_resolved), 1)
            resolved = found.symmetry_resolved[0]
            self.assertEqual(resolved["starting_serials"], [8, 9])
            self.assertEqual(resolved["atom_type"], "OA")
            self.assertIn("minimal total squared displacement", resolved["convention"])
            # A small translation keeps each oxygen nearest to itself.
            self.assertEqual(resolved["assignment"], {"8": 8, "9": 9})

    def test_the_convention_chooses_the_nearer_assignment_when_they_swap(self):
        # Writing the two oxygens in each other's place is indistinguishable
        # from them having exchanged positions, which is exactly why this is a
        # convention and not a measurement.
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(
                Path(temporary),
                lambda p: p,
                resolve=True,
            )
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(
                found.symmetry_resolved[0]["assignment"], {"8": 8, "9": 9}
            )

    def test_the_limitation_is_stated_in_the_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(
                Path(temporary), lambda p: p, resolve=True
            )
            self.assertIn("lower bound", found.as_dict()["limitation"])


# ---------------------------------------------------------------------------
# A second, smaller molecule whose root carries two sibling branches. The
# order sibling branches are written in is the one thing about a PDBQT that
# nothing establishes Vina's writer preserves, and equivalent branches written
# the other way round are the case an order-dependent correspondence gets
# wrong while every invariant inside each branch still agrees.
# ---------------------------------------------------------------------------

# serial: (generic name, AutoDock type, coordinates). Atoms 1-2 are the root,
# 3-4 branch A, 5-6 branch B, and 7 is branch B's third atom when a test
# wants the two branches to differ in size.
# The root's second atom is a nitrogen so that the two root atoms are
# themselves distinguishable: two equivalent atoms anywhere in the molecule
# fail the correspondence closed, which would mask what these tests are about.
SIBLING_SPECS = {
    1: ("C", "C",  ( 0.000,  0.000,  0.000)),
    2: ("N", "N",  ( 0.000,  0.000,  1.470)),
    3: ("O", "OA", ( 1.000,  0.000, -0.700)),
    4: ("H", "HD", ( 1.500,  0.700, -1.000)),
    5: ("O", "OA", (-1.000,  0.000, -0.700)),
    6: ("H", "HD", (-1.500,  0.700, -1.000)),
    7: ("C", "C",  (-1.600, -1.000, -1.300)),
}

# Branch B's hydrogen moved so that the two branches differ geometrically:
# the O-H distance becomes 1.049 A against branch A's 0.911 A.
DISTINCT_SIBLING_HYDROGEN = (-1.500, 0.700, -1.300)


def sibling_positions(distinguishable=False):
    coordinates = {
        serial: np.asarray(spec[2], dtype=float)
        for serial, spec in SIBLING_SPECS.items()
    }
    if distinguishable:
        coordinates[6] = np.asarray(DISTINCT_SIBLING_HYDROGEN, dtype=float)
    return coordinates


def sibling_pdbqt(swap=False, coordinates=None, distinguishable=False,
                  extra=False, nest=False):
    """A PDBQT whose root carries two sibling branches.

    Each branch is an O-H pair attached to the root carbon at the same bond
    length, so with the defaults the two are *equivalent*: same attachment,
    same atom types, same internal geometry. No invariant a PDBQT carries can
    tell them apart, and the only thing that distinguishes them in the file is
    which was written first.

    `swap` writes branch B first. `distinguishable` moves branch B's hydrogen
    so the branches differ geometrically. `extra` gives branch B a third atom
    so they differ in size. `nest` attaches branch B to branch A's oxygen
    instead of to the root, which makes a different tree of the same atoms.
    """
    coordinates = coordinates or sibling_positions(distinguishable)
    branch_b_serials = [5, 6, 7] if extra else [5, 6]

    def atom(serial):
        name, atom_type, _ = SIBLING_SPECS[serial]
        return atom_line(
            serial, name, "UNL", 1, "", atom_type, coordinates[serial], 0.0
        )

    def branch(first, second, serials):
        lines = [f"BRANCH {first:>3} {second:>3}"]
        lines.extend(atom(serial) for serial in serials)
        lines.append(f"ENDBRANCH {first:>3} {second:>3}")
        return lines

    first_branch = branch(1, 3, [3, 4])
    second_branch = branch(3 if nest else 1, 5, branch_b_serials)
    if nest:
        # Branch B becomes a child of branch A: same atoms, different tree.
        first_branch = (
            first_branch[:-1] + second_branch + first_branch[-1:]
        )
        blocks = first_branch
    else:
        blocks = (
            second_branch + first_branch if swap
            else first_branch + second_branch
        )
    lines = ["ROOT", atom(1), atom(2), "ENDROOT"] + blocks + ["TORSDOF 2"]
    return "\n".join(lines) + "\n"


def sibling_files(directory, swap=False, after=None, **kwargs):
    """Write a starting ligand and a written pose of the sibling molecule."""
    before = sibling_pdbqt(**kwargs)
    pose = sibling_pdbqt(swap=swap, coordinates=after, **kwargs)
    return (
        write(directory, "siblings.pdbqt", before),
        write(directory, "siblings_pose.pdbqt", pose),
    )


def sibling_correspondence(directory, swap=False, resolve=False, **kwargs):
    ligand, pose = sibling_files(directory, swap=swap, **kwargs)
    return sps.establish_correspondence(
        sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
        sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
        resolve_symmetric_atoms=resolve,
    )


def path_based_group_matcher(before_atoms, before_tree, after_atoms,
                             after_tree, tolerance=None):
    """The defective matcher: pair rigid groups by the path they sit at.

    This is what the module did before the sibling-branch defect was found,
    reinstated here so the defect itself is under test rather than only its
    absence. A group's path is built from the order its siblings appear in, so
    this pairs branch A with whatever branch happens to have been written
    first.
    """
    return {path: path for path in before_tree.groups}, [], []


class SiblingBranchTests(unittest.TestCase):
    """Sibling-branch order is not evidence, and is not treated as any."""

    def test_the_two_branches_of_the_fixture_really_are_equivalent(self):
        # Without this the tests below would prove much less: they would be
        # consistent with the branches being quietly distinguishable.
        with tempfile.TemporaryDirectory() as temporary:
            ligand, _pose = sibling_files(Path(temporary))
            atoms = sps.pose_atoms(ligand)
            tree = sps.parse_rigid_groups(ligand)
            keys = sps.group_structural_keys(atoms, tree)
            signatures = sps.subtree_signatures(atoms, tree)
            self.assertEqual(keys["root.1"], keys["root.2"])
            self.assertEqual(
                0.0,
                float(
                    np.max(
                        np.abs(signatures["root.1"] - signatures["root.2"])
                    )
                ),
            )
            # And a cross-assignment is visible: the two oxygens are 2 A apart.
            self.assertAlmostEqual(
                float(
                    np.linalg.norm(
                        np.asarray(atoms[3]["position"], dtype=float)
                        - np.asarray(atoms[5]["position"], dtype=float)
                    )
                ),
                2.0,
                places=6,
            )

    def test_pairing_branches_by_written_path_cross_assigns_a_swapped_pair(self):
        # The defect, demonstrated. With the branches written in the opposite
        # order and no coordinate changed at all, a path-keyed correspondence
        # pairs each branch with the other one, passes its own verification
        # because every invariant inside each branch still agrees, and reports
        # 2 A of displacement for atoms that never moved.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with mock.patch.object(
                sps, "match_rigid_groups", path_based_group_matcher
            ):
                found = sibling_correspondence(directory, swap=True)
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(found.pairs, {1: 1, 2: 2, 3: 5, 4: 6, 5: 3, 6: 4})
            ligand, pose = sibling_files(directory, swap=True)
            before = sps.pose_atoms(ligand)
            after = sps.pose_atoms(pose)
            def moved(serial):
                return float(
                    np.linalg.norm(
                        np.asarray(
                            after[found.pairs[serial]]["position"], dtype=float
                        )
                        - np.asarray(before[serial]["position"], dtype=float)
                    )
                )

            apparent = max(moved(serial) for serial in found.pairs)
            self.assertAlmostEqual(apparent, 3.0, places=6)
            self.assertAlmostEqual(moved(3), 2.0, places=6)

    def test_a_swapped_pair_of_equivalent_branches_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = sibling_correspondence(Path(temporary), swap=True)
            self.assertFalse(found.established)
            self.assertEqual(found.pairs, {})
            reason = " ".join(found.reasons)
            self.assertIn("cannot be told apart", reason)
            self.assertIn("root.1", reason)
            self.assertIn("root.2", reason)

    def test_the_symmetric_atom_flag_does_not_rescue_a_whole_branch(self):
        # Pairing two atoms by minimal displacement is a convention this
        # module will record. Pairing two whole branches that way would also
        # fix every atom beneath them, so it is refused instead.
        with tempfile.TemporaryDirectory() as temporary:
            found = sibling_correspondence(
                Path(temporary), swap=True, resolve=True
            )
            self.assertFalse(found.established)
            self.assertIn(
                "does not extend to whole branches", " ".join(found.reasons)
            )

    def test_equivalent_branches_are_refused_even_written_in_order(self):
        # The order happening to agree is not evidence that it was preserved,
        # so the unswapped file fails closed in exactly the same way.
        with tempfile.TemporaryDirectory() as temporary:
            found = sibling_correspondence(Path(temporary), swap=False)
            self.assertFalse(found.established)
            self.assertIn("cannot be told apart", " ".join(found.reasons))

    def test_geometrically_distinct_branches_are_paired_across_a_swap(self):
        # Branch B's O-H distance differs by 0.138 A, which no rigid-body
        # motion and no torsion rotation can produce, so the branches are
        # distinguishable and the swap must be seen through.
        with tempfile.TemporaryDirectory() as temporary:
            found = sibling_correspondence(
                Path(temporary), swap=True, distinguishable=True
            )
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(
                found.pairs, {serial: serial for serial in range(1, 7)}
            )
            self.assertEqual(found.group_pairs["root.1"], "root.2")
            self.assertEqual(found.group_pairs["root.2"], "root.1")
            self.assertEqual(found.max_intra_group_deviation_angstrom, 0.0)

    def test_branches_of_unequal_size_are_paired_across_a_swap(self):
        # A path-keyed scheme refuses this one outright, because the group at
        # path root.1 holds two atoms in one file and three in the other.
        with tempfile.TemporaryDirectory() as temporary:
            found = sibling_correspondence(
                Path(temporary), swap=True, extra=True
            )
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(
                found.pairs, {serial: serial for serial in range(1, 8)}
            )

    def test_the_matched_branches_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = sibling_correspondence(
                Path(temporary), swap=True, distinguishable=True
            )
            recorded = found.as_dict()
            self.assertEqual(
                recorded["rigid_group_pairs"],
                {"root": "root", "root.1": "root.2", "root.2": "root.1"},
            )
            self.assertFalse(recorded["sibling_branch_order_assumed"])
            self.assertIn(
                "Branch order in the file is not used", recorded["method"]
            )

    def test_the_same_atoms_in_a_different_tree_are_refused(self):
        # Branch B attached to branch A's oxygen rather than to the root: the
        # same atoms, the same group sizes, a different molecule's tree.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(directory, "l.pdbqt", sibling_pdbqt())
            pose = write(directory, "pose.pdbqt", sibling_pdbqt(nest=True))
            found = sps.establish_correspondence(
                sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
                sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
            )
            self.assertFalse(found.established)
            self.assertIn("not the same tree", " ".join(found.reasons))

    def test_a_structurally_distinct_sibling_swap_is_seen_through(self):
        # The main fixture's two deepest branches are the carboxylate and the
        # serine side chain. They differ in atom types, so swapping the order
        # they are written in must change nothing at all.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record, pose = prepared(
                directory, after_coordinates=positions(), swap_siblings=True
            )
            found = sps.establish_correspondence(
                sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
                sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
            )
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(
                found.pairs, {serial: serial for serial in ATOM_SPECS}
            )
            self.assertEqual(
                found.group_pairs["root.1.1.1"], "root.1.1.2"
            )
            displacement = sps.displacement_report(
                sps.pose_atoms(ligand), sps.pose_atoms(pose), found,
                sps.load_identity_recovery(ligand, record),
            )
            self.assertEqual(
                displacement["internal_backbone_deformation"]["rmsd_angstrom"],
                0.0,
            )


class HeavyAtomSelectionTests(unittest.TestCase):
    """Heavy atoms come from the AutoDock type, never from the atom name."""

    def test_hydrogens_are_excluded_by_type_and_heavy_atoms_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record, _ = prepared(directory)
            recovery = sps.load_identity_recovery(ligand, record)
            serials, report = sps.heavy_atom_serials(
                sps.pose_atoms(ligand), recovery
            )
            self.assertEqual(serials, HEAVY_SERIALS)
            self.assertEqual(report["heavy_atoms"], 11)
            self.assertEqual(report["hydrogen_atoms"], 1)
            self.assertIn("atom type", report["selected_by"])

    def test_a_heavy_atom_named_like_a_hydrogen_is_still_heavy(self):
        # A constructed case, and the point of it is narrow: it shows the atom
        # name is not load-bearing. The previous implementation selected heavy
        # atoms with `atom_name.startswith("H")`, which on a file like this
        # would silently drop a real oxygen from the box check and from every
        # clash count.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(
                directory, "l.pdbqt", ligand_pdbqt(names={11: "HG"})
            )
            record = write(directory, "l.json", ligand_record())
            serials, report = sps.heavy_atom_serials(sps.pose_atoms(ligand))
            self.assertIn(11, serials)
            self.assertEqual(report["heavy_atoms"], 11)
            by_name = [
                serial for serial in sps.pose_atoms(ligand)
                if not str(
                    sps.pose_atoms(ligand)[serial]["atom_name"]
                ).upper().startswith("H")
            ]
            self.assertNotIn(11, by_name)

    def test_a_blank_atom_type_is_fatal_rather_than_assumed_heavy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(directory, "l.pdbqt", ligand_pdbqt(types={6: ""}))
            with self.assertRaises(sps.ScoringError) as raised:
                sps.heavy_atom_serials(sps.pose_atoms(ligand))
            self.assertIn("blank AutoDock atom type", str(raised.exception))

    def test_a_type_disagreeing_with_the_recovered_element_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(directory, "l.pdbqt", ligand_pdbqt(types={11: "HD"}))
            record = write(directory, "l.json", ligand_record())
            recovery = sps.load_identity_recovery(ligand, record)
            with self.assertRaises(sps.ScoringError) as raised:
                sps.heavy_atom_serials(sps.pose_atoms(ligand), recovery)
            self.assertIn("disagree", str(raised.exception))

    def test_the_constructed_oxt_is_among_the_atoms_evaluated(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record, _ = prepared(directory)
            recovery = sps.load_identity_recovery(ligand, record)
            _serials, report = sps.heavy_atom_serials(
                sps.pose_atoms(ligand), recovery
            )
            self.assertTrue(report["constructed_c_terminal_atom_evaluated"])
            self.assertEqual(report["expected_heavy_atom_identities"], 11)
            self.assertEqual(report["evaluated_heavy_atom_identities"], 11)
            self.assertEqual(report["expected_but_not_evaluated"], [])

    def test_a_file_of_only_hydrogens_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(
                directory, "l.pdbqt",
                ligand_pdbqt(types={s: "HD" for s in ATOM_SPECS}),
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.heavy_atom_serials(sps.pose_atoms(ligand))
            self.assertIn("no heavy atoms", str(raised.exception))


class BoxContainmentTests(unittest.TestCase):
    def test_the_whole_peptide_sits_inside_the_recorded_30_angstrom_box(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record, _ = prepared(directory)
            recovery = sps.load_identity_recovery(ligand, record)
            containment = sps.pose_box_containment(
                sps.pose_atoms(ligand), BOX_RECORD, recovery
            )
            self.assertTrue(containment["fully_contained"])
            self.assertEqual(containment["heavy_atoms"], 11)
            self.assertIn("inside the box", containment["consequence"])
            self.assertTrue(
                containment["heavy_atom_selection"][
                    "constructed_c_terminal_atom_evaluated"
                ]
            )

    def test_an_atom_outside_the_box_is_reported_with_its_consequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            moved = positions()
            moved[9] = moved[9] + np.asarray([40.0, 0.0, 0.0])
            ligand = write(
                directory, "l.pdbqt", ligand_pdbqt(coordinates=moved)
            )
            record = write(directory, "l.json", ligand_record())
            recovery = sps.load_identity_recovery(ligand, record)
            containment = sps.pose_box_containment(
                sps.pose_atoms(ligand), BOX_RECORD, recovery
            )
            self.assertFalse(containment["fully_contained"])
            self.assertEqual(containment["atoms_inside"], 10)
            self.assertIn("not a score of this pose", containment["consequence"])

    def test_an_unevaluated_heavy_atom_is_refused_rather_than_ignored(self):
        # The ligand record describes eleven heavy atoms; this PDBQT carries
        # ten. A containment check on the ten it happens to have would report
        # the pose as contained.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand = write(directory, "l.pdbqt", ligand_pdbqt())
            record = write(directory, "l.json", ligand_record())
            recovery = sps.load_identity_recovery(ligand, record)
            del recovery.by_serial[9]
            recovery.heavy_atoms_absent_from_pdbqt = ["SER2:OXT"]
            atoms = sps.pose_atoms(ligand)
            with self.assertRaises(sps.ScoringError) as raised:
                sps.pose_box_containment(atoms, BOX_RECORD, recovery)
            self.assertIn("not evaluated against the box", str(raised.exception))


class SuperpositionTests(unittest.TestCase):
    """Kabsch superposition, checked against an independent implementation."""

    def three_sets(self):
        before = np.asarray(
            [positions()[s] for s in BACKBONE_SERIALS], dtype=float
        )
        return before

    def test_a_pure_translation_is_fitted_exactly(self):
        before = self.three_sets()
        after = before + np.asarray([1.5, -2.0, 0.75])
        fit = sps.kabsch_superposition(after, before)
        self.assertAlmostEqual(fit.rmsd, 0.0, places=9)
        self.assertAlmostEqual(fit.rotation_degrees, 0.0, places=6)
        self.assertAlmostEqual(
            fit.centroid_shift_angstrom, float(np.linalg.norm([1.5, -2.0, 0.75])),
            places=9,
        )

    def test_a_pure_rotation_is_fitted_exactly_and_its_angle_recovered(self):
        before = self.three_sets()
        centroid = before.mean(axis=0)
        rotation = rotation_matrix((0.2, -0.4, 0.9), 37.5)
        after = (before - centroid) @ rotation.T + centroid
        fit = sps.kabsch_superposition(after, before)
        self.assertAlmostEqual(fit.rmsd, 0.0, places=9)
        self.assertAlmostEqual(fit.rotation_degrees, 37.5, places=5)

    def test_the_result_agrees_with_gemmis_own_superposition(self):
        # An independent implementation, from a library this project already
        # depends on. If both agree on the residual for a deformed set, the
        # arithmetic here is not merely self-consistent.
        import gemmi

        before = self.three_sets()
        after = before.copy()
        after[3] += np.asarray([0.8, -0.3, 0.2])
        after[6] += np.asarray([-0.4, 0.5, -0.1])
        after = (after - after.mean(axis=0)) @ rotation_matrix(
            (0.1, 0.9, 0.3), 22.0
        ).T + after.mean(axis=0) + np.asarray([2.0, 1.0, -1.0])
        fit = sps.kabsch_superposition(after, before)
        result = gemmi.superpose_positions(
            [gemmi.Position(*row) for row in before],
            [gemmi.Position(*row) for row in after],
        )
        self.assertAlmostEqual(fit.rmsd, float(result.rmsd), places=6)

    def test_a_reflection_is_never_returned_as_a_fit(self):
        # A mirror image would superpose perfectly under an improper rotation.
        # Fitting one would make a D-peptide look like the L-peptide it came
        # from, so the determinant is corrected and the residual stays large.
        before = self.three_sets()
        after = before * np.asarray([1.0, 1.0, -1.0])
        fit = sps.kabsch_superposition(after, before)
        self.assertGreater(fit.rmsd, 0.1)
        self.assertAlmostEqual(float(np.linalg.det(fit.rotation)), 1.0, places=9)

    def test_two_atoms_cannot_define_a_fit(self):
        before = self.three_sets()[:2]
        with self.assertRaises(sps.ScoringError) as raised:
            sps.kabsch_superposition(before, before)
        self.assertIn("cannot define a rigid-body fit", str(raised.exception))

    def test_mismatched_shapes_are_refused(self):
        before = self.three_sets()
        with self.assertRaises(sps.ScoringError):
            sps.kabsch_superposition(before, before[:-1])

    def test_non_finite_coordinates_are_refused(self):
        before = self.three_sets().copy()
        after = before.copy()
        after[0, 0] = np.nan
        with self.assertRaises(sps.ScoringError):
            sps.kabsch_superposition(after, before)


class NearestSignatureTests(unittest.TestCase):
    """Two tolerances, and the band between them where the evidence still decides.

    The candidacy tolerance is generous so that the true pairing is never
    excluded; the distinctness tolerance is tight so that atoms are only
    paired on a difference well above coordinate noise. Between them lies the
    case that matters for the real target.
    """

    def correspondence(self, directory, overrides, translation=(0.4, 0.0, 0.0)):
        before = positions(overrides)
        ligand = write(
            directory, "l.pdbqt", ligand_pdbqt(coordinates=rounded(before))
        )
        pose = write(
            directory, "pose.pdbqt",
            ligand_pdbqt(
                coordinates=rounded(
                    rigidly_moved(before, translation=translation)
                ),
                index_map=False,
            ),
        )
        return sps.establish_correspondence(
            sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
            sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
        )

    def test_the_two_tolerances_are_separate_and_ordered(self):
        self.assertLess(
            sps.SIGNATURE_DISTINCTNESS_TOLERANCE_ANGSTROM,
            sps.RIGID_DISTANCE_TOLERANCE_ANGSTROM,
        )

    def test_the_real_carboxylate_geometry_is_resolved_on_the_evidence(self):
        # Measured, not invented: on the real aligned `spec_0574` pose the
        # C-terminal carbonyl oxygen sits 1.2211 A from its carbon and OXT is
        # constructed at 1.25 A. A single 0.05 A tolerance would have called
        # these two oxygens indistinguishable and failed closed on the real
        # file; 0.029 A is fifteen times the coordinate noise, so the
        # signatures do separate them.
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(Path(temporary), REAL_CARBOXYLATE)
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(found.pairs[8], 8)
            self.assertEqual(found.pairs[9], 9)
            resolved = {
                entry["starting_serial"]: entry
                for entry in found.resolved_by_nearest_signature
            }
            self.assertIn(8, resolved)
            self.assertLess(resolved[8]["signature_mismatch_angstrom"], 0.01)
            self.assertGreater(resolved[8]["next_best_mismatch_angstrom"], 0.02)

    def test_a_difference_too_small_to_pair_on_fails_closed(self):
        # The same two atoms 0.010 A apart. Nothing has gone wrong with the
        # file; the information simply is not there, and the module says so
        # instead of choosing.
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(Path(temporary), BORDERLINE_CARBOXYLATE)
            self.assertFalse(found.established)
            self.assertIn("cannot be told apart", " ".join(found.reasons))

    def test_a_borderline_pair_is_still_resolvable_by_the_stated_convention(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            before = positions(BORDERLINE_CARBOXYLATE)
            ligand = write(
                directory, "l.pdbqt", ligand_pdbqt(coordinates=rounded(before))
            )
            pose = write(
                directory, "pose.pdbqt",
                ligand_pdbqt(
                    coordinates=rounded(
                        rigidly_moved(before, translation=(0.4, 0, 0))
                    ),
                    index_map=False,
                ),
            )
            found = sps.establish_correspondence(
                sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
                sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
                resolve_symmetric_atoms=True,
            )
            self.assertTrue(found.established, found.reasons)
            self.assertEqual(len(found.symmetry_resolved), 1)

    def test_the_record_distinguishes_evidence_from_convention(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            on_evidence = self.correspondence(directory, REAL_CARBOXYLATE)
            self.assertEqual(on_evidence.symmetry_resolved, [])
            self.assertEqual(len(on_evidence.resolved_by_nearest_signature), 2)
            summary = on_evidence.as_dict()
            self.assertEqual(
                summary["signature_distinctness_tolerance_angstrom"],
                sps.SIGNATURE_DISTINCTNESS_TOLERANCE_ANGSTROM,
            )

    def test_a_perfectly_symmetric_pair_is_never_resolved_on_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            found = self.correspondence(Path(temporary), SYMMETRIC_CARBOXYLATE)
            self.assertFalse(found.established)
            self.assertEqual(found.resolved_by_nearest_signature, [])


class DisplacementTests(unittest.TestCase):
    """Rigid-body motion, internal deformation and side chains, kept apart."""

    def report(self, directory, after, resolve=False):
        ligand, record, pose = prepared(directory, after_coordinates=after)
        recovery = sps.load_identity_recovery(ligand, record)
        before_atoms = sps.pose_atoms(ligand)
        after_atoms = sps.pose_atoms(pose)
        found = sps.establish_correspondence(
            before_atoms, sps.parse_rigid_groups(ligand),
            after_atoms, sps.parse_rigid_groups(pose),
            resolve_symmetric_atoms=resolve,
        )
        return found, sps.displacement_report(
            before_atoms, after_atoms, found, recovery
        )

    def test_an_unmoved_pose_reports_zero_everywhere(self):
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(Path(temporary), positions())
            self.assertTrue(report["measured"])
            self.assertEqual(
                report["receptor_frame"]["all_heavy_atoms"]["maximum_angstrom"],
                0.0,
            )
            self.assertEqual(
                report["internal_backbone_deformation"][
                    "rmsd_after_superposition_angstrom"
                ],
                0.0,
            )
            self.assertEqual(report["rigid_body"]["centroid_shift_angstrom"], 0.0)

    def test_a_pure_translation_is_rigid_body_and_not_deformation(self):
        # Acceptance: a rigidly translated peptide must not be reported as
        # having deformed its backbone.
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(
                Path(temporary), rigidly_moved(positions(), translation=(2.5, 0, 0))
            )
            self.assertAlmostEqual(
                report["internal_backbone_deformation"][
                    "rmsd_after_superposition_angstrom"
                ],
                0.0, places=2,
            )
            self.assertAlmostEqual(
                report["rigid_body"]["centroid_shift_angstrom"], 2.5, places=2
            )
            self.assertAlmostEqual(
                report["rigid_body"]["rotation_degrees"], 0.0, places=1
            )
            # The receptor-frame figure still shows the 2.5 A that happened.
            self.assertAlmostEqual(
                report["receptor_frame"]["backbone"]["rmsd_angstrom"], 2.5,
                places=2,
            )

    def test_a_pure_rotation_is_rigid_body_and_not_deformation(self):
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(
                Path(temporary),
                rigidly_moved(positions(), degrees=25.0, axis=(0.3, 0.5, 0.81)),
            )
            self.assertAlmostEqual(
                report["internal_backbone_deformation"][
                    "rmsd_after_superposition_angstrom"
                ],
                0.0, places=2,
            )
            self.assertAlmostEqual(
                report["rigid_body"]["rotation_degrees"], 25.0, places=1
            )
            self.assertGreater(
                report["receptor_frame"]["backbone"]["rmsd_angstrom"], 0.5
            )

    def test_a_genuine_backbone_deformation_is_nonzero(self):
        # Rotating a backbone subtree about its rotatable bond is what a
        # torsion change does, and it is the one motion that reshapes the
        # predicted backbone. No rigid-body fit can absorb it.
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(
                Path(temporary), torsion_rotated(positions(), "root.1.1.1", 40.0)
            )
            self.assertGreater(
                report["internal_backbone_deformation"][
                    "rmsd_after_superposition_angstrom"
                ],
                0.1,
            )
            self.assertIsNotNone(
                report["internal_backbone_deformation"]["largest_residual_atom"]
            )

    def test_a_side_chain_change_leaves_the_backbone_geometry_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(
                Path(temporary), torsion_rotated(positions(), "root.1.1.2", 50.0)
            )
            self.assertAlmostEqual(
                report["internal_backbone_deformation"][
                    "rmsd_after_superposition_angstrom"
                ],
                0.0, places=3,
            )
            self.assertEqual(
                report["receptor_frame"]["backbone"]["maximum_angstrom"], 0.0
            )
            self.assertGreater(
                report["side_chain"]["receptor_frame"]["rmsd_angstrom"], 0.5
            )
            self.assertGreater(
                report["side_chain"]["after_backbone_superposition"][
                    "rmsd_angstrom"
                ],
                0.5,
            )

    def test_receptor_frame_and_aligned_side_chain_figures_are_both_reported(self):
        # Acceptance: the two are separate measurements. Under a whole-body
        # translation the receptor-frame figure is the translation and the
        # aligned figure is zero, which is the clearest case of why one
        # number could not serve for both.
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(
                Path(temporary), rigidly_moved(positions(), translation=(2.0, 0, 0))
            )
            side = report["side_chain"]
            self.assertAlmostEqual(
                side["receptor_frame"]["rmsd_angstrom"], 2.0, places=2
            )
            self.assertAlmostEqual(
                side["after_backbone_superposition"]["rmsd_angstrom"], 0.0,
                places=2,
            )
            self.assertIn("no superposition", side["receptor_frame"]["convention"])
            self.assertIn(
                "fitted on the backbone",
                side["after_backbone_superposition"]["convention"],
            )

    def test_the_backbone_split_uses_recovered_names_not_the_generic_ones(self):
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(
                Path(temporary), torsion_rotated(positions(), "root.1.1.2", 50.0)
            )
            self.assertEqual(report["receptor_frame"]["backbone"]["atoms"], 9)
            self.assertEqual(report["receptor_frame"]["side_chain"]["atoms"], 2)
            named = {
                entry["atom"] for entry in
                report["receptor_frame"]["largest_movements"]
            }
            self.assertIn("SER2:OG", named)
            self.assertNotIn("UNL1:O", named)

    def test_oxt_is_counted_as_backbone(self):
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(Path(temporary), positions())
            self.assertIn(
                "OXT", report["receptor_frame"]["backbone_atom_names"]
            )
            self.assertEqual(report["receptor_frame"]["backbone"]["atoms"], 9)

    def test_hydrogens_are_excluded_from_every_statistic(self):
        with tempfile.TemporaryDirectory() as temporary:
            moved = positions()
            moved[12] = moved[12] + np.asarray([0.0, 0.0, 0.0])
            _found, report = self.report(Path(temporary), moved)
            self.assertEqual(
                report["receptor_frame"]["all_heavy_atoms"]["atoms"], 11
            )
            self.assertIn("excluded", report["hydrogens"])

    def test_the_rmsd_is_a_root_mean_square_and_not_a_mean(self):
        # Two atoms of the side chain moved by 0 and 2 A: mean 1.0, RMSD
        # sqrt(2). The distinction matters because a few large movements are
        # what a relaxation has to be judged on.
        statistics = sps._statistics([0.0, 2.0])
        self.assertEqual(statistics["mean_angstrom"], 1.0)
        self.assertAlmostEqual(statistics["rmsd_angstrom"], 1.4142, places=3)

    def test_an_unestablished_correspondence_produces_no_measurement(self):
        # Fail closed: a record that says the comparison could not be made,
        # rather than a pass or an exception.
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(Path(temporary), positions(), resolve=False)
            self.assertTrue(report["measured"])
            moved = positions()
            moved[4] = moved[4] + np.asarray([0.6, 0.0, 0.0])
            _found2, broken = self.report(Path(temporary), moved)
            self.assertFalse(broken["measured"])
            self.assertIn("not established", broken["reason"])
            self.assertNotIn("receptor_frame", broken)

    def test_a_torsion_change_can_look_like_a_small_rigid_body_rotation(self):
        # Worth asserting because it is a real property of the decomposition
        # and not a defect: changing a torsion moves part of the backbone, so
        # the best rigid fit of the whole backbone is not the identity. The
        # internal residual is what distinguishes the two cases, and it is far
        # larger here than for any pure rigid motion.
        with tempfile.TemporaryDirectory() as temporary:
            _found, report = self.report(
                Path(temporary), torsion_rotated(positions(), "root.1.1.1", 40.0)
            )
            self.assertGreater(report["rigid_body"]["rotation_degrees"], 1.0)
            self.assertGreater(
                report["internal_backbone_deformation"][
                    "rmsd_after_superposition_angstrom"
                ],
                0.1,
            )


def receptor_atom_line(serial, name, residue_name, residue_number, element,
                       position, record="ATOM  "):
    """One PDB ATOM/HETATM record, with the element column Gemmi reads.

    Columns: record 1-6, serial 7-11, name 13-16, residue name 18-20, chain
    22, residue number 23-26, coordinates 31-54, occupancy 55-60, B 61-66,
    element 77-78.
    """
    name_field = name if len(name) == 4 else f" {name:<3}"
    x, y, z = (float(v) for v in position)
    return (
        f"{record}{serial:>5} {name_field}{'':1}{residue_name:>3} "
        f"A{residue_number:>4}{'':1}   "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}"
    )


def receptor_atoms():
    """Receptor atoms placed against the test peptide to create known clashes.

    Each is positioned relative to a named peptide atom, so the overlaps are
    deliberate: a histidine against the serine hydroxyl, a glycine alpha
    carbon against a backbone carbonyl oxygen, a glutamate against the
    constructed OXT, and the ANP gamma phosphate against the serine beta
    carbon. The ANP record is a HETATM with residue name ANP, which is how the
    prepared receptor stores it, under the same author chain as the protein.
    """
    p = positions()
    return [
        (1, "NE2", "HIS", 348, "N", p[11] + np.asarray([1.60, 0.0, 0.0]), "ATOM  "),
        (2, "CE1", "HIS", 348, "C", p[11] + np.asarray([2.70, 0.9, 0.0]), "ATOM  "),
        (3, "ND1", "HIS", 348, "N", p[11] + np.asarray([9.00, 0.0, 0.0]), "ATOM  "),
        (4, "CA", "GLY", 28, "C", p[4] + np.asarray([0.0, 2.60, 0.0]), "ATOM  "),
        (5, "N", "GLY", 28, "N", p[4] + np.asarray([0.0, 8.00, 0.0]), "ATOM  "),
        (6, "OE2", "GLU", 352, "O", p[9] + np.asarray([2.60, 0.0, 0.0]), "ATOM  "),
        (7, "PG", "ANP", 603, "P", p[10] + np.asarray([3.40, 0.0, 0.0]), "HETATM"),
        (8, "O2G", "ANP", 603, "O", p[10] + np.asarray([1.70, 0.0, 0.0]), "HETATM"),
        (9, "O1G", "ANP", 603, "O", p[10] + np.asarray([7.00, 0.0, 0.0]), "HETATM"),
    ]


def receptor_pdb(omit=()):
    """The prepared receptor structure the clash analysis measures against.

    `omit` drops atoms, which is how a structure that disagrees with the
    scored PDBQT is built: an atom the PDBQT has and this does not is a heavy
    atom the clash report cannot see.
    """
    lines = [
        receptor_atom_line(serial, name, residue, number, element, position, record)
        for serial, name, residue, number, element, position, record
        in receptor_atoms()
        if serial not in omit
    ]
    return "\n".join(lines) + "\nEND\n"


def receptor_pdbqt(shift=None, omit=(), hydrogen=True):
    """The same receptor as a Meeko-style PDBQT, with one polar hydrogen added.

    The added hydrogen is why the consistency check expects unpaired atoms:
    receptor preparation adds polar hydrogens that the source structure does
    not carry. `shift` displaces every atom, for the case where the two
    files are not the same model; `omit` drops atoms, for the case where the scored
    receptor is missing something the clash analysis measures; `hydrogen`
    turns off the added polar hydrogen.
    """
    offset = np.zeros(3) if shift is None else np.asarray(shift, dtype=float)
    types = {"N": "NA", "C": "C", "O": "OA", "P": "P"}
    lines = []
    for serial, name, residue, number, element, position, record in receptor_atoms():
        if serial in omit:
            continue
        lines.append(
            atom_line(
                serial, name, residue, number, "A", types[element],
                np.asarray(position, dtype=float) + offset,
            ).replace("ATOM  ", record, 1)
            if record == "HETATM"
            else atom_line(
                serial, name, residue, number, "A", types[element],
                np.asarray(position, dtype=float) + offset,
            )
        )
    if hydrogen:
        p = positions()
        lines.append(
            atom_line(
                10, "HE2", "HIS", 348, "A", "HD",
                p[11] + np.asarray([2.20, 0.0, 0.0]) + offset,
            )
        )
    return "\n".join(lines) + "\n"


class ClashComparisonTests(unittest.TestCase):
    """Protein and cofactor kept separate, with the project's overlap definition."""

    def setup_files(self, directory):
        ligand, record, _ = prepared(directory)
        recovery = sps.load_identity_recovery(ligand, record)
        atoms = sps.pose_atoms(ligand)
        serials, _ = sps.heavy_atom_serials(atoms, recovery)
        structure = write(directory, "receptor.pdb", receptor_pdb())
        protein, cofactor, partition = sps.receptor_site_sets(structure, "ANP")
        return atoms, serials, recovery, protein, cofactor, partition

    def report(self, directory, moved=None):
        atoms, serials, recovery, protein, cofactor, _ = self.setup_files(directory)
        coordinates = None
        if moved is not None:
            coordinates = {
                serial: np.asarray(atoms[serial]["position"], dtype=float)
                + np.asarray(moved, dtype=float)
                for serial in serials
            }
        sites = sps.peptide_atom_sites(
            atoms, serials, recovery, "receptor frame", coordinates
        )
        return sps.clash_report_for_pose(
            sites, protein, cofactor, "ANP", "test pose"
        )

    def test_the_receptor_partition_separates_protein_from_the_cofactor(self):
        with tempfile.TemporaryDirectory() as temporary:
            _a, _s, _r, protein, cofactor, partition = self.setup_files(
                Path(temporary)
            )
            self.assertEqual(len(protein), 6)
            self.assertEqual(len(cofactor), 3)
            self.assertEqual(partition["non_amino_acid_components"], {"ANP": 3})
            self.assertEqual(partition["author_chains"], ["A"])

    def test_protein_and_cofactor_clashes_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(Path(temporary))
            self.assertEqual(report["protein"]["target"], "MurC protein")
            self.assertEqual(report["cofactor"]["target"], "ANP cofactor")
            self.assertGreater(report["protein"]["clash_count"], 0)
            self.assertGreater(report["cofactor"]["clash_count"], 0)
            self.assertEqual(
                report["summary"]["total_clashes"],
                report["protein"]["clash_count"]
                + report["cofactor"]["clash_count"],
            )
            self.assertEqual(
                sorted(report["summary"]["clashes_by_target"]),
                ["ANP cofactor", "MurC protein"],
            )

    def test_clashing_atoms_are_named_by_their_recovered_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(Path(temporary))
            labels = {
                pair["first"]["label"]
                for pair in report["protein"]["clashing_pairs"]
            }
            self.assertTrue(
                any(label.endswith("/OG") for label in labels), labels
            )
            self.assertFalse(
                any("UNL" in label for label in labels), labels
            )

    def test_a_backbone_clash_is_distinguished_from_side_chain_clashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(Path(temporary))
            self.assertGreaterEqual(report["summary"]["backbone_clashes"], 1)
            self.assertGreaterEqual(report["summary"]["side_chain_clashes"], 1)
            self.assertFalse(report["summary"]["side_chain_only"])

    def test_clashes_involving_the_constructed_oxt_are_accounted_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(Path(temporary))
            accounting = report["constructed_oxt_accounting"]
            self.assertEqual(accounting["atom"], "OXT")
            self.assertGreaterEqual(accounting["clashes_involving_it"], 1)
            self.assertEqual(
                accounting["total_excluding_it"],
                accounting["total_clashes"] - accounting["clashes_involving_it"],
            )
            self.assertIn("before OXT existed", accounting["why"])

    def test_the_thresholds_are_recorded_with_their_inequalities(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report(Path(temporary))
            thresholds = report["thresholds"]
            self.assertEqual(thresholds["clash_overlap_tolerance_angstrom"], 0.4)
            self.assertEqual(thresholds["contact_cutoff_angstrom"], 4.0)
            self.assertIn("strictly", thresholds["contact_inequality"])
            self.assertIn("r_i + r_j - d", thresholds["overlap_definition"])
            self.assertIn("Bondi", thresholds["radii_source"])

    def test_moving_the_peptide_away_removes_the_clashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            initial = self.report(directory)
            relaxed = self.report(directory, moved=(0.0, 0.0, 9.0))
            comparison = sps.clash_comparison(initial, relaxed)
            self.assertTrue(comparison["definitions_identical"])
            self.assertEqual(comparison["counts_optimised"]["total"], 0)
            self.assertLess(comparison["change"]["total"], 0)
            self.assertIn("does not establish", comparison["interpretation"])

    def test_a_comparison_without_an_optimised_pose_says_so(self):
        with tempfile.TemporaryDirectory() as temporary:
            comparison = sps.clash_comparison(self.report(Path(temporary)), None)
            self.assertIsNone(comparison["change"])
            self.assertIn("no second pose", comparison["note"])

    def test_two_reports_with_different_thresholds_are_not_differenced(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            atoms, serials, recovery, protein, cofactor, _ = self.setup_files(
                directory
            )
            sites = sps.peptide_atom_sites(atoms, serials, recovery, "frame")
            strict = sps.clash_report_for_pose(
                sites, protein, cofactor, "ANP", "a", 4.0, 0.4
            )
            loose = sps.clash_report_for_pose(
                sites, protein, cofactor, "ANP", "b", 4.0, 0.8
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.clash_comparison(strict, loose)
            self.assertIn("different thresholds", str(raised.exception))

    def test_an_absent_cofactor_is_undefined_rather_than_clash_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            _protein, cofactor, partition = sps.receptor_site_sets(
                structure, "UMA"
            )
            self.assertEqual(cofactor, [])
            self.assertIn("undefined", partition["component_warning"])

    def test_an_untabulated_element_is_fatal_rather_than_given_a_radius(self):
        # The project's live case: the manganese ions it removes have no
        # settled single radius, and a guess would enter every overlap.
        with self.assertRaises(peptide_contacts.ContactError):
            peptide_contacts.radius_for_element("MN")

    def test_an_unknown_autodock_type_has_no_element_and_is_fatal(self):
        with self.assertRaises(sps.ScoringError) as raised:
            sps.element_for_atom({"atom_type": "Mn"}, None)
        self.assertIn("element table", str(raised.exception))

    def test_the_element_comes_from_the_recovered_identity_when_available(self):
        identity = vpp.AtomIdentity(
            residue_name="SER", residue_seqid=2, atom_name="OG", element="O"
        )
        self.assertEqual(
            sps.element_for_atom({"atom_type": "OA"}, identity), "O"
        )
        self.assertEqual(sps.element_for_atom({"atom_type": "A"}, None), "C")


class ReceptorConsistencyTests(unittest.TestCase):
    def test_the_same_model_in_both_files_is_consistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            pdbqt = write(directory, "receptor.pdbqt", receptor_pdbqt())
            report = sps.receptor_model_consistency(pdbqt, structure)
            self.assertTrue(report["checked"])
            self.assertTrue(report["coordinates_consistent"])
            self.assertEqual(report["paired_atoms"], 9)
            self.assertEqual(report["only_in_pdbqt"], 1)

    def test_a_displaced_receptor_pdbqt_is_reported_as_inconsistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            pdbqt = write(
                directory, "receptor.pdbqt", receptor_pdbqt(shift=(0.5, 0, 0))
            )
            report = sps.receptor_model_consistency(pdbqt, structure)
            self.assertFalse(report["coordinates_consistent"])
            self.assertGreater(report["max_deviation_angstrom"], 0.4)
def receptor_manifest(structure, pdbqt, retained=("ANP",), condition="anp-present"):
    """A receptor manifest as `prepare_receptor.py` writes one.

    Only the fields this module reads are given real values: the schema, the
    SHA-256 of both outputs, and the retained components. The remaining
    required fields are present because `read_manifest` insists on them, which
    is itself the check that a manifest of another kind is refused.
    """
    return json.dumps(
        {
            "schema": "mur-ligase-receptor/1",
            "receptor_condition": condition,
            "selections": {"author_chain": "A"},
            "source_structure": {"path": "data/1P3D.cif"},
            "outputs": {
                "prepared_pdb": str(structure),
                "prepared_pdb_sha256": sps.file_sha256(structure),
                "receptor_pdbqt": str(pdbqt),
                "receptor_pdbqt_sha256": sps.file_sha256(pdbqt),
            },
            "checks": {"pocket_before": {"n_found": 32}},
            "retained_components": {
                "by_component": {name: 1 for name in retained},
                "expected": list(retained),
                "in_receptor_pdbqt": list(retained),
            },
        },
        indent=2,
    )


class ReceptorPrerequisiteTests(unittest.TestCase):
    """A clash comparison against a receptor that is not the scored one is refused.

    The score comes from the receptor PDBQT and the clash distances from the
    prepared structure. Where those are not the same model, nothing in the
    clash report shows it -- the report measures the structure, and comes back
    looking exactly as it would if all were well. These tests show that, and
    then show the check that stops it being reported.
    """

    def consistency(self, directory, **kwargs):
        structure = write(
            directory, "receptor.pdb", receptor_pdb(omit=kwargs.pop("omit_pdb", ()))
        )
        pdbqt = write(directory, "receptor.pdbqt", receptor_pdbqt(**kwargs))
        return sps.receptor_model_consistency(pdbqt, structure), structure

    def partition(self, structure, component="ANP"):
        _protein, _cofactor, report = sps.receptor_site_sets(structure, component)
        return report

    def blockers(self, directory, component="ANP", expected=None, **kwargs):
        consistency, structure = self.consistency(directory, **kwargs)
        return sps.receptor_clash_prerequisites(
            consistency, self.partition(structure, component), component, expected
        )

    def test_the_matching_pair_has_no_blockers(self):
        # The fixture receptor holds three ANP atoms, stated explicitly: the
        # count is required evidence, not an optional extra.
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(self.blockers(Path(temporary), expected=3), [])

    def test_added_polar_hydrogens_remain_permissible(self):
        # The receptor PDBQT carries a polar hydrogen the structure does not.
        # That is what receptor preparation does, and it must not block.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            consistency, structure = self.consistency(directory)
            self.assertEqual(consistency["only_in_pdbqt_hydrogens"], 1)
            self.assertEqual(consistency["only_in_pdbqt_non_hydrogen"], [])
            self.assertEqual(
                sps.receptor_clash_prerequisites(
                    consistency, self.partition(structure), "ANP", 3
                ),
                [],
            )

    def test_a_displaced_receptor_blocks_the_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            blockers = self.blockers(
                Path(temporary), expected=3, shift=(0.5, 0, 0)
            )
            self.assertEqual(len(blockers), 1)
            self.assertIn("disagree on the coordinates", blockers[0])

    def test_nothing_in_the_clash_report_reveals_a_displaced_receptor(self):
        # The previously unsafe behaviour. The structure is untouched, so the
        # clash report is bit-for-bit the one the consistent pair produces,
        # although the receptor Vina scored has moved half an angstrom. The
        # report cannot be the thing that catches this.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record, _ = prepared(directory)
            recovery = sps.load_identity_recovery(ligand, record)
            atoms = sps.pose_atoms(ligand)
            serials, _ = sps.heavy_atom_serials(atoms, recovery)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            protein, cofactor, _partition = sps.receptor_site_sets(structure, "ANP")
            sites = sps.peptide_atom_sites(
                atoms, serials, recovery, "receptor frame"
            )
            report = sps.clash_report_for_pose(
                sites, protein, cofactor, "ANP", "test pose"
            )
            self.assertGreater(report["summary"]["total_clashes"], 0)
            displaced = write(
                directory, "receptor.pdbqt", receptor_pdbqt(shift=(0.5, 0, 0))
            )
            consistency = sps.receptor_model_consistency(displaced, structure)
            self.assertFalse(consistency["coordinates_consistent"])
            # Same numbers, different receptor. Only the consistency check
            # knows, which is why it is a blocker and not a limitation.
            again = sps.clash_report_for_pose(
                sites, protein, cofactor, "ANP", "test pose"
            )
            self.assertEqual(
                again["summary"]["total_clashes"],
                report["summary"]["total_clashes"],
            )

    def test_a_receptor_pdbqt_missing_a_heavy_atom_blocks_the_comparison(self):
        # The scored receptor is missing an atom the clash report measures, so
        # the report counts overlaps against an atom Vina never saw.
        with tempfile.TemporaryDirectory() as temporary:
            blockers = self.blockers(Path(temporary), omit=(1,))
            self.assertTrue(
                any("absent from the scored receptor" in b for b in blockers),
                blockers,
            )
            self.assertIn("HIS", " ".join(blockers))

    def test_an_extra_heavy_atom_in_the_receptor_pdbqt_blocks_the_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            blockers = self.blockers(Path(temporary), omit_pdb=(6,))
            self.assertTrue(
                any("non-hydrogen atom" in b for b in blockers), blockers
            )
            self.assertIn("GLU", " ".join(blockers))

    def test_a_receptor_without_the_cofactor_blocks_the_comparison(self):
        # The structure has no ANP while the scored receptor does, so a
        # cofactor clash count of zero would describe nothing. Both failures
        # are caught: the absent component and the atoms only in the PDBQT.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            blockers = self.blockers(directory, omit_pdb=(7, 8, 9))
            joined = " ".join(blockers)
            self.assertIn("no ANP atoms", joined)
            self.assertIn("undefined rather than zero", joined)
            self.assertIn("non-hydrogen atom", joined)

    def test_a_cofactor_of_the_wrong_size_blocks_the_comparison(self):
        # The real receptor holds 31 ANP atoms. A receptor holding a different
        # number is not the cofactor model the comparison was set up for.
        with tempfile.TemporaryDirectory() as temporary:
            blockers = self.blockers(Path(temporary), expected=31)
            self.assertEqual(len(blockers), 1)
            self.assertIn(
                "3 ANP atoms, not the 31 stated with "
                "--expected-component-atoms",
                blockers[0],
            )

    def test_the_expected_count_agreeing_does_not_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(self.blockers(Path(temporary), expected=3), [])

    def test_omitting_the_cofactor_expectation_blocks_the_comparison(self):
        # Neither the count nor the manifest: the cofactor's completeness is
        # then not established by anything, and a count of three atoms is
        # just a count.
        with tempfile.TemporaryDirectory() as temporary:
            blockers = self.blockers(Path(temporary))
            self.assertEqual(len(blockers), 1)
            self.assertIn("neither --expected-component-atoms nor", blockers[0])
            self.assertIn("31 heavy atoms", blockers[0])

    def test_a_consistently_truncated_cofactor_cannot_pass_silently(self):
        # The case the expectation exists for. One ANP atom is missing from
        # BOTH receptor files, so the two agree with each other perfectly:
        # every atom pairs, nothing is missing on either side, the coordinates
        # match. Nothing that compares the two files can see it.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb(omit=(9,)))
            pdbqt = write(
                directory, "receptor.pdbqt", receptor_pdbqt(omit=(9,))
            )
            consistency = sps.receptor_model_consistency(pdbqt, structure)
            self.assertTrue(consistency["coordinates_consistent"])
            self.assertEqual(consistency["only_in_structure"], 0)
            self.assertEqual(consistency["only_in_pdbqt_non_hydrogen"], [])
            partition = self.partition(structure)
            self.assertEqual(partition["component_atoms"], 2)
            # Stating the count catches it.
            blockers = sps.receptor_clash_prerequisites(
                consistency, partition, "ANP", 3
            )
            self.assertEqual(len(blockers), 1)
            self.assertIn("holds 2 ANP atoms, not the 3", blockers[0])
            # And omitting the count does not let it through either.
            self.assertTrue(
                sps.receptor_clash_prerequisites(consistency, partition, "ANP")
            )

    def test_a_truncated_cofactor_also_fails_the_manifest_digests(self):
        # The other way of establishing the cofactor. Truncating ANP changes
        # the bytes of both receptor files, so the manifest written for the
        # untruncated pair no longer describes them.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            pdbqt = write(directory, "receptor.pdbqt", receptor_pdbqt())
            manifest = write(
                directory, "receptor.manifest.json",
                receptor_manifest(structure, pdbqt),
            )
            full = sps.cofactor_provenance(
                "ANP", self.partition(structure), None, manifest, pdbqt,
                structure,
            )
            self.assertTrue(full["established"], full["reasons"])
            self.assertIn("validated receptor bytes", full["basis"])
            # Now truncate both files, leaving the manifest as it was.
            structure.write_text(receptor_pdb(omit=(9,)))
            pdbqt.write_text(receptor_pdbqt(omit=(9,)))
            truncated = sps.cofactor_provenance(
                "ANP", self.partition(structure), None, manifest, pdbqt,
                structure,
            )
            self.assertFalse(truncated["established"])
            joined = " ".join(truncated["reasons"])
            self.assertIn("receptor_pdbqt_sha256", joined)
            self.assertIn("prepared_pdb_sha256", joined)

    def test_a_manifest_not_naming_the_cofactor_does_not_establish_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            pdbqt = write(directory, "receptor.pdbqt", receptor_pdbqt())
            manifest = write(
                directory, "receptor.manifest.json",
                receptor_manifest(structure, pdbqt, retained=[]),
            )
            found = sps.cofactor_provenance(
                "ANP", self.partition(structure), None, manifest, pdbqt,
                structure,
            )
            self.assertFalse(found["established"])
            self.assertIn(
                "does not record ANP as retained", " ".join(found["reasons"])
            )

    def test_a_manifest_of_the_wrong_kind_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            pdbqt = write(directory, "receptor.pdbqt", receptor_pdbqt())
            manifest = write(
                directory, "receptor.manifest.json",
                json.dumps({"schema": "something-else/1"}),
            )
            found = sps.cofactor_provenance(
                "ANP", self.partition(structure), None, manifest, pdbqt,
                structure,
            )
            self.assertFalse(found["established"])
            self.assertIn("not usable", " ".join(found["reasons"]))

    def test_the_stated_count_and_the_manifest_agree_on_the_real_convention(self):
        # Both routes are available and either suffices; neither is implied by
        # the other.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            pdbqt = write(directory, "receptor.pdbqt", receptor_pdbqt())
            manifest = write(
                directory, "receptor.manifest.json",
                receptor_manifest(structure, pdbqt),
            )
            partition = self.partition(structure)
            by_count = sps.cofactor_provenance("ANP", partition, 3)
            by_manifest = sps.cofactor_provenance(
                "ANP", partition, None, manifest, pdbqt, structure
            )
            self.assertTrue(by_count["established"])
            self.assertTrue(by_manifest["established"])
            self.assertIn("--expected-component-atoms", by_count["basis"])
            self.assertIn("manifest", by_manifest["basis"])

    def test_an_uncheckable_receptor_pair_blocks_the_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(directory, "receptor.pdb", receptor_pdb())
            missing = directory / "absent.pdbqt"
            consistency = sps.receptor_model_consistency(missing, structure)
            self.assertFalse(consistency["checked"])
            blockers = sps.receptor_clash_prerequisites(
                consistency, self.partition(structure), "ANP"
            )
            self.assertEqual(len(blockers), 1)
            self.assertIn("not established", blockers[0])

    def test_no_check_at_all_blocks_the_comparison(self):
        self.assertTrue(sps.receptor_clash_prerequisites(None, None, "ANP"))

    def test_a_protein_free_structure_blocks_the_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            structure = write(
                directory, "receptor.pdb", receptor_pdb(omit=(1, 2, 3, 4, 5, 6))
            )
            pdbqt = write(
                directory, "receptor.pdbqt",
                receptor_pdbqt(omit=(1, 2, 3, 4, 5, 6), hydrogen=False),
            )
            consistency = sps.receptor_model_consistency(pdbqt, structure)
            blockers = sps.receptor_clash_prerequisites(
                consistency, self.partition(structure), "ANP"
            )
            self.assertTrue(
                any("no protein atoms" in b for b in blockers), blockers
            )


class RecordingEngine:
    """A stub standing in for `vina.Vina`, which records what was called.

    **It computes nothing.** The numbers it returns are fixed constants, not
    scores of anything, and they exist only so the surrounding bookkeeping can
    be exercised where Vina is not installed. `dock` raises, because a search
    must never be substituted for fixed-pose scoring, and a test that merely
    checked `dock` was not in a call list could pass while it was.
    """

    def __init__(self):
        self.calls = []

    def score(self):
        """A stub, not Vina. Returns fixed numbers that are not scores."""
        self.calls.append("score")
        return [-5.0, -6.0, 1.0]

    def optimize(self, max_steps=None):
        self.calls.append(("optimize", max_steps))
        return [-7.0, -8.0, 1.0]

    def write_pose(self, path, overwrite=False):
        self.calls.append(("write_pose", overwrite))
        Path(path).write_text(ligand_pdbqt(index_map=False))

    def dock(self, *args, **kwargs):
        raise AssertionError(
            "dock() was called: a conformational search must never stand in "
            "for fixed-pose scoring"
        )


class ScoringCallTests(unittest.TestCase):
    """`score()` and `optimize()` are different measurements, kept apart."""

    def test_fixed_pose_scoring_calls_score_and_nothing_else(self):
        engine = RecordingEngine()
        record = sps.score_fixed_pose(engine)
        self.assertEqual(engine.calls, ["score"])
        self.assertEqual(record["vina_call"], "Vina.score()")
        self.assertFalse(record["search_performed"])
        self.assertFalse(record["optimisation_performed"])
        self.assertEqual(record["raw_components"], [-5.0, -6.0, 1.0])
        self.assertIn("no atom was moved", record["pose"])

    def test_local_optimisation_calls_optimize_then_writes_the_pose(self):
        engine = RecordingEngine()
        with tempfile.TemporaryDirectory() as temporary:
            pose = Path(temporary) / "relaxed.pdbqt"
            record = sps.locally_optimise(engine, pose, max_steps=50)
            self.assertEqual(
                engine.calls, [("optimize", 50), ("write_pose", False)]
            )
            self.assertTrue(pose.is_file())
            self.assertEqual(record["vina_call"], "Vina.optimize()")
            self.assertTrue(record["optimisation_performed"])
            self.assertFalse(record["search_performed"])
            self.assertEqual(len(record["pose_file_sha256"]), 64)
            self.assertEqual(record["max_steps"], 50)

    def test_vinas_own_default_step_count_is_used_when_none_is_given(self):
        engine = RecordingEngine()
        with tempfile.TemporaryDirectory() as temporary:
            sps.locally_optimise(engine, Path(temporary) / "p.pdbqt")
            self.assertEqual(engine.calls[0], ("optimize", None))

    def test_no_code_path_in_this_module_calls_dock(self):
        # The guard that matters, because it covers paths no test exercises:
        # the module's own source must not contain a dock() call at all.
        source = Path(sps.__file__).read_text()
        self.assertNotIn(".dock(", source)
        self.assertNotIn("engine.dock", source)
        self.assertIn(
            "dock() is never called", source
        )

    def test_the_two_results_never_share_a_field(self):
        engine = RecordingEngine()
        initial = sps.score_fixed_pose(engine)
        with tempfile.TemporaryDirectory() as temporary:
            optimised = sps.locally_optimise(
                engine, Path(temporary) / "p.pdbqt"
            )
        self.assertNotEqual(
            initial["total_kcal_per_mol"], optimised["total_kcal_per_mol"]
        )
        self.assertEqual(initial["vina_call"], "Vina.score()")
        self.assertEqual(optimised["vina_call"], "Vina.optimize()")


class OutputPlanningTests(unittest.TestCase):
    """Overwrite protection, and a fixed-pose run that needs no pose file."""

    def test_a_fixed_pose_run_plans_no_optimised_pose_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            planned = sps.plan_outputs(Path(temporary), "spec_0574", False)
            self.assertIsNone(planned["optimised_pose"])
            self.assertEqual(planned["record"].name, "spec_0574_score.json")

    def test_an_existing_optimised_pose_does_not_block_a_fixed_pose_run(self):
        # Acceptance: the fixed-pose path must not require, or be blocked by,
        # an optimised-pose file. The earlier version reserved that filename
        # on every run, so a second fixed-pose score in the same directory
        # failed for a file it was never going to write.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "spec_0574_optimised.pdbqt").write_text("stale\n")
            planned = sps.plan_outputs(directory, "spec_0574", False)
            self.assertIsNone(planned["optimised_pose"])

    def test_an_existing_record_is_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "spec_0574_score.json").write_text("{}\n")
            with self.assertRaises(sps.ScoringError) as raised:
                sps.plan_outputs(directory, "spec_0574", False)
            self.assertIn("Refusing to overwrite", str(raised.exception))

    def test_overwrite_is_allowed_when_asked_for_deliberately(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "d_score.json").write_text("{}\n")
            planned = sps.plan_outputs(directory, "d", False, overwrite=True)
            self.assertEqual(planned["record"].name, "d_score.json")

    def test_an_optimisation_run_reserves_both_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "d_optimised.pdbqt").write_text("stale\n")
            with self.assertRaises(sps.ScoringError) as raised:
                sps.plan_outputs(directory, "d", True)
            self.assertIn("d_optimised.pdbqt", str(raised.exception))


class VerificationRecordTests(unittest.TestCase):
    def test_a_verified_record_is_summarised_into_the_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "check.json", verification_record())
            summary = sps.read_verification_record(path)
            self.assertTrue(summary["verified"])
            self.assertEqual(summary["torsdof"], TORSDOF)
            self.assertEqual(len(summary["sha256"]), 64)
            self.assertEqual(summary["backbone_phi_psi_count"], 2)

    def test_an_unverified_ligand_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "check.json", verification_record(verified=False)
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.read_verification_record(path)
            self.assertIn("unverified", str(raised.exception))

    def test_a_record_of_the_wrong_kind_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(
                Path(temporary), "check.json",
                verification_record(schema="mur-ligase-peptide-score/2"),
            )
            with self.assertRaises(sps.ScoringError) as raised:
                sps.read_verification_record(path)
            self.assertIn("not a peptide PDBQT verification record",
                          str(raised.exception))

class VerificationBindingTests(unittest.TestCase):
    """A verified record is evidence about one file, and only about that file."""

    def files(self, directory, record_text=None, coordinates=None):
        ligand = write(
            directory, "ligand.pdbqt",
            ligand_pdbqt(coordinates=coordinates) if coordinates
            else ligand_pdbqt(),
        )
        ligand_record_path = write(directory, "ligand.json", ligand_record())
        check = write(
            directory, "check.json", record_text or verification_record()
        )
        return ligand, ligand_record_path, check

    def read(self, directory, record_text=None, design_id="spec_test",
             coordinates=None):
        ligand, ligand_record_path, check = self.files(
            directory, record_text, coordinates
        )
        return sps.read_verification_record(
            check, ligand, ligand_record_path, design_id
        )

    def test_the_record_for_this_file_binds_and_says_how(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.read(Path(temporary))
            binding = summary["bound_to_scored_input"]
            self.assertTrue(binding["bound"])
            self.assertTrue(all(check["agrees"] for check in binding["checks"]))
            self.assertEqual(
                {check["quantity"] for check in binding["checks"]},
                {
                    "atoms_in_pdbqt", "heavy_atoms_in_pdbqt",
                    "atom_type_census", "torsdof_declared",
                    "torsdof_declaration", "branch_records",
                },
            )

    def test_a_record_for_a_ligand_of_another_size_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sps.ScoringError) as raised:
                self.read(
                    Path(temporary),
                    verification_record(overrides={"atoms_in_pdbqt": 56}),
                )
            self.assertIn("atoms_in_pdbqt", str(raised.exception))
            self.assertIn("does not belong to the file", str(raised.exception))

    def test_the_record_for_the_other_torsion_state_is_refused(self):
        # The two variants of one design differ in nothing but their torsion
        # tree, which is exactly what a score of the wrong one would hide: the
        # flexible ligand declares TORSDOF 30 over 30 BRANCH records, the
        # backbone-rigid one 14 over 14.
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sps.ScoringError) as raised:
                self.read(
                    Path(temporary),
                    verification_record(
                        overrides={
                            "torsdof": {
                                "declared": 30,
                                "declaration": "present",
                                "branch_records": 30,
                            }
                        }
                    ),
                )
            message = str(raised.exception)
            self.assertIn("torsdof_declared", message)
            self.assertIn("branch_records", message)

    def test_a_record_of_a_different_composition_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sps.ScoringError) as raised:
                self.read(
                    Path(temporary),
                    verification_record(
                        overrides={"atom_type_census": {"C": 5, "N": 3, "OA": 3, "HD": 1}}
                    ),
                )
            self.assertIn("atom_type_census", str(raised.exception))

    def test_a_record_naming_another_pdbqt_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sps.ScoringError) as raised:
                self.read(
                    Path(temporary),
                    verification_record(pdbqt="spec_0574_flexible.pdbqt"),
                )
            self.assertIn("spec_0574_flexible.pdbqt", str(raised.exception))

    def test_a_record_for_another_design_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sps.ScoringError) as raised:
                self.read(
                    Path(temporary), verification_record(design_id="spec_0001")
                )
            self.assertIn("spec_0001", str(raised.exception))

    def test_a_record_made_without_the_ligand_record_is_refused(self):
        # Without it stage 4 had no chemical identities, so whatever it checked
        # was not what this module relies on.
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sps.ScoringError) as raised:
                self.read(
                    Path(temporary),
                    verification_record(overrides={"ligand_record": None}),
                )
            self.assertIn("without a ligand record", str(raised.exception))

    def test_a_record_lacking_a_bindable_quantity_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(sps.ScoringError) as raised:
                self.read(
                    Path(temporary),
                    verification_record(overrides={"atom_type_census": None}),
                )
            self.assertIn("does not state atom_type_census", str(raised.exception))

    def test_a_directory_difference_is_recorded_and_not_refused(self):
        # The same record is read where it was written and where the scoring
        # runs, so the paths differ legitimately. The file names must still be
        # the same file names.
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.read(
                Path(temporary),
                verification_record(pdbqt="/epicac/ligands/ligand.pdbqt"),
            )
            binding = summary["bound_to_scored_input"]
            self.assertTrue(binding["bound"])
            self.assertFalse(binding["paths_identical"])

    def test_a_record_read_without_the_scored_file_is_not_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write(Path(temporary), "check.json", verification_record())
            summary = sps.read_verification_record(path)
            self.assertFalse(summary["bound_to_scored_input"]["bound"])
            self.assertIn(
                "not established",
                summary["bound_to_scored_input"]["reason"],
            )

    def test_the_checksum_gap_is_stated_rather_than_implied(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.read(Path(temporary))
            limitation = summary["bound_to_scored_input"]["limitation"]
            self.assertIn("no checksum", limitation)
            self.assertIn("--predicted-structure", limitation)

    def test_moved_coordinates_pass_the_binding_but_fail_the_recheck(self):
        # The gap the binding cannot close, and the check that closes it. The
        # census, counts and torsion state are untouched, so the record still
        # binds; the atom has moved 0.5 A, and comparing against the structure
        # stage 3 wrote finds it.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            moved = positions()
            moved[11] = moved[11] + np.asarray([0.5, 0.0, 0.0])
            summary = self.read(directory, coordinates=rounded(moved))
            self.assertTrue(summary["bound_to_scored_input"]["bound"])
            source = write(directory, "source.pdb", peptide_source_pdb())
            ligand = directory / "ligand.pdbqt"
            comparison = sps.recheck_coordinate_preservation(
                sps.pose_atoms(ligand),
                sps.load_identity_recovery(ligand, directory / "ligand.json"),
                source,
            )
            self.assertFalse(comparison["coordinates_preserved"])
            self.assertAlmostEqual(
                comparison["max_deviation_angstrom"], 0.5, places=3
            )
            self.assertEqual(comparison["max_deviation_atom"], "SER2:OG")

    def test_the_unmoved_file_passes_the_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, ligand_record_path, _check = self.files(directory)
            source = write(directory, "source.pdb", peptide_source_pdb())
            comparison = sps.recheck_coordinate_preservation(
                sps.pose_atoms(ligand),
                sps.load_identity_recovery(ligand, ligand_record_path),
                source,
            )
            self.assertTrue(comparison["coordinates_preserved"])
            self.assertEqual(comparison["max_deviation_angstrom"], 0.0)
            self.assertEqual(comparison["heavy_atoms_compared"], 11)


class ValidatedWorkflowTests(unittest.TestCase):
    """A score without the evidence must not look like a score with it."""

    def bound_verification(self, directory):
        ligand, ligand_record_path, _ = prepared(directory)
        check = write(directory, "check.json", verification_record())
        return sps.read_verification_record(
            check, ligand, ligand_record_path, "spec_test"
        )

    def preserved_recheck(self, directory):
        ligand, ligand_record_path, _ = prepared(directory)
        source = write(directory, "source.pdb", peptide_source_pdb())
        return sps.recheck_coordinate_preservation(
            sps.pose_atoms(ligand),
            sps.load_identity_recovery(ligand, ligand_record_path),
            source,
        )

    def test_both_arguments_are_required(self):
        for verification, predicted, expected in (
            (None, None, ["--verification-record", "--predicted-structure"]),
            (Path("c.json"), None, ["--predicted-structure"]),
            (None, Path("s.pdb"), ["--verification-record"]),
        ):
            refusal = sps.validated_workflow_requirements(
                verification, predicted, False
            )
            self.assertIsNotNone(refusal)
            for fragment in expected:
                self.assertIn(fragment, refusal)

    def test_both_arguments_given_is_not_refused(self):
        self.assertIsNone(
            sps.validated_workflow_requirements(
                Path("c.json"), Path("s.pdb"), False
            )
        )

    def test_the_opt_in_flag_waives_the_requirement(self):
        self.assertIsNone(
            sps.validated_workflow_requirements(None, None, True)
        )

    def test_the_refusal_names_the_opt_in_rather_than_dead_ending(self):
        refusal = sps.validated_workflow_requirements(None, None, False)
        self.assertIn("--unvalidated-exploratory", refusal)

    def test_both_pieces_of_evidence_make_a_validated_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            status = sps.validation_status(
                self.bound_verification(directory),
                self.preserved_recheck(directory),
            )
            self.assertTrue(status["validated"])
            self.assertTrue(status["label"].startswith("VALIDATED"))
            self.assertEqual(status["missing"], [])
            self.assertTrue(
                status["evidence"]["stage_4_verification_bound_to_this_file"]
            )
            self.assertTrue(
                status["evidence"]["coordinates_rechecked_against_prediction"]
            )

    def test_a_missing_verification_record_is_unvalidated(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = sps.validation_status(
                None, self.preserved_recheck(Path(temporary))
            )
            self.assertFalse(status["validated"])
            self.assertEqual(status["label"], "UNVALIDATED EXPLORATORY SCORE")
            self.assertIn("no stage-4", " ".join(status["missing"]))

    def test_a_missing_predicted_structure_is_unvalidated(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = sps.validation_status(
                self.bound_verification(Path(temporary)), None
            )
            self.assertFalse(status["validated"])
            self.assertIn(
                "no predicted structure", " ".join(status["missing"])
            )

    def test_an_unbound_verification_record_is_not_evidence(self):
        # Read without the file it describes, so it was never checked against
        # it. A verified record alone is not validation of this score.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            check = write(directory, "check.json", verification_record())
            unbound = sps.read_verification_record(check)
            status = sps.validation_status(
                unbound, self.preserved_recheck(directory)
            )
            self.assertFalse(status["validated"])
            self.assertIn("was not bound", " ".join(status["missing"]))

    def test_coordinates_that_do_not_match_are_not_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            status = sps.validation_status(
                self.bound_verification(directory),
                {"coordinates_preserved": False, "max_deviation_angstrom": 0.5},
            )
            self.assertFalse(status["validated"])
            self.assertIn("do not match", " ".join(status["missing"]))

    def test_the_opt_in_flag_cannot_make_a_record_look_validated(self):
        # The flag waives the requirement for evidence. It does not change
        # what was established, which is what the label reports.
        status = sps.validation_status(None, None, exploratory_opt_in=True)
        self.assertFalse(status["validated"])
        self.assertTrue(status["exploratory_opt_in"])
        self.assertEqual(status["label"], "UNVALIDATED EXPLORATORY SCORE")

    def test_the_flag_does_not_withdraw_validation_that_was_earned(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            status = sps.validation_status(
                self.bound_verification(directory),
                self.preserved_recheck(directory),
                exploratory_opt_in=True,
            )
            self.assertTrue(status["validated"])

    def test_the_label_never_implies_verification_passed(self):
        status = sps.validation_status(None, None)
        self.assertIn("must never be read as meaning it passed", status["meaning"])
        self.assertIn("does NOT mean PEP-001 verification failed", status["meaning"])

    def test_the_verdict_is_the_records_third_field(self):
        # Before any number, so it cannot be missed by a reader who stops at
        # the score.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, ligand_record_path, _ = prepared(directory)
            receptor = write(directory, "receptor.pdbqt", receptor_pdbqt())
            record = sps.build_record(
                "spec_test", ligand, receptor, BOX_30A, BOX_CENTRE, BOX_SIZE,
                1, 2, {"fully_contained": True}, TORSDOF,
                {"total_kcal_per_mol": -5.0}, None,
                ligand_record=ligand_record_path,
                identity_recovery=sps.load_identity_recovery(
                    ligand, ligand_record_path
                ),
            )
            self.assertEqual(
                list(record)[:3], ["schema", "design_id", "validation"]
            )
            # Nothing supplied, so the record defaults to unvalidated rather
            # than to silence.
            self.assertFalse(record["validation"]["validated"])


class OptimisedPoseValidityTests(unittest.TestCase):
    """An optimised score computed outside the box is not a score."""

    def containment(self, directory, moved=None):
        coordinates = positions()
        if moved is not None:
            coordinates[9] = coordinates[9] + np.asarray(moved, dtype=float)
        ligand = write(
            directory, "l.pdbqt", ligand_pdbqt(coordinates=coordinates)
        )
        record_path = write(directory, "l.json", ligand_record())
        return sps.pose_box_containment(
            sps.pose_atoms(ligand),
            BOX_RECORD,
            sps.load_identity_recovery(ligand, record_path),
        )

    def test_a_contained_optimised_pose_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            containment = self.containment(Path(temporary))
            optimised = {"total_kcal_per_mol": -7.0}
            self.assertIsNone(
                sps.record_optimised_validity(optimised, containment)
            )
            self.assertTrue(optimised["score_valid"])
            self.assertNotIn("validity", optimised)

    def test_an_optimised_pose_outside_the_box_blocks_and_is_marked_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            containment = self.containment(Path(temporary), moved=(40.0, 0, 0))
            self.assertFalse(containment["fully_contained"])
            optimised = {"total_kcal_per_mol": -7.0}
            blocker = sps.record_optimised_validity(optimised, containment)
            self.assertIsNotNone(blocker)
            self.assertIn("affinity maps are not defined", blocker)
            self.assertIn("local optimisation produced", blocker)
            self.assertFalse(optimised["score_valid"])
            self.assertIn("must not be quoted", optimised["validity"])
            # The number is kept so the failure is visible, not deleted.
            self.assertEqual(optimised["total_kcal_per_mol"], -7.0)

    def test_the_run_cannot_report_success_for_an_invalid_optimised_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            containment = self.containment(directory, moved=(40.0, 0, 0))
            optimised = {"total_kcal_per_mol": -7.0}
            blocker = sps.record_optimised_validity(optimised, containment)
            ligand, ligand_record_path, _ = prepared(directory)
            receptor = write(directory, "receptor.pdbqt", receptor_pdbqt())
            record = sps.build_record(
                "spec_test", ligand, receptor, BOX_30A, BOX_CENTRE, BOX_SIZE,
                1, 2, {"fully_contained": True}, TORSDOF,
                {"total_kcal_per_mol": -5.0}, optimised,
                ligand_record=ligand_record_path,
                identity_recovery=sps.load_identity_recovery(
                    ligand, ligand_record_path
                ),
            )
            record["blocking_issues"] = [blocker]
            self.assertEqual(sps.exit_code_for(record), 1)
            self.assertFalse(
                record["after_local_optimisation"]["score_valid"]
            )

    def test_a_clean_record_exits_zero(self):
        self.assertEqual(sps.exit_code_for({"blocking_issues": []}), 0)
        self.assertEqual(sps.exit_code_for({}), 0)

    def test_any_blocking_issue_prevents_success(self):
        self.assertEqual(
            sps.exit_code_for({"blocking_issues": ["something was not established"]}),
            1,
        )


class RecordTests(unittest.TestCase):
    """What the output record must carry for the result to be interpretable."""

    def record(self, directory, **kwargs):
        ligand, ligand_record_path, _ = prepared(directory)
        receptor = write(directory, "receptor.pdbqt", receptor_pdbqt())
        recovery = sps.load_identity_recovery(ligand, ligand_record_path)
        defaults = dict(
            ligand_record=ligand_record_path,
            torsdof_declaration="present",
            identity_recovery=recovery,
            engine_description="RecordingEngine, a stub: not Vina",
        )
        defaults.update(kwargs)
        return sps.build_record(
            "spec_test", ligand, receptor, BOX_30A, BOX_CENTRE, BOX_SIZE,
            1, 2, {"fully_contained": True}, TORSDOF,
            {"total_kcal_per_mol": -5.0, "vina_call": "Vina.score()"},
            {"total_kcal_per_mol": -7.0, "vina_call": "Vina.optimize()"},
            **defaults,
        )

    def test_the_coordinate_recheck_is_absent_rather_than_assumed(self):
        # Silence here would read as "the coordinates are the predicted ones".
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            recheck = record["coordinate_preservation_recheck"]
            self.assertFalse(recheck["checked"])
            self.assertIn("--predicted-structure", recheck["reason"])

    def test_a_coordinate_recheck_that_was_made_is_carried(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, ligand_record_path, _ = prepared(directory)
            source = write(directory, "source.pdb", peptide_source_pdb())
            comparison = sps.recheck_coordinate_preservation(
                sps.pose_atoms(ligand),
                sps.load_identity_recovery(ligand, ligand_record_path),
                source,
            )
            record = self.record(directory, coordinate_recheck=comparison)
            carried = record["coordinate_preservation_recheck"]
            self.assertTrue(carried["coordinates_preserved"])
            self.assertEqual(carried["heavy_atoms_compared"], 11)
            self.assertEqual(carried["source"], str(source))

    def test_the_verification_binding_is_part_of_the_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, ligand_record_path, _ = prepared(directory)
            check = write(directory, "check.json", verification_record())
            summary = sps.read_verification_record(
                check, ligand, ligand_record_path, "spec_test"
            )
            record = self.record(directory, verification=summary)
            binding = record["stage_4_verification"]["bound_to_scored_input"]
            self.assertTrue(binding["bound"])
            self.assertIn("re-derived from the file being scored", binding["method"])

    def test_the_prediction_and_the_relaxed_pose_are_separate_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            self.assertEqual(record["initial_pose"]["total_kcal_per_mol"], -5.0)
            self.assertEqual(
                record["after_local_optimisation"]["total_kcal_per_mol"], -7.0
            )
            self.assertEqual(record["schema"], "mur-ligase-peptide-score/2")

    def test_a_run_without_optimisation_records_none_not_an_empty_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            record["after_local_optimisation"] = None
            self.assertIsNone(record["after_local_optimisation"])

    def test_every_input_is_checksummed(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            for key in (
                "ligand_pdbqt_sha256", "receptor_pdbqt_sha256",
                "box_record_sha256", "ligand_record_sha256",
            ):
                self.assertEqual(len(record["inputs"][key]), 64, key)

    def test_the_record_states_that_no_search_was_performed(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            self.assertFalse(record["scoring_parameters"]["search_performed"])
            self.assertIn(
                "dock() is never called", record["scoring_parameters"]["note"]
            )

    def test_the_engine_is_named_so_a_stub_cannot_pass_as_vina(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            self.assertIn("stub", record["scoring_parameters"]["engine"])
            self.assertNotIn("vina", record["software"])

    def test_the_atom_identity_method_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            identity = record["atom_identity"]
            self.assertTrue(identity["established"])
            self.assertEqual(identity["atoms_identified"], 12)
            self.assertIn("INDEX MAP", identity["source"])

    def test_an_absent_verification_record_is_stated_as_a_consequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            self.assertFalse(record["stage_4_verification"]["supplied"])
            self.assertIn(
                "was not confirmed here",
                record["stage_4_verification"]["consequence"],
            )

    def test_a_supplied_verification_record_is_carried_into_the_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            check = write(directory, "check.json", verification_record())
            record = self.record(
                directory,
                verification=sps.read_verification_record(check),
            )
            self.assertTrue(record["stage_4_verification"]["verified"])

    def test_the_score_is_labelled_as_not_an_affinity(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            limitations = " ".join(record["limitations"])
            self.assertIn("not a measured binding free energy", limitations)
            self.assertIn("not evidence of binding", limitations)
            self.assertIn("inhibition", limitations)
            self.assertIn("rigid throughout", limitations)
            self.assertIn("chemically unvalidated", limitations)

    def test_the_torsion_caveat_names_both_hazards(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(Path(temporary))
            caveat = record["ligand_torsions"]["caveat"]
            self.assertIn("phi and psi", caveat)
            self.assertIn("rigid-body degrees of freedom", caveat)

    def test_additional_limitations_are_appended_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.record(
                Path(temporary), limitations=["no clash analysis was made"]
            )
            self.assertEqual(
                record["limitations"][0], "no clash analysis was made"
            )
            self.assertGreater(len(record["limitations"]), 1)


class MappedIdentityTests(unittest.TestCase):
    def test_identities_are_carried_to_the_pose_through_the_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            renumber = {serial: 13 - serial for serial in ATOM_SPECS}
            ligand, record_path, pose = prepared(
                directory, after_coordinates=positions(),
                renumber=renumber, reverse_within_groups=True,
            )
            recovery = sps.load_identity_recovery(ligand, record_path)
            found = sps.establish_correspondence(
                sps.pose_atoms(ligand), sps.parse_rigid_groups(ligand),
                sps.pose_atoms(pose), sps.parse_rigid_groups(pose),
            )
            derived = sps.mapped_recovery(recovery, found)
            # Serial 9 of the starting file is OXT; in the pose it is serial 4.
            self.assertEqual(recovery.by_serial[9].label, "SER2:OXT")
            self.assertEqual(derived.by_serial[4].label, "SER2:OXT")
            self.assertIn("derived", derived.method)

    def test_identities_cannot_be_carried_without_a_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record_path, _ = prepared(directory)
            recovery = sps.load_identity_recovery(ligand, record_path)
            with self.assertRaises(sps.ScoringError) as raised:
                sps.mapped_recovery(recovery, sps.Correspondence())
            self.assertIn("without an established correspondence",
                          str(raised.exception))


class VinaAbsenceTests(unittest.TestCase):
    @unittest.skipIf(sps.VINA_AVAILABLE, "Vina is installed here")
    def test_a_missing_vina_is_a_blocker_and_says_so(self):
        with self.assertRaises(sps.ScoringError) as raised:
            sps.require_vina()
        message = str(raised.exception)
        self.assertIn("blocker", message)
        self.assertIn("not a skippable check", message)

    def test_the_module_never_simulates_a_score_when_vina_is_absent(self):
        # There is no fallback scoring path: the only numbers the module can
        # report come from an engine object the caller supplied.
        source = Path(sps.__file__).read_text()
        self.assertNotIn("VINA_AVAILABLE else", source)
        self.assertIn("raise ScoringError", source)
        self.assertEqual(
            source.count("def require_vina"), 1
        )


class CommandLineTests(unittest.TestCase):
    """The command line, as far as it can be driven without Vina."""

    def test_the_parser_accepts_the_documented_invocation(self):
        with self.assertRaises(SystemExit) as raised:
            sps.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_the_ligand_record_is_a_required_argument(self):
        with self.assertRaises(SystemExit) as raised:
            sps.main(
                [
                    "--ligand", "l.pdbqt", "--receptor", "r.pdbqt",
                    "--box-record", str(BOX_30A), "--design-id", "d",
                    "--out-dir", "out",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    @unittest.skipIf(sps.VINA_AVAILABLE, "Vina is installed here")
    def test_a_run_without_vina_exits_two_and_simulates_nothing(self):
        # Acceptance: a missing dependency is an explicit failure. Nothing is
        # written, and no number is produced.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ligand, record, _ = prepared(directory)
            receptor = write(directory, "receptor.pdbqt", receptor_pdbqt())
            out = directory / "scores"
            code = sps.main(
                [
                    "--ligand", str(ligand),
                    "--ligand-record", str(record),
                    "--receptor", str(receptor),
                    "--box-record", str(BOX_30A),
                    "--design-id", "spec_test",
                    "--out-dir", str(out),
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
