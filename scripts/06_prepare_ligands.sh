#!/usr/bin/env bash
# Build peptide ligands from sequence and convert them to PDBQT.
#
# usage:   ./06_prepare_ligands.sh OUTDIR SEQ [SEQ ...]
# example: ./06_prepare_ligands.sh pep TMGFTAPRFPHY GSQAGPSG
#
# Structures are built with PyMOL's `fab`, which needs the SYSTEM python.
# Conda shadows it, so this script deliberately runs pymol with a cleaned
# PATH rather than relying on the caller having deactivated conda.
#
# Conversion is done with OpenBabel end to end, NOT via Meeko. Meeko's
# RDKit sanitisation rejects some peptide geometries with
#   "Explicit valence for atom # N, 4, is greater than permitted"
# and adding pH-based protonation (-p 7.4) makes it worse (N with five
# bonds). The failures are geometry- rather than composition-dependent:
# of six peptides with identical composition, three converted and three
# did not. Always check the output count.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 OUTDIR SEQ [SEQ ...]" >&2
    exit 1
fi

OUTDIR="$1"; shift
mkdir -p "$OUTDIR"

PML="${OUTDIR}/build.pml"
: > "$PML"
for SEQ in "$@"; do
    echo "fab ${SEQ}, ${SEQ}, ss=2"          >> "$PML"
    echo "save ${OUTDIR}/${SEQ}.pdb, ${SEQ}" >> "$PML"
    echo "delete all"                        >> "$PML"
done

echo "building $# peptide(s) with PyMOL"
PATH=/usr/bin:/bin pymol -cq "$PML"

echo "converting to PDBQT with OpenBabel"
for SEQ in "$@"; do
    obabel "${OUTDIR}/${SEQ}.pdb" -O "${OUTDIR}/${SEQ}.pdbqt" \
           -h --partialcharge gasteiger 2>/dev/null
done

echo
N_IN=$#
N_OUT=$(ls "${OUTDIR}"/*.pdbqt 2>/dev/null | wc -l)
echo "requested $N_IN, produced $N_OUT"
[ "$N_IN" -eq "$N_OUT" ] || echo "WARNING: some conversions failed" >&2

echo
echo "torsional degrees of freedom:"
grep -H TORSDOF "${OUTDIR}"/*.pdbqt | sed 's|.*/||'
echo
echo "AutoDock Vina's practical limit is about 32 torsions. A 12-mer"
echo "peptide carries 42-43 and is outside that range; results from such"
echo "ligands should be treated as unreliable."
