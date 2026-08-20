#!/usr/bin/env bash
# Download a structure from the RCSB PDB.
#
# usage:   ./01_fetch_structure.sh PDBID [outdir]
# example: ./01_fetch_structure.sh 1P3D data

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 PDBID [outdir]" >&2
    exit 1
fi

PDBID=$(echo "$1" | tr '[:lower:]' '[:upper:]')
OUTDIR="${2:-.}"
mkdir -p "$OUTDIR"
OUT="${OUTDIR}/${PDBID}.cif"

if [ -f "$OUT" ]; then
    echo "already present: $OUT"
    exit 0
fi

echo "fetching $PDBID"
wget -q "https://files.rcsb.org/download/${PDBID}.cif" -O "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
