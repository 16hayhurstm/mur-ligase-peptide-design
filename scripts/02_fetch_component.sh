#!/usr/bin/env bash
# Download a Chemical Component Dictionary entry from the RCSB.
#
# usage:   ./02_fetch_component.sh COMPONENT [outdir]
# example: ./02_fetch_component.sh UMA components
#
# The CCD entry is the authoritative source for a ligand's atom names,
# elements, connectivity, bond orders and declared stereochemistry. It is NOT
# the authority for the protonation state to model: a deposited entry is the
# neutral molecule.
#
# CCD entries are revised. Each carries a `_chem_comp.pdbx_modified_date`, and
# a later revision can change atom names or bond orders. The SHA-256 printed
# here is what pins the version actually used, and it is recorded in every
# ligand manifest, so a re-fetch that changes the hash is a change of input
# and not a detail.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 COMPONENT [outdir]" >&2
    exit 1
fi

COMPONENT=$(echo "$1" | tr '[:lower:]' '[:upper:]')
OUTDIR="${2:-.}"
mkdir -p "$OUTDIR"
OUT="${OUTDIR}/${COMPONENT}.cif"

if [ -f "$OUT" ]; then
    echo "already present: $OUT"
else
    echo "fetching component $COMPONENT"
    wget -q "https://files.rcsb.org/ligands/download/${COMPONENT}.cif" -O "$OUT"
    echo "wrote $OUT"
fi

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUT"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$OUT"
else
    echo "NOTE no sha256 tool found; record the hash before relying on this file" >&2
fi

grep -E "_chem_comp\.(id|name|formula|pdbx_formal_charge|pdbx_modified_date)" \
    "$OUT" || true
