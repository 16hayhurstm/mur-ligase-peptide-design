#!/usr/bin/env bash
# Run a BoltzGen design job against a spec written by make_spec.py.
#
# usage:   ./03_boltzgen_design.sh SPEC.yaml OUTDIR [num_designs]
# example: ./03_boltzgen_design.sh murc.yaml out_murc 100
#
# Requires the boltzgen conda environment. Expect roughly an hour for 100
# designs on an RTX A6000; run under tmux so it survives a dropped
# connection.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 SPEC.yaml OUTDIR [num_designs]" >&2
    exit 1
fi

SPEC="$1"
OUTDIR="$2"
N="${3:-100}"

if [ ! -f "$SPEC" ]; then
    echo "error: spec not found: $SPEC" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null; then
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "GPU memory in use: ${USED} MiB"
    if [ "$USED" -gt 2000 ]; then
        echo "WARNING: GPU already busy - this machine is shared" >&2
    fi
fi

echo "validating spec"
boltzgen check "$SPEC"

echo
echo "running $N designs -> $OUTDIR"
boltzgen run "$SPEC" \
    --output "$OUTDIR" \
    --protocol peptide-anything \
    --num_designs "$N" \
    --budget 20 \
    --use_kernels false

echo
echo "done. sequences and metrics:"
echo "  $OUTDIR/final_ranked_designs/all_designs_metrics.csv"
echo "design-stage structures for the occupancy filter:"
echo "  $OUTDIR/intermediate_designs/"
