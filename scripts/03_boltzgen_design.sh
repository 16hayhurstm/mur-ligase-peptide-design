#!/usr/bin/env bash
# Run a BoltzGen design job against a spec written by make_spec.py.
#
# usage:   ./03_boltzgen_design.sh SPEC.yaml OUTDIR [num_designs]
# example: ./03_boltzgen_design.sh murc.yaml out_murc 100
#
# Requires the boltzgen conda environment. Run under tmux.
# Estimate duration from measured local runs, not a fixed GPU runtime.
# Optional: [num_designs] --visualize [builder selection options]

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 SPEC.yaml OUTDIR [num_designs] [--visualize [selection options]]" >&2
    exit 1
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPEC="$1"
OUTDIR="$2"
shift 2
N=100
if [[ $# -gt 0 && "$1" != --* ]]; then
    N="$1"
    shift
fi
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "num_designs must be positive" >&2; exit 1; }
VISUALIZE=false
VIS_ARGS=()
if [[ $# -gt 0 ]]; then
    [[ "$1" == --visualize ]] || { echo "Unknown argument: $1" >&2; exit 1; }
    VISUALIZE=true
    shift
    VIS_ARGS=("$@")
fi

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

if [[ "$VISUALIZE" == true ]]; then
    bash "$SCRIPT_DIR/04_visualize_designs.sh" "$OUTDIR" \
        "$OUTDIR/visualization" "${VIS_ARGS[@]}"
fi
