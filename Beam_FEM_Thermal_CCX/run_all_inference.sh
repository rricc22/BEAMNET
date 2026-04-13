#!/bin/bash
# Run inference + visualize for all trained models.
# Each model gets its own subfolder under saves/.

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
SAVES="$ROOT/saves"
NORM="$SAVES/norm_params.npz"

for MODEL in "$SAVES"/thermal_pinn_*.pt; do
    NAME=$(basename "$MODEL" .pt | sed 's/thermal_pinn_//')
    OUTDIR="$SAVES/$NAME"
    mkdir -p "$OUTDIR"

    echo "========================================================"
    echo "Model : $NAME"
    echo "Outdir: $OUTDIR"
    echo "========================================================"

    python "$ROOT/src/inference.py" \
        --model  "$MODEL" \
        --outdir "$OUTDIR"

    python "$ROOT/utils/visualize_results.py" \
        --model  "$MODEL" \
        --outdir "$OUTDIR"

    echo ""
done

echo "All done. Results in $SAVES/<model_name>/"
