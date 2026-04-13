#!/usr/bin/env bash
# =============================================================================
# Beam FEM Thermal CCX — full pipeline runner
#
# Usage:
#   ./run_pipeline.sh              # run all steps
#   ./run_pipeline.sh --from 3    # start from step 3
#   ./run_pipeline.sh --only 5    # run only step 5
#
# Steps:
#   1  freecadcmd  01_GENERATE_REFERENCE_BEAM.py
#   2  freecadcmd  02_MESH_REFERENCE_BEAM.py
#   3  python3     03_GENERATE_ELMER_INPUTS_THERMAL.py
#   4  python3     04_RUN_AND_CONVERT.py
#   5  python3     scripts/make_video.py
#   6  python3     src/train.py
#   7  python3     src/inference.py
#   8  python3     src/inference.py --vtu
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

FREECADCMD=/snap/bin/freecad.cmd

# ── Argument parsing ──────────────────────────────────────────────────────────
FROM_STEP=1
ONLY_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift 2 ;;
        --only) ONLY_STEP="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $*"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAILED:${NC} $*"; exit 1; }

run_step() {
    local step="$1"
    local label="$2"
    local cmd="$3"

    if [[ -n "$ONLY_STEP" && "$step" != "$ONLY_STEP" ]]; then return; fi
    if [[ -z "$ONLY_STEP" && "$step" -lt "$FROM_STEP" ]]; then return; fi

    local logfile="$LOG_DIR/step_$(printf '%02d' "$step").log"
    log "Step $step — $label"
    echo "  cmd : $cmd"
    echo "  log : $logfile"

    if eval "$cmd" > "$logfile" 2>&1; then
        log "Step $step done."
    else
        fail "Step $step exited with error. See $logfile"
    fi
    echo
}

# ── Pipeline ──────────────────────────────────────────────────────────────────
echo "=============================================="
echo "  Beam FEM Thermal CCX Pipeline"
echo "  Project : $SCRIPT_DIR"
if [[ -n "$ONLY_STEP" ]]; then
    echo "  Mode    : only step $ONLY_STEP"
elif [[ "$FROM_STEP" -gt 1 ]]; then
    echo "  Mode    : from step $FROM_STEP"
else
    echo "  Mode    : full run (steps 1–8)"
fi
echo "=============================================="
echo

run_step 1 "Generate reference beam CAD" \
    ""$FREECADCMD" '$SCRIPT_DIR/scripts/01_GENERATE_REFERENCE_BEAM.py'"

run_step 2 "Mesh reference beam (Elmer format, ~10 000 nodes)" \
    ""$FREECADCMD" '$SCRIPT_DIR/scripts/02_MESH_REFERENCE_BEAM.py'"

run_step 3 "Generate Elmer SIF files (525 cases, 5 materials)" \
    "python3 '$SCRIPT_DIR/scripts/03_GENERATE_ELMER_INPUTS_THERMAL.py'"

run_step 4 "Run Elmer simulations + convert to VTK" \
    "python3 '$SCRIPT_DIR/scripts/04_RUN_AND_CONVERT.py'"

run_step 5 "Generate FEM temperature overview video" \
    "python3 '$SCRIPT_DIR/scripts/make_video.py'"

run_step 6 "Train thermal beam PINN" \
    "python3 '$SCRIPT_DIR/src/train.py'"

run_step 7 "Inference — metrics + PNG comparisons" \
    "python3 '$SCRIPT_DIR/src/inference.py'"

run_step 8 "Inference — write comparison VTU files" \
    "python3 '$SCRIPT_DIR/src/inference.py' --vtu"

echo "=============================================="
log "Pipeline complete."
echo "Logs in: $LOG_DIR"
echo "=============================================="
