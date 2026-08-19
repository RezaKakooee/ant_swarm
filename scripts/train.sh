#!/usr/bin/env bash
# Train on any machine. No cluster, no job scheduler, no conda required.
#
# Usage:
#   scripts/train.sh                                   # SAC, default config
#   scripts/train.sh train_sac pnas_dyn_geo_v2         # the dynamic PNAS maze
#   scripts/train.sh train_ppo pnas_kin_geo_v2 ppo.timesteps=5e6 run.wandb=false
#
#   BACKGROUND=1 scripts/train.sh train_sac pnas_dyn_geo_v2
#       runs detached, survives closing the terminal, logs to storage_local/logs/
#
# The config can be a name from configs/rl/ (no .yaml) or a path to any yaml.
# Anything of the form key=value is passed straight to the training script.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

ALGO="${1:-train_sac}"; [ $# -gt 0 ] && shift || true
CONFIG="${1:-}"
if [[ -n "$CONFIG" && "$CONFIG" != *=* ]]; then shift; else CONFIG=""; fi
OVERRIDES=("$@")

case "$ALGO" in
    train_sac|sac) SCRIPT="scripts/rl/train_sac.py" ;;
    train_ppo|ppo) SCRIPT="scripts/rl/train_ppo.py" ;;
    */*)           SCRIPT="$ALGO" ;;
    *)             SCRIPT="scripts/rl/${ALGO%.py}.py" ;;
esac
[ -f "$SCRIPT" ] || { echo "no such training script: $SCRIPT" >&2; exit 1; }

# --- resolve the config -----------------------------------------------------
CFG_ARGS=()
CFG_FILE="configs/rl/config.yaml"
if [ -n "$CONFIG" ]; then
    if [ -f "$CONFIG" ]; then
        CFG_FILE="$CONFIG"
        export ANT_SWARM_CONFIG="$ROOT/$CONFIG"
    else
        CFG_FILE="configs/rl/${CONFIG%.yaml}.yaml"
        [ -f "$CFG_FILE" ] || { echo "no such config: $CFG_FILE" >&2; exit 1; }
        CFG_ARGS=(--config-name "${CONFIG%.yaml}")
    fi
fi

# --- build the geodesic field if this config needs one ----------------------
FIELD="$(grep -E '^\s*geodesic_field:' "$CFG_FILE" | head -1 | sed -E 's/.*geodesic_field:\s*//; s/["'"'"']//g' | tr -d '\r')"
MODE="$(grep -E '^\s*reward_mode:' "$CFG_FILE" | head -1 | sed -E 's/.*reward_mode:\s*//' | tr -d '\r ')"
if [ "$MODE" = "geodesic" ] && [ -n "$FIELD" ] && [ ! -f "$FIELD" ]; then
    echo "This config uses the geodesic reward, but the field is missing:"
    echo "  $FIELD"
    echo "Building it now (about a minute, one time only)..."
    mkdir -p "$(dirname "$FIELD")"
    "$PY" scripts/rl/gen_geodesic_field.py "$CFG_FILE" \
        --out "$FIELD" --inflate 0.002 --dx 0.003 --dth 2
    echo
fi

# --- run --------------------------------------------------------------------
echo "script : $SCRIPT"
echo "config : $CFG_FILE"
[ ${#OVERRIDES[@]} -gt 0 ] && echo "override: ${OVERRIDES[*]}"

if [ "${BACKGROUND:-0}" = "1" ]; then
    mkdir -p storage_local/logs
    LOG="storage_local/logs/$(date +%Y%m%d_%H%M%S)_$(basename "${SCRIPT%.py}").out"
    nohup "$PY" -u "$SCRIPT" "${CFG_ARGS[@]}" "${OVERRIDES[@]}" > "$LOG" 2>&1 &
    echo "pid    : $!"
    echo "log    : $LOG"
    echo "follow : tail -f $LOG"
    echo "stop   : kill $!"
else
    exec "$PY" -u "$SCRIPT" "${CFG_ARGS[@]}" "${OVERRIDES[@]}"
fi
