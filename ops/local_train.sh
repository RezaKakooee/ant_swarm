#!/usr/bin/env bash
# Local/Azure equivalent of ops/sb_train.sh. It launches training detached
# from the terminal and keeps the same script/config/override argument shape.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${ANT_SWARM_CONDA_ENV:-antswarm}"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

usage() {
    cat <<'EOF'
Usage: ops/local_train.sh [train_ppo|train_sac] [config.yaml] [key=value ...]

Examples:
  ops/local_train.sh train_sac
  ops/local_train.sh train_sac configs/rl/pnas_kin_geo.yaml
  ops/local_train.sh train_sac configs/rl/pnas_kin_geo.yaml sac.timesteps=5e6

The job runs in the background and survives terminal/SSH disconnects.
Set ANT_SWARM_CONDA_ENV to use a Conda environment other than "roboverse".
EOF
}

resolve_script() {
    local script="$1"
    case "$script" in
        */*) ;;
        *) script="scripts/rl/${script}" ;;
    esac
    script="${script%.py}.py"
    printf '%s\n' "$script"
}

run_job() {
    local run_id="$1"
    local py_script="$2"
    local cfg_arg="$3"
    shift 3
    local extra_args=("$@")
    local output_dir="$PROJECT_ROOT/storage_local/sci_out"
    local output_file="$output_dir/${run_id}.out"

    mkdir -p "$output_dir"
    exec >> "$output_file" 2>&1

    if [ -n "$cfg_arg" ]; then
        export ANT_SWARM_CONFIG="$cfg_arg"
    else
        unset ANT_SWARM_CONFIG || true
    fi
    export ANT_SWARM_RUN_ID="$run_id"
    export DS_BUILD_OPS=0
    export DS_SKIP_CUDA_CHECK=1
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

    # Load Conda in this non-interactive, detached shell.
    for p in "/home2/reza/miniconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$p" ]; then source "$p"; conda activate "$CONDA_ENV" 2>/dev/null || true; break; fi
    done
    export PATH="/home2/reza/.conda/envs/$CONDA_ENV/bin:$PATH"

    echo "========================================"
    echo "Ant Swarm RL Training (local/Azure)"
    echo "========================================"
    echo "Job ID : $run_id"
    echo "PID    : $BASHPID"
    echo "Script : $py_script"
    echo "Config : ${ANT_SWARM_CONFIG:-config.yaml (default)}"
    echo "Host   : $(hostname)"
    echo "Conda  : $CONDA_ENV"
    echo "Python : $(command -v python3)"
    echo "Start  : $(date --iso-8601=seconds)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';' -)"
    else
        echo "GPU    : nvidia-smi not available"
    fi
    echo ""

    cd "$PROJECT_ROOT"
    local job_status=0
    python3 -u "$py_script" "${extra_args[@]}" || job_status=$?
    echo ""
    echo "End    : $(date --iso-8601=seconds)"
    echo "Exit   : $job_status"
    return "$job_status"
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

# Internal detached-worker mode. The public invocation below starts this mode
# with nohup so closing VS Code or SSH does not terminate training.
if [ "${1:-}" = "--worker" ]; then
    shift
    run_job "$@"
    exit
fi

PY_SCRIPT="$(resolve_script "${1:-train_ppo}")"
CFG_ARG="${2:-}"
EXTRA_ARGS=("${@:3}")

if [ ! -f "$PROJECT_ROOT/$PY_SCRIPT" ]; then
    echo "Error: training script not found: $PROJECT_ROOT/$PY_SCRIPT" >&2
    exit 2
fi

CFG_TAG=""
if [ -n "$CFG_ARG" ]; then
    if [ ! -f "$CFG_ARG" ]; then
        echo "Error: config file not found: $CFG_ARG" >&2
        exit 2
    fi
    CFG_ARG="$(readlink -f "$CFG_ARG")"
    CFG_TAG="__$(basename "$CFG_ARG" .yaml)"
fi

if [ ! -d "/home2/reza/.conda/envs/$CONDA_ENV" ] && ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda environment '$CONDA_ENV' not found." >&2
    exit 2
fi

OUTPUT_DIR="$PROJECT_ROOT/storage_local/sci_out"
CURRENT_DATE="$(date +%Y%m%d_%H%M)"
LOCAL_JOB_ID="local-${BASHPID}"
RUN_ID="ant__${CURRENT_DATE}__${LOCAL_JOB_ID}__$(basename "$PY_SCRIPT" .py)${CFG_TAG}"
OUTPUT_FILE="$OUTPUT_DIR/${RUN_ID}.out"

mkdir -p "$OUTPUT_DIR"
nohup setsid "$PROJECT_ROOT/ops/local_train.sh" --worker \
    "$RUN_ID" "$PY_SCRIPT" "$CFG_ARG" "${EXTRA_ARGS[@]}" \
    </dev/null >/dev/null 2>&1 &
JOB_PID=$!

echo "Submitted local job $RUN_ID"
echo "PID    : $JOB_PID"
echo "Log    : $OUTPUT_FILE"
echo "Watch  : '$PROJECT_ROOT/ops/watch_local.sh'"
echo "Follow : tail -f '$OUTPUT_FILE'"
echo "Stop   : kill -- -$JOB_PID"
