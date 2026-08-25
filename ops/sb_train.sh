#!/bin/bash
#SBATCH --job-name=ant
#SBATCH --qos=rtx4090-1day
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null


# repo root: the directory sbatch was submitted from, else this script's parent
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Usage: sbatch ops/sb_train.sh [train_ppo|train_sac] [config.yaml] [key=value ...]
#   arg1: training script — a name resolved under scripts/rl/ (default train_ppo),
#         or an explicit path like scripts/rl/train_sac.py
#   arg2: optional config variant — exported as ANT_SWARM_CONFIG so parallel
#         sweep jobs each read their own config instead of the project default
#   rest: optional hydra-style overrides passed to the script, e.g.
#         sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml sac.timesteps=5e6
PY_SCRIPT="${1:-train_ppo}"
case "$PY_SCRIPT" in
    */*) ;;                                  # explicit path: use as given
    *)   PY_SCRIPT="scripts/rl/${PY_SCRIPT}" ;;
esac
PY_SCRIPT="${PY_SCRIPT%.py}"
CFG_ARG="${2:-}"
CFG_TAG=""
if [ -n "$CFG_ARG" ]; then
    export ANT_SWARM_CONFIG="$(readlink -f "$CFG_ARG")"
    CFG_TAG="__$(basename "$CFG_ARG" .yaml)"
fi
EXTRA_ARGS=("${@:3}")               # hydra-style key=value overrides


run_job() {
    echo "========================================"
    echo "Ant Swarm RL Training"
    echo "========================================"
    echo "Job ID : $SLURM_JOB_ID"
    echo "Script : $PY_SCRIPT"
    echo "Config : ${ANT_SWARM_CONFIG:-config.yaml (default)}"
    echo "Node   : $SLURM_NODELIST"
    echo "GPUs   : $CUDA_VISIBLE_DEVICES"
    echo "Start  : $(date)"
    echo ""

    module load CUDA/12.1 2>/dev/null || module load cuda 2>/dev/null || true
    if [ -z "${CUDA_HOME:-}" ] && command -v nvcc &>/dev/null; then
        export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
    export DS_BUILD_OPS=0
    export DS_SKIP_CUDA_CHECK=1
    # ANT_SWARM_FORCE_CPU=1 -> run without a GPU. These jobs are CPU-bound
    # (pure-NumPy env), so this avoids GPU queues and contention entirely.
    if [ "${ANT_SWARM_FORCE_CPU:-0}" = "1" ]; then
        export CUDA_VISIBLE_DEVICES=""
    fi
    # W&B: remote dashboard is the record; stage local files in job scratch
    # (deleted automatically when the job ends) instead of storage_local
    export WANDB_DIR="${TMPDIR:-/tmp}"
    export MUJOCO_GL=egl
    export PYTHONNOUSERSITE=1

    # Conda location and env are configurable for other servers:
    #   ANT_SWARM_CONDA_SH  (default: $HOME/miniconda3/etc/profile.d/conda.sh)
    #   ANT_SWARM_CONDA_ENV (default: roboverse)
    source "${ANT_SWARM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${ANT_SWARM_CONDA_ENV:-roboverse}"

    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    echo "Python : $(which python3)"
    echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo ""

    python3 -u "$PROJECT_ROOT/$PY_SCRIPT".py "${EXTRA_ARGS[@]}"

    echo ""
    echo "End : $(date)"
    echo "Done."
}



# Mint ONE run id up front (see ant_swarm/run_id.py): the .out log, the
# storage_local run dir, and the wandb run all share this exact name.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output_dir="${PROJECT_ROOT}/storage_local/sci_out"
current_date=$(date +%Y%m%d_%H%M)
job_id=${SLURM_JOB_ID}

export ANT_SWARM_RUN_ID="ant__${current_date}__${job_id}__$(basename "$PY_SCRIPT")${CFG_TAG}"
output_file="${output_dir}/${ANT_SWARM_RUN_ID}.out"

mkdir -p ${output_dir}

run_job > "${output_file}" 2>&1
