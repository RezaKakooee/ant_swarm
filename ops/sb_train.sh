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


PROJECT_ROOT="/scicore/home/graber0001/kakooe0000/ant_swarm"

PY_SCRIPT="train_sac"


run_job() {
    echo "========================================"
    echo "Ant Swarm RL Training"
    echo "========================================"
    echo "Job ID : $SLURM_JOB_ID"
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
    export MUJOCO_GL=egl
    export PYTHONNOUSERSITE=1

    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate roboverse

    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    echo "Python : $(which python3)"
    echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo ""

    python3 -u "$PROJECT_ROOT/$PY_SCRIPT".py 

    echo ""
    echo "End : $(date)"
    echo "Done."
}



# Generate output filename
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output_dir="${PROJECT_ROOT}/storage_local/sci_out"
current_date=$(date +%Y%m%d_%H%M)
job_id=${SLURM_JOB_ID}

output_file="${output_dir}/ant__${current_date}__${job_id}__${PY_SCRIPT}.out"

mkdir -p ${output_dir}

run_job > "${output_file}" 2>&1
