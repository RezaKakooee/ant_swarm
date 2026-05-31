#!/bin/bash
#SBATCH --job-name=ant_swarm_ppo
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

echo "========================================"
echo "Ant Swarm PPO Training"
echo "========================================"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Start  : $(date)"
echo ""

module load CUDA/12.1 2>/dev/null || module load cuda 2>/dev/null || true
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc &>/dev/null; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi\
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

python3 -u "$PROJECT_ROOT/train_ppo.py" \
    --timesteps 50000000 \
    --n-envs 8 \
    --n-steps 4096 \
    --render-freq 500000

echo ""
echo "End : $(date)"
echo "Done."
