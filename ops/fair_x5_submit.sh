#!/bin/bash
# Chapter 04 §12, experiment 1: the single ant with 5 envs per worker.
# Same submit line as jobs 242212/242213 (see `sacct --format=SubmitLine`),
# plus --envs-per-worker 5. A second job evaluates every checkpoint on the
# held-out seed blocks (60000+, 70000+) when the training job ends.
#
#   bash ops/fair_x5_submit.sh
set -euo pipefail
ROOT=/home2/reza/ant_swarm
ENVV="export PATH=/home2/reza/.conda/envs/antswarm/bin:\$PATH; export PYTHONPATH=$ROOT; export WANDB_ENTITY=kakooee; export OMP_NUM_THREADS=1; cd $ROOT"
TAG=marl_v2_h4_x5

TRAIN=$(sbatch --parsable -M cluster -p performance -c 32 --mem=48G -t 24:00:00 -J fair_single_x5 \
  -o $ROOT/storage_local/sci_out/fair_single_x5_%j.out \
  --wrap "$ENVV; export ANT_SWARM_RUN_ID=ant__\$(date +%Y%m%d_%H%M)__\${SLURM_JOB_ID}__$TAG; \
python -u scripts/rl/train_marl_v2.py --config configs/rl/marl_v2_1ant.yaml --workers 30 --history 4 \
--timesteps 20000000 --eval-every 250000 --eval-episodes 30 --seed 31000 --device cpu \
--envs-per-worker 5 --ckpt-every 1000000")
TRAIN=${TRAIN%%;*}
echo "train job: $TRAIN"

# afterany: also runs if training hits the 24 h limit, so partial checkpoints get scored
EVAL=$(sbatch --parsable -M cluster -p performance -c 4 --mem=8G -t 6:00:00 -J fair_single_x5_eval \
  --dependency=afterany:$TRAIN \
  -o $ROOT/storage_local/sci_out/fair_single_x5_eval_%j.out \
  --wrap "$ENVV; RUN=\$(ls -d storage_local/ant__*__${TRAIN}__$TAG | head -1); echo run=\$RUN; \
python -u scripts/rl/eval_marl_v2.py \$RUN/ckpt_*.pt \$RUN/best.pt \$RUN/final.pt \
storage_local/ant__20260909_2334__242212__marl_v2_h4/final.pt \
storage_local/ant__20260909_2337__242213__marl_v2_h4/final.pt \
--history 4 --out \$RUN/heldout_200.json")
EVAL=${EVAL%%;*}
echo "eval job:  $EVAL (after $TRAIN)"
