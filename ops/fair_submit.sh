#!/bin/bash
# Submit one MARL v2 "fair recipe" run (chapter 04 §11-§13) plus a held-out
# eval job that scores every checkpoint on seeds 60000+ and 70000+ when the
# training job ends. Same submit line as jobs 242212/242213/242283.
#
#   bash ops/fair_submit.sh <config.yaml> <tag> [extra train args...]
#   bash ops/fair_submit.sh configs/rl/marl_v2_10ants.yaml marl_v2_h4_10ants
#   bash ops/fair_submit.sh configs/rl/marl_v2_1ant.yaml  marl_v2_h4_x5 --envs-per-worker 5
#   FAIR_SCRIPT=scripts/rl/train_masac.py bash ops/fair_submit.sh configs/rl/marl_v2_5ants_nomap.yaml masac_h4_ind_nomap --utd 0.05
set -euo pipefail
CFG=${1:?config}; TAG=${2:?tag}; shift 2; EXTRA="$*"
SCRIPT=${FAIR_SCRIPT:-scripts/rl/train_marl_v2.py}   # FAIR_SCRIPT=scripts/rl/train_masac.py for the SAC trainer
MEM=${FAIR_MEM:-48G}                                     # FAIR_MEM=40G etc.
ROOT=/home2/reza/ant_swarm
ENVV="export PATH=/home2/reza/.conda/envs/antswarm/bin:\$PATH; export PYTHONPATH=$ROOT; export WANDB_ENTITY=kakooee; export OMP_NUM_THREADS=1; cd $ROOT"

TRAIN=$(sbatch --parsable -M cluster -p performance -c 32 --mem=$MEM -t 24:00:00 -J fair_$TAG \
  -o $ROOT/storage_local/sci_out/fair_${TAG}_%j.out \
  --wrap "$ENVV; export ANT_SWARM_RUN_ID=ant__\$(date +%Y%m%d_%H%M)__\${SLURM_JOB_ID}__$TAG; \
python -u $SCRIPT --config $CFG --workers 30 --history 4 \
--timesteps 20000000 --eval-every 250000 --eval-episodes 30 --seed 31000 --device cpu \
--ckpt-every 1000000 $EXTRA")
TRAIN=${TRAIN%%;*}
echo "train job: $TRAIN"

# afterany: also runs if training hits the 24 h limit, so partial checkpoints get scored
EVAL=$(sbatch --parsable -M cluster -p performance -c 4 --mem=8G -t 8:00:00 -J fair_${TAG}_eval \
  --dependency=afterany:$TRAIN \
  -o $ROOT/storage_local/sci_out/fair_${TAG}_eval_%j.out \
  --wrap "$ENVV; RUN=\$(ls -d storage_local/ant__*__${TRAIN}__$TAG | head -1); echo run=\$RUN; \
python -u scripts/rl/eval_marl_v2.py \$RUN/ckpt_*.pt \$RUN/best.pt \$RUN/final.pt \
storage_local/ant__20260909_2337__242213__marl_v2_h4/final.pt \
storage_local/ant__20260910_1013__242283__marl_v2_h4_x5/final.pt \
--config $CFG --history 4 --out \$RUN/heldout_200.json")
EVAL=${EVAL%%;*}
echo "eval job:  $EVAL (after $TRAIN)"
