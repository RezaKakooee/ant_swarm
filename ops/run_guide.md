# Ant-Swarm Ops Run Guide

How to run, monitor, and post-process experiments. All commands run from the
repo root on the cluster login node, with the `roboverse` conda env:

```bash
cd ~/ant_swarm
conda activate roboverse
```

## Install on a new server

```bash
git clone https://github.com/RezaKakooee/ant_swarm.git && cd ant_swarm
conda create -n antswarm python=3.10 -y && conda activate antswarm
pip install -r requirements.txt          # pure Python, no MuJoCo
wandb login                              # or add run.wandb=false to commands

# the geodesic field is not in git (storage_local/ is ignored) — regenerate (~1 min):
python scripts/rl/gen_geodesic_field.py configs/rl/pnas_kin_geo_v2.yaml \
    --out storage_local/fields/pnas_gap015.npz --inflate 0.002 --dx 0.003 --dth 2

# smoke test, then train:
python scripts/rl/train_sac.py --config-name pnas_dyn_geo_v2 run.wandb=false sac.timesteps=2000
python scripts/rl/train_sac.py --config-name pnas_dyn_geo_v2
```

Without SLURM, use `ops/local_train.sh`; it accepts the same arguments as the
SLURM wrapper, runs detached, and writes to the same log directory.

```
ops/
├── sb_train.sh            # SLURM training job (a100, 1 day) — the main launcher
├── local_train.sh         # detached local/Azure equivalent (same arguments)
├── fsrun.sh               # interactive srun shell on a compute node
├── py_runner.sh           # nohup-run python scripts on the login node (survives logout)
├── render_success.sh      # nohup wrapper for scripts/il/render_success.py
├── combine_successes.sh   # nohup wrapper for scripts/il/combine_successes.py
├── convert_successes.sh   # nohup wrapper for scripts/il/convert_successes.py
├── watch.sh / watch_py.sh # simple process watchers
└── run_guide.md           # this file
```

Python entry points live in `scripts/{rl,il,heuristic}`; configs in
`configs/{rl,il,heuristic}` (see their READMEs). Reusable logic lives in the
`ant_swarm` package.

## Train on the cluster

```bash
# default config (configs/rl/config.yaml)
sbatch ops/sb_train.sh train_sac
sbatch ops/sb_train.sh train_ppo

# a config variant (parallel sweeps = submit several of these)
sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml

# variant + hydra-style overrides
sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml sac.timesteps=5e6

# other GPU partitions: sbatch flags override the script's #SBATCH headers.
# These MLP jobs barely use the GPU — rtx4090 is a good fit and a shorter queue.
sbatch --partition=rtx4090 --qos=rtx4090-1day --gres=gpu:1 \
    ops/sb_train.sh train_sac configs/rl/pnas_dyn_geo_v2.yaml
# also available: titan (titan-1day), l40s (l40s-1day); -6hours/-1week QOS variants
```

### MARL v2 (PPO, centralised critic; chapter 04 §10-§12)

These runs are CPU-bound (NumPy env). Use many CPU workers and no GPU.
`ops/fair_x5_submit.sh` holds the exact submit line; the short form is:

```bash
sbatch -M cluster -p performance -c 32 --mem=48G -t 24:00:00 -J <name> \
    -o storage_local/sci_out/<name>_%j.out \
    --wrap "export PATH=/home2/reza/.conda/envs/antswarm/bin:\$PATH; export PYTHONPATH=$PWD; \
            export WANDB_ENTITY=kakooee; export OMP_NUM_THREADS=1; cd $PWD; \
            python -u scripts/rl/train_marl_v2.py --config configs/rl/marl_v2_5ants.yaml \
            --workers 30 --history 4 --timesteps 20000000 --eval-every 250000 --eval-episodes 30 \
            --seed 31000 --device cpu"

# switches: --envs-per-worker 5 (5 actor rows per worker step; `steps` = worker steps)
#           --ckpt-every 1000000 (default; writes ckpt_<steps>.pt)   --resume <ckpt>
#           --warm-start <single-ant BC .pt>   --stages 2,3,5   --init-actor <pt>
# smoke test first (fails on a traceback AND on exit 124):
timeout 900 python scripts/rl/train_marl_v2.py --config configs/rl/marl_v2_1ant.yaml --workers 2 \
    --history 4 --timesteps 4096 --rollout-steps 256 --eval-every 1024 --eval-episodes 2 \
    --heldout-episodes 2 --no-wandb --out /tmp/smoke; echo exit=$?
```

One run id (see `ant_swarm/run_id.py`) names three things identically:

- log:       `storage_local/sci_out/<run_id>.out`
- run dir:   `storage_local/<run_id>/`  (checkpoints, renders, successes, tb,
             train.log, `code/` snapshot with the RESOLVED config)
- wandb run: https://wandb.ai/kakooee/ant_swarm — same name

## Train locally (debug)

```bash
python scripts/rl/train_sac.py --config-name pnas_kin_geo \
    run.wandb=false sac.timesteps=50000
```

## Train on Azure / without SLURM

```bash
# These return immediately; training continues after terminal/SSH disconnects.
ops/local_train.sh train_sac
ops/local_train.sh train_sac configs/rl/pnas_kin_geo.yaml
ops/local_train.sh train_sac configs/rl/pnas_kin_geo.yaml sac.timesteps=5e6

# The command prints the exact log path, PID, and stop command. Monitor with:
tail -f storage_local/sci_out/<run_id>.out
```

The runner uses the `roboverse` Conda environment by default. Override it for
one invocation with, for example:

```bash
ANT_SWARM_CONDA_ENV=ant_swarm ops/local_train.sh train_sac ...
```

## Monitor

```bash
squeue -u $USER -n ant                              # job states
tail -f storage_local/sci_out/<run_id>.out          # live log
grep "\[curriculum\]" storage_local/sci_out/<run_id>.out   # stage progress
```

What to watch: `curriculum/difficulty` should walk toward its target
(reverse mode: spawn_x → ~0.3); `rollout/success_rate` per stage;
`eval/*` is pinned at the FULL hard task. A run prints
`[curriculum] TARGET MASTERED` and stops itself when the real task is solved
(≥ `stop_success` over `stop_window` episodes).

## Evaluate a checkpoint

```bash
# Standalone evaluation with automatic GIF and MP4 recording:
python scripts/rl/evaluate.py storage_local/<run> --episodes 5

# Or via train_sac:
python scripts/rl/train_sac.py --config-name pnas_dyn_geo_v2 \
    run.eval=true run.eval_model=storage_local/<run>/checkpoints/best/best_model.zip run.eval_episodes=5
```

MARL v2 actors (`best.pt`, `final.pt`, `ckpt_*.pt`) are scored on held-out
seed blocks (60000+, 70000+; 100 episodes each) with:

```bash
python scripts/rl/eval_marl_v2.py storage_local/<run>/final.pt --history 4 --out storage_local/<run>/heldout_200.json
# success vs actor rows across runs (chapter 04 §12.1 figure):
python scripts/tools/samples_curve.py --run single=storage_local/<run_a> --run swarm=storage_local/<run_b> --out figures.png
```

Quote held-out numbers only. Best-of-N on one 30-episode block inflates.

## Watch what was learned

```bash
# one solved episode → GIF (uses the run's own geometry snapshot)
python scripts/il/render_success.py storage_local/<run>/successes/success_*.json

# grid of solutions from a combined file (n, tile size, frame cap adjustable)
python scripts/il/render_success.py <combined.json> --grid 9 --tile 240

# policy GIFs from the true full-task spawn appear automatically during
# training in storage_local/<run>/renders/ every run.render_freq steps
```

## Success-file housekeeping

Every solved episode is saved as a small replayable JSON (init pose + actions).
Mastered runs can accumulate tens of thousands:

```bash
python scripts/il/combine_successes.py storage_local/<run>   # chunk into combined_*.json (deletes originals)
```

## Random baseline

```bash
python scripts/heuristic/random_agent.py heuristic.episodes=3 heuristic.gif=false
```

## Geodesic reward field

`env.reward_mode: geodesic` needs a precomputed distance field. Regenerate it
whenever walls, T-shape, goal, or `goal_track` change:

```bash
python scripts/rl/gen_geodesic_field.py configs/rl/pnas_kin_geo.yaml \
    --out storage_local/fields/pnas_gap016.npz --inflate 0.003
```

`--inflate` keeps razor-thin channels out of the field (routes need real
clearance). The config's `env.geodesic_field` must point at the output file.

## Interactive sandbox

Open `interactive/interactive_t.html` in a browser (rigid-body drag/rotate,
scroll-wheel rotates while dragging). After changing a config, refresh its
copy of the parameters:

```bash
ANT_SWARM_CONFIG=configs/rl/pnas_kin_geo.yaml python interactive/gen_web_config.py
```

## When a node fails

Checkpoints save every 50k steps; success JSONs are written immediately, so a
`NODE_FAIL` loses at most 50k steps. Resume by warm-starting a new job:

```bash
sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml \
    run.init_from=storage_local/<dead run>/checkpoints/sac_<last>_steps.zip
```

Note: for SAC this restores weights, not the replay buffer.
