# Ant-Swarm Ops Run Guide

How to run, monitor, and post-process experiments. All commands run from the
repo root on the cluster login node, with the `roboverse` conda env:

```bash
cd ~/ant_swarm
conda activate roboverse
```

```
ops/
├── sb_train.sh            # SLURM training job (a100, 1 day) — the main launcher
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
python scripts/rl/train_sac.py --config-name pnas_kin_geo \
    run.eval=true run.eval_model=storage_local/<run>/checkpoints/best/best_model.zip
```

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
