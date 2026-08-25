# Installation guide

Tested with: Python 3.10, torch 2.7, stable-baselines3 2.7.1, gymnasium 1.3
(the exact pins are in `requirements.txt`). The env is pure Python/NumPy —
no MuJoCo, no compiled extensions. A GPU is NOT needed (CPU is faster here).

## 1. Set up the environment

```bash
conda create -n antswarm python=3.10 -y
conda activate antswarm
cd <repo root>
pip install -r requirements.txt
```

Notes:
- On the cluster we reuse the existing env `roboverse` instead (same pins).
- `wandb` is optional. Without an account, set `run.wandb=false` on every run.

## 2. Verify the install

```bash
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from ant_swarm import AntSwarmEnv, load_config
env = AntSwarmEnv(config=load_config(None), seed=0)
obs, _ = env.reset()
print("env OK, obs size:", obs.size)
EOF
```

Expected output: `env OK, obs size: 25` (27 with `observe_linear_velocity`).

## 3. Generate the geodesic field (needed for geodesic reward modes)

The field is NOT in git (binary, geometry-specific). Regenerate it after any
geometry change:

```bash
python scripts/rl/gen_geodesic_field.py --inflate 0.002 --dx 0.003 --dth 2
```

Check the output says `reachable 100.0%`. Coarser grids silently disconnect.
Sparse-only configs (e.g. `gen_h_her_sparse.yaml`) do not need a field.

## 4. Run a first training (smoke test, ~2 min on CPU)

```bash
export ANT_SWARM_CONFIG="$PWD/configs/rl/gen_e_scratch_randall.yaml"
python scripts/rl/train_sac.py sac.timesteps=5000 run.wandb=false
```

Results land in `storage_local/<run_id>/` (checkpoints, renders, train.log).

## 5. Real runs

| where | command |
|---|---|
| any machine | `bash scripts/train.sh` (auto-builds a missing field) |
| SLURM cluster | `sbatch ops/sb_train.sh train_sac configs/rl/<cfg>.yaml` (CPU: add `--gres=NONE --export=ALL,ANT_SWARM_FORCE_CPU=1`) |

- Config selection: the `ANT_SWARM_CONFIG` env var, or the second sbatch arg.
- All features (reward mode, curriculum, HER, intrinsic bonus, gSDE, random
  start/goal) are config switches — master table in the header of
  `configs/rl/gen_h_her_sparse.yaml`.

## 6. Optional: the interactive sandbox

`interactive/` holds the browser version of the maze (feel the task by hand).
Open its HTML page directly; `gen_web_config.py` regenerates its geometry
from the yaml config.

## Moving to a new server

The repo is path-portable: all code resolves paths relative to the repo root,
and `storage_local/` (results, fields) is created where the repo lives.

1. Clone the repo, do steps 1-3 above (the geodesic field must be rebuilt).
2. No SLURM? Use `scripts/train.sh` — plain bash, uses `$PYTHON` or `python`.
3. SLURM with different partitions? Edit the `#SBATCH` header of
   `ops/sb_train.sh`, or override on the command line
   (`sbatch --partition=... --qos=... ops/sb_train.sh ...`).
4. Different conda setup? Set `ANT_SWARM_CONDA_SH` (path to conda.sh) and
   `ANT_SWARM_CONDA_ENV` (env name) — used by all `ops/*.sh` scripts.
5. Old checkpoints and configs reference `storage_local/...` relatively, so
   copied run folders keep working if placed under the new repo root.

## Common problems

| symptom | fix |
|---|---|
| `ModuleNotFoundError: ant_swarm` | run from the repo root, or `export PYTHONPATH=$PWD` |
| `ModuleNotFoundError: success_replay_buffer` when loading a checkpoint | add `scripts/rl` to `sys.path` before `SAC.load` |
| geodesic field error / wrong geometry | regenerate the field (step 3) |
| HER crash "Unable to sample before the end of the first episode" | set `sac.learning_starts` > `env.max_steps` (500) |
| training uses GPU and is slow | set `CUDA_VISIBLE_DEVICES=""` — CPU is faster for this env |
