# scripts

Entry points, grouped by method family (library code stays in `ant_swarm/`):

- `rl/` — RL training and its tooling: `train_ppo.py`, `train_sac.py`,
  shared SB3 callbacks (`train_utils.py`), and `gen_geodesic_field.py`
  (precomputes the geodesic-reward distance field).
- `il/` — demonstration-data pipeline: replay/render saved success
  trajectories (`render_success.py`), consolidate (`combine_successes.py`),
  convert legacy files (`convert_successes.py`). BC/LfD training will land here.
- `heuristic/` — non-learning baselines: `random_agent.py`.

All scripts are run from the repo root, e.g.:

    python scripts/rl/train_sac.py
    python scripts/heuristic/random_agent.py heuristic.episodes=3
    python scripts/il/render_success.py <success.json>

Configs live in `configs/<family>/` (OmegaConf/Hydra). Select a variant and
override values from the CLI:

    python scripts/rl/train_sac.py --config-name pnas_kin_geo sac.timesteps=5e6
    python scripts/heuristic/random_agent.py heuristic.episodes=3

or via the env var / SLURM wrapper:

    ANT_SWARM_CONFIG=configs/rl/pnas_kin_geo.yaml python scripts/rl/train_sac.py
    sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml [key=value ...]

Each run writes a `train.log` (loguru) plus a `code/config.yaml` snapshot of the
RESOLVED config (with CLI overrides applied) into its run dir.
