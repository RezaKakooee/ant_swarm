# configs

YAML configs, grouped like `scripts/`:

- `rl/config.yaml` — the project default (the single source of truth the
  `ant_swarm` package loads when `ANT_SWARM_CONFIG` is not set).
- `rl/pnas_*.yaml` — the PNAS-replica experiment variants
  (kin+geodesic, kin+sparse control, dyn+geodesic).
- `heuristic/random_agent.yaml` — config for the random baseline.
- `il/bc.yaml` — placeholder for future behavior-cloning training.

Per-run configs used by past experiments are snapshotted inside each run dir
(`storage_local/<run>/code/config.yaml`) — those, not these, are what replay
tooling reads.

Configs are plain yaml, loaded with OmegaConf; entry scripts compose them
with Hydra, so any value can be overridden on the command line:

    python scripts/rl/train_sac.py --config-name pnas_kin_geo env.reward_mode=sparse
    ANT_SWARM_CONFIG=configs/rl/pnas_kin_geo.yaml python scripts/rl/train_sac.py
    sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml sac.timesteps=5e6
