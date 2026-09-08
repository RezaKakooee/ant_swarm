# Handoff — where we are (2026-08-25)

Short state file for the next chat.
Full stories: `docs/project_journey/01_goal_generalisation.md` and
`02_self_learning_limit.md`. Experiment tables:
`docs/GENERALISATION_EXPERIMENTS.md`, `docs/SELF_LEARNING_EXPERIMENTS.md`.

## Status in three lines

1. **Single agent, fixed task: solved** (100%, ants' maneuver verified by replay).
2. **Random start + random goal: solved** (99-100%) — two-leg reward
   (`geodesic_exit`) + continuous goals; from scratch in one run (variant E);
   both routes possible (`curriculum.route: up` → variant G).
3. **Self-learning without a teacher: fails, measured.** HER, intrinsic bonus,
   gSDE, Go-Explore — every method stalls at frontier x=0.900 (second slit).
   The task needs a teacher.

## Everything is a config switch

Master table in the header of `configs/rl/gen_h_her_sparse.yaml`. Highlights:
reward (`env.reward_mode`: sparse/shaped/geodesic/geodesic_exit), curriculum
(`curriculum.enabled`, `route`, `mirror_anchors`, `final_spawn_x_range`),
random start (`spawn.resample_each_reset`), random goal (`goal.random_box`),
HER (`sac.her.enabled`), intrinsic (`env.intrinsic`), gSDE (`sac.use_sde`),
success replay (`sac.success_replay`), resume warm-up
(`sac.resume_warmup_steps`), Go-Explore (`go_explore:` + `run_go_explore.py`).

## Key checkpoints (each under `<run>/checkpoints/best/`)

| what | run dir in `storage_local/` |
|---|---|
| E — random everything, from scratch, down-route | `ant__20260822_1754__21107972__*gen_e_scratch_randall` |
| G — same but up-route | `ant__20260824_1124__21243500__*gen_g_uponly` |
| F — mirror curriculum, 100% benchmark | `ant__20260823_1120__21144812__*gen_f_mirror` |
| single-goal source run | `ant__20260814_1230__local-1427853__*pnas_dyn_geo_v2__best` |

## Cluster habits

- CPU beats GPU here (pure-NumPy env, tiny MLP): `--partition=scicore
  --qos=1day` (or `1week`), `--gres=NONE`, `ANT_SWARM_FORCE_CPU=1`,
  ~40-56 fps. `scicore-fast`/`fast` caps at 6h ≈ 1.2M steps.
- Launch: `sbatch ops/sb_train.sh train_sac configs/rl/<cfg>.yaml`.
- W&B: remote dashboard is the record; local files go to job scratch.

## Next steps (in rough order)

1. **Human demos**: record ~10 plays in `interactive/` sandbox, convert to
   success JSONs, seed via `run.seed_successes_from` (needs
   `sac.success_replay: true`, HER off), sparse SAC. Tests "human teacher,
   no BFS". No converter sandbox→JSON exists yet — must be written.
2. **Multi-agent** (`ants.n >= 2`, dynamic): the real goal. Central controller
   first; per-ant observation layout is already built.
3. Optional: fold chapters 01+02 into the blog; publish the toolkit to the
   public repo (`python tools/publish.py`, then manual push — blogs/docs/ops
   stay private).

## Traps already paid for (do not relearn)

1. Distance shaping is deceptive here; per-goal geodesic fields do NOT
   generalise (0% on unseen goals). Use `geodesic_exit` + continuous goals.
2. SB3 resume starts with an EMPTY buffer and the saved LR — always use
   `sac.resume_warmup_steps` (50k) and check the LR log line.
3. HER needs `learning_starts` > `env.max_steps` (500).
4. Judge runs by per-condition metrics (per-goal, frontier_x), never by saved
   videos (biased sample) or a mixed rolling average.
5. The BFS grid has a tiny down-route bias (asymmetric discretisation); it
   decides ties but cannot overturn a learned skill.
6. `pkill -f <pattern>` kills your own shell if the pattern is in the command
   line. Analysis scripts live in `storage_local/analysis/`, not /tmp.
7. Old runs held 400k+ small files; big cleanup was running 2026-08-25 —
   check `du -sh storage_local` before assuming space.
