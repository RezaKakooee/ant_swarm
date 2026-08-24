# Self-learning experiments: no teacher, no hand-made curriculum

Question (2026-08-24): can the agent learn the maze itself, with sparse reward
and a tiny solution space — no geodesic field, no pose-path curriculum?

## The usual options in RL

| family | idea | fit for our maze |
|---|---|---|
| 1. HER (hindsight replay) | Relabel every failed episode as a success for the goal it actually reached. The agent always gets learning signal. | **Best fit.** Our task is goal-conditioned already. HER builds an implicit curriculum: first it learns to reach easy places, and the frontier slowly crosses the slits. |
| 2. Intrinsic reward (RND, curiosity, count-based) | Add a bonus for visiting states the agent has not seen. | Good helper. Pushes the load into the corridor "for fun". Alone it rarely finishes a precise maneuver. |
| 3. Go-Explore | Archive visited states, teleport back to the frontier, explore from there. Then make the found solution robust. | Very strong for exactly this problem type. But it exploits the resettable simulator — philosophically it *rediscovers* our BFS path instead of being given it. |
| 4. Demonstrations (BC + RL, SACfD) | A few human demos seed the replay buffer; RL refines. | Very practical for us: the interactive sandbox already exists — ~10 recorded plays would do. But it is a teacher again, just a human one. |
| 5. Auto-curriculum (reverse from goal, Goal-GAN) | The algorithm generates start states or goals of medium difficulty by itself. | Works — but it is still a curriculum, only machine-made. Depends on what "remove curriculum" means. |
| 6. Model-based RL (Dreamer, MPC) | Learn a world model, plan through it. | Sample-efficient, but planning does not solve exploration by itself. Heavy machinery for this task. |
| 7. Better noise (pink noise, SDE, options) | Structured exploration noise instead of white noise. | Cheap to try, small effect alone. |

## What we implemented (all config switches)

- **HER**: `sac.her.enabled` — `GoalObsWrapper` (scripts/rl/goal_env.py) + SB3 `HerReplayBuffer`. Relabeled goals may be anywhere the load reached. Verified: 25.9% of sampled transitions carry positive reward even under a random policy. Pitfall found: `learning_starts` must exceed `env.max_steps`.
- **Intrinsic reward**: `env.intrinsic.enabled`, `mode: count | rnd` (scripts/rl/intrinsic.py). Pose-based, training env only, works for SAC and PPO, composes with HER. Verified: bonus decays with familiarity in both modes.
- **Go-Explore phase 1**: `go_explore:` section + `scripts/rl/run_go_explore.py` (module scripts/rl/go_explore.py). Archive over the pose grid, return-then-explore with bit-exact restore-by-replay. Solutions are saved in the training success-JSON format, so phase 2 = `run.seed_successes_from`. Sanity run: frontier past the first wall (x=0.82) in 30k random steps.

The full feature-switch table lives in the header of `configs/rl/gen_h_her_sparse.yaml`.

## Experiments (running, launched 2026-08-24)

| variant | job | method | budget |
|---|---|---|---|
| H | 21281398 | sparse + HER | 10M steps / 3 days |
| I | 21281502 | sparse + HER + count intrinsic | 10M steps / 3 days |
| J | 21281670 | Go-Explore phase 1, random actions | 3M steps / 6 h |
| K | 21282433 | sparse + HER + structured noise (gSDE, `sac.use_sde`) | 10M steps / 3 days |
| L | 21282577 | sparse + HER + intrinsic + gSDE (everything combined) | 10M steps / 3 days |

Reading the ablation: I climbs but H does not → the bonus pushes the frontier
through the first slit. Both climb → HER alone suffices. Neither climbs → we
learned the honest limit; next step is J's solutions seeding a policy
(`run.seed_successes_from`) or human demos from the sandbox.

Results: pending.
