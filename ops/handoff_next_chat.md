# Handoff — where we are (2026-09-10)

Short state file for the next chat. Simple English.
Full stories: `docs/project_journey/01`–`04`. Diagnoses: `CLAUDE_FINDINGS.md`,
`CODEX_FINDINGS.md`. The original problem statement (now partly outdated):
`CLAUDE_HANDOFF.md`. Previous version of this file: `handoff_next_chat_2026-08-25.md`.

## Status in five lines

1. **Single agent, random start + random goal: solved.** BC on the 34,777 demos
   gives 88-95%; residual RL on top gives **97%** (chapter 03). Geodesic reward
   alone, from scratch, no curriculum, also reaches 100% (chapter 01 addendum).
2. **Swarm, 5 ants, each sees only its own row, distilled from an oracle: 91%**
   held-out (chapter 04 §9). The oracle = single-ant BC + least-norm force split.
3. **Same swarm, then PPO with an anchor: 84%.** RL did not help it (§10).
4. **Pure decentralised RL from zero (MAPPO): 96%** held-out for 5 ants, 88%
   for 1 ant, same recipe and seed (chapter 04 §11). An earlier seed gave 30%,
   so seed variance is large. Configs `marl_v2_5ants.yaml` / `marl_v2_1ant.yaml`.
   The swarm's actor is not bigger (shared network); it gets 5 actor rows per
   env step instead of 1. **§12 tested this.** The single ant with 5 envs per
   worker (same rows, same updates as the swarm) ends at **100%** held-out
   (job 242283), above the swarm's 96%. So the 88% was a sample budget. The
   swarm is still faster early: 93% vs 80% held-out at 3M steps with equal
   rows. Quote: one ant 100%, swarm 96%, at equal rows.
5. **Unseen attachment layouts: 0%** for every swarm policy. Not addressed.

## Constraints the user holds

No curriculum over the maze. No map / waypoints at test time. Each ant sees
only its own observation row. No ant-count input. A teacher at *training*
time (demos, the oracle) has been accepted; say so when you use one.

## Everything is a config switch

| what | where |
|---|---|
| ant count and explicit attachment points | `ants.n`, `ants.offsets` (spread layout: `configs/rl/marl_5ants_spread.yaml`) |
| absolute velocity (not divided by ant count) | `env.velocity_scale_ants: 1` |
| contact-point velocity block (obs 27→29) | `env.observe_contact_velocity: true` |
| device | `device: cpu|cuda|auto` in fine-tune configs; `cuda` fails loudly if absent |
| MARL v2 arms | `scripts/rl/train_marl_v2.py --warm-start <single-ant BC .pt> | --stages 2,3,5 | --init-actor <pt> | --sanity-reward push` |
| envs per worker (5 actor rows per worker step for one ant) | `train_marl_v2.py --envs-per-worker 5` (§12); `steps` = worker steps |
| periodic checkpoints and resume | `train_marl_v2.py --ckpt-every 1000000` (default on) writes `ckpt_<steps>.pt`; `--resume <ckpt>` continues in the same folder |
| held-out scoring of any MARL v2 `.pt` | `scripts/rl/eval_marl_v2.py <run>/final.pt --history 4` (seeds 60000+, 70000+, 100 each) |
| success vs actor rows (§12.1 table + figure) | `scripts/tools/samples_curve.py --run single=<dir> --run swarm=<dir> --out <png>` |
| DAgger distillation (the 91% swarm) | `scripts/rl/dagger_shared_ant.py --config configs/rl/marl_5ants_spread.yaml --history 4` |
| single-agent residual RL | `scripts/rl/finetune_bc.py --config configs/rl/ft_residual.yaml` |

## Key checkpoints

| policy | path |
|---|---|
| single-ant BC, corrected goals (the oracle's brain) | `storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt` |
| single-ant residual RL, 97% | `storage_local/ant__20260906_0950__240957__finetune_bc_residual/best.zip` |
| swarm, distilled, 91% held-out | `storage_local/ant__20260907_1256__241627__dagger_shared_n5_h4/shared_ant_policy.pt` |
| swarm, distilled + PPO, 84% | `storage_local/ant__20260908_0327__241788__marl_v2_warm_h4/best.pt` |
| swarm, pure RL, 96% | `storage_local/ant__20260909_2337__242213__marl_v2_h4/final.pt` |
| single ant, pure RL (same recipe), 88% | `storage_local/ant__20260909_2334__242212__marl_v2_h4/final.pt` (note 2334, not 2337) |
| single ant, 5 envs per worker, **100%** held-out (§12.2) | `storage_local/ant__20260910_1013__242283__marl_v2_h4_x5/final.pt`; `ckpt_*.pt` every 1M steps; `heldout_200.json` has every checkpoint |

Caches: `storage_local/cache/replay_full_goals_v2.npz` is the corrected 4.91M
transitions. `replay_4000.npz` and `replay_full.npz` carry the stale goal — do
not use.

## Cluster habits

- GPUs come from Slurm, not the login node. Only **calc-g-002 and calc-g-003**
  have a driver new enough for this env's torch (CUDA 13); on the others torch
  silently falls back to CPU. Use `-w calc-g-002`.
- RL here is env-bound (NumPy env): use many CPU workers, not a GPU. BC on the
  cache is the opposite (209 s on a GPU).
- `sbatch ops/sb_train.sh <script.py> "" <args>` takes an explicit script path.
  The §11/§12 MARL runs used a plain `sbatch --wrap` on `-p performance -c 32
  --mem=48G` (no GPU); `ops/fair_x5_submit.sh` is the exact form, and
  `sacct -j <id> --format=SubmitLine%300` recovers any past submit line.
- Smoke-test every patched script before `sbatch`, and make the guard fail on a
  timeout (exit 124) as well as on a traceback. Both have bitten.
- Evaluate on seeds training never touched. Best-of-N on one fixed block
  inflates: 98% became 91%, 100% became 84%.

## W&B

Account `kakooee` (key in `~/.netrc`; `WANDB_ENTITY=kakooee` exported). All
project `.env` files were aligned on 2026-09-07. The pasted key should be
rotated. `scripts/tools/wandb_backfill.py` pushes a finished run's
`results.json`. Runs: `https://wandb.ai/kakooee/ant_swarm` (pure RL:
`runs/v5edvbtn`; distilled swarm: `runs/pju6bigj`; §11 single `runs/uote3o0p`,
swarm `runs/cn1wjwl9`; §12 single ×5: `runs/dini8iz1`).

## Next steps (in rough order)

0. **Why is the swarm faster early?** §12.2 left this open: at 3M steps and
   equal rows and updates, swarm 93% vs single ×5 80%. Candidate: five pushes
   at five points give torque and translation at once. A test: the swarm run
   with periodic checkpoints (now default) and a second seed of each arm.
1. **Unseen layouts** — every swarm policy is 0% off its training layout. The
   fix needs each ant's force to depend on where the *others* sit, which its
   row does not contain (chapter 04 §8). Untested: training across layouts
   with the oracle labeller; team-size staging now that workers are distinct.
2. **Seed variance in pure RL** — one seed 30%, another 96%. Run 3-5 seeds
   before quoting a number. The §12 gaps (100% vs 96% at 20M steps; 93% vs
   80% at 3M steps) are single-seed numbers too.
3. **Contact recovery** — the residual-RL single agent still jams at slit 1 in
   3% of episodes; the swarms' failures are the same jam.
4. Delete the three stray `gen-xr` W&B copies (needs the old key, only in
   `~/.netrc.bak_*`).

## Traps already paid for (do not relearn)

- `ObservationModel` once held the goal by reference: every cached observation
  had one frozen goal. Fixed; the cache audit in `CODEX_FINDINGS.md` proves it.
- The old evaluator used the first N demo episodes — all one start pose.
  `pick_eval_episodes()` fixes it.
- A worker pool passes the same `initargs` to every process: thirty workers were
  one trajectory copied thirty times. `train_marl_v2.py` seeds per worker now.
  A fit that is "too good" (MSE 0.00000) is the tell.
- Regress the force **vector**, never the push **angle**: a small vector error is
  tens of degrees when |f| ≈ 0.18.
- A distilled student is not an oracle: DAgger must be labelled by the split
  controller, not by a copy of itself.
- Velocity divided by `n_ants` hides the team size and breaks transfer across N.
- PPO on a good policy without an anchor collapses it (87% → 0%); with an
  anchor it is neutral at best (91% → 84%). Distillation is the result; RL is
  not the improvement here.
- Landmarks by the load's *tip*, not its centre: the T is 0.33 m long.
- A run folder's timestamp is the *start* minute, not the submit minute: the
  §11 single run is `ant__20260909_2334__242212__...`, the swarm `..._2337__242213__...`.
  Find folders by job id (`ls -d storage_local/ant__*__<jobid>__*`), not by time.
- With `--envs-per-worker E`, `steps` on the W&B x-axis are worker steps. Env
  steps are `E × steps` (logged as `samples/env_steps`). Compare runs with
  different E on `samples/actor_rows`, not on `steps`.
