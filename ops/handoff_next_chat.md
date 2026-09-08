# Handoff — where we are (2026-09-08)

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
4. **Pure decentralised RL from zero (MAPPO): ~30%** held-out, first non-zero
   result ever here; plateaus there after 40M steps (§10).
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
| DAgger distillation (the 91% swarm) | `scripts/rl/dagger_shared_ant.py --config configs/rl/marl_5ants_spread.yaml --history 4` |
| single-agent residual RL | `scripts/rl/finetune_bc.py --config configs/rl/ft_residual.yaml` |

## Key checkpoints

| policy | path |
|---|---|
| single-ant BC, corrected goals (the oracle's brain) | `storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt` |
| single-ant residual RL, 97% | `storage_local/ant__20260906_0950__240957__finetune_bc_residual/best.zip` |
| swarm, distilled, 91% held-out | `storage_local/ant__20260907_1256__241627__dagger_shared_n5_h4/shared_ant_policy.pt` |
| swarm, distilled + PPO, 84% | `storage_local/ant__20260908_0327__241788__marl_v2_warm_h4/best.pt` |
| swarm, pure RL, 31% | `storage_local/ant__20260908_0327__241789__marl_v2_h4/best.pt` (continuation 241856: 29%) |

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
- Smoke-test every patched script before `sbatch`, and make the guard fail on a
  timeout (exit 124) as well as on a traceback. Both have bitten.
- Evaluate on seeds training never touched. Best-of-N on one fixed block
  inflates: 98% became 91%, 100% became 84%.

## W&B

Account `kakooee` (key in `~/.netrc`; `WANDB_ENTITY=kakooee` exported). All
project `.env` files were aligned on 2026-09-07. The pasted key should be
rotated. `scripts/tools/wandb_backfill.py` pushes a finished run's
`results.json`. Runs: `https://wandb.ai/kakooee/ant_swarm` (pure RL:
`runs/v5edvbtn`; distilled swarm: `runs/pju6bigj`).

## Next steps (in rough order)

1. **Unseen layouts** — every swarm policy is 0% off its training layout. The
   fix needs each ant's force to depend on where the *others* sit, which its
   row does not contain (chapter 04 §8). Untested: training across layouts
   with the oracle labeller; team-size staging now that workers are distinct.
2. **Pure-RL plateau at 30%** — untested levers: entropy schedule, larger
   rollouts per update, reward scaling, team-size staging 2→3→5.
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
