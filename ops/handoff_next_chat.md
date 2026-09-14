# Handoff — project closed (2026-09-14)

**The project was closed on 2026-09-14.** The final summary with every number
to quote is chapter 04 §16 (`docs/project_journey/04_*.md`). This file says
where everything is, in case the project is reopened. Simple English.
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
   **Ten ants (§13): 92% at the best checkpoint (3.8M steps), 80% at the
   end.** As fast as five ants per step, then a slow decline: mean push
   falls to 0.25, log-std to -4.1, training return drops too. Config
   `marl_v2_10ants.yaml`, job 242469.
   **One network per ant (§14): 5 ants 98%, 10 ants 94%** held-out. Slower
   per step than a shared network, same end level, and no late decline for
   10 ants. Switch `--independent-actors`.
   **No geodesic field (§15): 0%** for PPO and for SAC with a success
   buffer, 5 independent ants, 20M steps. Chapter 02's verdict holds.
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
| one network per ant, own init, small width/activation diversity | `train_marl_v2.py --independent-actors` (§14); also `train_masac.py` (default on) |
| multi-agent SAC with success replay buffer | `scripts/rl/train_masac.py --config <yaml> --utd 0.05 --batch 512 --init-alpha 0.01` (§15) |
| no geodesic field, Euclidean reward | `configs/rl/marl_v2_5ants_nomap.yaml` (§15) |
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
| swarm, 10 ants, 92% best / 80% final held-out (§13) | `storage_local/ant__20260911_0016__242469__marl_v2_h4_10ants/best.pt` (3.8M steps), `final.pt`; `heldout_200.json` has every checkpoint |
| swarm, 5 independent networks, **98%** held-out (§14) | `storage_local/ant__20260911_1512__242601__marl_v2_h4_ind/final.pt` (`best.pt` is only 85%) |
| swarm, 10 independent networks, 94% held-out (§14) | `storage_local/ant__20260911_1512__242603__marl_v2_h4_10ants_ind/final.pt` (`best.pt` at 9M: 97%) |
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
  --mem=48G` (no GPU). `ops/fair_submit.sh <config> <tag> [args]` submits one
  such run plus a held-out eval job of every checkpoint after it.
  `FAIR_SCRIPT=scripts/rl/train_masac.py` picks the SAC trainer; `FAIR_MEM=40G`
  lowers the memory request (shared-actor runs use ~10 GB, the 5-envs-per-worker
  run used 33 GB). The SAC run needs the full 24 h for 20M steps at utd 0.05.
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
swarm `runs/cn1wjwl9`; §12 single ×5: `runs/dini8iz1`; §13 ten ants:
`runs/17ox4c9t`; §14 independent 5/10: `runs/gcdhpkkt`, `runs/bh8mzgt6`;
§15 no map PPO/SAC: `runs/c3q0urrz`, `runs/93h8spsf`).

## If the project is reopened

Three things are open. Everything else is done or was shown not to work.

1. **Unseen attachment layouts** (0% for every swarm policy, chapter 04 §8).
2. **Learning without the geodesic field** (0%, chapter 04 §15 and chapter
   02). Needs a new idea for the signal, not another algorithm.
3. **Seeds.** Every number is one seed. Run 3 seeds of the three headline
   runs before publishing: single ×5 (100%), 5 shared (96%), 5 independent (98%).

## Next steps as they stood before closing (kept for reference)

0. **Why do ten shared-network ants decline after 5M steps (§13)?** Peak
   94%, final 80%. Ten independent networks do not decline (94% at 20M, §14),
   so weight sharing is part of it. Levers not tried on the shared run:
   entropy floor, lower learning rate after the peak, a second seed. Also
   open from §12.2: why the 5-ant swarm is faster early than the single ant
   at equal rows and updates (93% vs 80% at 3M steps).
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
- `best.pt` is the first checkpoint to hit the top 30-episode score and is
  often worse held-out than `final.pt` (85% vs 98% in §14). Quote the held-out
  table, never `best.pt` alone.
- The held-out eval job needs `results.json` to find the config. A run that
  hits the time limit has none; `fair_submit.sh` now passes `--config`.
- Without the geodesic field, nothing learns the slit (§15, chapter 02). Do
  not spend more nodes on "pure" reward variants without a new idea for the
  signal.
- With `--envs-per-worker E`, `steps` on the W&B x-axis are worker steps. Env
  steps are `E × steps` (logged as `samples/env_steps`). Compare runs with
  different E on `samples/actor_rows`, not on `steps`.
