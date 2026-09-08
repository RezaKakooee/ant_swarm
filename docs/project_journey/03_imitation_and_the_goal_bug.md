# 03 — Imitation learning, and the one line that hid it

Period: 2026-08-25 → 2026-09-06.
Question: chapter 02 ended with "the task needs a teacher". With 34,777 expert
demonstrations, can behavioural cloning thread both slits?
Answer: **yes — 88-95% with BC alone, 97% after residual RL fine-tuning. But
it took eleven days, because a stale goal reference made every BC run look
like a failure.**

Evidence used for every number:

| source | what it holds |
|---|---|
| `storage_local/datasets/successes_v1/dataset.npz` | 34,777 demos, 4,914,568 transitions |
| `CLAUDE_HANDOFF.md` | the original problem statement and the pre-fix results table |
| `CODEX_FINDINGS.md` | the root-cause diagnosis and the fix |
| `CLAUDE_FINDINGS.md` | the failed diagnostics, and which of them are void |
| `storage_local/ant__20260905_1322__240805__train_goal_bc/` | checkpoint, config, `results.json`, rollout GIFs |
| Slurm jobs 240805 (train), 240868 + 240887 (fresh-start eval) | run records |
| `storage_local/ant__2026090[56]_*__finetune_bc_{residual,gaussian}` | the four RL fine-tuning runs |
| `scripts/rl/bc_finetune.py`, `configs/rl/ft_*.yaml` | the fine-tuning module and its two arms |

## Key terminology

(Terms from chapters 01 and 02 are not repeated.)

- **BC** — behavioural cloning: supervised regression from observation to expert action. No reward, no rollouts during training.
- **Open-loop replay** — feeding a demo's recorded action sequence back into the env. Succeeds 100/100, because the env is deterministic.
- **Covariate shift** — the policy drifts into states the expert never visited, where its training says nothing.
- **Start-pose group** — all demo episodes sharing one initial pose. The unit that must be held out, because episodes inside a group are near-duplicates.
- **Stale goal** — an observation built from a goal value captured earlier and never refreshed.
- **Residual RL** — the BC policy is frozen; RL learns only a bounded correction on top, `a = clip(a_bc + 0.15·tanh(z))`. At zero correction it is exactly the BC policy.
- **Anchor** — a penalty `MSE(π_mean, π_bc)` on replay states, holding the fine-tuned policy near the frozen reference. For a fixed-variance Gaussian it is the KL up to a constant.
- **Best-by-eval** — evaluate every 20k steps on fresh starts and keep the best checkpoint, so fine-tuning cannot return a policy worse than it started with.

## 1. Setup

**Believed:** the demos were the teacher chapter 02 asked for, so BC should work
and any gap would be a covariate-shift problem.

**Did:** BC on 0.5M and 4.9M transitions, perturbation-augmented BC, DAgger,
HG-DAgger, frame-stacked BC (K=4), warm-started SAC.

**Result:** open-loop replay 100%. Every closed-loop policy 0-5%. The load
reached slit 1 and stopped. This is the state recorded in `CLAUDE_HANDOFF.md`.

## 2. Eleven days of diagnosing the wrong thing

Six hypotheses were tested and all came back negative:

| hypothesis | test | result |
|---|---|---|
| the observation lacks a contact cue | predict torque sign from obs | AUC 0.93 — cue present |
| MSE averages the bimodal recovery torque | 21 discrete bins + cross-entropy | magnitude fixed, success unchanged |
| the demos mix two hidden strategies | add the route label | AUC 0.934 → 0.935 |
| compounding error per step | action chunking, H = 1, 8, 32, 64 | identical |
| the policy freezes in a deadlock | action-noise sweep | jamming 100% → 53%, success still 0% |
| more data | 0.57M → 4.91M transitions | identical |

Only one intervention ever moved the number: replacing the goal vector in
`obs[4:6]` with a direction from the geodesic field took success from 0% to 84%.
That was read as evidence for waypoint conditioning. It was evidence that
`obs[4:6]` was broken. **That was the moment to check the input, and it was
missed.**

## 3. The root cause

`ant_swarm/observation.py` held the goal by reference at construction:

```python
self.goal = layout.goal          # snapshot of the reference
```

Every replay and evaluation script then did `env.layout.goal = np.asarray(...)`,
which **rebinds** `layout.goal` to a new array. `obs_model.goal` kept pointing at
the original. Reward, termination and rendering used the requested goal. The
policy's observation did not.

Reconstructing the goal the policy actually saw, from `obs[4:6]`:

| quantity | value |
|---|---|
| goal seen by the policy | x = 1.51753 ± **0.00000**, y = 0.52424 ± **0.00000** |
| true goal range | x = 1.050 … 1.550, y = 0.100 … 0.620 |
| mean error | 0.315 m |
| max error | 0.627 m |

Zero variance across 4.9M transitions. Every BC policy in this project was asked
to reach a moving goal while being told the goal never moves.

A second defect followed: `np.asarray(goals[i])` aliased the dataset row, so a
later reset wrote through it and changed the in-memory demo target. The dataset
file on disk was not damaged.

## 4. The fix and the corrected result

`Layout` now owns its goal array and its setter writes in place;
`ObservationModel.goal` is a property reading the live value.
`env.reset(options={'init_pose': ..., 'goal': ...})` sets pose, momentum, step
counter, previous distance, reward state and goal together.

Retrained plain BC, unchanged in every other respect:

| test set | seeds | success | mean distance | jammed |
|---|---|---|---|---|
| withheld demo start-pose groups | — | 91 / 100 | 0.1031 m | 9 |
| fresh random starts | 10000-10099 | 88 / 100 | 0.1106 m | 12 |
| fresh random starts | 20000-20099 | **95 / 100** | 0.0780 m | 5 |
| expert action replay (reference) | — | 100 / 100 | 0.0518 m | 0 |
| old checkpoint, correct goals at inference | — | 0 / 100 | 0.5999 m | 99 |

MLP, one action per step, 4,885,903 training transitions, 40 epochs, batch 4096,
lr 1e-3, 209 s on one GPU. Sparse reward, unchanged full maze, ordinary 27-D
observation. **No geodesic field, no waypoints, no curriculum, no action noise.**
The last row is the control: the same architecture trained on stale goals scores
0% even when given correct observations at test time.

## 5. A second bug: the evaluation tested one start pose

`evaluate_full()` used `for ep in range(n)` with `init_poses[ep]`. The dataset is
stored run by run and repeats start poses:

| range | distinct start poses |
|---|---|
| first 20 episodes | **1** |
| first 50 episodes | **1** |
| first 1000 episodes | 486 |
| all 34,777 | 15,758 |

534 episodes are needed before 20 distinct starts appear. So every success rate
in `CLAUDE_HANDOFF.md` measured one start pose with N different goals. All
evaluators now select distinct start poses and hold out whole groups.

## 6. RL fine-tuning: residual works, full-policy collapses

The remaining BC failures all jam at slit 1, and the demos contain no wedged
states, so BC cannot learn recovery. Sparse-reward RL from scratch never found a
success (chapter 02) — but starting from a 94% policy, nine rollouts in ten
succeed and the critic has signal from step one. Two arms, one config switch
(`finetune.mode`) apart, everything else shared:

| arm | actor | freedom |
|---|---|---|
| `residual` | frozen BC + bounded correction (±0.15) | cannot destroy the base |
| `gaussian` | ordinary SAC actor, distilled from BC | full |

Both anchored to the frozen BC, SAC, sparse reward, 400k steps, 100 fresh
random starts (seed 30000). No field, no curriculum, no new demos.

**First run — both arms made things worse.**

| arm | before | after |
|---|---|---|
| residual | 94.0% | 64.0% |
| gaussian | 56.0% | 0.0% |

Three causes, all self-inflicted: the critic is random at step 0 under sparse
reward and the actor followed it anyway; the anchor was annealed 1.0 → 0.1, so
the policy was barely held after 200k steps; and the *last* checkpoint was
returned, not the best. (The gaussian arm also started at 56%, not 94%: 3k
distillation steps did not reproduce BC.)

**Second run — three fixes, same method, same steps.**

| fix | effect |
|---|---|
| `critic_warmup_steps: 50000` | critic trains alone first; actor frozen |
| anchor held constant | no drift after the anneal |
| best-by-eval checkpoint | cannot return worse than the start |

| arm | before | last | best |
|---|---|---|---|
| **residual** | 94.0% | 97.0% | **97.0%** |
| gaussian | 87.0% | 0.0% | 87.0% |

Residual: mean distance 0.0745 → 0.0662 m, three of six failures fixed, and
`last == best`, so the policy was stable throughout. Gaussian still collapsed
to 0% — full freedom plus exploration noise breaks a 15 mm slit crossing — and
was saved only by the checkpoint rule. Checkpoint:
`ant__20260906_0950__240957__finetune_bc_residual/best.zip`. Rollout GIFs in its
`renders/`.

**Why this matters beyond the 3 points.** It is the only RL that has ever
improved anything in this project, and it did so by *not* trusting RL: a
frozen base, a bounded correction, and a safety net. That recipe is what
chapter 04 proposes for the swarm.

**A cluster fact found on the way.** The `antswarm` env's torch is built for
CUDA 13.0 (driver ≥ 610). Nodes calc-g-004/008/010 run driver 575, report no
CUDA, and torch silently falls back to CPU — the first fine-tuning run went
that way unnoticed. Working GPU nodes: calc-g-002, calc-g-003. `ant_swarm/compute.py`
now fails loudly on `device: cuda` when CUDA is absent; every config defaults to
`cpu`, which is right for RL here (the NumPy env dominates) and wrong for BC on
cached data (209 s on GPU).

## What worked

1. **Reconstructing an observation channel from the data itself.** Recovering the
   goal from `obs[4:6]` and checking its variance took one short script and
   settled the question outright.
2. **A control arm.** Evaluating the old checkpoint with correct observations
   (0/100) proved the fault was in training data, not at inference.
3. **Two independent test sets plus an exact-repeat check**, so 88-95% is not one
   lucky split.
4. **Caching the replay.** Rebuilding observations takes 4.3 h on one core and
   ~5 min on 60. Every later experiment ran in minutes.
5. **Reporting before / last / best.** The gaussian collapse (87% → 0%) is
   visible in the table instead of hidden behind a single "after" number.

## Remaining limitations

- **Contact recovery is unsolved.** All remaining failures jam near the **first**
  slit, ending 0.50-0.69 m from the goal. None fail later in the maze. The demos
  contain only successes, so BC never sees a wedged load. Rollout GIFs (5
  successes, 5 failures) are in `storage_local/ant__20260905_1322__240805__train_goal_bc/fresh_starts_seed20000/`.
- **One training seed** for BC and for each fine-tuning arm.
- **Same maze.** No evidence about a different wall layout.
- **The negative results in section 2 are void**, not disproved. They all ran on
  stale-goal data. Anything reused from them — especially the action-noise sweep,
  the only one that speaks to contact recovery — must be repeated on corrected
  data.

## Exact state right now (2026-09-06)

- **Running:** nothing.
- **Artifacts:** `storage_local/ant__20260905_1322__240805__train_goal_bc/` holds `bc_goal_fixed.pt`, `config.yaml`, `results.json`,
  `training.json`, and both fresh-start evaluations with GIFs.
- **Caches:** `storage_local/cache/replay_full_goals_v2.npz` is correct and audited
  (0 bad rows). `replay_4000.npz` and `replay_full.npz` carry stale goals — do not
  use them.
- **Tests:** 17 pass, including new regressions for goal synchronisation, dataset
  ownership, deterministic reset and train/test group separation.
- **Best single-agent policy:** residual RL, 97% on 100 fresh starts —
  `ant__20260906_0950__240957__finetune_bc_residual/best.zip`.
- **Verdict feeding forward:** the teacher chapter 02 asked for was enough. What
  the task never needed was a new algorithm. What it needed was a correct
  observation — and then a bounded correction on top. Next: the swarm (chapter 04).
