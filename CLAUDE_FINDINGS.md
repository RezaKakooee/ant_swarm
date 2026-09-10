# Ant-Swarm — corrected findings, 2026-09-05

**This document was rewritten after `CODEX_FINDINGS.md` identified the root
cause. Most of the earlier version was invalid. What follows separates what
survives from what does not.**

---

## 1. The root cause (found in CODEX_FINDINGS.md, not here)

`ant_swarm/observation.py:46` held the goal by reference at construction:

```python
self.goal = layout.goal          # snapshot of the reference
```

Every replay and evaluation script then did `env.layout.goal = np.asarray(...)`,
which **rebinds** `layout.goal` to a new array. `obs_model.goal` kept pointing at
the original. Distance, termination and rendering used the requested goal; the
policy's observation did not.

Verified against `storage_local/cache/replay_4000.npz`:

| Quantity | Value |
|---|---|
| goal seen by the policy | x = 1.51753 ± **0.00000**, y = 0.52424 ± **0.00000** |
| true goal range | x = 1.050 … 1.550, y = 0.100 … 0.620 |
| mean error | 0.315 m |
| max error | 0.627 m |

Zero variance. Every policy trained on these caches was told the goal never
moves. Fixed upstream: `ObservationModel.goal` is now a property reading the live
`layout.goal`, and `Layout.goal`'s setter writes in place.

**Corrected result: retrained plain BC succeeds on 88-95 of 100 episodes across
three separate test sets.** The task is solved by plain BC. No new algorithm was
needed.

| Test set | Seeds | Success | Mean distance | Jammed |
|---|---|---|---|---|
| Withheld demo start-pose groups | - | 91 / 100 | 0.1031 m | 9 |
| Fresh random starts | 10000-10099 | 88 / 100 | 0.1106 m | 12 |
| Fresh random starts | 20000-20099 | **95 / 100** | 0.0780 m | 5 |
| Expert action replay (reference) | - | 100 / 100 | 0.0518 m | 0 |

Training: MLP, one action per step, 4,885,903 transitions, 40 epochs, batch
4096, lr 1e-3, 209 s on one GPU. Sparse reward, unchanged full maze, ordinary
27-D observation. No geodesic field, no waypoints, no curriculum, no action
noise. The only change was rebuilding the cache with correct goal vectors.

---

## 2. What in the earlier analysis is VOID

Every experiment below trained on observations with a frozen goal, so none of
them is evidence about behavioural cloning.

| Claim previously made | Status |
|---|---|
| "8.7x more data changes nothing" | void |
| "discrete torque bins do not help" | void |
| "action chunking H = 1…64 does not help" | void |
| "the deterministic deadlock is the failure mode" | void — an artifact of a policy chasing a fixed phantom goal |
| "noise unjams but does not help" | void |
| "BC has saturated; try offline RL (IQL / AWAC)" | void, and the recommendation was wrong |
| "root causes 1–4 of the handoff are ruled out" | void — untestable on corrupted data |

The waypoint result (0% → 84%) is also explained by this bug. Substituting a
live-computed vector into `obs[4:6]` bypassed the stale goal channel. Most of
that gain was repairing the broken input, not route knowledge from the map — a
correct goal vector alone gives 91%.

---

## 3. What still stands

### 3.1 The evaluation tested ONE start pose

Independent of the goal bug. `evaluate_full()` in
`scripts/il/train_full_dataset.py` used `for ep in range(n)` with
`init_poses[ep]`. The dataset repeats start poses heavily:

| Range | Distinct start poses |
|---|---|
| first 20 episodes | **1** |
| first 50 episodes | **1** |
| first 1000 episodes | 486 |
| all 34,777 | 15,758 |

534 episodes are needed before 20 distinct starts appear. So the handoff's
"5% SR (1/20)" was one success out of 20 goals from a single start pose.
`CODEX_FINDINGS.md` adopts the same fix (distinct start poses, group-held-out).

### 3.2 Landmarks must use the leading tip, not the centre

The T is 0.3315 m long, so its centre sits ~0.166 m behind the tip that clears a
wall. A rollout ending at centre x = 0.5934 reads as "never reached slit 1", but
0.5934 + 0.166 = 0.759 and slit 1 is at 0.758 — the load is against the wall.

### 3.3 Why the eval was not reproducible

Two identical no-noise runs gave 0.6547 and 0.6351. Cause now known: each
`env.reset()` sampled a fresh random goal, which became the frozen goal for that
run. A symptom of §1.

---

## 4. What I got wrong, and why

The bug was in a file I read in full early on. I traced the goal vector's *use*
(handoff root cause #3: "the goal vector pulls the load into the wall") but never
checked that the value was correct. When the waypoint substitution jumped to 84%,
I read it as evidence for route conditioning; it was evidence that `obs[4:6]` was
broken. That was the moment to check, and I did not.

General lesson for this repo: before comparing algorithms, verify that each
observation channel carries what its name says. A one-line reconstruction of the
goal from `obs[4:6]` would have caught this on day one.

---

## 5. Files added here

Still usable, but any result they produced before the goal fix must be rerun.

| File | Purpose |
|---|---|
| `scripts/il/build_replay_cache.py` | parallel replay of the full dataset (~5 min on 60 cores vs 4.3 h single-core) |
| `scripts/il/train_chunked_bc.py` | action-chunked BC; contains `pick_eval_episodes()` |
| `scripts/il/diagnose_failure_location.py` | where a rollout dies, tip-based landmarks |
| `scripts/il/eval_noise_sweep.py` | action-noise sweep |
| `scripts/il/test_torque_observability.py` | is the torque sign predictable from the observation |
| `scripts/il/train_discrete_torque_bc.py` | A/B: discrete torque head vs MSE |
| `scripts/il/train_waypoint_bc.py` | A/B: waypoint vs goal conditioning |

`storage_local/cache/replay_4000.npz` and `replay_full.npz` carry the stale goal
vectors. Use `replay_full_goals_v2.npz` from `CODEX_FINDINGS.md` instead, or
rebuild.

Cluster note that still holds: GPUs come from Slurm, not the login node.
Replay-only jobs need cores, not a GPU — request CPUs on an idle GPU node with
no `--gres`.

---

## 6. Storage layout fixed

Results were written to `storage_local/experiments/goal_fix_bc/`, which does not
match this repo's convention. Run folders live directly under `storage_local/`
and are named by the run id from `ant_swarm/run_id.py`:

    ant__<YYYYMMDD_HHMM>__<slurm job id | local>__<script>[__single|multi][__config]

Moved to `storage_local/ant__20260905_1322__240805__train_goal_bc/` (training was Slurm job 240805). `storage_local/experiments/`
was removed. `train_goal_bc.py` now calls `build_run_id()` so new runs land in
the right place by themselves; `ANT_SWARM_RUN_ID` from `ops/sb_train.sh` still
takes priority. Paths in `evaluate_goal_bc.py` and `CODEX_FINDINGS.md` were
updated. 17 unit tests pass after the move.

`evaluate_goal_bc.py` gained `--render-count N`, which saves N successes and N
failures as GIFs instead of one of each.

---

## 7. Contact recovery: residual RL fine-tuning, 94% → 97%

Documented in full in `docs/project_journey/03_imitation_and_the_goal_bug.md`
§6. Short version: freeze the BC policy, learn a bounded ±0.15 correction with
sparse-reward SAC, anchor to the frozen reference, warm up the critic for 50k
steps, keep the best-by-eval checkpoint. 97% on 100 fresh starts. The
full-policy variant collapses to 0%. Checkpoint:
`storage_local/ant__20260906_0950__240957__finetune_bc_residual/best.zip`.

The multi-agent work is in
`docs/project_journey/04_multi_agent_and_the_cancelling_sum.md`. As of
2026-09-10: a swarm of 5 ants, each seeing only its own row, distilled from an
oracle reaches 91%; pure decentralised RL from zero reaches **96%** (and 88%
for a single ant under the same recipe, §11). Earlier 0% and 30% results came
from a worker-seed bug and from seed variance.

## 8. Open question, as of 2026-09-05: contact recovery

All remaining failures are the same failure. On seed 20000 the 5 failed episodes
end at 0.50-0.69 m from the goal, and every one jams near the **first** slit.
None fail later in the maze.

| GIF | Episode | Final distance |
|---|---|---|
| failure_1.gif | 0 | 0.6042 m |
| failure_2.gif | 25 | 0.5226 m |
| failure_3.gif | 35 | 0.5784 m |
| failure_4.gif | 65 | 0.6908 m |
| failure_5.gif | 68 | 0.5007 m |

Rollout GIFs (5 successes, 5 failures) are in
`storage_local/ant__20260905_1322__240805__train_goal_bc/fresh_starts_seed20000/`.

The demos contain only successes, so BC never sees a wedged load and has no
recovery behaviour. This is the one open problem. Note the earlier
action-noise sweep is not evidence here: it ran on the corrupted cache and
should be repeated on corrected data before drawing any conclusion.

Reproduce:

```bash
sbatch -M cluster -p performance -c 8 --mem=16G -t 1:00:00 \
  --wrap "export MUJOCO_GL=egl; python -u scripts/il/evaluate_goal_bc.py \
    --seed 20000 --episodes 100 --workers 8 --render --render-count 5 \
    --out storage_local/ant__20260905_1322__240805__train_goal_bc/fresh_starts_seed20000"
```
