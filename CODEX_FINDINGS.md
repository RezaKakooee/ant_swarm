# Goal observation bug and corrected baseline — 2026-09-05

## Confirmed root defect

`ObservationModel` retained `layout.goal` by reference at construction. Most
imitation replay/evaluation scripts subsequently assigned a new array to
`env.layout.goal`. Distance, termination and rendering then used the requested
goal, while policy observations still used the first randomly sampled goal.

All 4,914,564 transitions in `storage_local/cache/replay_full.npz` have incorrect
goal vectors. Reconstructing the observed goal from the tracked big-cap point
and obs[4:6] gives approximately **(1.5175362, 0.5242438)** throughout the cache.
The true goal varies per episode. Mean error: **0.3136312 m**; maximum:
**0.6304046 m**. This is not evidence that a correctly conditioned BC policy
cannot solve the task. The previous algorithm comparisons must be reconsidered.

A second defect followed from the assignment: `np.asarray(goals[i])` aliases
the dataset row. A subsequent random reset wrote through that row, modifying
the in-memory demo target. The original dataset file was not changed.

## Implemented fixes

- Layout owns its goal array; assignment copies coordinates and preserves the
  reference. ObservationModel reads the current layout goal, including through
  its compatibility setter.
- `env.reset(options={"init_pose": pose, "goal": goal})` initializes pose,
  momentum, step counter, previous distance, reward state and goal together.
  Values use world units; explicit values bypass their random samplers.
- Main BC evaluators use distinct start poses and exact resets.
- `scripts/il/repair_replay_goals.py` audits every cached transition and can
  repair only obs[4:6] using original demo targets and the configured tracked
  point. It retains the original cache and changes no actions or physics.
- Cache-backed chunked, torque-head and waypoint experiments reject stale goal
  vectors before training. Existing old checkpoints still require retraining.
- `scripts/il/train_goal_bc.py` reproduces the H=1, 3x256 MLP baseline with
  whole start-pose groups held out and per-episode JSON results. It uses sparse
  reward, the unchanged full maze and ordinary 27-D observations. No geodesic
  map, waypoint oracle, curriculum or action noise is used.

## Verification

- 17 tests pass, including existing replay-buffer/curriculum tests and new goal
  synchronization, dataset ownership, deterministic reset, cache reconstruction,
  scaled geometry, tracked-point and train/test group-separation regressions.
- Fresh replay of 8 episodes spread over the dataset: 8 successes; 1,131 cached
  observations checked; maximum absolute error after repair: 1.1920929e-7.
- All goal vectors in `storage_local/cache/replay_full_goals_v2.npz` pass the
  exact audit. The original corrupt cache is retained as evidence.

## Experiment

Slurm job 240805 completed 40 epochs of unchanged H=1 BC on 4,885,903 training
transitions. All episodes sharing any of 100 selected test starts are excluded
from training (seed 20260905). The test also replays the expert and runs the old
H=1 checkpoint with correct observations. The old checkpoint may have trained
on these starts and is a diagnostic control, not a held-out comparison.

Artifacts: `storage_local/ant__20260905_1322__240805__train_goal_bc/`.

Commands:

```bash
python scripts/il/repair_replay_goals.py \
  --cache storage_local/cache/replay_full.npz \
  --out storage_local/cache/replay_full_goals_v2.npz
sbatch --time=00:30:00 --cpus-per-task=8 --mem=16G \
  ops/sb_train.sh scripts/il/train_goal_bc.py "" --epochs 40 --workers 8
python -m unittest discover -s tests -v
```

Use a new `--out` directory for a new training run. For an exact repeat of
recorded evaluation, pass `--evaluate-only`; the checkpoint holdout is checked.

## Corrected held-out result

| Policy | Successes / 100 | Mean final distance | Jammed |
|---|---:|---:|---:|
| Expert action replay | 100 | 0.05179 m | 0 |
| BC retrained on corrected goals | **91** | **0.10310 m** | **9** |
| Old BC checkpoint, correct goals at inference | 0 | 0.59995 m | 99 |

The main failure is solved by correcting training observations and retraining;
a new RL algorithm is not necessary to obtain high success. The remaining
9 failures all jam around the first slit. This is not a claim of 100% reliability
or generalization to a different maze. The 91% estimate uses one training seed
and 100 withheld start-pose groups from the demonstration distribution.

The trained checkpoint is
`storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt` and its environment
configuration is saved alongside it as `config.yaml`. Full metrics, exact start
IDs, and per-episode outcomes are in `results.json`.

Fresh random-start evaluation and repeatability check:

```bash
sbatch --time=00:15:00 --gres=gpu:0 --cpus-per-task=8 --mem=12G \
  ops/sb_train.sh scripts/il/evaluate_goal_bc.py "" --repeat --render
```

## Independent fresh-start validation

Slurm job 240868 evaluated the frozen checkpoint on 100 new random starts/goals
from the original full-maze configuration (reset seeds 10000 through 10099).
Result: **88/100 successes**, mean final distance **0.1106168 m**, 12 jammed.
Repeating the complete evaluation produced identical per-episode results,
including distances, steps and failure landmarks. No training or model
selection used these fresh-start outcomes.

Fresh-start poses/goals, individual results, and repeatability status are saved
under `storage_local/ant__20260905_1322__240805__train_goal_bc/fresh_starts/`. The held-out demo
example animations are `storage_local/ant__20260905_1322__240805__train_goal_bc/success.gif` and `storage_local/ant__20260905_1322__240805__train_goal_bc/failure.gif`.

The two independent evaluation sets therefore give 91% (withheld demo starts)
and 88% (new random starts). Fixing the stale goal is sufficient for high BC
success, but contact recovery on the remaining failures is still unresolved.
