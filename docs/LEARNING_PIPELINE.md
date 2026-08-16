# Narrow-path RL pipeline

The canonical dynamic task uses `configs/rl/pnas_dyn_geo_v2.yaml`. It is built
for a sparse terminal objective whose successful motion lies in a narrow part
of pose space.

## What is implemented

1. `gen_geodesic_field.py` runs a collision-aware BFS over `(x, y, theta)` and
   saves both distance and the original reachable mask.
2. `pose_path` extracts one strictly descending solution path and places 16
   complete pose anchors `(x, y, theta)` along it, from easy to hard. The maze
   never changes.
3. Geodesic pretraining rewards only true progress along that route plus the
   real terminal success bonus.
4. A stage advances only after at least 90 successes in the last 100 episodes.
   Elapsed time can log a stall, but can never promote a failed stage.
5. SAC retains complete successful episodes in a separate replay ring and
   reserves 25% of each minibatch for them.
6. PPO exploration is bounded (`log_std` in `[-3, -0.5]`), with low entropy
   annealed to zero and gSDE disabled. SAC uses bounded actions, automatic
   entropy from a small initial coefficient, and an explicit target entropy.
7. Dynamic observations include world-frame linear velocity. Set
   `env.observe_linear_velocity=false` only for legacy 25-feature checkpoints.

Evaluation and rendered policy checks are pinned to the final hard pose. They
do not report an easier curriculum-stage score as full-task success.

## Build the field

Regenerate after changing the maze, T geometry, goal, or tracked goal point:

```bash
python scripts/rl/gen_geodesic_field.py configs/rl/pnas_dyn_geo_v2.yaml \
  --out storage_local/fields/pnas_gap015.npz \
  --inflate 0.002 --dx 0.003 --dth 2
```

The planner refuses unreachable starts. Every selected anchor is also checked
by the real environment collision model before training.

## Phase 1: geodesic pretraining

```bash
ops/local_train.sh train_sac configs/rl/pnas_dyn_geo_v2.yaml
```

The run saves:

- `checkpoints/sac_final.zip`
- `checkpoints/sac_final_replay_buffer.pkl`
- periodic model/replay recovery checkpoints
- reward-independent successful action trajectories under `successes/`

## Phase 2: sparse fine-tuning

Use the geometry-identical sparse config. Transfer only the actor and replay
saved successful actions through the sparse environment so their rewards are
recomputed correctly:

```bash
ops/local_train.sh train_sac configs/rl/pnas_dyn_sparse_v2.yaml \
  run.transfer_actor_from=storage_local/<geo-run>/checkpoints/sac_final.zip \
  run.seed_successes_from=storage_local/<geo-run>/successes
```

Do not load a geodesic replay buffer into sparse training. Its stored rewards
belong to a different objective; actor-only transfer leaves critics, target
critics, optimizers, and entropy tuning fresh.

## Resume an interrupted phase

For the same config and reward phase, restore the model and its replay buffer:

```bash
ops/local_train.sh train_sac configs/rl/pnas_dyn_geo_v2.yaml \
  run.resume_model=storage_local/<run>/checkpoints/sac_resume_500000_steps.zip \
  run.resume_replay=storage_local/<run>/checkpoints/sac_resume_replay_buffer_500000_steps.pkl \
  curriculum.start_stage=<stage-at-that-checkpoint>
```

`sac.timesteps` is the number of additional steps on resume. Curriculum
callback state is not stored inside an SB3 checkpoint, so copy the last stage
number from the source log into `curriculum.start_stage`. Full resume and actor
transfer are mutually exclusive.

## Compatibility

Changing motion mode, ant count, observation shape, action shape, geometry, or
reward mode invalidates a full resume. Start a new phase instead. In
particular, checkpoints made before linear velocity was added have 25 input
features and cannot be loaded into the default 27-feature dynamic policy.
