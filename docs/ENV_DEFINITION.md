# Ant-Swarm T-Barrier — Environment Definition

**Package:** `ant_swarm/` (pure Python/NumPy — no physics engine, no display dependency)
**Registered ids:** `AntSwarmBarrier-v0` (gymnasium 5-tuple API), `AntSwarmBarrier-v0-compat` (classic gym 4-tuple)
**Parameters:** all values below come from `configs/rl/config.yaml` (single source of truth) and reflect its current state; the geometry ones are multiplied by the global `scene_scale` (1.0).

---

## Overview

`n` ant agents are rigidly attached to a T-shaped object and must move it across
a 2-D arena, through the narrow gap of a two-column barrier, until a tracked
point on the T (by default the **big-cap centre**) is within `reach_radius` of
the goal. The gap (0.15) is narrower than the big cap (0.18), so a straight push
cannot succeed — the T must be threaded diagonally, rotating while translating.

An episode **terminates** on success and **truncates** at `env.max_steps` (500).

---

## World

| Parameter | Value | Description |
|---|---|---|
| `world.width × height` | 1.25 × 0.72 m | Origin (0, 0) at bottom-left; x → right, y → up. |
| `scene_scale` | 1.0 | Uniform scale hook: lengths ×s, inertia ×s², forces ×s. |

## Barrier walls

Two vertical columns (`walls.x_columns` = 0.525, 0.735), each made of an upper
and a lower segment that leave a horizontal passage in the middle:

| Parameter | Value | Description |
|---|---|---|
| `walls.length` | 0.285 m | Length of each of the 4 segments (upper ones touch the top edge, lower ones the bottom). |
| `walls.thickness` | 0.02 m | Same as the T thickness. |
| gap | **0.15 m** | `height − 2·length`, vertically centred; computed, not configured. |
| `walls.render_extra` | 0.20 m | Cosmetic outward extension when rendering only. |

`Layout.wall_heads` precomputes the four **gap-facing segment tips** — the
corners the T must clear — used by the observation's barrier features.

## T-shape

Three axis-aligned rectangles in the object's local frame, sharing one
`thickness` (0.02 m): a stem along local x with a cap at each end.

| Part | Local centre | Half-size |
|---|---|---|
| Stem (`stem_len` 0.265) | (0, 0) | (0.1325, 0.01) |
| Big cap (`cap_big_len` 0.18) | (−0.1325, 0) | (0.01, 0.09) |
| Small cap (`cap_small_len` 0.09) | (+0.1325, 0) | (0.01, 0.045) |

Collision against walls is an exact SAT test (oriented rects vs wall AABBs) in
`geometry.obb_aabb_overlap`.

## Spawning

`sample_free_pose` rejection-samples a collision-free pose: x uniform in
`spawn.x_range` (0.06–0.40, left of the barrier), y uniform inside `margin`
(0.06), angle uniform in ±π/2; up to `max_tries` (500) attempts, then a
fallback at the band centre with angle 0. The pose is sampled **once per env
instance** and reused every reset, unless the reverse curriculum enables
per-episode resampling (see hooks below).

## Ants

Ants are rigid attachment points on the T (they never detach); their world
positions are recomputed from the T pose each step.

* `n == 1` → the stem centre (origin) — current setting
* `n == 2` → the two stem↔cap junctions
* `n ≥ 3` → random points on the T's perimeter, uniform by arc length

## Action space

Set by `motion.mode`:

**`dynamic`** (current) — forces + momentum.
Per ant: `[push angle ∈ [−π, π], magnitude ∈ [0, 1]]`; force =
`physics.push_strength · magnitude` (0.0005 N) along the angle, applied at the
ant's attachment point, so off-centre ants create torque. With a single ant and
`ants.single_agent_spin: true` the action gains a third component
`spin ∈ [−1, 1]` adding direct torque `spin · spin_strength`
(`spin_strength: null` → `push_strength · stem_len / 2`), since a centred point
force alone cannot rotate the body. Shape: `(n_ants, 2)` or `(1, 3)`.

**`kinematic`** — one command `[direction ∈ [−π, π], rotation ∈ [−1, 1]]` for
the whole T: translate `step_len` (0.01) along `direction`, rotate
`rot_step · rotation` (0.10 rad). No mass/momentum; collisions handled by
rejecting the offending component (full move → translate-only → rotate-only →
stay), so the T slides along walls.

## Physics (dynamic mode)

Per env step, the summed wrench integrates the T as a rigid body:

```
vel     = linear_friction  · vel     + F / object_mass        # 0.96, 0.5 kg
ang_vel = angular_friction · ang_vel + τ / object_inertia     # 0.94, 0.01 kg·m²
```

The step is split into `substeps` (10) to prevent tunnelling through the thin
walls. On wall overlap in a substep: try x-only motion, then y-only, else fully
revert; the blocked velocity component is scaled by `restitution_wall` (−0.2).
Rotation causing overlap is reverted independently. The world boundary is soft:
corners are clamped `boundary_margin` (0.025) inside, with `restitution_bound`
(−0.15) on contact.

## Observation space

Dynamic mode uses `Box(-1, 1, shape=(n_ants, 27))`; kinematic mode and
`env.observe_linear_velocity: false` use the legacy 25-float shape.

| # | Feature | Notes |
|---|---|---|
| 0–1 | attachment offset (local x, y) | normalised by stem/cap half-lengths |
| 2–3 | T centre (x, y) | normalised by world size |
| 4–5 | goal − tracked point (dx, dy) | measured from the goal-tracked point, normalised |
| 6–7 | sin(angle), cos(angle) | T orientation |
| 8 | angular velocity | clipped ±1 at 0.05 rad/step |
| 9–10 | linear velocity (world x, y) | dynamic mode only; clipped ±1 at theoretical full-push steady speed |
| 11–26 | tip→head distances | 4 T arm tips (big-cap top/bottom, small-cap top/bottom) × 4 wall heads, row-major, ÷ world diagonal |

The barrier block is identical across ants (it depends only on the T pose); it
gives a sharp "arm about to clip a wall corner" cue that the threading maneuver
can condition on. Training scripts wrap the env in `FlattenObservation`.

## Reward and goal

The goal-tracked point is set by `env.goal_track` (`big_cap` | `center` |
`small_cap`; current `big_cap`). Distance-to-goal, success, the observation's
goal vector, and `info["object_distance"]` all use this same point — tracking
the big cap forces the agent to lead with the *hard* end, removing the
small-cap-first exploit.

`env.reward_mode`:

* **`sparse`** (current): `reward_success` (1.0) when the tracked point is
  within `goal.reach_radius` (0.05) of `goal.pos` (1.05, 0.36); else 0.
* **`shaped`**: adds `reward_progress_coef` (0.1) × per-step distance decrease.

`info` per step: `object_center`, `object_angle`, `object_distance`, `step`,
`wall_len`, `gap`.

## Curriculum hooks

The env exposes two hooks, both taking effect on the **next reset** (used by
`train_utils.CurriculumCallback`; the current config runs `reverse` mode):

* `set_wall_length(v)` — rebuild the layout at wall length `v` (gap curriculum:
  wide → narrow).
* `set_spawn_x_range(lo, hi)` — sample a fresh spawn pose in that x-band every
  episode (reverse curriculum: start past the barrier near the goal, move the
  band back toward the full-task spawn as success rises).

## Rendering

`Renderer` produces a 650-px-wide RGB `uint8` frame with pure NumPy: grey
walls (with cosmetic extensions), green goal square, red T, cyan dot at the
goal-tracked point, white ant dots. `render_fps` metadata: 30.
