# Ant Swarm — T-Shape Transport Environment

**File:** `new_exps/ant_swarm_roboverse.py`  
**Simulator:** MuJoCo (kinematic playback — 2-D physics drives all motion, MuJoCo is the renderer only)

---

## Overview

A swarm of N (=10) ant-like agents are permanently attached to a rigid T-shaped object and collectively push it toward a goal position across a 2-D arena.  
Two pairs of vertical wall barriers divide the arena into a left zone and a right zone, creating narrow passages the T-shape must navigate through.

The physics are implemented entirely in Python (no MuJoCo contacts). Each simulation step:
1. Every ant applies a force at its attachment point on the T-shape.
2. The combined force + torque integrates the T-shape rigid-body dynamics.
3. The new pose is pushed into MuJoCo via `set_states` (kinematic override).
4. MuJoCo renders the frame.

---

## World

| Parameter | Value | Description |
|---|---|---|
| `world_size` | (1.00, 0.72) m | Width × height of the arena. Origin (0, 0) at bottom-left. |
| Coordinate axes | x → right, y → up | 2-D top-down world. |

---

## T-Shape Object

The movable object is a rigid T-shape composed of three axis-aligned rectangles in the object's local frame.

```
        [cap_big] ← big cap (left end of stem)
          |  |
          |  |            
==========|  |===========
          stem
==========    ===========
          |  |
          |  |        
       [cap_small]   ← small cap (right end of stem)
```

All three parts share a single `thickness` value (stem width = cap depth = wall thickness = 0.02 m).

| Parameter | Default | Description |
|---|---|---|
| `center` | `None` (random) | Initial position of the T-shape centre in world frame. Pass `(x, y)` to fix it; `None` → sampled by `sample_tshape_pose`. |
| `angle` | `None` (random) | Initial rotation in radians. Pass a float to fix it; `None` → sampled uniformly in [−π/2, π/2]. |
| `pose_seed` | `None` | RNG seed used when `center` or `angle` is `None`. |
| `stem_len` | 0.30 m | Length of the stem. |
| `cap_big_len` | 0.18 m | Length of the larger cap (at the left/minus end of the stem). |
| `cap_small_len` | 0.09 m | Length of the smaller cap (at the right/plus end of the stem). |
| `thickness` | 0.02 m | Uniform cross-section thickness for all three parts. |
| `tshape_z` | 0.02 m | Z position of the body origin in 3-D (floor contact point). |
| `tshape_height` | 0.04 m | Extrusion height of the object in 3-D. |

**Collision geometry (local frame)**

| Rect | Centre | Half-size |
|---|---|---|
| Stem | (0, 0) | (0.15, 0.01) |
| Big cap | (−0.15, 0) | (0.01, 0.09) |
| Small cap | (0.15, 0) | (0.01, 0.045) |

### Random initial pose — `sample_tshape_pose`

When `center=None` or `angle=None`, the pose is drawn by `sample_tshape_pose` (rejection sampling):

```python
center, angle = sample_tshape_pose(
    rects       = cfg.rects,       # scaled collision rects
    walls_aabb  = ...,             # scaled wall AABBs
    world_size  = cfg.world_size,
    rng         = np.random.default_rng(pose_seed),
    angle_range = (-π/2, π/2),    # avoids upside-down T-shapes
    margin      = 0.06,            # clearance from world boundary
    max_tries   = 500,
)
```

A candidate is accepted only if the T-shape (at that pose) does not overlap any wall AABB and stays at least `margin` metres inside the world boundary. If no valid pose is found in `max_tries` attempts the function falls back to the hard-coded default `(0.22, 0.36) / −35°`.

**Manual constraint (if pose is fixed):** `center_x + 0.157 < 0.45` (i.e. `center_x < 0.29`) at angle = −35° to keep the small cap clear of the left gate wall.

---

## Walls (Gates)

Two vertical gate walls divide the arena. Each gate consists of an **upper** and a **lower** segment, creating a horizontal passage.

| Parameter | Default | Description |
|---|---|---|
| `wall_len` | 0.30 m | Length of each wall segment. |
| `thickness` | 0.02 m | Wall thickness (same as T-shape thickness). |
| `wall_height` | 0.08 m | 3-D height of wall objects. |
| `wall_render_extra` | 0.20 m | Extra length added outward for 3-D rendering only, so walls visually reach the scene boundary. Does **not** affect 2-D collision. |

**Default gate positions** (`wall_positions`)

| Segment | Centre (x, y) | Y extent | Role |
|---|---|---|---|
| Left gate — upper | (0.46, 0.57) | [0.42, 0.72] | Touches top boundary |
| Left gate — lower | (0.46, 0.15) | [0.00, 0.30] | Touches bottom boundary |
| Right gate — upper | (0.67, 0.57) | [0.42, 0.72] | Touches top boundary |
| Right gate — lower | (0.67, 0.15) | [0.00, 0.30] | Touches bottom boundary |

**Passage gap:** y = [0.30, 0.42] → **0.12 m** wide at each gate.

Wall collision uses AABB overlap test against the T-shape's bounding boxes. Ants are not subject to wall collision (they move with the T-shape).

---

## Ant Agents

10 ants are permanently attached to the T-shape. They never detach or move independently.

| Parameter | Default | Description |
|---|---|---|
| `n_ants` | 10 | Number of ants. |
| `ant_radius` | 0.012 m | Sphere radius (≈ real fire-ant body length). |
| `ant_z` | 0.04 m | Z position in 3-D = T-shape centre height (`tshape_z + tshape_height/2`). Ants are **beside** the T-shape, not on top. |
| `ant_mass` | 0.001 kg | Mass per ant (~1 mg, real ant scale). Does not enter current physics. |
| `push_strength` | 2 × 10⁻⁶ N | Force magnitude each ant applies per step. |

### Attachment points

Each ant is assigned a fixed point on the **perimeter** of one of the T-shape's three rectangles, sampled uniformly by perimeter length at construction time. The attachment point is in the T-shape's local frame and does not change during an episode.

### Action space

Each ant chooses one of 9 discrete directions per step:

| Index | Direction |
|---|---|
| 0 | Stay (no force) |
| 1 | −Y (down) |
| 2 | +Y (up) |
| 3 | −X (left) |
| 4 | +X (right) |
| 5 | −X −Y (diagonal) |
| 6 | +X −Y (diagonal) |
| 7 | −X +Y (diagonal) |
| 8 | +X +Y (diagonal) |

Diagonal actions are normalised to unit length.

### Heuristic policy

Each ant independently samples a push direction biased toward `(goal − object_centre)` with Gaussian noise (σ = 0.7). This gives genuine directional diversity — some ants push toward the goal, others sideways or in opposing directions — while the net average force still moves the T-shape toward the goal.

```
v_i  =  desired  +  N(0, 0.7)      # per-ant noisy direction
action_i  =  argmax over 9 dirs of  dot(dir, v_i / |v_i|)
```

---

## Physics

All dynamics are implemented in Python. MuJoCo is used only as a renderer.

### Integration

At each step, ant forces are accumulated and the T-shape is integrated:

```
total_force  = Σ_i  push_strength × dir_i
total_torque = Σ_i  (attachment_world_i − centre) × force_i

vel      = linear_friction  × vel  + total_force  / object_mass
ang_vel  = angular_friction × ang_vel + total_torque / object_inertia
```

Integration uses **10 sub-steps** to prevent tunnelling through thin walls.

### Object physics parameters

| Parameter | Default | Description |
|---|---|---|
| `object_mass` | 0.5 kg | T-shape mass. |
| `object_inertia` | 0.04 kg·m² | Moment of inertia (scales as s² with `scene_scale`). |
| `linear_friction` | 0.96 | Velocity damping per step (≈ ground friction). |
| `angular_friction` | 0.94 | Angular velocity damping per step. |

**Terminal velocity** (10 ants, all aligned):
```
v_terminal = (n_ants × push_strength) / (object_mass × (1 − linear_friction))
           = (10 × 2e-6) / (0.5 × 0.04)
           ≈ 0.001 m/step  (≈ 1 mm per frame)
```

### Wall collision

Sub-step collision uses AABB overlap between each of the T-shape's 3 rects and each wall's AABB. On overlap:
1. Try sliding along X only.
2. Try sliding along Y only.
3. If both fail — fully revert and damp velocity by −0.2.

Rotation is independently reverted if it causes overlap.

### World boundary

Soft boundary: T-shape corners are clamped to `[margin, W−margin] × [margin, H−margin]` with `margin = 0.025 m`. Velocity is damped by −0.15 on boundary contact.

---

## Task

| Parameter | Default | Description |
|---|---|---|
| `goal` | (0.20, 0.54) m | Target position for the T-shape centre. |

**Episode:** The heuristic runs for `n_steps = 260` steps (≈ 8.7 s at 30 fps). No explicit termination condition — the episode always runs to completion.

---

## Scene Scale

All spatial dimensions scale uniformly with `scene_scale` (default 1.0). Forces scale linearly with `s`, inertia scales as `s²`, preserving qualitatively identical dynamics at any world size.

```bash
# Default 1 m × 0.72 m world
python new_exps/ant_swarm_roboverse.py --headless

# Double the world size
python new_exps/ant_swarm_roboverse.py --headless --scene-scale 2.0
```

---

## Output

| File | Content |
|---|---|
| `new_exps/output/ant_swarm_mujoco.mp4` | Top-down 1280×832 video at 30 fps |
| `new_exps/output/ant_swarm/i_shape.xml` | Auto-generated MJCF for the T-shape |
| `new_exps/output/scene_check_s1.0.png` | 2-D geometry diagram (no MuJoCo needed) |

Run the geometry checker without MuJoCo:
```bash
python new_exps/visualize_scene.py --no-show
```
