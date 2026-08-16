# Ant-Swarm T-Barrier — Conceptual Overview

A companion to [ENV_DEFINITION.md](ENV_DEFINITION.md) (the precise parameter
spec). This document explains **what the environment is, why it is built the
way it is, and how the pieces fit together** — the mental model behind the code.

---

## What this is

A gym/gymnasium recreation of the cooperative-transport **"piano-movers"
experiment** from Ofer Feinerman's lab (Dreyer et al., *PNAS* 2025): longhorn
crazy ants (*Paratrechina longicornis*) — and, for comparison, human groups —
maneuver a T-shaped load through two narrow slits in successive walls. In the
real system no individual plans the path; each ant applies a small force at its
attachment point, transiently informed ants steer, and the rest align with the
force they feel through the load, giving the group an emergent directional
persistence (Gelblum et al., *Nature Communications* 2015).

The environment models exactly that setting: N ants rigidly attached to a rigid
T-shape in a 2-D arena apply point forces; the summed force + torque drives the
load's rigid-body dynamics. The real problem is inherently **multi-agent**; the
project deliberately starts **single-agent** (one controller, with an explicit
spin action) to isolate the maneuver-learning problem from the coordination
problem, then scales to the multi-agent force model (`ants.n >= 2`).

## The task, and why it is hard

The T spawns left of a two-column barrier and must reach a goal on the far
right. Success = the **big-cap centre** within `reach_radius` of the goal.

The geometry is chosen so the task cannot be solved greedily:

* the slit gap (0.15) is **narrower than the big cap** (0.18) — a straight push
  jams; the T must be threaded diagonally, rotating while translating;
* there are **two** gates with a corridor between them, so the maneuver must be
  executed, recovered from, and executed again;
* the T is **asymmetric** (big cap 0.18 vs small cap 0.09) — the small end fits
  through anything, which is precisely why the goal tracks the big end (see
  below).

This makes the env a **deceptive-exploration** benchmark rather than merely a
sparse-reward one: distance-to-goal shaping pulls the agent straight into the
wall, while the correct maneuver temporarily moves *against* that gradient.
Exploration must pass through a narrow bottleneck in state space, which is what
the curriculum, the success harvesting, and the barrier observation features
all target.

## Architecture — how the modules compose

Everything is parameterised by `configs/rl/config.yaml` (single source of truth, no CLI
args anywhere). The package is RL-library-agnostic; anything importing
stable-baselines3 lives outside it in `scripts/rl/train_utils.py`.

```
configs/rl/config.yaml ──► config.py (namespace)
                   │
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
  Layout        TShape       ActionModel ─┐
  (walls,       (geometry,   ObservationModel ─┼──► AntSwarmEnv (ant_swarm.py)
   heads,        collision,  RewardModel ─┘        │
   goal)         spawning)                         ▼
     └────────────►SwarmState (physics) ◄──────────┘
                     Renderer (pure-NumPy RGB)
```

* **`geometry.py`** — `LocalRect` + exact SAT overlap (oriented rect vs AABB).
  Pure functions, no config knowledge.
* **`layout.py`** — the static scene: world bounds, 4 wall segments (2 columns
  × upper/lower), the goal, and `wall_heads`: the 4 **gap-facing wall corner
  tips**, precomputed because they are what the T must clear.
* **`tshape.py`** — the load (3 rects: stem + 2 caps), pose transforms,
  wall-collision test, spawn sampling (`sample_free_pose`), and ant attachment
  placement (`make_attachment_offsets`).
* **`state.py`** — the only mutable thing: current T pose/velocities + ant world
  positions. `integrate()` advances one env step in 10 substeps with
  slide-along-wall collision response (try full move → x-only → y-only →
  revert + restitution; rotation reverted independently) and a soft world
  boundary. `apply_kinematic()` is the physics-free alternative.
* **`action.py` / `observation.py` / `reward.py`** — pluggable definitions of
  the three RL interfaces (details below).
* **`ant_swarm.py`** — composes the above into a gym `Env`, and exposes the two
  curriculum hooks (`set_wall_length`, `set_spawn_x_range`), both applied on
  the **next reset**.

## Control modes

`motion.mode` selects one of two interchangeable control models (checkpoints do
**not** transfer across modes — the action space changes):

* **`dynamic`** (current, faithful to the ants): per-ant `[push angle,
  magnitude]` point force at the attachment point; torque arises from
  off-centre pushes; the load carries momentum under linear/angular friction.
  With `ants.n == 1` the force acts at the stem centre and can produce no
  torque, so the action gains a third `spin ∈ [-1, 1]` component (explicit
  torque) — the single-agent concession that keeps rotation learnable.
* **`kinematic`** (simple): one `[direction, rotation]` command teleports the T
  by small increments, no momentum. Useful for isolating the geometric planning
  problem from momentum control; collision uses component rejection so the T
  slides along walls.

Ant placement: `n == 1` → stem centre; `n == 2` → the two stem↔cap junctions;
`n ≥ 3` → random perimeter points, uniform by arc length — mirroring how real
ants distribute around a load's rim.

## Observation design

Per ant, dynamic mode exposes 27 floats in ~[-1, 1]. The first 9 are the legacy
base features (own attachment offset, T centre, goal vector *measured from the
goal-tracked point*, sin/cos orientation, angular velocity), followed by 2
world-frame linear-velocity components and a 16-float **barrier block**:
distances from the T's 4 arm tips (big-cap top/bottom, small-cap top/bottom) to
the 4 wall heads.

The barrier block is the important design decision. Earlier training failed
with a policy that never rotated near the gap: the fixed layout was not in the
observation at all, so "about to clip a corner" had no direct signal. As a tip
approaches a wall head its distance → 0 — a sharp local cue that rotation can
be conditioned on. The block depends only on the T pose and is therefore shared
across ants; only the attachment offset differs per row, which is what a future
parameter-shared multi-agent policy would key on.

Linear velocity is normalised by the theoretical steady speed under aligned
full-strength pushes, so the policy can distinguish momentum from new control
input. Kinematic mode, which has no momentum, keeps the 25-float layout. Set
`env.observe_linear_velocity: false` for compatibility with old checkpoints.

## Reward and the anti-cheat

`reward_mode: sparse` (current) pays `1.0` on success only; `shaped` adds
potential-style progress `0.1 · (prev_dist − dist)`.

Goal distance is measured from the point chosen by `env.goal_track`
(**`big_cap`** by default). This closes an exploit actually observed in
training: with centre tracking, the agent learned to lead with the small cap
(which fits any gap), poke it through both slits, and score — never performing
the hard maneuver. Tracking the big cap makes the easy strategy worthless: the
agent must lead with the hard end. Reward, termination,
`info["object_distance"]`, and the observation's goal vector all use the same
tracked point (kept consistent on purpose), and the renderer marks it with a
cyan dot.

## Curriculum

Sparse reward + bottleneck exploration means the success signal may never be
seen from the full task. Two curricula (config `curriculum:`, executed by
`CurriculumCallback` in `scripts/rl/train_utils.py`) create winnable episodes first:

* **`gap`** — start with a wide gap (straight push possible) and narrow it by
  `step` each time rolling success ≥ threshold. Caveat learned the hard way:
  most stages sit in the "gap ≥ big cap" regime where threading isn't needed,
  so the hard skill only appears in the last stages.
* **`reverse`** (current) — pin the gap at the hard width and instead move the
  **spawn** backwards: the T starts *past* the barrier near the goal (success
  almost immediate), then the spawn band slides left toward the real start as
  success accumulates. The reward is never modified — only where episodes
  begin.
* **`pose_path`** (canonical dynamic task) — keep the hard maze fixed and move
  through complete `(x, y, theta)` anchors taken from one collision-free
  geodesic solution path.

In every mode the **eval env is pinned at the full hard task**, so eval metrics
always measure the real objective. A stalled stage is held and logged but never
force-advanced; training stops early only once the target is mastered
(`stop_success` over `stop_window` episodes).

## Training stack (outside the env package)

PPO (`scripts/rl/train_ppo.py`, 8 envs, bounded log-std + entropy annealing) and SAC
(`scripts/rl/train_sac.py`, single env) via stable-baselines3, both with:

* checkpoints + best-model eval, policy GIFs (disk/TensorBoard/W&B),
  success-rate + final-distance metrics;
* **success harvesting** (`SuccessTrajectoryCallback`): every successful
  episode saved as a minimal replayable JSON — init pose + action sequence +
  the difficulty it was solved at — with tolerance-based path dedup. Successes
  are rare and precious here; the files feed replay/rendering
  (`scripts/il/render_success.py`) and future BC/warm-starting;
* full reproducibility: each run dir snapshots the package source + the exact
  `config.yaml` used (`snapshot.save_code`).

## Gotchas worth remembering

* Both curriculum hooks take effect on the **next reset**, not immediately.
* The spawn pose is sampled **once per env instance** and reused each reset —
  per-episode resampling only happens under the reverse curriculum's
  `set_spawn_x_range`.
* Checkpoints are invalidated by anything that changes obs/action shapes or
  meaning: switching `motion.mode`, changing `ants.n`, past obs-layout changes,
  and the `goal_track` switch (same shapes, different objective — retrain).
* In `kinematic` mode the ants are cosmetic (one centralized command regardless
  of `n`); `ang_vel` is always 0 and the linear-velocity block is omitted.
* The SB3 setup is a **centralized** controller even for `n ≥ 2` (obs flattened,
  one network emits all ants' actions). Decentralized per-ant policies — the
  actual ant setting — are future work the per-ant observation rows already
  anticipate.
