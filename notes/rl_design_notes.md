# Ant-Swarm T-Barrier — RL Design Notes

Working notes on algorithm choice, reward, and observation design for the
T-shape threading task. Captures the reasoning, not just the conclusions.

---

## The task, characterised

- 2 ants rigidly attached to a T-shape; continuous per-ant action `[angle, magnitude]`.
- Must **thread the T diagonally** through a narrow gap, then reach a goal past the barrier.
- Geometry (verified): straight-push is **impossible** (big cap 0.18 > gap 0.15);
  diagonal threading **is** possible. So the task is solvable but requires a
  non-trivial rotate-while-translating maneuver.
- Reward = `0.1 * (prev_dist - dist)` shaping  +  `1.0` sparse bonus on reaching goal.

### Why it's hard — it's deception, not pure sparsity

The shaping is (approximately) **potential-based** (Φ = −dist), so in theory it
doesn't change the optimal policy — it just densifies the signal. The real
difficulty:

> The straight-line distance reward pulls the ants to shove the T **directly at
> the wall**, but threading requires briefly **rotating / moving sideways** —
> i.e. against the per-step gradient. This is an **exploration-through-a-
> bottleneck** problem, not a no-signal problem.

This reframing matters: tools that manufacture signal from nothing (HER) are
less critical than tools/strategies that fix exploration into the bottleneck.

---

## Algorithm choice

Current: PPO (`train_ppo.py`) and SAC (`train_sac.py`).

**SAC is the better default here:** off-policy (sample-efficient), continuous
actions, and **auto entropy tuning** keeps the policy stochastic — which is
exactly what helps escape the deceptive local trap.

### Recommended order of escalation (for sparse / deceptive goals)

1. **Run the SAC baseline first.** Threading is geometrically possible and SAC's
   entropy may crack it without extra machinery. Get the baseline before adding
   complexity.
2. **Curriculum** (highest leverage, lowest complexity for a *fixed* goal):
   start the goal just past the gap exit → expand outward to x=1.05 as success
   rate rises. Directly attacks the bottleneck.
3. **HER / TQC+HER** only if curriculum stalls. HER's superpower is *zero-signal*
   sparse rewards and wants a **multi-goal** framing
   (`achieved_goal`/`desired_goal`); with a single fixed goal + existing shaping
   its payoff shrinks and it's more code. TQC (from `sb3-contrib`) is
   distributional SAC — the "best benchmark recipe" move.
4. **RND** (curiosity / novelty bonus) — lighter than HER; rewards visiting novel
   states, which helps push through the threading corridor. No env changes.

**One-line take:** don't add HER reflexively. Watch SAC; if it stalls, add a
goal **curriculum** before reaching for HER.

---

## Observation space — do NOT go to image + CNN (for the fixed layout)

The state is already low-dimensional and ~Markov: T pose (center, angle),
goal, attachment offsets fully describe the controllable system. A CNN would
have to *re-derive* all that from pixels — inflating sample complexity on top of
the problem that's already the bottleneck (exploration). **Perception isn't
hard here; control is.**

> Rule of thumb: use images only when what you must perceive can't be cheaply
> handed over as numbers. Here it can.

**Image+CNN is only worth it if the layout is randomised** each episode (wall
positions, gap size, goal) — then the agent must perceive a novel obstacle it
can't memorise in the weights. Even then, privileged scalars (wall x, gap
centre/size) beat pixels for sample efficiency.

### Cheap, high-value obs fixes instead

1. **Add linear velocity `(vx, vy)`.** Current obs has `ang_vel` but **not**
   linear velocity — genuine partial observability, since the T carries momentum
   in the physics. Likely matters more than anything visual for a
   momentum-sensitive maneuver. ~2 floats, free.
2. **Wall-relative features.** The barrier isn't in the obs at all today — the
   agent memorises the fixed layout implicitly. Adding signed x-distance to the
   barrier + each cap's clearance to the gap edges gives spatial awareness
   cheaply, and is what makes randomised layouts learnable later.

Both are small, contained changes in `ant_swarm/observation.py`.

### IMPLEMENTED (2026-06-01): tip→wall-head distances

Observation grew **9 → 25** per ant. Added a 16-float barrier block:
distances from the T's **4 arm tips** (big-cap top/bottom, small-cap top/bottom)
to the **4 inner wall "heads"** (gap-facing wall corners), normalised by the
world diagonal. Row-major: tip0→head0..3, tip1→head0..3, …

- Wall heads precomputed in `Layout.wall_heads` (the inner gap-facing tip of each
  of the 4 wall segments — the corners the T must clear).
- Block is shared across ants (depends only on T pose), matching the existing
  per-ant layout where only the attachment offset differs.
- Rationale: as an arm tip nears a wall head the distance → 0, giving a sharp
  **local "about to clip" cue** the policy can condition rotation on — directly
  targeting the observed failure where neither PPO nor SAC rotates near the gap.

**Still NOT added** (candidates if this isn't enough): linear velocity `(vx, vy)`
(real partial-observability gap), signed x-distance to barrier plane, per-cap
gap-edge clearance.

> ⚠️ Obs size changed → **existing checkpoints are incompatible**; retrain from
> scratch.

---

## Motion modes (config `motion.mode`)

Two interchangeable control/physics models:

### kinematic (simple, current default)
- Action = `[direction ∈ [-π,π], rotation ∈ [-1,1]]`, one command for the whole T.
- Each step: translate `step_len` along `direction`, rotate `rot_step * rotation`.
  No forces / mass / friction / momentum.
- Collisions: **rejection** (full → translate-only → rotate-only → stay), so the T
  slides along walls instead of tunnelling.
- Rotation is immediate & responsive → much easier credit assignment than dynamic.
- Single centralized controller regardless of `ants.n` (n only affects ant-dot
  rendering / obs rows). `ang_vel` obs feature is always 0 here (harmless).

### dynamic (forces + momentum)
- Multi-agent: per-ant `[angle, magnitude]` → forces; rotation from *differential*
  forces. Single-agent (`n==1`): `[angle, magnitude, spin]` (spin → torque, since a
  central point force makes zero torque).

**Switching modes changes the action space → checkpoints are NOT transferable;
retrain.**

### Speeding up dynamic mode

Governing (terminal) relations:

```
terminal speed   = (push_strength / object_mass)    / (1 - linear_friction)
terminal ang_vel = (torque        / object_inertia) / (1 - angular_friction)
```

| Want | Knob | Note |
|------|------|------|
| faster translation | `push_strength` ↑ / `object_mass` ↓ | clean (no drift) |
|                    | `linear_friction` → 1 | higher top speed but **more momentum/drift** |
| faster rotation    | `object_inertia` ↓ | clean |
|                    | `angular_friction` → 1 | faster but more drift |
|                    | `spin_strength` (single agent) | now a config knob; `null` → `push_strength*stem/2` |

**Applied preset (2026-06-01):** `push_strength 0.00025→0.0005` (2× translation),
`object_inertia 0.04→0.01` (4× rotation). Single-agent 90° turn: **153 → 31 steps**.
Friction left unchanged to keep motion crisp (low drift).

---

## Horizon

`env.max_steps` cut **2000 → 500** (a good trajectory is ~150–350 steps).
Shorter = more episodes/attempts per compute, less discounting of the terminal
success bonus, tighter credit assignment. `gamma` lowered **0.999 → 0.99**
(effective horizon ~100) to match. Once successes accrue, right-size `max_steps`
to ~2× their median `length` (now recorded — see below).

---

## Goal-tracking point — the "small-cap cheat" fix (IMPLEMENTED 2026-06-02)

**Problem observed:** with distance measured from the T *centre*, and an asymmetric
T (big cap 0.18, small cap 0.09) + a two-column barrier, the agent learned a cheat:
lead with the **small cap** (always fits any gap), poke it through hole 1, rotate in
the corridor, poke through hole 2 — never actually threading the big cap. Worse, the
gap-curriculum spends 7 of 9 levels in the "gap ≥ big cap" regime where *no* threading
is needed at all, so the hard skill is only required in the last 1–2 stages and the
learned strategy doesn't transfer.

**Fix:** measure goal distance from the **big-cap centre** instead of the T centre.
Config `env.goal_track: big_cap | center | small_cap` (default `big_cap`).

- Reward shaping, success/termination, `info["object_distance"]`, and the observation's
  `goal_dx,goal_dy` **all** use the tracked point (kept consistent).
- To get the big-cap centre to the goal the agent must **lead with the big (hard) end**
  (orient ~180°); verified geometrically reachable (small cap trails inside the world).
  The small-cap-first strategy now earns no progress and scores no success → cheat gone.
- `TShape.track_local_point(which)` returns the local point; `SwarmState.tracked_world()`
  exposes it; the renderer draws it as a **cyan dot**.
- Obs dim unchanged (25) — but the *meaning* of the goal-vector and the reward changed.

> This is orthogonal to (and lighter than) switching the curriculum to grow
> `cap_big_len`; it removes the exploit directly. Both could be combined later.

## Curriculum learning (gap-size) — IMPLEMENTED

Config `curriculum:` (read by both train scripts):

```yaml
curriculum:
  enabled: true
  start_wall_len: 0.205        # easy: wide gap (~0.31, straight-push OK)
  target_wall_len: 0.285       # hard: narrow gap (~0.15, threading required)
  step: 0.01                   # narrow by this per advance  (8 stages)
  success_threshold: 0.7       # advance when rolling success rate >= this
  window: 100                  # episodes in the success window
  max_steps_per_stage: 2000000 # stall safety: force-advance if a stage drags (null = off)
```

- **Mechanism:** training barrier starts wide/easy → narrows one `step` whenever
  rolling success ≥ `success_threshold` over `window` episodes → until `target`.
- `CurriculumCallback` (in `train_utils.py`) changes difficulty via the env hook
  `AntSwarmEnv.set_wall_length(v)`, **applied on next reset** (`_apply_pending`
  rebuilds layout / obs model / renderer and respawns — obs/action shapes
  unchanged, so the policy is unaffected).
- **Eval is pinned at `target_wall_len`** → eval metrics always reflect the real
  (hard) task; you watch eval success climb as the gap closes.
- When `enabled: true`, the static `walls.length` is **overridden** by the schedule.
- **Stall safety:** if a stage doesn't reach the threshold within
  `max_steps_per_stage`, it **force-advances** anyway — a too-hard stage can't park
  the curriculum forever (which would waste the rest of training at a sub-target gap).
- Logs `curriculum/wall_len`, `curriculum/success_rate`, and per-advance
  `curriculum/stage_steps` + `curriculum/stage_idx` (watch how long each stage takes).

### Budgeting training steps (multi-stage!)

Curriculum = training **N tasks in sequence** (here 8 stages), so the budget must
cover all stages **plus** real training at the final hard gap. Rough shape:
`total ≈ Σ(steps to master each stage) + buffer for the target stage`; early
wide-gap stages are quick, the last threading stages dominate.

- **PPO** default `--timesteps 50M` — generous (~17 h at ~800 fps; fits 1-day SLURM).
- **SAC** default bumped **5M → 15M** — 5M split across 8 stages left too little for
  the hard end.
- ⚠️ `max_steps_per_stage * n_stages` bounds the worst-case traversal (2M×8 = 16M):
  fine for PPO, **tight for SAC** — if SAC must not be cut off mid-curriculum, lower
  `max_steps_per_stage` (~1M) or raise SAC timesteps.
- Use the logged `curriculum/stage_steps` from a real run to right-size next time.

## Training infrastructure (callbacks & reproducibility)

- **`save_code(run_dir)`** snapshots `ant_swarm/` package + `config.yaml` + the entry
  script into `<run>/code/` — every run is self-reproducible.
- **`EpisodeMetricsCallback`** logs `rollout/success_rate` + `rollout/final_dist_mean`
  (reward/length already come from `VecMonitor`).
- **`SuccessTrajectoryCallback`** saves every successful episode to
  `<run>/successes/*.npz` (obs, actions, rewards + metadata). Successes are rare
  and precious here → reuse for BC / demos / warm-start / HER. **Each file records
  `timesteps` (training step found at) and `wall_len`/`gap` (curriculum difficulty
  solved at)** — essential context, since a solve at a wide gap ≠ a solve at the
  hard gap.
- **`RenderCallback`** saves a policy GIF every `render_freq` steps (disk + TB + W&B).
- Run dirs/names: `ant__<ts>__<jobid>__<script>__<single|multi>`; W&B logs into the
  run dir; outputs under `storage_local/`.

---

## TL;DR

- Keep SAC; run a baseline before adding machinery.
- If it stalls: **goal curriculum** > HER/TQC > RND.
- **No CNN** unless you randomise the scene.
- **DONE:** added 16 tip→wall-head distances (obs 9→25) to give the threading
  cue. Next obs candidates if needed: linear velocity, barrier-plane distance.
- **Motion:** `kinematic` mode (direct `[direction, rotation]`, no physics) is the
  simple default; `dynamic` (forces/momentum) still available. Dynamic sped up via
  `push_strength`/`object_inertia`; `spin_strength` now a config knob.
- **Horizon** cut to 500 (`gamma` 0.99); **gap-size curriculum** implemented
  (easy→hard, eval pinned hard, stall-safety force-advance + per-stage logging);
  successful trajectories saved with the training step + difficulty they were found at.
- **Budget for stages:** PPO 50M, SAC 15M; watch `curriculum/stage_steps` to
  right-size. Worst-case traversal = `max_steps_per_stage × n_stages`.
- **Goal tracked from big-cap centre** (`env.goal_track: big_cap`) removes the
  small-cap cheat — agent must lead with the hard end. Changed reward/objective →
  prefer training from scratch over warm-starting the old (cheat-trained) policy.
- **No argparse** — all run/algo settings live in `config.yaml` (`run:`,`ppo:`,`sac:`);
  set `run.eval`/`run.eval_model` to evaluate, `run.init_from` to warm-start.
