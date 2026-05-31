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

---

## TL;DR

- Keep SAC; run a baseline before adding machinery.
- If it stalls: **goal curriculum** > HER/TQC > RND.
- **No CNN** unless you randomise the scene. Instead add **linear velocity** (a
  real partial-observability gap) and a couple of **gap-clearance scalars**.
