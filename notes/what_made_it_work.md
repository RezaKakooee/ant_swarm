# What Made It Work — Simple Explanation

The single-agent PNAS piano-movers task is solved, in dynamic mode (real forces
and momentum). This file explains, in simple words, the three changes that made
it work. Full status and history: [../ops/handoff.md](../ops/handoff.md).

Result: 100% success, mastered at 247,519 steps. Verified by replaying the last
25 solutions: **25/25 use the real ants' maneuver** — big head into slit 1, turn
in the middle chamber, small head out of slit 2. No shortcuts.

---

## 1. Better curriculum (pose-path)

**Before.** We started the T at a random spot inside an x-band and hoped it was
useful. Many of those starts were bad — impossible to solve, or already
touching a wall.

**Now.** We take the real solution path (from the BFS route through the free
pose space) and pick 16 points along it. Each stage starts the T exactly on the
correct route, only further back than the stage before.

**Why it helps.** The agent always practices the true maneuver, never a
shortcut. A stage advances only when the agent really masters it (90% success
over 100 episodes) — never because time ran out.

Code: `scripts/rl/pose_curriculum.py`, config `curriculum.mode: pose_path`.

## 2. Memory of past wins (success replay buffer)

**Before.** SAC learns from one big pool of old experience. Wins through the
slit are rare, so they drown in that pool and get forgotten.

**Now.** Successful episodes are also kept in a separate pool, and 25% of every
learning batch is taken from it.

**Why it helps.** In simple words: the agent is not allowed to forget how it
passed the slit.

Code: `scripts/rl/success_replay_buffer.py`.

## 3. The agent can now feel its own speed

**Before.** In dynamic mode the T has momentum — it slides and drifts. But the
agent could only see its position, not its speed. That is like driving while
seeing where you are, but not how fast you move.

**Now.** Linear velocity (vx, vy) is part of the observation (25 → 27 numbers).

**Why it helps.** The agent can correct for drift, and the state is Markov
again (see "partial observability" in [rl_concepts.md](rl_concepts.md)).

Config: `env.observe_linear_velocity: true`.

---

## One-paragraph version (for a paper or slide)

> We replaced the random-start curriculum with anchors sampled along the true
> solution path, kept a dedicated buffer of successful episodes (25% of each
> minibatch), and added linear velocity to the observation to remove partial
> observability under momentum.

## What was already there before these three

These three changes sit on top of the earlier ones (see
[tutorial.md](tutorial.md) and [rl_design_notes.md](rl_design_notes.md)):

- **Geodesic reward** — reward measures progress along the real route through
  the maze, not straight-line distance to the goal. Straight-line distance is
  deceptive here: it pushes the T into the wall and punishes the turn.
- **Goal tracked from the big cap** — removes the old "small-cap cheat".
- **Sparse success bonus + saving every solved episode** for later reuse.
