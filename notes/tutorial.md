# Tutorial notes

Longer explanations of ideas we use in this project. Simple English.
(Concept one-liners live in [rl_concepts.md](rl_concepts.md); hard words in [english_words.md](english_words.md).)

---

## 1. Geodesic reward shaping

### The problem with the current shaped reward

Our reward uses straight-line distance — "how far is the goal, as the bird
flies". But the bird flies over walls; the T cannot. So the reward praises
pushing straight into the wall, and it *punishes* the correct turn, because
during the turn the straight-line distance grows. The signal is not just weak —
it points the wrong way at the most important moment.

### The idea: route distance, not bird distance

Think of GPS navigation. Your GPS never shows bird distance. It shows the
length of the real road route. When you drive away from your destination to
reach a highway ramp, bird distance goes UP, but route distance goes DOWN —
you are making real progress along the only possible way. Geodesic reward is
exactly this: replace bird distance with route distance.

### What is the "route" for our T?

A pose of the T is three numbers: position x, y and angle θ. Imagine a 3-D grid
of all poses. Mark each pose "free" (no collision) or "blocked". A legal motion
of the T is a walk through neighboring free cells — one tiny move or one tiny
rotation per step. This grid is the **configuration space**; we already build
it in the geometry scanner.

### Building the distance field (once)

Run **BFS** from the goal pose: a wave spreading through free cells only. When
the wave reaches a cell, the cell gets a number — "how many small
moves/rotations to the goal along a legal route". The wave flows through the
slits, around the walls, through the turn, so the numbers contain all the
geometry. Save this 3-D table as a `.npz` file. That is the whole
precomputation.

### Using it (every step)

```
reward = field[old pose] - field[new pose]   (+ success bonus at the end)
```

One table lookup per step. Consequences:

- The T tilts in place in front of slit 1 → straight-line distance unchanged,
  but the route number DROPS → positive reward. **Turning counts as progress.**
- The T moves left to line up the long bar → bird distance grows, route number
  drops → still positive reward.
- The T pokes the small head into slit 1 (the dead end) → the route number
  JUMPS UP (the legal route from there is: back out, turn, re-enter) →
  immediate negative reward. **The trap punishes itself.**

### Why it is safe

This is potential-based shaping: reward = the change of one fixed function
Φ(state), here Φ = −field. A known theorem (Ng, Harada, Russell 1999) says such
shaping cannot change which policy is optimal. It only makes the good policy
easier to find.

### Why it is cheap for us

The expensive part — the free-pose grid — already exists (the scanner). We add
one BFS (minutes, once per geometry), one `.npz` file, and a small reward mode
in `RewardModel` that looks up the nearest grid cell for the current pose.

### The honest caveat

The field is tied to one fixed geometry. If walls change (curriculum stage,
randomization), each stage needs its own field. Fine for a few stages; not for
full randomization. Our reverse curriculum keeps the geometry fixed, so the
two combine perfectly.

---

## 2. "Is this not model-based RL?" — a fair question

Short answer: the *policy* stays model-free, but the *training signal* now uses
a model. We moved along the spectrum, and it is worth being honest about it.

### What "model-based" normally means

In model-based RL, the **agent** has or learns a model of the environment and
uses it to plan or imagine ahead (MPC, Dyna, MuZero). At decision time, the
agent asks the model "what happens if...".

### What we are doing

Our agent (SAC) stays model-free: it sees an observation, outputs an action,
learns from rewards. It never plans, never imagines. But **we, the trainers**,
used perfect knowledge of the env — the wall map and the goal — to precompute
the geodesic field and bake it into the reward. This is called **privileged
information at training time**: the teacher knows the map; the student policy
only ever sees its own observations.

So the correct label is something like: *a model-free policy trained with a
model-informed (planner-informed) reward*.

### The spectrum

```
pure model-free  ──────────────────────────────────────►  pure planning
sparse reward        geodesic-shaped reward           follow the BFS path,
+ exploration        (our plan)                       no learning at all
```

Notice the right end: with a perfect model and a fixed task, we could skip RL
entirely — the BFS path IS a solution. So why still do RL?

1. **Closed-loop robustness.** The BFS path is one open-loop trajectory. A
   policy learns to reach the goal from *everywhere*, and to recover after
   errors.
2. **Dynamic mode.** The grid knows geometry, not physics. With momentum and
   forces (and later, many ants pulling), the model is only partial — planning
   alone is not enough, learning is.
3. **The scientific goal.** We want decentralized multi-agent controllers and
   their learning dynamics — the interesting part is *how* coordination
   emerges, not just *that* the maze is solvable.

### The purity scale of our tools

Each training trick assumes a different amount of env knowledge:

| Tool | What it assumes |
|---|---|
| Sparse reward + exploration bonus (RND) | nothing about the env |
| HER | ability to relabel goals |
| Reverse curriculum | ability to set the start state (simulator reset) |
| Geodesic shaping | full map + goal (a model) |
| BC from BFS path | full map + a planner |

The lower rows learn faster but "know" more. If one day we randomize the maze
or go toward real-world settings, we must climb back up this table — the
crutches that need a map disappear first. This is a real trade-off, not a
flaw: for now the goal is to get the maneuver learned at all; later we can
remove crutches one by one and measure what still works.
