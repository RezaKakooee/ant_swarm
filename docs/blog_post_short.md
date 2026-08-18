# Teaching One Agent the Puzzle Ants Solve Every Day

**A weekend project. The short version.**
*(The long version, with all the failures and figures, is in
[BLOG_POST.md](blog_post.md).)*

---

## The puzzle

▶ **[Watch the ants first](../storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2__best/redvid_io_ants_solving_a_puzzle.mp4)** — 30 seconds.

A group of crazy ants carries a T-shaped load through a box with three rooms.
Two walls divide the rooms, and each wall has one narrow slit. The load is
wider than the slit.

They push the load in **big head first**, turn it inside the middle room, and
take it out **small head first**. No ant is in charge. No ant can see the whole
maze.

![The task and its words](figures/task_vocabulary.png)

This is a real experiment — **Dreyer et al., PNAS 2025**. The same maze was
given to ants and to groups of people. Ants get *better* in bigger groups.
People get worse.

Why it is hard, in three facts:

- The **big head is wider than the slit**. You cannot push the load straight in.
- The **middle room is shorter than the load**. You cannot freely spin it there.
- We checked every angle: at **no fixed angle** can the load cross the maze. It
  must turn *while* it moves. This is the classic **piano-movers problem**.

## Try it yourself (2 minutes)

```bash
git clone https://github.com/RezaKakooee/ant_swarm.git
# open interactive/interactive_t.html in a browser
```

Left-drag to move, scroll to rotate. Take the load through. Then try it small
head first — you will get stuck. That dead end matters later.

## Why I did this

I saw the video and could not stop thinking about it. A few milligrams of
insect, no leader, no plan, solving a motion-planning problem we teach in
robotics courses.

So I rebuilt the maze as a gym environment on my weekends, with one question:
**can a learning agent find the same maneuver by itself?**

The real system is multi-agent, and that is where this is going. But two hard
problems at once — the maneuver *and* the coordination — is a good way to learn
nothing. So: **one agent first.** That is this post.

## What failed

The environment is a small 2-D rigid-body simulator (exact rectangle
collisions, pure NumPy). The agent controls the load either directly
(*kinematic*) or by pushing with real forces and momentum (*dynamic*, closest
to the ants). Algorithm: SAC, an ordinary MLP policy.

Then the honest first try — train it and see:

| Attempt | Steps | Result |
|---|---|---|
| Random policy | — | 0% |
| SAC / PPO, reward only at the goal | 13M | 0% |
| PPO, **plus** a reward for getting closer to the goal | 14M | 0% |

The third row is the interesting one. Rewarding "get closer to the goal" is the
standard fix for a rare reward. Here it makes things **worse**:

- Straight-line distance points **through the wall**. The correct maneuver —
  tilt, back up, line the big head up — *increases* it. So the reward
  **punishes the correct move**. It is not a weak reward, it is a **lying** one.
- The free space is tiny: only ~30% of all poses are legal, and the channel
  through a slit is about **2.5 mm** wide. Random exploration never finds it.
- We also caught a **cheat**: the agent poked the *small* head through both
  slits, which fits anywhere, and scored well without doing the real maneuver.
  Fixed by measuring the distance from the **big head**, which must arrive.

## What worked: give the agent a teacher

The agent cannot *find* the solution. So it needs to practise — a curriculum.
But a curriculum needs to know what "easier" means, and where the solution is.

Our teacher is **BFS** (breadth-first search). We take every possible pose of
the load — position and angle — mark the legal ones, and run BFS outward from
the goal through legal poses only. Now every pose knows *how many legal moves
it is from the goal, along the real route*.

That one computation paid three times:

1. **A reward that tells the truth.** Reward = progress in route distance, not
   straight-line distance. Like a car navigation system: driving away from your
   destination to reach the motorway still counts as progress. Now **turning is
   progress**, and poking into the dead end **punishes itself**.
2. **Curriculum stages on the real route.** We take 16 poses along the BFS
   solution and use them as starting points: first almost at the goal, then
   step by step further back, until the real start. Every stage is on the
   correct route — so the agent never practises a shortcut. A stage advances
   only on real mastery (90% success), never because time ran out.
3. **Proof** the maze is solvable at all.

Two more fixes for the physics version:

- **Do not forget the rare win.** Successful episodes are kept in a separate
  pool, and 25% of every training batch comes from it.
- **Let the agent feel its speed.** With momentum, the load drifts — but the
  observation had position and angle, not velocity. We added it.

## Result

**100% success, mastered at 247,519 steps** — 56× fewer steps than the
14-million-step run that learned nothing.

![The solved task](../storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2__best/eval/eval_ep01_len153_ret1.76.gif)

And the part that matters: we replayed the last 25 solutions and checked which
head goes through each slit first.

```
slit 1 first :  BIG    25 / 25
slit 2 first :  small  25 / 25
```

**Every solution uses the ants' maneuver.** Big head in, turn in the middle
room, small head out — in real physics, with no shortcuts.

## Next

Many agents, one load. First a central controller, then decentralized policies
where each ant sees only its own view and feels the others only through the
load it is holding. That is the setting the real ants live in, and the question
this project was always about:

> How does a group with no leader and no plan discover a maneuver that none of
> its members understands?

---

*Full write-up with every failure, figure and number: [BLOG_POST.md](blog_post.md)
· Code: [github.com/RezaKakooee/ant_swarm](https://github.com/RezaKakooee/ant_swarm)*
