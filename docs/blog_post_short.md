# Teaching One Agent the Puzzle That Ants Solve Every Day

<video src="../storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2__best/redvid_io_ants_solving_a_puzzle.mp4" controls width="720"></video>

▶ **[Watch this first](../storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2__best/redvid_io_ants_solving_a_puzzle.mp4)** — 30 seconds of
real ants.

A group of crazy ants is carrying a T-shaped load through a box with three
rooms. Two walls divide the rooms, and each wall has one narrow slit. **The
load is wider than the slit.**

Watch what they do: they push the load in **big head first**, turn it inside
the middle room, then take it out **small head first**. Nobody is in charge.
No ant can see the whole maze. The maneuver is what *comes out* of the group.

This is a real experiment — **Dreyer et al., PNAS 2025**, from Ofer
Feinerman's lab. The same maze was given to ants and to groups of people. Ants
get **better** in bigger groups. People get **worse**.

I spent a few weekends trying to teach one reinforcement-learning agent to do
the same thing. It failed for a long time. This is the short story of why, and
what finally worked. *(The full version, with every experiment, is in
[BLOG_POST.md](blog_post.md).)*

---

## 1. The task, and the words for it

![The task and its words](figures/task_vocabulary.png)

Everything below uses these words: the **load** with its **big head** and
**small head**, three **rooms**, two **walls**, two **slits**, a **start pose**
and a **goal**.

Now the numbers that make it hard:

![The maze and its key dimensions](figures/maze_geometry.png)

- The **big head (0.175) does not fit the slit (0.150)**. You cannot push the
  load straight through — it has to be tilted and threaded.
- The **middle room (0.221) is shorter than the load (0.359)**. So you cannot
  freely spin it in there either. The turn has to borrow space from both slits.

And one more fact, which we measured rather than guessed:

![The legal poses, at four fixed angles](figures/config_space.png)

Each panel fixes the load's angle and shows where its centre may sit — white is
legal, black collides. We tested **every** angle: at **no fixed angle** can the
load get from the left room to the right room. Turning is not an optimisation
here, it is the only way through. That is the classic **piano-movers problem**.

## 2. Try it yourself (2 minutes)

```bash
git clone https://github.com/RezaKakooee/ant_swarm.git
# then open interactive/interactive_t.html in any browser
```

Left-drag to move, scroll to rotate, **R** to reset. The load and the walls are
rigid, so you cannot cheat through them.

Take the load to the goal. Then try it **small head first** — you will get
stuck. That dead end matters later in this story.

## 3. Why I did this, and how I started

I saw the ant video and could not stop thinking about it. A few milligrams of
insect, no leader, no plan, solving a motion-planning problem that we teach in
robotics courses. So I rebuilt the maze as a gym environment on my weekends,
with one question: **can a learning agent find the same maneuver by itself?**

The real system is multi-agent, and that is where this is going. But it hides
two hard problems inside each other — the **maneuver** and the
**coordination** — and trying to learn both at once is a good way to learn
nothing. So: **one agent first**. That is this post.

The environment is a small 2-D rigid-body simulator in pure NumPy, with exact
rectangle collisions. The agent either moves the load directly (*kinematic*) or
pushes it with real forces and momentum (*dynamic*, the mode closest to the
ants). The algorithm is SAC with an ordinary MLP policy — the difficulty here
is not the model.

## 4. What failed

The honest first move is to just train it and look.

| Attempt | Steps | Result |
|---|---|---|
| Random policy | — | 0% |
| SAC / PPO, reward only at the goal | 13M | 0% |
| PPO, **plus** a reward for getting closer to the goal | 14M | 0% |

![Failed dynamic policy](../storage_local/ant__20260602_2331__13328496__train_ppo__single/renders/policy_14000000.gif)

*PPO after 14 million steps. It learned exactly one thing: drive at the goal.
It pushes the load into the wall and stays there until the episode ends.*

The third row is the interesting one. Rewarding "get closer to the goal" is the
standard fix when the real reward is too rare. Here it makes things **worse**:

![Straight-line distance vs route distance](figures/reward_euclidean_vs_geodesic.png)

*Left: straight-line distance to the goal. The colour flows smoothly through
the walls as if they were not there — so the reward tells the agent to drive
into them, and it **punishes** the tilt-and-back-up that the real maneuver
needs. That reward is not weak, it is **lying**.*

There were two more problems underneath:

- **The free space is tiny.** Only ~30% of all poses are legal at all, and the
  channel through a slit is about **2.5 mm** wide. Random exploration simply
  never finds it, and then has to follow it for ~150 steps in the right order.
- **The agent found a cheat.** Our first "successes" were fake:

![The small-head cheat](../storage_local/ant__20260601_1714__13102512__train_sac__single/success_gifs/cheat_smallhead.gif)

*A "94% success" policy from that era, at the full narrow gap. Look at which
end leads: the small head goes first through both slits, and the big head never
enters. We checked the last 12 solutions of that run — 12 out of 12 did this.
The score was real; the behaviour was not what we wanted.*

The fix was to measure the distance from the **big head**, so the hard end is
the one that must arrive. The cheat disappeared — and the success rate honestly
dropped back to 0%.

## 5. What worked: move the start, and hire a teacher

The agent cannot **find** the solution by luck. So it has to practise — a
curriculum. That raises two questions.

**First: what do we make easier?** The obvious idea is to widen the slit and
narrow it slowly. It is a trap: a wider slit is not the same task, it is a
*different* task where the small-head trick works. The agent learns that trick
and it does not transfer back. So the rule became:

> Never change the maze in a way that changes which solutions exist. Change
> **where the episode starts** instead.

Keep the maze at full difficulty, start the load almost at the goal, and move
the start backwards as the agent succeeds. That alone solved the kinematic
version:

| First success (step 1,403) | After mastery (step 421,814) |
|---|---|
| ![first](../storage_local/ant__20260812_0033__20309480__train_sac__single__sac_kin_rev/success_gifs/1_first__step1403_len403.gif) | ![final](../storage_local/ant__20260812_0033__20309480__train_sac__single__sac_kin_rev/success_gifs/5_final__step421814_len79.gif) |
| Lucky wandering, 403 steps, started next to the goal | The real task, from the real start, 79 steps — near optimal |

**Second: backwards along what?** We were dropping the load at random spots in
a band, and many of those poses were useless — impossible, or already touching
a wall. We needed something that actually **knows the maze**.

That teacher is **BFS** (breadth-first search). Take every possible pose of the
load — position *and* angle — keep the legal ones, and spread a wave outward
from the goal through legal poses only. When it stops, every pose knows *how
many legal moves it is from the goal along the real route*.

One computation, three payoffs:

1. **A reward that tells the truth** — the right panel of the figure above.
   Reward = progress in **route** distance. Like a car navigation system:
   driving away from your destination to reach the motorway still counts as
   progress. Now **turning is progress**, and poking into the dead end
   **punishes itself**.
2. **Curriculum stages that sit on the real route:**

![The 16 curriculum stages](figures/curriculum_anchors.png)

*The 16 training stages, drawn as the load itself. Read it from dark blue
(stage 0, almost at the goal) to dark red (stage 15, the real start) and you
are reading the solution backwards — the turn included. The agent practises
this sequence in reverse. A stage advances only on real mastery (90% success),
never because time ran out.*

3. **Proof** that the maze is solvable at all.

Even then, the kinematic policy found one more legal-but-unwanted trick — poke
the small head in, then pirouette inside the slit (the real experiment's
transparent covers prevent this):

| The maneuver we want | The trick the agent found |
|---|---|
| ![big first](../storage_local/ant__20260812_2304__20381854__train_sac__single__pnas_sac_kin_rev_geo/success_gifs/final_bigfirst.gif) | ![pirouette](../storage_local/ant__20260812_2304__20381854__train_sac__single__pnas_sac_kin_rev_geo/success_gifs/final_pirouette.gif) |
| Big head enters, turn in the middle, small head exits | Small head pokes in, then spins inside the slit |

Putting the curriculum stages exactly on the BFS route is what finally removed
it — the agent only ever practises the canonical maneuver.

Two last fixes for the physics version:

- **Do not forget the rare win.** Successful episodes are kept in a separate
  pool, and 25% of every training batch is drawn from it. In plain words: the
  agent is not allowed to forget how it got through the slit.
- **Let the agent feel its own speed.** With momentum the load drifts, but the
  observation had position and angle and **not velocity** — like driving while
  seeing where you are but not how fast you move. We added it.

## 6. Result

**100% success, mastered at 247,519 steps** — about two hours on one machine,
and **56× fewer steps** than the 14-million-step run that learned nothing.

![The solved task](../storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2__best/eval/eval_ep01_len153_ret1.76.gif)

*One evaluation episode from the real start pose: big head into the first slit,
turn in the middle room, small head out of the second. 153 steps.*

A score of 100% was not enough for me, though — I wanted to know **what** it
actually does. So we replayed the last 25 solutions and measured which head
crosses each slit first:

```
slit 1 first :  BIG    25 / 25
slit 2 first :  small  25 / 25
```

**Every single solution uses the ants' maneuver.** Big head in, turn in the
middle room, small head out — in real physics, with no shortcuts.

Here is the whole journey in one picture:

![All experiments](figures/experiments_overview.png)

*Every experiment we ran. Red bars never solved the maze, no matter how long
they ran. The blue bar is the final solution — the smallest bar on the chart.*

## 7. What I would tell my past self

1. **Do not make the environment easier if that changes the solution.** Move
   the *start*, not the walls.
2. **A dense reward only helps if it points the right way.** Straight-line
   distance was worse than no reward at all.
3. **Measure your solution space before blaming your algorithm.** "A 2.5 mm
   channel in a 3-D pose space" explains the failures better than any
   hyperparameter sweep.
4. **Check what the agent did, not just its score.** Two of our "successes"
   were cheats.
5. **A good teacher pays several times.** One BFS gave us the proof, the
   reward, and the curriculum.

## 8. Next: back to the ants

Single agent is done. Now the interesting part: **many agents, one load**.
First a central controller, then decentralized policies where each ant sees
only its own view and feels the others only through the load it is holding.
That is the setting the real ants live in, and the question this project was
always about:

> How does a group with no leader and no plan discover a maneuver that none of
> its members understands?

---

*Full write-up: [BLOG_POST.md](blog_post.md) · Code and environment:
[github.com/RezaKakooee/ant_swarm](https://github.com/RezaKakooee/ant_swarm)*
