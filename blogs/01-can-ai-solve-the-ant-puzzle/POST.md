# Can an AI agent solve the piano-movers puzzle that ants solve?

## 1. The ants

I'm sure many of you have seen this video, showing a group of ants carrying a
T-shaped load through a box.

It is amazing to see how a few milligrams of insect, with no leader and no plan,
solve a motion-planning problem that we teach in robotics courses.

So I asked myself: can AI agents do the same?

<video src="assets/redvid_io_ants_solving_a_puzzle.mp4" controls width="720"></video>

▶ [assets/redvid_io_ants_solving_a_puzzle.mp4](assets/redvid_io_ants_solving_a_puzzle.mp4)

There might already be some works implementing this task. I did not even search
about it, because I wanted to do it myself — I was sure I would learn a lot
along the way.

## 2. The environment

So I rebuilt the maze as a gym environment, to see if RL agents can learn the
same maneuver or not.

The real system is multi-agent, and that is where this task gets even more
interesting, because the research behind this video [**Dreyer et al., PNAS
2025**] shows that ants get **better** in bigger groups. People get **worse!**

However, I started with a single agent, which should be easier for RL.

The environment is a small 2-D rigid-body simulator in pure NumPy,
with exact rectangle collisions.

<img src="assets/task_vocabulary.png" alt="The task and its words" width="720">

A single agent either moves the load directly (*kinematic*) or pushes it with real
forces and momentum (*dynamic*, the mode closest to the ants).

## 3. Try it yourself

If you want to try how this env works, I also built a simple but interactive
version of it here:

```bash
git clone https://github.com/RezaKakooee/ant-piano-movers-rl.git
# then open interactive/interactive_t.html in any browser
```

Left-drag to move the load, scroll to rotate it, **R** to reset.

Feeling the maze by hand is the fastest way to understand why an agent
struggles with it.

## 4. Now let's see how the env components are defined

**Observation.** Instead of giving the agent an image of the maze, I took the observation simple, as a collection of features — 25 numbers in total:

| Block                        | Count | Meaning                       |
| ---------------------------- | ----- | ----------------------------- |
| Attachment offset            | 2     | where this ant holds the load |
| Load position                | 2     | where the T is                |
| Goal vector                  | 2     | from the load to the goal     |
| Orientation                  | 2     | sin θ, cos θ                |
| Angular velocity             | 1     | how fast it spins             |
| Tip → slit-corner distances | 16    | 4 arm tips × 4 slit corners  |

The last block matters. Without it the agent has no idea where the walls are,
so it cannot know that it is about to hit one. And I did not want an image,
because then the agent would first have to learn to *see*. The hard part here
is not seeing, it is control.

**Action.** I implemented two versions. In kinematic mode the action is two
numbers: a direction to move in, and how much to rotate.

```
action = [ direction in [-pi, pi],  rotation in [-1, 1] ]
```

Each step the load moves 0.01 along `direction` and turns `0.1 * rotation`.
There is no mass and no momentum — the load simply moves.

In dynamic mode the agent pushes instead:

```
action per ant = [ push angle,  magnitude ]      (+ spin, for a single agent)
```

Having physics is even more interesting, and much closer to the ants: the load
has mass, it drifts, it overshoots. (The spin is there because a single agent
sits in the middle of the load, and pushing from the middle can never turn it.)

| Physics               | Value |
| --------------------- | ----- |
| Mass                  | 0.5   |
| Moment of inertia     | 0.01  |
| Linear friction       | 0.96  |
| Angular friction      | 0.94  |
| Substeps per env step | 10    |

**Reward.** For now I kept it as simple as possible: **+1 when the load reaches
the goal, and 0 everywhere else**. Nothing for getting closer, nothing for
trying. An episode ends after 500 steps if the agent did not make it.

## 5. The agent

For the RL agent I picked two algorithms: **PPO**, which is on-policy, and
**SAC**, which is off-policy. I wanted to see both.

The network is as simple as it gets: an MLP. Both algorithms are actor-critic,
so one part picks the action and another part judges it. They are two separate
small networks: PPO uses 64x64 for each, SAC uses 256x256. (SAC actually keeps
two critics instead of one)

Nothing special. No vision, no memory. If this task turns out to be hard, it
should not be because of the model.

## 6. Experiment 1: make it easy first

My first idea was the obvious one: start with a **wide** slit, and make it
narrower step by step as the agent gets better.

It worked. The agent passed every stage and reached the real narrow width.

Then I looked at what it actually does:

<img src="assets/cheat_smallhead.gif" alt="The small-head cheat" width="720">

It never does the maneuver. It slides the load through sideways, **small head
first**. The small head fits everywhere, so that is what the agent learned in
the wide stages — and it kept doing it. My reward did not complain, because I
only asked for the centre of the load to reach the goal.

So by widening the slit I did not make the task easier. I made it a
**different** task, with a solution the real task does not have.

## 7. Experiment 2: narrow slit, and ask for the big head

I narrowed the slit to make the task harder. And to encourage the agent to go
with the big head, I changed what counts as success: now the **big head** has
to arrive at the goal, not the place where the agent sits.

Now the agent has to do the real maneuver. But both PPO and SAC failed to learn
the task:

| Run                        | Steps  | Success |
| -------------------------- | ------ | ------- |
| PPO dynamic, sparse reward | 13.21M | 0%      |
| SAC dynamic, sparse reward | 2.94M  | 0%      |

### Why did it fail?

My first guess was the **sparse reward**. The agent is only paid when it
finishes. Before that, every state looks the same: reward 0. There is nothing
to climb, so there is nothing to learn from.

So I added a shaped reward: a small reward for every step that brings the load
closer to the goal. Now there is a signal everywhere, not only at the end.

It failed again:

| Run                        | Steps  | Success |
| -------------------------- | ------ | ------- |
| PPO dynamic, shaped reward | 14.06M | 0%      |

<img src="assets/policy_14000000.gif" alt="PPO after 14 million steps" width="720">

After 14 million steps it has learned exactly one thing: drive at the goal. It
pushes the load against the wall and waits there until the episode ends.

So the sparse reward was not the whole story. My second guess: **the space of
solutions is too tiny.** In most positions and angles the load would simply
overlap a wall, so they are impossible. And to pass a slit, the load has to be
in the right place *and* at the right angle at the same time — very few
combinations do that. How tight is it?
If I make the walls 3 mm thicker, the maze has no solution at all. Random
exploration does not find something like this.

## 8. Experiment 3: a curriculum

If the agent cannot find the solution by itself, it has to practise. It needs a
curriculum.

The two simple ways to build curriculum are:

**1. Widen the slit** and narrow it step by step. But in
Experiment 1 we saw that this teaches the agent to cheat.

**2. Cut the task into pieces.** Let it first learn to reach the goal from
inside the goal room. Then from inside the corridor. Then from around the real
start. And finally stitch the pieces together, so it can do the whole thing.

The maze never changes here — only where the episode begins:

| 1. start in the goal room                                                                                       | 2. start in the corridor                                                                                         | 3. start at the real start                                                                                       |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| <img src="assets/1_first__step1403_len403.gif" alt="a" width="720"> | <img src="assets/2_early__step150542_len31.gif" alt="b" width="720"> | <img src="assets/5_final__step421814_len79.gif" alt="c" width="720"> |
| easy — even a clumsy policy stumbles into the goal                                                             | the hard part: the turn between the two slits                                                                    | the whole task, from the real start                                                                              |

Then I ran SAC, and it was successful.

But when I looked closely, it was still cheating.

My first thought was that my arena was to blame — I had drawn it by hand, not
from the paper. So I rebuilt the maze with the real numbers:

<img src="assets/maze_dimensions.png" alt="The maze, with the real numbers" width="720">

It did not help. In the new maze the agent went in small head first again — all
20 solutions I checked.

<img src="assets/cheat_accurate_maze.gif" alt="Still cheating in the accurate maze" width="720">

Two things were wrong. My slit is still a bit too generous — in the paper it is
0.81 of the big head, in mine 0.86 — so this shortcut stays open. And my
curriculum dropped the load at a **random place and a random angle**, so nothing
ever asked it to reach the slit with the big head in front.

The stages were mine. They were not taken from a real solution of the maze.


## 9. Experiment 4: give the agent a teacher

At this point the agent had two problems. It did not know which moves were real progress, and my curriculum did not know which starting poses were useful.

So I decided to give it a teacher. Not a teacher that tells it exactly what action to take, but one that already knows a real route through the maze.

For that, I needed a map of every situation the load can be in. Here a "situation" is not only its position. Its angle matters too: the load can fit at one angle and hit a wall at another, even when its centre is in the same place. So each point on this map is a full pose: x, y, and angle.

Then I used BFS, or breadth-first search. You can think of it as flooding this map backwards from the goal. The flood can move from one pose to another only when that small move is legal and does not cross a wall.

When the flood is finished, every reachable pose has a number. The number says how many small legal moves are left before the goal. A pose with distance 100 is closer than one with distance 101, even if the straight-line distance says otherwise.

This one map gave me both things I was missing.

**A better reward.** My old reward used the straight-line distance to the goal. But the straight line goes through the walls. That is why the agent was rewarded for pushing against a wall. It was also punished for turning away from the goal, even though that turn is exactly what solves the maze.

The BFS number does not have this problem, because it counts the moves along the real route. So the new reward is simple: the agent gains when the number goes down, and loses when it goes up. Turning the load to line it up with the slit brings the number down, so turning now pays. Pushing the small head into the dead end sends it up, so the dead end costs.

**Better starting poses.** BFS also gives me a valid path from the start to the goal. I picked 16 poses along that path and used them as the curriculum stages.

The 16 curriculum stages

Read the colours from dark blue to dark red. Stage 0 is almost at the goal, and stage 15 is the real start. In other words, the picture shows the solution backwards, including the important turn between the two slits.

The agent learns it in the same order: first the last piece, then the piece before it, and so on. It moves to the next stage only when it reaches 90% success on the current one. A stage is never skipped just because it is taking too long.

## 10. Two last fixes, and the result

With the teacher in place I moved to the hard mode: **dynamic**, with real
forces and momentum. Two more things were needed there.

**Do not forget the rare win.** SAC learns from a big pool of past experience.
Wins through the slit are rare, so they drown in that pool and get forgotten. So
I keep successful episodes in a second pool, and force 25% of every training
batch to come from it.

**Let the agent feel its own speed.** With momentum the load drifts and
overshoots, but my observation only had position and angle — not speed. That is
like driving while you can see where you are, but not how fast you are moving.
Adding the velocity took the observation from 25 numbers to 27.

Then it worked. **100% success, learned in 247,519 steps** — about two hours on
one machine, and 56 times fewer steps than the 14-million-step run that learned
nothing.

<img src="assets/eval_ep01_len153_ret1.76.gif" alt="The solved task" width="720">

But the number I actually cared about was not the success rate. It was *how* it
solves the maze. So I replayed the last 25 solutions and checked which head goes
through each slit first:

```
slit 1 entered first by :  BIG head    25 / 25
slit 2 exited  first by :  small head  25 / 25
```

Every single one. Big head into the first slit, turn in the corridor, small head
out of the second — the same maneuver as the ants, in real physics, with no
shortcuts.

## 11. What is next

One agent can now do it. But one agent is not what the ants do.

The next step is **many agents on the same load**. Each one pushes at its own
point, sees only its own local view, and feels the others only through the load
it is holding. No leader, no plan, no messages — the same setting the ants are
in. That is the part I am really curious about: whether a group can find this
maneuver together, and whether it gets better with more agents, as the ants do
and people do not.

There is also plenty left to improve in what already exists. My slit is still a
bit wider than the paper's. The teacher is a strong crutch — I would like to see
how far one can get with a weaker one. And I only ran the teacher setup with SAC, never
with PPO.

## Addendum: what happened after this post

This post ended with one agent, one start, one goal. Three things happened
next, before the [swarm post](https://rezakakooee.github.io/ant-swarm-rl/).
They are the bridge between the two, so here they are, in the order they
happened.

### A. Any start, any goal

The agent above knew exactly one start pose and one goal. I wanted one that
can begin anywhere in the start room, at any angle, and reach a goal drawn
anywhere in the goal room.

First I measured how far the fixed-goal agent already got, with no extra
training:

| setting | success |
|---|---|
| its own task: fixed start, fixed goal | 100% |
| random start, fixed goal | 62% |
| fixed start, random goal | 12% |
| random start and random goal | 6% |

Random starts were half solved. Random goals were the real problem. And the
reason was simple once I found it: during training the goal never moved, so
the agent never learned to read the goal input. An input that never changes
is an input the network ignores.

The obvious fix is to make a BFS map for every goal. I did that for five
goals, trained on them one after the other, and got 100% on the two goals
the agent practised first and 0% on the other three. It had memorised, not
learned.

<img src="assets/any_start_any_goal.png" alt="The maze with the random start zone, the random goal box, and the two legs of the reward" width="720">

The fix that worked is a reward in two legs. The hard part of the maze, the
two slits and the turn, is the same for every goal. So the first leg uses
one map, flooded from a fixed exit point just past the second slit. Once the
whole load is through the second slit, the second leg takes over: plain
straight-line distance to the actual goal, which is safe in an open room.
One map, learned once, for every goal.

With this reward, and goals drawn from the whole box so there was nothing to
memorise, the agent reached **99 to 100%** on random starts and random
goals. The curriculum from section 8 turned out to be optional here: the same
run with no curriculum at all also reached 100%, at about 910,000 steps.
What made the task learnable from scratch was the map in the reward, not
the stages.

### B. Trying to drop the teacher

The map is a strong crutch. It knows the whole maze. I wanted to know how
far an agent gets with no map at all: sparse reward, a point only when the
load reaches the goal, random start and random goal, no curriculum. Plain
SAC had already failed at this in section 7. So I tried the four standard
tools for hard-exploration tasks, each as a switch in the same trainer.

<img src="assets/four_methods.png" alt="Four sketches: hindsight goal relabelling, an exploration bonus for new poses, exploration noise held for many steps, and Go-Explore's return-then-explore loop" width="720">

- **Hindsight experience replay (HER).** The agent almost never reaches the
  goal, so it almost never sees a reward. HER cheats in a useful way. After
  a failed episode, it pretends that wherever the load ended *was* the goal,
  and stores the episode again with that goal. Now every episode teaches
  something: how to get to that place. The hope is that "how to get to
  places" adds up to "how to get to the goal".
- **An exploration bonus.** A small extra reward for every pose of the load
  the agent has not seen often. Poses it visits all the time pay nothing.
  A new pose pays well. This pushes the agent toward the unknown even when
  the real reward is silent.
- **gSDE, slower exploration noise.** Ordinary RL adds a fresh random wobble
  to every action. That is fine for small steps, but a wobble that changes
  every step averages out to nothing over a long push. gSDE picks one random
  direction and keeps it for 64 steps, so the agent tries real, sustained
  moves.
- **Go-Explore.** The strongest of the four. Keep an archive of every pose
  the load has reached, with the exact actions that got there. Pick a
  promising pose, replay those actions to return to it exactly, then push
  randomly from there and add anything new to the archive. It never
  forgets a place it has been.

<img src="assets/frontier_wall.png" alt="The maze with the region reached by every method shaded: everything up to x = 0.900, in front of the second slit" width="720">

None of them crossed the second slit. The two Go-Explore runs got furthest,
to x = 0.900, just inside the corridor. The archive grew to 14,000 poses and
that frontier did not move by a millimetre. The HER runs barely left the
first room: they never got a crossing to relabel. The bonus ran out of new
poses on the near side. gSDE brought the load to the slit and no further.

| method | steps | success |
|---|---|---|
| HER | 1.7M | 0% |
| HER + exploration bonus | 1.6M | 0% |
| HER + gSDE | 2.0M | 0% |
| all three | 1.4M | 0% |
| Go-Explore, two grid sizes | 3.0M and 4.9M | 0 crossings |

The second slit needs the load in the right place *and* at the right angle
at the same time, and only a thin set of poses does both. Random pushes
never produce even one crossing, so none of these methods ever gets the one
example it needs. More samples do not help. A signal that says "turn away
from the goal now" is what is missing, and none of these tools provides
it.

### C. A teacher from demonstrations, and the bug that hid it

If the map cannot be dropped, maybe it can be hidden. Use it once, to make
demonstrations, then train a policy that copies them and never sees the map
again.

A planner that walks the load downhill on the map solved 34,777 episodes,
random starts and random goals, every step recorded. Then behaviour
cloning: a network that takes the load's pose, speed and the goal, and
outputs the planner's action. No reward, no map, no curriculum.

It scored 0%. Not once or twice: for eleven days, in every variant I could
think of. More data, five million transitions instead of half a million.
Action chunks. A frame stack. Noise on the actions. A discrete torque head.
DAgger. Each one 0 to 5%. The load reached the first slit and stopped.

The cause was one line of code. The observation module stored the goal by
reference when the environment was built. Every script that changed the goal
later replaced the array instead of writing into it. So the reward saw the
new goal, the success check saw the new goal, the renders showed the new
goal. The observation kept the first one. For all 4.9 million stored
transitions, the goal input was the same two numbers.

<img src="assets/frozen_goal.png" alt="The goal box with the real goals of the demonstrations scattered across it, and the single frozen goal the network saw" width="720">

Every policy in those eleven days was asked to reach a moving goal while
being told that the goal never moves. Of course it stopped at the first
slit: from there on, the route depends on the goal.

Two things I keep from this. First, the one experiment that ever moved the
number, replacing the goal input with a direction from the map, took
success from 0% to 84%. I read it as "the policy wants waypoints". It was
telling me "the goal input is broken". When one input swap changes
everything, check the input. Second, my evaluation had its own bug: it took
the first N episodes of the dataset, and the first 534 episodes all share
one start pose. Every success rate I had quoted was measured on one start.

With both fixed, and nothing else changed, the same behaviour cloning
scored **88 to 95%** on fresh random starts and goals. Then I put RL on top,
two ways. Let it change the whole policy: it collapsed to 0%, saved only by
keeping the best checkpoint. Let it add only a small bounded correction to
the frozen cloned policy: **97%**.

<img src="assets/il_numbers.png" alt="Bar chart: copying with the bug 0 percent, copying fixed 88 to 95 percent, RL with full freedom 0 percent, RL as a bounded correction 97 percent" width="720">

<img src="assets/residual_success.gif" alt="The cloned policy with the bounded RL correction solving a fresh episode" width="720">

This 97% policy, trained without ever seeing the map, is the single-agent
brain that the swarm post starts from.

## Come and play with it

The code is on GitHub, and it is open:

**[github.com/RezaKakooee/ant-piano-movers-rl](https://github.com/RezaKakooee/ant-piano-movers-rl)**

It is a normal gym environment, so you can drop your own algorithm on it in a
few lines. Everything in this post — the maze, the configs, the curriculum, the
BFS teacher — is in there, and the browser sandbox needs no install at all.

If you find a better way, break something, or make the agent cheat in a way I
did not think of, I would love to hear about it. Open an issue, send a pull
request, or just write to me. And if you work on collective behaviour or
multi-agent RL and this is close to your area, please get in touch — I would
enjoy the conversation.
