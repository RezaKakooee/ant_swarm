# 04 — Multi-agent: the physics is fine, the sum is not

Period: 2026-09-05 → 2026-09-06.
Question: chapter 03 solved the task with one ant. Can N ants, each seeing only
its own observation and deciding alone, do the same?
Answer: **not yet. Five decentralised attempts all scored 0%. But the obstacle
was measured, and it is not the one anyone expected: per-ant accuracy is fine,
and the summed torque is not.**

Evidence used for every number:

| source | what it holds |
|---|---|
| `storage_local/ant__20260905_22*__local__multiagent_split*` | wrench-split probe + GIFs at n = 1…100 |
| `storage_local/ant__2026090[56]_*__shared_ant_*` | six supervised runs (3 heads × 2 history lengths) |
| `storage_local/ant__20260906_1429__24106[34]__marl_n4` | the two MARL arms |
| `scripts/rl/test_multiagent_split.py` | the reference controller |
| `scripts/rl/train_shared_ant_policy.py` | shared per-ant policy, all three heads |
| `scripts/rl/train_marl.py` | parameter-sharing PPO |

## Key terminology

(Terms from chapters 01-03 are not repeated.)

- **Wrench** — the (force, torque) pair applied to the load. The single ant commands one directly; N ants can only produce one through the geometry of their pushes.
- **Arm** `a_i` — vector from the load centre to ant i's attachment point.
- **Wrench split** — solving `sum f_i = F`, `sum a_i x f_i = T` for the per-ant forces. 3 equations, 2n unknowns; the least-norm solution is taken.
- **Parameter sharing** — one network, run once per ant on that ant's own row. Permutation-equivariant and independent of N.
- **Catastrophic cancellation** — a sum of large opposing terms with a small net result. The signal cancels; the errors do not.

## 1. What changes with N ≥ 2

| n | observation | action |
|---|---|---|
| 1 | (1, 27) | (1, 3) — angle, force, **spin** |
| ≥2 | (n, 27) | (n, 2) — angle, force |

With one ant the env grants a direct spin torque. With two or more there is no
spin action at all: **every bit of rotation must come from pushing at different
points**. That is the coordination problem, and it is the whole chapter.

Rows of the observation differ in exactly one place: `obs[0:2]`, the ant's own
attachment offset. Everything else — load pose, goal vector, velocity, barrier
distances — is shared.

## 2. The physics is fine (option C)

Drive N ants with the solved single-ant policy by splitting its wrench:

| ants | SR % | mean dist | force-capped steps |
|---|---|---|---|
| 1 | 95.0 | 0.0649 | 0% |
| 10 | 95.0 | 0.0649 | 0% |
| 50 | 95.0 | 0.0649 | 0% |
| 100 | 95.0 | 0.0649 | 0% |

Identical to four decimal places, and nothing ever hit the per-ant force cap.
**N ants can reproduce any wrench the single ant can, up to at least 100.** So
multi-agent is a learning problem, not a capability problem.

The least-norm solution has a closed form. With `abar` the mean arm and
`ac_i = a_i - abar`:

    f_i = F/n + lambda * perp(ac_i),      lambda = (T - abar x F) / sum_j |ac_j|^2

An equal share of the push, plus a sideways push scaled by the ant's arm
*measured from the centroid*. Verified against `lstsq` to 1e-16 on asymmetric
arms. **The `abar` terms are not optional.** The first version of the
decentralised base (§7) dropped them — exact for the symmetric n=2 layout, and
wrong for random attachment points: at n=4 the torque flipped sign and the
controller scored 3% where the exact split scores ~95%. The two parts are
orthogonal once centred: the first adds no torque, the second adds no force.

**A bug found here, worth keeping.** `ObservationModel` normalises linear
velocity by `push_strength * n_ants / mass / damping`. The scale is proportional
to N, so a policy trained at one ant count reads velocities at the wrong size at
another. The first probe scored 0% for n ≥ 2 purely because of this; undoing the
factor restored 90%. Any transfer across N must handle it.

## 3. Three supervised attempts, all 0%

One shared network, each ant seeing only its own row, no ant-count input (the
user's explicit choice). Trained on n = 2,4,8,16,32; tested on 2,3,4,10,50,100.

| head | what the network outputs | n=2 | n=4 | n=100 |
|---|---|---|---|---|
| `direct` | `f_i` | 0.0% | 0.0% | 0.0% |
| `wrench` | `(c_F, lambda)`, then the exact formula | 0.0% | 10.0% | 0.0% |
| `wrenchlog` | the same with a log scale + relative loss | 0.0% | 0.0% | 0.0% |

History (K=4) moved n=2 from 0% to 6.7% once and changed nothing else.

## 4. The measured obstacle

The fit is good: training MSE 7e-4, each ant 1-5% accurate. So the failure is
not representation. Measuring the **summed** wrench on the teacher's own
trajectory:

| ants | per-ant error | total force error | total **torque** error |
|---|---|---|---|
| 2 | 4.6% | 13.7% | **90.2%** |
| 4 | 3.2% | 16.0% | 77.2% |
| 10 | 1.6% | 19.0% | 76.5% |
| 50 | 0.9% | 48.4% | 136.9% |
| 100 | 1.3% | 136.0% | **356.7%** |

Errors do not cancel. They add. At n=100 each ant is 1.3% wrong and the load
receives a torque 357% wrong.

**Why torque fails first.** The push term `F/n` is large and points the same way
for every ant, so errors are small beside it. The turn term
`lambda * perp(a_i)` points a different way for each ant: it is a sum of large
opposing contributions with a small net result. The signal cancels, the errors
survive. This is catastrophic cancellation, and it worsens with N because each
ant's share shrinks faster than its error does.

**One thing that is NOT the problem.** A probe trained to predict the swarm size
from a single ant's own row:

| history | R² | correct ant count |
|---|---|---|
| 1 frame | 0.864 | 84.9% |
| 4 frames | 0.944 | 93.1% |

The ants can read N from the load's response — `obs[9:11]` is velocity divided
by a scale proportional to N. **No ant-count input is needed.** The information
is there; the arithmetic of assembling it is what breaks.

## 5. Two MARL arms, both 0%

Parameter-sharing PPO, shared actor-critic, team reward, 4 ants, 3M steps.
Each ant sees only its own row.

| ants | sparse reward | geodesic_exit reward |
|---|---|---|
| 2 | 0.0% | 0.0% |
| 4 (trained) | 0.0% | 0.0% |
| 10 | 0.0% | 0.0% |
| 50 | 0.0% | 0.0% |
| 100 | 0.0% | 0.0% |

The sparse arm returned exactly 0.000 for 3M steps — not one success, the same
wall as chapter 02, now confirmed for multi-agent. The geodesic arm did pick up
the shaping signal (episode return ~0.11) but never converted it into a success,
and its mean distance (0.62-0.75) is worse than the supervised runs (0.53).

**The curriculum, added (job 241648).** The last untested cell: five ants (spread
layout, §9), own row only, `geodesic_exit` reward **and** the chapter-01
pose-path curriculum, driven from the PPO loop with the same rules (a stage
advances at 90% over 100 episodes that started at that stage). 6M steps.

| what | result |
|---|---|
| stage 0 (anchor nearest the goal) | mastered at 493k steps |
| stage 1 | held for the remaining 5.5M steps — pinned success oscillated 0.2-0.88, never 0.9 over 100 |
| stages mastered, of 16 | 1 |
| real task (random starts), 30-episode evals | 0.0% at every checkpoint; final **0.0%**, mean distance 0.5969 m |

So the curriculum does not rescue decentralised PPO either: it cannot stabilise
the second-easiest sub-task. Pure decentralised RL from scratch has now failed
three ways — sparse, geodesic, geodesic + curriculum.

**The single-agent control, run later (job 241646, chapter 01 addendum).** The
same `geodesic_exit` reward, from scratch, no curriculum, ONE ant: 100% by
~910k steps. So the reward alone is sufficient for one ant and insufficient for
five. The gap is the decentralisation, not the reward and not the curriculum.

RL was chosen precisely because it does not have to reproduce an exact formula —
the reward measures the load, not per-ant accuracy. That weaker requirement was
still not enough within 3M steps.

## What worked

1. **Probing feasibility before spending compute.** Option C took an hour, cost
   no training, and proved the physics allows the task at any N. Without it,
   five failures would have been ambiguous.
2. **Measuring the summed wrench, not just the loss.** Training MSE said the
   policy was good; the wrench measurement said exactly where and how it was
   bad. That one table is the chapter's finding.
3. **Testing the assumption instead of arguing about it.** The swarm-size probe
   settled the ant-count question with a number rather than an opinion.
4. **A closed-form reference controller** that works at every N — useful later
   as a teacher, an upper bound, or a sanity check.

## 6. On reflection: two flaws in the attempts above

1. **The `wrench` head was not identifiable.** It was supervised on `f_i`, not
   on `(c_F, λ)`. For one ant, `f_i = c_F + λ·perp(a_i)` is two equations in
   three unknowns, so every ant was free to pick a different `(c_F, λ)` that
   fits its own target. The coefficients were never forced to agree across
   ants — which is exactly what the torque needs. Supervising the coefficients
   directly (the teacher knows them) would close that hole. That negative result
   is therefore weaker than section 3 makes it look.
2. **The MARL critic was decentralised.** Proper MAPPO trains a *centralised*
   critic on the global state and executes decentralised actors. Section 5 ran
   the weak variant: each ant's critic saw only its own row. Credit assignment
   under a team reward was correspondingly poor.

## 7. Proposed next design (not run)

Every attempt so far asks each ant to *invent* its share of a global wrench from
scratch each step; the shares must then sum exactly. That is fragile by
construction. The alternative is the recipe that actually worked in chapter 03:
a frozen base that is roughly right, plus a bounded learned correction.

| piece | what it does | why |
|---|---|---|
| base: single-ant BC on the ant's own row | gives `(F, T)` for the load | 88-95%; each ant can rebuild the n=1 observation from its row (§2) |
| base: estimate N from velocity | the ant's own swarm-size guess | R² 0.94, no input needed (§4) |
| base: the split formula with the ant's own arm | `f_i = F/n + λ·perp(a_i)` | exact, no learning |
| residual actor, ±0.15 | learned correction on `[angle, force]` | the one RL recipe that worked (chapter 03 §6) |
| centralised critic, training only | sees all rows | fixes credit assignment; execution stays decentralised |

**Why it should escape the cancelling sum.** All ants estimate N from the *same*
shared observation, so their errors are *correlated*. A shared N error scales
every ant's force by the same factor — a uniform scaling of the wrench, which
keeps its *direction*. Independent per-ant errors broke direction; correlated
ones do not. And the residual closes the loop: an under-turning load shows up
in `obs[8]` and gets corrected, which open-loop regression could never do.

**First step, before any RL:** run the decentralised base alone — BC per row,
N estimate, formula — with no learning. Hours, not days. 50%+ means residual RL
has a strong start; 0% means the N estimate is the bottleneck and must be fixed
first. Same move as §2, one level down.

Constraints kept at execution: own row only, no ant-count input, no map, no
curriculum. Grey areas, unchanged from before: the single-ant BC teacher (the
chapter 03 demos) and a centralised critic that sees the global state during
training only.

## 8. The base controller, run (first step of §7)

`scripts/rl/test_decentralised_base.py`, job 241562. Each ant, from its own row
only: single-ant BC → `(F, T)`; a swarm-size estimate from the velocity channel
(4-frame probe, held-out R² 0.981); the exact split with its own arm. No learning
in the loop. Global knowledge is removed one piece at a time, 30 episodes per cell:

| level | each ant knows | n=2 | n=4 | n=10 | n=50 | n=100 |
|---|---|---|---|---|---|---|
| 0 | true N, true centroid + spread | 100 | 100 | 100 | 100 | 100 |
| 1 | true N, **geometry constants** | 100 | 10.0 | 36.7 | 96.7 | 96.7 |
| 2 | **estimated N**, geometry constants | 36.7 | 0.0 | 30.0 | 63.3 | 73.3 |
| — | N the ants believed at level 2 | 79 | 38 | 25 | 76 | 106 |

Three things fall out.

1. **Level 0 is exact.** The per-ant formula with true globals reproduces the
   centralised split at every N. (It did not on the first run: the mean-arm term
   was missing — see §2.)
2. **The geometry constants fail at small N.** Replacing the true attachment
   centroid by the rule's *expected* centroid is fine at n=50-100 (law of large
   numbers) and wrong at n=4-10, where four random points sit far from their
   expectation. Nothing decentralised can recover the true centroid from one row.
3. **The N estimate is wildly off in closed loop** — 79 at n=2, 38 at n=4 —
   despite R² 0.98 on held-out data. First suspicion: `ObservationModel` divides
   velocity by `n_ants`, the total force is `n_true × (F / N_est)`, so the
   observed velocity is `∝ F / N_est` and the probe would just read back the
   ants' own assumption. **Tested and refuted.** Force the ants to assume
   `N_assumed`, run the base, ask the probe:

   | n_true | N_assumed | 10 | 25 | 50 | 100 | 200 |
   |---|---|---|---|---|---|---|
   | 50 | N_read | 35.0 | 43.6 | **51.8** | 42.2 | 32.9 |

   The reading does not follow the assumption. It stays near 50 when the
   assumption is right and collapses toward a mid-range value when it is wrong.
   The linear-response argument fails because the maze is contact-rich: a
   wrench sized for the wrong N jams the load, velocity falls to ~0 in both
   directions (|obs vel| 0.0029 at N_assumed=10, 0.0074 at 50, 0.0015 at 200),
   and a probe trained only on correctly-driven loads has nothing to read. The
   same test at n_true=4 was confounded the same way — the level-1 controller
   scores 10% there, so most steps are stuck — and gave readings that ran
   *opposite* to the assumption.

   So the failure is **distribution shift, not unobservability**: the probe was
   trained on exact-split rollouts, where N leaks through the per-ant force
   size and the load never jams. In the loop, from a wrong first guess, the
   load jams, the probe defaults to ~30-40, and at small n_true that default is
   the fixed point. At n=50-100 the default is close enough and the loop is
   self-consistent (63-73%).

**Two counter redesigns, both worse (jobs 241565, 241568).** Fix (i): train the
counter on rollouts where the ants *assume* a random N (log-uniform 1-200,
re-drawn every 25 steps), so it sees jammed and mis-driven loads. Fix (ii): feed
it its own previous estimate. Each ant keeps its own running estimate from a
prior of 16 — the earlier run had quietly averaged estimates across ants.

| level 2, counter | n=2 | n=4 | n=10 | n=50 | n=100 |
|---|---|---|---|---|---|
| naive (exact-split data) | 36.7 | 0.0 | 30.0 | **63.3** | **73.3** |
| randomised data + feedback | 36.7 | 3.3 | 20.0 | 50.0 | 23.3 |
| randomised data, no feedback | 33.3 | 3.3 | 20.0 | 40.0 | 20.0 |
| level 1 ceiling (true N) | 100 | 10.0 | 36.7 | 96.7 | 96.7 |

All three counters score R² 0.97-0.98 on held-out data. In the loop the two
redesigns believe N ≈ 9-40 whatever the truth is. The ablation pins it on the
data: filling the training set with jammed, mis-driven loads — states that carry
no information about N — teaches the counter to output the average label. The
feedback input then just lets it stay there. The naive counter, trained only on
correctly-driven loads, at least reads the well-driven signature when the loop
happens to be close, which is why it holds at n=50-100 and nowhere else.

**Lesson, stated once.** Offline accuracy of a component says nothing about its
behaviour inside the loop it will run in. This is the third time in this chapter
(supervised heads → summed torque; counter v1 → jammed loads; counter v2/v3 →
regression to the mean), each one level further down.

**The signal itself (job 241569).** `env.velocity_scale_ants: 1` makes the
velocity channel absolute instead of divided by `n_ants`. The naive counter's
offline R² falls from 0.98 to **0.54**, and that is the finding: with the exact
split, a correctly driven load moves *identically at every N*. The only
N-signal the earlier counters ever had was the normalisation — a bookkeeping
artefact, not physics. Remove it and there is nothing to read. N is visible
only when the wrench is *wrong* (response ≠ command), and reading it then
requires knowing the command — the feedback design, which regressed to the mean
on jammed data. Four counters, four failures, and a reason.

| level 2, absolute velocity, naive counter | n=2 | n=4 | n=10 | n=50 | n=100 |
|---|---|---|---|---|---|
| success % | 0.0 | 0.0 | 30.0 | 70.0 | 50.0 |
| N the ants believed | 6 | 37 | 25 | 29 | 27 |

Note the n=50 cell: 70% with a belief of 29 — a factor-2 magnitude error
tolerated. So: how much does the count matter? Jobs 241577-81, no counter at
all, every ant assumes a fixed N:

| assumed N → / true N ↓ | 5 | 15 | 30 | 60 | 120 |
|---|---|---|---|---|---|
| 2 | 3.3 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 10 | 10.0 | 6.7 | 0 | 0 | 0 |
| 50 | 0 | 0 | 46.7 | **90.0** | 3.3 |
| 100 | 0 | 0 | 3.3 | 53.3 | **90.0** |

No constant covers more than one swarm size. The band for ≥50% is roughly
**0.6×-1.2×** of the true N; assuming too *many* (under-pushing) is far worse
than too few. Small N fails at every assumption — the centroid problem (level
1) plus N-sensitivity on top.

**Conclusion of §8.** The decentralised base as designed in §7 cannot work
across swarm sizes without an N input: the ant needs N to within ~±40%, and N is
not recoverable from its own row while the swarm is driving the load correctly
(absolute-velocity counter, R² 0.54). This is a property of the exact split,
not of any counter — the better the swarm performs, the less its own motion
reveals about its size. Two ways out, both changing the design: give the ant N
(which the user has ruled out), or drop the formula-based base entirely and let
a closed-loop policy discover behaviour that does not need N — for example,
ants that push, watch the load's response, and scale by it. That is MARL again,
and MARL from scratch has failed twice here (§5).

## 9. MARL without a formula: DAgger at fixed N=5 — the first non-zero result

**In plain terms — what this method is, and is not.**

| piece | what it is |
|---|---|
| Brain | the single-agent IL checkpoint (chapter 03). Given the load's state, it says how the load should move: one push, one turn. |
| Teacher | maths (least-norm split) divides that into five pushes, one per ant. Needs all five positions — centralised. ~92%. |
| Student | a **new** shared network. Each ant feeds it only its own row and gets only its own push. No formula, no ant count, no view of the others. Trained by copying the teacher with DAgger. ~91%. |
| RL | **not used.** Proposed as the next step (residual on top of the student), not run. |

So this is *not* "fine-tune the single-agent checkpoint with RL in a swarm". The
checkpoint is never changed; a new per-ant network is trained from scratch by
imitation. One line: **a centralised controller copied into a decentralised one.**

The user's call after §8: a learned ant, no formula and no N at execution, at
about five ants rather than a hundred. Third attempt at a decentralised
learner; what differs from the two that failed (§3, §5):

| earlier | now |
|---|---|
| from scratch (RL) or plain BC | **DAgger**: roll out the student, label its states with the teacher, retrain |
| N varied 2-32 in one regression | **one fixed N=5** — no 1/N scaling in the targets |
| — | history option (4 frames) |

Teacher = the split controller (single-ant BC + least-norm split), used only as
a labeller. Each ant runs one shared MLP on its own row and outputs its own
force. Attachment layout fixed (seed 0); teacher scores 95% on it.
`scripts/rl/dagger_shared_ant.py`, jobs 241582 (h=1), 241583 (h=4), 40 episodes
per round, 30-episode evals.

| round | samples | h=1 SR % | h=4 SR % |
|---|---|---|---|
| 0 (BC only) | 34k | 0.0 | 0.0 |
| 2 | 234k | 3.3 | 0.0 |
| 4 | 430k | 0.0 | 13.3 |
| 5 | 525k | 23.3 | 30.0 |
| 6 | 600k | 36.7 | 43.3 |
| 7 | 680k | 43.3 | 50.0 |
| 8 | 755k | 40.0 | **73.3** |
| teacher | — | 95.0 | 95.0 |

**16 rounds, 60 episodes each, 50-episode evals (job 241598, h=4):**

| round | 0 | 2 | 4 | 7 | 10 | 11 | 15 | 16 | teacher |
|---|---|---|---|---|---|---|---|---|---|
| student SR % | 0 | 16 | 46 | 60 | 70 | 84 | **92** | 84 | 93.3 |

The student reaches the teacher. Round-to-round spread is about ±10 points at
50 episodes, so 84-92 is the plateau, not a drop. (The round-15 weights were
overwritten by round 16 — the script now keeps best-by-eval.)

**Spread layout (job 241627).** The seed-0 layout had put all five ants on the
big cap by chance. The user asked for 2 on the big cap, 2 on the small cap, 1 at
the centre (`ants.offsets` in config, new; `configs/rl/marl_5ants_spread.yaml`).
Same run otherwise:

| round | 0 | 2 | 3 | 5 | 11 | 14 | 16 | teacher |
|---|---|---|---|---|---|---|---|---|
| student SR % | 0 | 50 | 82 | 92 | 96 | 98 | **98** | 98.3 |

**The 98 is inflated, and so is the teacher's 98.3 — by selection.** Every round
was scored on the *same* 50 episodes and the best round kept; the maximum of
seventeen noisy evaluations is not the policy's rate. On two fresh 100-episode
blocks the chosen checkpoint never saw:

| seeds | student | teacher |
|---|---|---|
| 60000-60099 | 89% | 94% |
| 70000-70099 | 93% | 90% |

**Honest headline: student ≈ 91%, teacher ≈ 92%, on held-out seeds.** The
conclusion — the student matches the teacher — survives; the absolute number
does not. It also learns faster on this layout (82% at round 3 vs 28% on the
clustered one): symmetric arms put the torque leverage at both ends of the T and
make the mean arm zero. The script now rotates the eval block every round and
reports the chosen checkpoint on a held-out block, teacher alongside. Fourth
instance in this chapter of a measured number not surviving contact with fresh
data; the fix each time was the same — evaluate on what training never touched.

**Three readings.**

1. **Plain BC is 0% here too** — the same zero every shared-policy run in §3
   produced. DAgger is what moved it. So covariate shift was a large part of
   those failures, not only the cancelling sum. (The two are related: a student
   that is slightly wrong drifts into states the teacher never visited, where
   it is more wrong.)
2. **History matters** (73 vs 43 at round 8), and with 16 rounds the h=4
   student plateaus at the teacher's level (84-92 vs 93).
3. **Unseen attachment layouts: 0%** for both. The policy learned these five
   positions, not a position→force rule. Training across layouts would bring
   back the centroid problem (§8): the right force for one row depends on
   where the *other* ants sit, which that row does not contain. For now the
   claim is scoped to a fixed layout.

Honest framing: the deployed ant has no formula and no N, but it is distilled
from a centralised solution. The claim, if it holds, is "a decentralised swarm
can be distilled from a centralised one", not "learned from nothing". Residual
RL on top (§7's second half) would loosen that.

## 10. What the field does, and MARL v2

After three zeros from decentralised RL (§5), a literature pass on decentralised
cooperative transport. Seven sources; three go through narrow passages.

| ingredient | literature | what §5 ran |
|---|---|---|
| critic | centralised / asymmetric, training only. decPLM tried the decentralised variant: "unstable due to non-stationarity" | decentralised |
| what an agent senses | the load's motion **at its own contact point** — "the object is the communication medium" (Wang & Schwager); the ant model's uninformed carriers align with the force felt at their attachment | load-centre velocity divided by N — the normalisation that hid N (§8) |
| starting point | warm-started actor (previous team-size stage, or a frozen pretrained skill) | from zero |
| team size | staged 1→2→3→4 with a distillation loss (cable-towed); or train N=2, deploy 2-10 (decPLM) | fixed |
| data | 2048 parallel envs; 50M steps / 8 days | 6M steps, one env |
| reward | dense progress + sparse success | same |

Two mechanisms worth stealing. (i) Wang & Schwager's followers read the object's
local velocity at their own contact and rotate their force toward it; forces
align exponentially, faster with more robots, and no follower needs N, the
object, or the goal — which dissolves §8's wall (an ant cannot count the swarm)
rather than solving it. (ii) Nobody starts from zero; our own chapter 03 found
the same (residual on a good base: 94→97; free RL on it: collapse), and a ~91%
decentralised student exists (§9).

Caveat: the narrow-passage papers pass gaps by re-forming (side-by-side →
front-rear); none rotates a load through a 15 mm slit. This task is harder than
anything found.

**MARL v2** (`scripts/rl/train_marl_v2.py`, `configs/rl/marl_v2_5ants.yaml`):

| piece | implementation |
|---|---|
| A centralised critic | one team value from all ants' rows; actor still sees its own row only |
| B contact-point velocity | new observation block `v_centre + ω × arm`, absolute units (`env.observe_contact_velocity`, `velocity_scale_ants: 1`; obs 27→29) |
| C warm start | distil the DAgger student into the actor, keep an MSE anchor to it during PPO |
| D parallel envs | 30 worker processes, each with its own env |
| E team-size staging | 2→3→5, actor carried over, critic rebuilt — a curriculum over N, not over the maze; the user allowed it |

Reward `geodesic_exit`, no maze curriculum, evaluation always the real task.

**The first warm arm was not a warm start — caught and cancelled after 4 min.**
Its distillation fitted the student's labels to MSE 0.00017, yet the actor
scored 0% on the real task while the student it copied scores 93% in the same
env (checked through the identical row conversion). Two causes, both mine: the
distillation states came from rollouts of the *random* actor, not the student —
a near-perfect fit on the wrong distribution, the §9 covariate-shift trap one
level up — and a single-frame actor was fitted to a 4-frame student (§9: h=1
43% vs h=4 91%). Fix: the actor takes a 4-frame history; distillation is
DAgger-style (round 0 the student drives, then the actor drives and the student
labels, refit on everything, stop when the actor is within 10 points of the
student), and only then PPO with the anchor.

**The second warm arm failed the same way, and the cause was found offline.**
Seven DAgger rounds (538k samples on the student's own states, the §9 recipe
that had reached 91%) left the actor at 0%. Not the ±π wrap (only 1.2% of the
student's pushes sit near it). On identical rows and labels: the v2 actor,
the same actor trained 10× longer, and the §9 architecture all bottom out at
MSE 0.012-0.040 — 40× the §9 floor — and all score 0% closed-loop. Swap the
*target* from the push **angle** to the force **vector** and the same rows fit
to 0.0008. An angle is an ill-conditioned function of a vector whose mean
magnitude is 0.18: a small force error is tens of degrees, five ants' worth of
it is the cancelling sum of §4. The §9 student regressed the vector; the v2
actor had regressed the angle. Fixed: the actor now outputs a force vector,
the env receives `(atan2, |f|)`.

**The from-scratch arms were collapsing to "do nothing"** — return 0.13 → 0.000
within 600k steps, mean distance drifting to ~1.0 m, worse than random. Not the
reward: measured over 8 episodes, frozen 0.000 < random 0.047 < all-push-+x
0.129, so freezing is not what the reward asks for. One code-level suspect was
real and is fixed regardless: PPO recomputed old log-probs by `atanh` of the
stored *squashed* action, which is exact only away from saturation and gives
vanishing gradients once an output saturates — a self-reinforcing drift into
`force = 0`. The buffer now stores the pre-tanh sample and the log-prob is
exact. Whether that was the whole mechanism is what the relaunched arms show;
`|f|` and `log_std` are now logged every rollout.

Three arms, all vector-action, 4-frame history, from 2026-09-07 23:14:

| arm | pieces | job | W&B |
|---|---|---|---|
| warm | A+B+C+D + history | 241760 | https://wandb.ai/kakooee/ant_swarm/runs/2ct753bw |
| scratch | A+B+D + history | 241761 | https://wandb.ai/kakooee/ant_swarm/runs/5v1o33bj |
| staged 2→3→5 | A+B+D+E + history | 241762 | https://wandb.ai/kakooee/ant_swarm/runs/zsxgxb5g |

Warm vs scratch differ in one thing.

**The vector-action warm arm failed too — and the cause is the labeller, not
the fit.** Seven DAgger rounds now fit to MSE 0.00006 (better than the
student's own training fit) and the actor stays at 0%. Ruled out offline, on
identical episodes and labels: the target (vector now), the network, the
training length, the ±π wrap (1.2%), the observation (27-D N-normalised rows
and 29-D absolute rows both fit to ~0.002 and both score 0% after one round;
velocity saturation 0.0% / 0.5%). Along the actor's own trajectory the gap to
the student's label is 0.048 per ant at step 0 — the fit is just not precise
enough, as §9's round 0 was not either (0%). What lifted §9 to 91% were DAgger
rounds labelled by the **split controller, a true oracle** that is right at
any state. The v2 warm start labelled the actor's drifted states with the
**DAgger student** — a policy that is right only on its own distribution. Its
labels off-distribution are wrong, and fitting them to 0.00006 reproduces
wrong. Fixed: the warm start and the PPO anchor now use the split-controller
oracle (job 241765, https://wandb.ai/kakooee/ant_swarm/runs/8vxhq7ub). A control runs alongside: the unchanged §9 trainer
on the v2 env's 29-D rows (job 241764) — if it climbs as §9 did, the v2
observation is confirmed harmless and the oracle was the whole difference.

**Control result (241764):** it climbs exactly as §9 did — 3% → 87% over
eight rounds, held-out **90%** on 50 episodes (teacher 98%). The 29-D
absolute-velocity observation is harmless; the oracle labeller is what works.
Whatever still fails in v2 is inside its own distillation/PPO path.

**Not settled after all.** At 1.6M steps the exact-log-prob arms looked healthy
(|f| rising, log_std steady at −1.0); by 10.7M (scratch) and 13.0M (staged)
both had converged to the same dead state as before — episode return exactly
0.000, log_std sliding to −2.2 / −2.9, the staged arm having promoted 2→3→5
on the step budget, never on success. Cancelled (241761, 241762). So the
atanh pitfall was worth fixing but was *not* the collapse mechanism. Two
parameterisations and two log-prob paths all end at return 0 on a reward
under which a random policy scores 0.047 — the update is moving *away* from
random. Suspect, untested: something in the v2 PPO update itself (team
advantage, critic slicing) rather than in the policy head. Tested with a
known-solvable reward (`--sanity-reward push`: reward = the team's mean
|force|, optimum ≈ 1.41): the same update drives |f| monotonically 0.43 →
0.94 in 400k steps. **The update is not broken.** The collapse to zero motion
is a property of the real reward under uncoordinated pushing — a random
5-ant policy earns almost nothing (0.047 per episode), the gradient toward
coordination is invisible from there, and the entropy shrinks onto whatever
does no harm. The literature's rule — never from zero — holds here with a
number attached. No further from-scratch runs; everything rides on the warm
start.

Results: pending.

**Differential tests (local, identical oracle-DAgger data, 29-D rows, Ring).**
With a *deterministic* actor collecting the DAgger states, both students
climb — v2 actor + v2 `distil()`: 0→0→5→20%; §9 net + §9 fit: 0→10→10→35%
(the control's early curve). With *sampled* collection (std 0.135 pre-tanh)
they still climb, later: 0→0→0→15% and 0→0→0→35%. So neither the actor,
nor `distil()`, nor sampled collection explains the v2 script's failure.
Relaunched with deterministic collection, the §9 body and the §9 fitting
schedule (job 241778): seventeen rounds, aggregate MSE **0.00000**, actor
**0%** at every round; cancelled. A student that memorises 1.3M oracle-labelled
rows to zero error and transfers nothing to `evaluate()` can only mean the
rows the worker records are not the rows `evaluate()` feeds the actor —
a mechanical train/test mismatch that exists only in the worker path, which
the harness never uses.

**Found.** Not a row mismatch — an in-process lockstep test showed worker rows
and evaluator rows identical to the last bit. The worker pool passed the same
`initargs` to every process, so all 30 workers ran the **same seed**: the same
env, the same spawn sequence and, with deterministic collection, the same
states. Measured: three workers, 4,500 recorded rows, **775 unique**. Every
"76,800-sample" round was 2,560 samples copied thirty times; seventeen rounds
of "1.3M samples" were ~43k unique rows — the harness's *round-1* volume,
where the harness itself was still at 0-10%. A student memorising 43k rows to
MSE 0.00000 and transferring nothing is exactly what was observed. The same
duplication fed PPO thirty identical trajectories per batch, which is at least
consistent with the from-scratch arms' entropy collapse onto "do nothing".
Fix: one line, a per-worker seed (`current_process()._identity`); after it,
4,500 of 4,500 rows unique. Relaunched from 2026-09-08 with everything else as
before (vector action, exact log-prob, deterministic DAgger collection, §9
body and fit, oracle labeller, 30 distinct workers):

| arm | job | W&B |
|---|---|---|
| warm, oracle-labelled, 4-frame | 241788 | https://wandb.ai/kakooee/ant_swarm/runs/7blr1m6k |
| scratch, 4-frame | 241789 | https://wandb.ai/kakooee/ant_swarm/runs/v5edvbtn |

Lesson worth its own line: every diagnostic I ran in-process used one env, so
none could see a bug that only exists across processes. The number that gave
it away was the fit being *too good*.

**With distinct workers the warm start exists (241788).** Oracle-labelled
DAgger distillation into the shared per-ant actor, 30 workers, deterministic
collection, 4-frame history:

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| actor SR % | 0 | 20 | 50 | 60 | 60 | 80 | 80 | 77 | **93.3** | 98.9 |

Stopped at "within 10 points of the oracle" and handed to PPO, whose first
sampled rollouts (log_std −2) succeed at 0.86-0.91. This is §9's result
reproduced inside the RL pipeline — a decentralised swarm at the oracle's
level — and it is the first time an RL phase here has started from a working
multi-agent policy.

**Both arms completed (2026-09-08), 20M env steps each, 30 distinct workers.**

*Warm arm (241788).* PPO with the oracle anchor neither destroyed nor clearly
improved the 93% start. Thirty-episode evaluations peaked at 100% several times
(best-by-eval kept one of those), sat at 83-97% for most of the run, and decayed
to 47-73% in the last quarter — a slow erosion the anchor did not prevent.
Best checkpoint on 100 held-out episodes: **88%**. The §9 student scored 89/93%
on its own held-out blocks, so the RL phase added nothing measurable; the
distilled policy is the result, RL is at best neutral on it here.

*Scratch arm (241789).* **The first pure decentralised RL success in this
project.** Zero for the first ~12M steps, then a rise to 40-47% by 20M, still
climbing at the end (46.7% at the last two evaluations); best checkpoint on
100 held-out episodes: **30%**. Same reward, same critic design and same
observation as the runs that scored 0% in §5 and earlier in this section — the
difference is thirty *different* worker streams instead of one copied thirty
times. Continued from its final actor for 20M more steps (job 241856,
https://wandb.ai/kakooee/ant_swarm/runs/3xawtaj6), critic re-initialised.

Fair comparison on identical fresh seed blocks (60000+, 70000+; 200 episodes):

| policy | seeds 60000+ | seeds 70000+ | mean |
|---|---|---|---|
| §9 student — oracle-distilled, no RL (241627) | 89% | 93% | **91%** |
| v2 warm best — distilled, then PPO with anchor (241788) | 87% | 82% | 84% |
| v2 scratch best — pure decentralised RL, 20M steps (241789) | 30% | 32% | 31% |

So on equal footing the RL phase *cost* the distilled swarm about seven points,
and its in-training peaks of 100% were the 30-episode winner's curse once more.
The ordering to quote: distillation 91%, distillation+RL 84%, pure RL 31%.
Rollout GIFs of both v2 policies are in each run's `renders/`.

*Continuation of the pure-RL arm (241856): 20M more steps from the 20M-step
actor, critic re-initialised.* It did not keep climbing: 80 evals: first 47%, median 17%, 10-90% band 10-37%, max 50%, last 23%.
Held-out best: **29%** — the same as before the continuation (30-31%). With
this recipe — MAPPO with a centralised critic, geodesic reward, 30 workers,
20-40M steps — pure decentralised RL saturates at about 30% held-out and
oscillates between 10% and 50% on 30-episode evaluations. Not zero, not
solved.

**The oracle-anchored arm (241765), after its distillation stayed at 0%, ran
PPO with an MSE anchor to the oracle for 7.3M steps: 29 of 29 evaluations at
0%, entropy collapsed (log_std −4.4), return ≤ 0. Cancelled. An oracle anchor
does not lift a 0% start either — the actor must be *at* the oracle's level
before RL begins, which is exactly what its distillation failed to deliver.

Sources: decPLM (arXiv 2509.14342); cable-towed load MARL (2503.18221);
dual-quadruped narrow environments (2602.16353); shape-formation MARL
(2606.09610); DQN rod through a doorway (2007.09243); Wang & Schwager,
"Multi-robot manipulation without communication" (DARS 2014); Gelblum et al.
2015 (PMC4525283); Feinerman lab, piano-movers ants vs humans (PNAS 2024).

## Remaining limitations

- **Decentralised policies now exist at three levels (§9, §10).** Distilled
  from the oracle: 91% held-out. Distilled then PPO: 84%. Pure decentralised RL
  from zero: 31% and climbing at 20M steps. Unseen attachment layouts: still 0%. Everything before it was 0%, and the
  formula-based base (§8) needs an N the ant cannot obtain. The only working
  controller is centralised: one brain computing a wrench and dividing it.
- **Not tried:** the residual and critic parts of §7 (blocked: the base needs N, §8);
  supervising `(c_F, λ)` directly (§6.1);
  letting ants specialise (some push, some turn); curriculum over N; longer
  MARL runs than 3M steps; communication between ants.
- **Single seed** for every arm.
- **n=1 is degenerate** for a push-only swarm: the lone attachment is the centre
  of the stem, so its arm is zero and it can produce no torque at all. It is
  excluded from every push-only test.

## Exact state right now (2026-09-08)

- **Running:** nothing.
- **MARL v2 runs:** `ant__20260908_0327__241788__marl_v2_warm_h4` (distilled+PPO,
  held-out 84%), `ant__20260908_0327__241789__marl_v2_h4` (pure RL, 31%) and its
  continuation 241856 (29%); `best.pt`/`final.pt`, `results.json`, `renders/` in
  each. Trainer: `scripts/rl/train_marl_v2.py`; config `configs/rl/marl_v2_5ants.yaml`.
- **This session's runs:** `ant__20260907_*__decentralised_base_*` (levels
  0-2, four counters, the fixed-N map); `env.velocity_scale_ants` added to
  `observation.py`; chapter 04 §2 corrected (mean-arm term).
- **Artifacts:** the split-probe GIFs (n = 1,2,4,10,50,100), six shared-policy
  runs, two MARL runs — all under `storage_local/` by run id.
- **Single-agent line, for contrast:** BC 88-95% (chapter 03), plus residual RL
  fine-tuning at **97%** (`ant__20260906_0950__240957__finetune_bc_residual`).
  Full-policy Gaussian fine-tuning collapsed to 0% and was saved only by
  best-by-eval checkpointing.
- **Verdict feeding forward:** the obstacle has a name and a number. Any next
  attempt should be judged on whether it reduces the summed torque error, not on
  whether it lowers the training loss — the two are close to unrelated here.
  The next attempt is §7, and its first step costs hours, not days.
