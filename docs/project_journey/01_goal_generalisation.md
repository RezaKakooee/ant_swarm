# 01 — Goal generalisation: from one fixed goal to random start + random goal

Period: 2026-08-21 → 2026-08-24 (chapter closed; all seven variants finished).
Scope: making the solved single-goal task general. Earlier history (how the
single-goal task was solved) is in `ops/handoff_next_chat.md`,
`notes/what_made_it_work.md`, and the blog.

Evidence used for every number in this file:

| source | what it holds |
|---|---|
| `storage_local/sci_out/ant__2026082*_*.out` | training logs (job IDs in the filenames) |
| `sacct` on jobs 21048333-35, 21056020-21, 21072278, 21107972, 21144812 | job states, run times |
| `configs/rl/gen_[a-f]_*.yaml` + each run's `code/config.yaml` snapshot | exact settings per run |
| `docs/GENERALISATION_EXPERIMENTS.md` | the summary table |
| session transcript + memory note `goal-generalisation.md` | in-session evals whose harness was scratch code, not committed (marked below) |

## Key terminology

- **Variant A/B/C/D/E/F/G** — the experiment configs, `configs/rl/gen_a_*` … `gen_g_*`. See the table in `docs/GENERALISATION_EXPERIMENTS.md`.
- **Zero-shot** — evaluating a checkpoint on a task setting it was never trained on, with no extra training.
- **Per-goal field** — one precomputed geodesic BFS field per goal position (the approach of B and C).
- **Two-leg reward** (`reward_mode: geodesic_exit`) — leg 1: shaping on ONE shared geodesic field toward a fixed exit point past the second wall; leg 2: once every corner of the load is past x = 0.989, plain distance shaping toward the actual goal. Both legs are potential-based.
- **Goal box** (`goal.random_box`) — each episode samples the goal uniformly from the rectangle [1.05, 1.55] × [0.10, 0.62] in the right room, instead of picking from a fixed list.
- **Resume warm-up** (`sac.resume_warmup_steps`) — after resuming a checkpoint, act with the loaded policy but do zero gradient updates for N steps while the (empty) replay buffer refills.
- **Free spawn** (`curriculum.final_spawn_x_range`) — after the last pose-curriculum anchor is mastered, switch training to fully random start poses instead of stopping.
- **5-goal benchmark** — in-session eval: 20 deterministic episodes on each of 5 fixed goals ([1.20,0.36], [1.15,0.22], [1.15,0.50], [1.42,0.22], [1.42,0.50]).

## 1. Starting point

**Believed:** the single-goal task is done. `ops/handoff_next_chat.md` records 100% success, mastered at 247,519 steps, 25/25 replayed solutions using the ants' maneuver. Source checkpoint: `storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2__best/checkpoints/best/best_model.zip` (saved at 220,000 timesteps, per the resume lines in later logs).

**Did:** defined three harder settings — A: random start, fixed goal; B: fixed start, random goal (5 fixed positions); C: both random.

**Measured (zero-shot, 50 episodes each; scratch harness, recorded in the session transcript of 2026-08-21, not committed):** baseline on its own task 100%; A 62% deterministic / 72% stochastic; B 12%; C 6-8%.

**Changed because of it:** random starts looked easy; random goals looked like the real problem. Both were still attacked the same naive way first (section 2). One trap in this eval itself: the first version of the harness showed 0% for the *baseline*, because without a fixed spawn each env instance samples one pose at construction and reuses it. Fix: pin `spawn.fixed_pose` when evaluating fixed-start settings. Rule: an eval harness is code too — it can be the thing that is broken.

## 2. Naive fine-tuning (jobs 21048333/34/35, cancelled)

**Believed:** resume the solved checkpoint, train on the new distribution, done.

**Did:** built per-goal geodesic fields for B/C (5 fields in `storage_local/fields/`), resumed the source checkpoint with unchanged hyper-parameters (lr 3e-4, `learning_starts: 10000`, per the runs' `code/config.yaml`).

**Result (logs `ant__20260821_1155__21048333/34/35__*.out`; all three CANCELLED by us at 1h35 to relaunch):**

| variant | rolling success at cancel | timesteps |
|---|---|---|
| A | 1.00 | ~496k |
| B | 0.21-0.23 | ~517k |
| C | 0.02 | ~525k |

**Changed because of it:** A was declared solved. For B/C we found two problems in how SB3 resumes (section 3) and added a goal curriculum (section 4).

## 3. Failure: what SB3 resume actually does

**What broke:** a resumed model starts with an EMPTY replay buffer and keeps the learning rate it was saved with. Its first gradient updates fit batches of 256 drawn from only a few fresh transitions.

**How found:** reading the resume path in `scripts/rl/train_sac.py` while asking why C *lost* ability (6% zero-shot → 2% after 500k steps of training).

**Interpretation (not isolated experimentally):** those early updates on a tiny, correlated buffer damage the loaded policy before real data arrives. C's drop from 6% to 2% is consistent with this; we did not run an ablation that separates it from the other changes made at the same time.

**What it cost:** one 3-job round (~1.5h × 3 on CPU) plus the misleading conclusion "fine-tuning barely works".

**Fixes (uncommitted, in `scripts/rl/train_sac.py` / `train_utils.py`):**
1. Re-apply the config learning rate on resume (log line: `Resume: learning rate 0.0003 -> 0.0001`, first seen in `ant__20260821_1400__21056020__*.out`).
2. `ResumeWarmupCallback`: act with the loaded policy, zero gradient steps for `resume_warmup_steps` (50k in later runs). We deliberately did NOT use SB3's own `learning_starts` for this — that takes *random* actions during warm-up, which throws away the policy being fine-tuned.

**Rule produced:** never resume an off-policy model without a buffer-refill warm-up, and check which learning rate a loaded model actually uses.

## 4. Failure: per-goal fields + goal curriculum (jobs 21056020/21)

**Believed:** with the resume fixes, a lower LR (1e-4), and a 3-stage goal curriculum (1 goal → 2 goals → all 5, promotion at 70% over 100 episodes), B and C would generalise.

**Did:** relaunched B and C on `scicore-fast` (6h QOS), 3M-step target.

**Result (logs `ant__20260821_1400__21056020/21__*.out`; both TIMEOUT at 6h per sacct):**

| | stage transitions | at timeout |
|---|---|---|
| B | stage 1 at 14:02 (success 1.00), stage 2 at 15:19 (0.70) | 1.43M steps, rolling success 0.42 |
| C | stage 1 at 14:04 (0.70), stage 2 at 16:13 (0.70) | 1.44M steps, rolling success 0.46 |

Success at the final stage was flat for ~800k steps (log values drift 0.31-0.49 with no trend), so the timeout is not what stopped progress.

**5-goal benchmark of the best checkpoints (in-session harness; also in `docs/GENERALISATION_EXPERIMENTS.md`):** B 40%, C 39% overall — and the split is the finding: 100% on the two goals practised alone in stages 0-1, **0% on all three goals introduced together in stage 2**. 2/5 = 40%: the "partial success" was no generalisation at all.

**Where the failures happen (in-session diagnostic, 8 episodes per goal on B's checkpoint):** on unseen goals the load never reaches the goal region — it stops before the first wall (ends near x≈0.55) or between the walls (x≈0.80). Changing the goal vector breaks the maneuver from the first step.

**A second measurement making the cause sharp (in-session, right-room spawn test):** spawning the load already past both walls, the original single-goal checkpoint reaches its trained goal 100% and every other goal 0%. **Interpretation:** the policy ignores the goal observation (goal_dx, goal_dy) — during single-goal training those inputs were constant, so nothing forced the network to read them. Per-goal reward fields make each goal a separate task with a separate reward landscape, so the shared skill (threading the slits) is never marked as shared.

**What it cost:** two 6h jobs plus the earlier round; ~1 day.

**Rules produced:**
- Rolling success on a mixed distribution can hide a bimodal truth. Always split the metric by condition (here: per goal).
- An input the policy never needed during training is an input the policy never learned to read. If generalisation over X is wanted, X must vary during training.
- Success videos are a biased sample: only successes get saved. Never judge a run by its saved videos (we almost did — B/C's videos looked perfect).

## 5. Two-leg reward, fine-tuned: variant D (job 21072278)

**Believed (hypothesis from section 4):** put the goal only where it is easy. One shared field for the slits; plain distance in the open room; continuous goals so memorising is impossible.

**Did:** implemented `reward_mode: geodesic_exit` (`ant_swarm/reward.py`), `goal.random_box` (`ant_swarm/ant_swarm.py`), config `gen_d_twoleg.yaml`; resumed C's best checkpoint (resume line: timesteps=940,000) with lr 1e-4 and 50k warm-up. Ran on `scicore` (1-day QOS), COMPLETED in 21h41m at 3.94M total steps (SB3 counts the 3M target as *additional* steps on resume — a surprise worth remembering).

**Result (log `ant__20260822_0035__21072278__*.out`):** final rollout success 0.99, final eval success 1.0 (eval = random start + random goal). 5-goal benchmark of `checkpoints/best`: **99%** (95% on [1.15,0.22], 100% on the rest). Render eval: 10/10 successes, GIFs in the run's `eval/` folder.

**Fairness note:** B's 40% benchmark ran from B's fixed training start; D and E were benchmarked with random starts (harder). The comparison direction is therefore conservative — the gap is, if anything, understated.

## 6. Two-leg reward, from scratch: variant E (job 21107972)

**Believed:** D still needed a chain of prior checkpoints. One self-contained run would be a cleaner recipe.

**Did:** `gen_e_scratch_randall.yaml` — no resume, lr 3e-4, two-leg reward, goal box, the 16-anchor reverse pose curriculum driving the spawn, plus a new curriculum option `final_spawn_x_range`: on mastery of the last anchor (90% over 200 episodes), switch to fully random starts and keep training. Eval env left unpinned all run, so it measures the final task from step one.

**Result (log `ant__20260822_1754__21107972__*.out`):**

| step | event |
|---|---|
| 0 → 690,608 | all 16 anchors mastered one by one (e.g. stage 10 in 60,850 steps at 0.90) |
| 690,608 | `TARGET MASTERED (success 0.99) -> FREE SPAWN` |
| ~820,000 | unpinned eval success reaches 1.00 |
| 820k → 2.85M | eval stays 0.99-1.00; flat |
| 2,846,038 | we cancelled the job (sacct: CANCELLED at 19h34m) — converged, nothing left to learn |

Best vs final: best-model saves stopped improving long before the cancel (15 `New best mean reward` events total); the final state equals the best state — no metric fell.

5-goal benchmark of `checkpoints/best`: **99%** (95% on [1.20,0.36], 100% on the other four). Render eval: 8/8, mean lengths 107-131 steps (GIF filenames). Last logged eval mean episode length: 116.

**Note on reading W&B for this run:** the W&B chart x-axis ("Step") is W&B's logging counter, not env timesteps; the log file's `total_timesteps` is the real scale. Convergence was at ~820k env steps, not ~20k.

## 7. Small failure: all training render GIFs showed the same episode

**What broke:** every 500k-step snapshot GIF in D's `renders/` shows the identical start and goal. **Why:** the render callback reset its env with the same fixed seed each time — intentional when the task was fixed, wrong once start and goal are random. **How found:** the user noticed all snapshots looked identical. **Cost:** none in results, but D's training snapshots are useless for judging diversity. **Fix (uncommitted):** derive the render seed from the current timestep in `train_sac.py` / `train_ppo.py`, keeping the fixed seed only for pinned curriculum scenes. **Rule:** a diagnostic that always shows the same slice of the distribution is not a diagnostic.

## 8. Route diversity: variant F (job 21144812, RUNNING)

**Observed (D/E renders + user):** the policy always turns the big head downward in the corridor. **Measured:** the geodesic field itself is symmetric — mean |phi(x,y,θ) − phi(x,H−y,−θ)| = 0.0085 over 2,594 reachable sample pairs (in-session check) — so the reward never preferred a route. **Interpretation:** the bias comes from the curriculum: the extracted descending path is one of two symmetric route families, and all anchors follow it.

**Did:** `curriculum.mirror_anchors` — mirror every anchor (y → H−y, θ → −θ; valid because walls, goal height and the T are symmetric about mid-height), and let each episode randomly practise either variant (`set_spawn_pose` now accepts several poses). Config `gen_f_mirror.yaml` = E + this flag. Launched 2026-08-23.

**Result (log `ant__20260823_1120__21144812__*.out`; job CANCELLED by us at 22h23m after convergence):** all anchors mastered at step 979,205 (E: 690,608 — both routes cost ~290k extra curriculum steps, confirming the earlier hypothesis); unpinned eval at 0.98-1.00 up to ~2.98M steps. 5-goal benchmark of `checkpoints/best`: **100%** (20 episodes per goal; vs E's 99% this is a one-episode difference — read as equal or slightly better).

**Route check (in-session, 40 identical random-start random-goal episodes per model):** F takes the down-route 40/40 — with the deterministic AND the stochastic policy. Same as E. **Conclusion (measurement):** mirrored anchors did not produce route diversity in the final policy.

**Training-history check (measurement; `route_classification.json` in the F run dir):** replaying F's saved successes shows the up-route was used only from stage-9 anchor spawns whose big head was already in the first slit (5 up vs 30 down there) — and **never** from a free left-room start (0 up in 500 randomly sampled of 14,920 such successes). The down preference existed at the decision point from the beginning; it is not a late collapse.

**Why down every time (measurement, in-session):** not chance. The BFS grid is not aligned with the maze's mid-height, so mirrored poses round into different cells. Comparing each anchor with its mirror: identical phi in the open room (stages 0-7), but **phi higher by 0.035-0.135 for the up-route poses through the slit and corridor (stages 8-14)** — on the grid, the upper passage is effectively narrower and every route through it longer. The shaping reward therefore pays slightly more for the down branch at the decision point in every episode; the critic, the argmax, and the success replay buffer all amplify that constant edge. E's path extractor followed the same gradient, which is why its single path was the down one too. **Untested fix:** regenerate the field with the y-grid symmetric about H/2, or symmetrise the field with its own mirror — then the tie is exact.

## 9. The up-route on purpose: variant G (job 21243500)

**Believed (from section 8):** the grid's down-bias wins ties between unlearned options; question — does it also block or overturn a *learned* up-route?

**Did:** `curriculum.route: up` — every anchor replaced by its mirror; the down path is never practised. Config `gen_g_uponly.yaml` = E + this flag; from scratch. Launched and cancelled after convergence on 2026-08-24 (log `ant__20260824_1124__21243500__*.out`).

**Result (measurements):**

| | E (down curriculum) | G (up curriculum) |
|---|---|---|
| curriculum mastered at | 690,608 steps | 783,446 steps |
| 5-goal benchmark | 99% | **100%** |
| route from free starts (40 eps, det./stoch.) | 40/40 down | **39/40 and 40/40 up** |

So: the up-route is fully learnable (the toll only costs ~93k extra curriculum steps), and ~560k steps of free-spawn training under the slightly down-favouring shaping did **not** pull G back to the down-route. **Interpretation:** the field's asymmetry decides ties between options the policy has not learned; it does not overturn a mastered skill.

**Practical outcome:** route diversity exists today as a pair of policies — E or F (down) + G (up) — without any new machinery. A single both-routes policy still needs a route-conditioned input (untested).

## What worked

1. Two-leg reward + continuous random goals: 39-40% → 99% on the 5-goal benchmark (B/C vs D/E).
2. From-scratch single-run recipe (E): reverse pose curriculum → free spawn; eval 100% from ~820k steps.
3. Resume warm-up + LR re-apply: C's fine-tune went from degrading (6%→2%) to working (as part of the D recipe; not isolated).
4. Per-condition evaluation (per-goal split, right-room spawn test) — every real diagnosis in this chapter came from one of these, not from rolling averages.

## Remaining limitations

- **Not verified:** goals outside the box [1.05,1.55]×[0.10,0.62]; goals in the *left* room or corridor; start poses in the right room; robustness to physics changes.
- **Not isolated:** the individual contribution of warm-up, LR drop, and goal curriculum in section 4's runs (changed together); the "empty-buffer updates wreck the policy" mechanism is an interpretation.
- **Answered since first draft:** F's mirrored curriculum does NOT yield a both-routes policy (40/40 down-route, even stochastic); it cost ~290k extra curriculum steps and gave a small (within-noise) benchmark gain. Untried ideas for real diversity: a route-conditioned input bit, a higher entropy floor, or two separate policies.
- **Unfair-comparison caveat:** B/C benchmarks used their own (easier, fixed-start for B) settings; D/E used random starts. Direction of the bias favours B/C.
- The ~1% failure residue of D and E sits on different goals (D: [1.15,0.22]; E: [1.20,0.36]); with 20 episodes per goal, 95% vs 100% is one episode — too small a sample to interpret.
- All zero-shot and per-goal numbers come from in-session scratch harnesses recorded in the session transcript; the harness itself is not committed. Re-running requires rebuilding it (~30 lines around `set_goal_choices` + `SAC.load`).

## Exact state right now (2026-08-23)

- **Running:** nothing. F cancelled 2026-08-23, G (job 21243500) cancelled 2026-08-24, both after convergence.
- **Done checkpoints (each `<run>/checkpoints/best/best_model.zip`):** A `ant__20260821_1155__21048333__*gen_a_randstart` (rolling 100%); D `ant__20260822_0035__21072278__*gen_d_twoleg` (99%); E `ant__20260822_1754__21107972__*gen_e_scratch_randall` (99%); F `ant__20260823_1120__21144812__*gen_f_mirror` (100%); G `ant__20260824_1124__21243500__*gen_g_uponly` (100%, up-route).
- **Code:** ALL generalisation changes are uncommitted (git status: modified `ant_swarm/{ant_swarm,reward,geodesic}.py`, `scripts/rl/{train_sac,train_ppo,train_utils,pose_curriculum}.py`, `ops/sb_train.sh`; new `scripts/rl/goal_curriculum.py`, `configs/rl/gen_[a-f]_*.yaml`). Last commit is 8fc46b7 (blog/public-release tooling).
- **Public repo:** `~/ant-piano-movers-rl` staged but unpushed; contains none of this chapter's changes.
- **Docs:** summary table in `docs/GENERALISATION_EXPERIMENTS.md`; this file is the first entry in `docs/project_journey/`.

## Addendum (2026-09-07): the curriculum was not necessary

Every geodesic-reward run in this chapter either resumed a solved checkpoint or
paired the reward with the pose-path curriculum. The cell "geodesic reward, from
scratch, **no** curriculum" was never run — it was assumed to fail. Job 241646
ran exactly variant E (`gen_e_scratch_randall.yaml`) with `curriculum.enabled=false`,
everything else identical: random start, random goal, `geodesic_exit` reward.

| steps | deterministic eval success | mean episode length | final distance |
|---|---|---|---|
| ~910k | **100%** | 124 | 0.0508 m |

Variant E with the curriculum needed 690k steps to master its last anchor and
~820k to reach 100% on the unpinned eval. Without it: 100% by ~910k. The
curriculum bought little, if anything, for the single agent. What made the task
learnable from scratch was the geodesic reward — the BFS field over load poses
that scores progress along the real route, turns included. (Straight-line
distance shaping is what "pushes the T into the wall and punishes the turn",
`notes/what_made_it_work.md`; that sentence had been remembered as being about
the geodesic reward.) Recorded here because chapter 04 §5 relies on it: for the
swarm, the same reward without a curriculum gave 0%, so the difference between
one ant and five is not the curriculum.
