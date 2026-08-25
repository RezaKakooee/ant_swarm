# 02 — The self-learning limit: sparse reward without a teacher

Period: 2026-08-24 → 2026-08-25 (chapter closed; all runs stopped).
Question: can the agent learn the maze with sparse reward only — no geodesic
field, no pose curriculum? Answer: **no. Every method stalled at the second
slit; the measured frontier is x = 0.900.**

Evidence used for every number:

| source | what it holds |
|---|---|
| `storage_local/sci_out/ant__2026082[45]_*__2128*/2129*.out` | run logs (job IDs in filenames) |
| `sacct` on jobs 21281398, 21281502, 21281670, 21282433, 21282577, 21295462 | states, run times |
| `configs/rl/gen_[h-l]*.yaml` | exact settings; master switch table in `gen_h_her_sparse.yaml` header |
| `docs/SELF_LEARNING_EXPERIMENTS.md` | options table + experiment summary |

## Key terminology

(Terms from `01_goal_generalisation.md` are not repeated.)

- **HER** — hindsight experience replay: every failed episode is relabeled as a success for the goal it actually reached (`sac.her`, `GoalObsWrapper` + SB3 `HerReplayBuffer`).
- **Intrinsic bonus** — exploration reward for novel load poses (`env.intrinsic`; `count` = pose-grid pseudo-counts, `rnd` = random network distillation). Training env only.
- **gSDE** — state-dependent exploration noise, held for 64 steps (`sac.use_sde`).
- **Go-Explore (phase 1)** — archive of visited pose cells; return to a cell exactly by replaying its action trail, then explore randomly (`go_explore:`, `scripts/rl/go_explore.py`).
- **Frontier** — largest load-centre x in the Go-Explore archive. The second wall's slit is at x ≈ 0.98.

## 1. Setup

**Believed:** HER was the best fit (goal-conditioned task), intrinsic bonus a helper, Go-Explore the strongest but "teacher-like" option (full options table: `docs/SELF_LEARNING_EXPERIMENTS.md`).

**Did:** implemented all of them as independent config switches, then launched an ablation: H = HER; I = HER+bonus; K = HER+gSDE; L = all three; J = Go-Explore with random actions. All sparse reward, random start + random goal, no curriculum, no geodesic field.

**Verified before launch (measurements):** HER relabeling put positive reward on 25.9% of sampled transitions under a random policy; both bonus modes decay with familiarity; found and fixed: HER needs `learning_starts` > `env.max_steps` (500).

## 2. Result: everything stalls at the same wall

| run | job | stopped at | success | key number |
|---|---|---|---|---|
| H (HER) | 21281398 | 1.7M steps | 0% | distance-to-goal flat at 1.20 |
| I (+bonus) | 21281502 | 1.6M | 0% | flat at ~1.20; bonus income decayed to ~0 |
| K (+gSDE) | 21282433 | 2.0M | 0% | flat at ~0.65 — closest, still stuck |
| L (all) | 21282577 | 1.4M | 0% | flat at ~0.80 |
| J (Go-Explore, 30° cells) | 21281670 | 3.0M (completed) | 0 solutions | frontier 0.900 from <0.7M on |
| J2 (Go-Explore, 10° cells, 100-step bursts) | 21295462 | 4.9M | 0 solutions | frontier 0.900; 14,058 cells (2.6× J) |

The stop decisions were made on flat trends (≥1M steps without change), not on impatience; H/I/L stopped first, K and J2 last.

## 3. Why it fails (measurement + interpretation)

**Measured:** the frontier is identical (0.900) for two different Go-Explore grids and never moves even though the archive keeps growing (new cells are variations inside the reached region). Random restarts *at* the frontier cannot thread the second slit.

**Interpretation:** the second slit requires a precise pose-plus-push sequence. Random or noisy actions never produce a single crossing, so: HER has no crossing to relabel; the intrinsic bonus exhausts the reachable region and goes quiet; gSDE gets the load to the maze but not through it. Sample count is not the missing ingredient — signal is.

**Diagnostic worth keeping:** a policy snapshot at 1.5M steps (L) ends its episodes repeating one constant action against a wall — the actor has converged to a local habit with no gradient pointing anywhere better.

## 4. Small fixes found on the way

- `SuccessTrajectoryCallback` crashed on HER's Dict observations (fixed: use the flat part).
- W&B was missing for the Go-Explore runner (added; runs after J2 will log `frontier_x` etc.).
- Local W&B staging files were ~14 GB of `storage_local`. Decision: the remote dashboard is the record; `WANDB_DIR` now points to job scratch (`ops/sb_train.sh`), and `wandb.init` no longer stages inside run dirs.
- Two self-inflicted tooling failures worth remembering: `pkill -f <pattern>` kills the shell whose command line contains the pattern (three background tasks died this way), and session-scratch files can vanish — analysis scripts now live in `storage_local/analysis/`.

## What worked

1. The modular switch design: five method combinations launched from configs only, no code edits between arms.
2. Cheap verification before launching (relabel check, bonus decay check) — no run died of a silent bug.
3. Trend-based stopping: ~2 days of planned compute cancelled once flatness was established.

## Remaining limitations

- **Not tried:** human demonstrations from the interactive sandbox (the designated next step); RND mode in a full run (only `count` ran); policy-guided Go-Explore (explore with K's actor instead of random actions); longer horizons than 500; reward for partial progress through the corridor.
- **Not isolated:** why K approaches closer than H (gSDE vs luck; single seed each).
- All arms ran one seed. The conclusion "stalls at 0.900" is robust across methods, but per-method numbers are single samples.

## Exact state right now (2026-08-25)

- **Running:** nothing. All six jobs stopped or completed.
- **Artifacts:** run dirs under `storage_local/ant__2026082[45]_*`; no success JSONs exist from this chapter (0 solutions everywhere).
- **Code:** committed in 49ead0c except: J2 config (`gen_j2_goexplore_fine.yaml`), W&B in `run_go_explore.py`, final doc updates — this chapter's closing commit.
- **Verdict feeding forward:** the task needs a teacher. Candidates ranked: human demos (sandbox) → Go-Explore with policy-guided exploration → keep the BFS geodesic teacher as the honest answer.
