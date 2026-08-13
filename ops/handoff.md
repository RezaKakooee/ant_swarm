# Project Handoff — What We Did and Where We Are

Status file for continuing on any machine (written 2026-08-13). Simple English.
How-to commands live in [run_guide.md](run_guide.md); concepts in
[../notes/rl_concepts.md](../notes/rl_concepts.md) and
[../notes/tutorial.md](../notes/tutorial.md).

## What this project is

A gym recreation of the "piano-movers" ant experiment (Dreyer et al., PNAS
2025): a T-shaped load must pass two narrow slits. The load must enter the
first slit with its BIG head, turn inside the narrow middle chamber, and exit
the second slit small-head-first. Long-term goal: multi-agent (many ants push
together). Current stage: single agent.

## Timeline of results

1. **Old env (world 1.25, gap 0.15)** — solved. SAC kinematic + reverse
   curriculum + sparse reward: 99% in 422k steps. PPO kinematic: 90% in 22.9M.
   Dynamic mode stalled; those runs died in a cluster node failure.
2. **PNAS replica built (v1: world 1.65, gap 0.16)** — exact dimensions taken
   from the paper's SI Table S1 + measured from Movie S1 frames.
3. **Geodesic reward built** — BFS distance field over the free pose space
   (`scripts/rl/gen_geodesic_field.py`, `reward_mode: geodesic`). Fixes the
   deceptive straight-line distance: turning counts as progress.
4. **v1 solved** — kin+geodesic: 100% at 322k steps. Sparse control: 100% at
   411k (geodesic ~25% faster). BUT the agent partly used a "pirouette"
   shortcut: poke the small head into the slit, rotate the stem inside it.
5. **v2 = final maze** (`configs/rl/pnas_kin_geo_v2.yaml`, wall 0.285 →
   gap 0.15). Kills most of the shortcut. **User-validated by hand in the
   sandbox**: big-first entry + corridor turn + small-first exit is possible;
   small-head entry is not (by hand). DECISION: v2 is canonical, no slit tabs.
6. **v2 solved (kinematic)** — job 20384223: 100% at 518k steps, 1,661 saved
   solutions. ~13/20 final solutions enter big-first; a residual ~7/20 use a
   tight small-first thread (accepted — impossible by hand, geometrically real).
7. **Now running: dynamic mode on v2** — job 20432145 (rtx4090, started
   2026-08-13 ~11:33). Dynamic = forces + momentum, the physics closest to
   real ants. This is the last single-agent milestone.

## Where things are

| Thing | Path |
|---|---|
| Env package | `ant_swarm/` (config, layout, tshape, state, action, obs, reward, render, run_id, log) |
| Canonical maze configs | `configs/rl/pnas_kin_geo_v2.yaml` (kinematic), `pnas_dyn_geo_v2.yaml` (dynamic) |
| Geodesic field builder | `scripts/rl/gen_geodesic_field.py` (field .npz is NOT in git — regenerate, see run_guide) |
| Train scripts | `scripts/rl/train_sac.py` / `train_ppo.py` (hydra CLI: `--config-name`, `key=value`) |
| Success replay/grids | `scripts/il/render_success.py` |
| Manual sandbox | `interactive/interactive_t.html` (rigid body, scroll = rotate while dragging, optional slit tabs) |
| Run outputs | `storage_local/<run_id>/` — checkpoints, successes (replayable JSONs), GIFs/MP4s, train.log, `code/` snapshot |
| W&B | https://wandb.ai/kakooee/ant_swarm (run name = run dir name = log name) |
| Design notes / glossary | `notes/rl_design_notes.md`, `notes/rl_concepts.md`, `notes/tutorial.md`, `notes/english_words.md` |

Key run dirs on the scicore server (not in git):

- `storage_local/ant__20260813_0158__20384223__train_sac__pnas_kin_geo_v2` — v2 kinematic, MASTERED
- `storage_local/ant__20260813_1133__20432145__train_sac__pnas_dyn_geo_v2` — v2 dynamic, running
- older: v1 runs 20381854/55/56, old-env sweep 2030947x/8x

## The method that works (recipe)

SAC + **reverse-start curriculum** (spawn walks from the goal back to the
start; env at full difficulty the whole time) + **geodesic reward**
(potential shaping on the BFS route distance) + success-trajectory harvesting.
Gap/easier-env curricula cause negative transfer here — do not use them.

## Hard-won lessons (do not relearn these)

1. Euclidean distance shaping is deceptive here; geodesic fixes it.
2. Simplifying the maze changes WHICH solutions exist (small-head shortcut)
   → negative transfer. Only the start pose may be made easier.
3. The geodesic field must be built with wall inflation (~2-3mm) or BFS routes
   through razor channels; and with a fine grid (`--dx 0.003 --dth 2`) or the
   field silently disconnects (44% reachable = broken; expect ~100%).
4. Scanner state-blocking tests cannot prove temporal order (which head enters
   first). Verify maneuvers by REPLAYING trajectories.
5. Config mutations for parallel jobs: every run snapshots its RESOLVED config
   to `<run>/code/config.yaml` — replay tools read that, never the live configs.
6. SAC warm-start (`run.init_from`) restores weights, not the replay buffer.
7. These jobs are CPU-bound (~55 steps/s on a100 nodes; the rtx node gave
   ~11/s, likely CPU contention). Give 8 real cores.

## Next steps (in order)

1. **Finish dynamic v2** (job 20432145 or rerun on a stronger CPU/a100).
   If it stalls: add linear velocity (vx, vy) to the observation — known
   partial-observability gap in dynamic mode.
2. **Multi-agent** (`ants.n >= 2`, dynamic): first centralized (current SB3
   setup already supports it), then parameter-shared decentralized policies
   (CTDE). The per-ant observation rows are already designed for this.
3. Optional: BC / demonstrations from the ~40k saved kinematic solutions
   (`configs/il/bc.yaml` is a placeholder; scripts/il has the data tools).
