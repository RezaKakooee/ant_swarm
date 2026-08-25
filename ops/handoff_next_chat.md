# Handoff — where we are (2026-08-25)

UPDATE 2026-08-25 — two chapters happened after the text below:
1. **Generalisation SOLVED**: random start + random goal at 99-100% via the
   two-leg reward (`geodesic_exit`) + continuous goals; from scratch in one
   run; both routes possible (up needs `curriculum.route: up`).
   Read: `docs/project_journey/01_goal_generalisation.md`.
2. **Self-learning limit MEASURED**: sparse reward without a teacher fails —
   HER / intrinsic / gSDE / Go-Explore all stall at frontier x=0.900.
   Read: `docs/project_journey/02_self_learning_limit.md`.
   Next candidate there: human demos from the interactive sandbox.
Everything is a config switch now — master table in the header of
`configs/rl/gen_h_her_sparse.yaml`. No jobs running.

Short state file for the next chat. Longer history: `ops/handoff.md`.
Plain-English summary of the method: `notes/what_made_it_work.md`.

## Status: single agent is DONE

The PNAS piano-movers task is solved in **dynamic** mode (real forces and
momentum): **100% success, mastered at 247,519 steps**. Verified by replaying
the last 25 solutions — **25/25 use the ants' maneuver** (big head into slit 1,
turn in the corridor, small head out of slit 2). No shortcuts.

Best run (copied from the Azure box, kept here):
`storage_local/ant__20260814_1230__local-1427853__train_sac__pnas_dyn_geo_v2__best`

What made it work (all in git): pose-path curriculum from a BFS solution,
success replay buffer (25% of each batch), linear velocity in the observation
(25 → 27 numbers), geodesic reward (BFS route distance).

## Two repos now

| repo | where | what |
|---|---|---|
| `ant_swarm` (private) | `~/ant_swarm` | working repo — everything, incl. `ops/` (SLURM), `blogs/`, `docs/`, `notes/` |
| `ant-piano-movers-rl` (public) | `~/ant-piano-movers-rl` | code only: `ant_swarm/`, `configs/`, `scripts/`, `interactive/`, README, CITATION.cff, LICENSE |

Publish flow (never push the private repo's history — it contains cluster paths
and personal notes):

```bash
cd ~/ant_swarm && python tools/publish.py      # allowlist export + scrubber
cp -r storage_local/public_export/. ~/ant-piano-movers-rl/
cd ~/ant-piano-movers-rl && git add -A && git commit -m "..." && git push
```

`tools/publish.py` REFUSES to export if anything mentions the cluster name,
usernames, `sbatch`/`squeue` or `/home/…`. Edit `ALLOW` there to change what
goes public. `blogs/`, `docs/`, `notes/`, `ops/` are deliberately NOT public.

## Blog

`blogs/01-can-ai-solve-the-ant-puzzle/`
- `POST.md` — the write-up (the one that matters)
- `ant-piano-movers-rl/` — static project page, copied into the site repo
  (`RezaKakooee/rezakakooee.github.io`), live at
  `rezakakooee.github.io/ant-piano-movers-rl/` with the sandbox at `/sandbox/`
- `make_figures.py` — regenerates every figure from the real env + BFS field
- older drafts (`BLOG_POST*.md`, `POST_REFINED.md`, `REPORT.md`) are superseded

## Next: multi-agent

`ants.n >= 2`, dynamic mode. First a central controller (the SB3 setup already
supports it), then parameter-shared decentralized policies where each ant sees
only its own observation row. The per-ant observation layout is already built
for this.

Known open ends:
- our slit is a bit wider than the paper's (0.86 of the big head vs 0.81), so a
  small-head route still exists geometrically; the curriculum, not the geometry,
  is what removed it
- the BFS teacher is a strong help — worth testing how far a weaker one gets
- PPO was never run with the teacher setup, only SAC

## Traps already paid for (do not relearn)

1. Straight-line distance shaping is DECEPTIVE here: it rewards pushing into the
   wall and punishes the turn. Use `reward_mode: geodesic`.
2. Never widen the slit to make it easier — that is a different task with a
   different solution (negative transfer). Move the START pose instead.
3. Check WHAT the agent does, not just its success rate. Two "successes" were
   cheats, caught only by replaying trajectories.
4. The geodesic field needs `--inflate 0.002 --dx 0.003 --dth 2`; the default
   coarse grid silently disconnects (look for "reachable 100%" in the output).
5. Regenerate the field after any geometry change; it is not in git.
