# Generalisation experiments (2026-08-21 → 2026-08-23)

All variants use the accurate PNAS maze, dynamic mode, SAC.
Fine-tune = resumed from an earlier solved checkpoint. Success = 20 deterministic episodes per goal on the 5-goal benchmark (random-goal variants), or rolling training success (A).

| variant | start | goal | how | result |
|---|---|---|---|---|
| A | random | fixed | fine-tune | 100% |
| B | fixed | random (5 goals) | fine-tune, one geodesic field per goal + goal curriculum | 40% |
| C | random | random (5 goals) | fine-tune, one geodesic field per goal + goal curriculum | 39% |
| D | random | random (continuous box) | fine-tune, two-leg reward (`geodesic_exit`) | 99% |
| E | random | random (continuous box) | from scratch, two-leg reward + reverse pose curriculum + free spawn | 99% |
| F | random | random (continuous box) | like E + mirrored curriculum routes (up-turn and down-turn) | 100% |
| G | random | random (continuous box) | like E but ONLY the up-route curriculum (mirrored anchors) | 100% |

B and C were not "partly good": each scored 100% on the goals seen alone during the curriculum and 0% on all others — memorised routes, goal input ignored. The two-leg reward (one shared field through the slits, plain distance after the exit, goals sampled continuously) is what made D and E generalise.

Route-diversity result: E and F take the down-route 40/40 (even stochastically) — a single SAC policy collapses to one route, and a small grid asymmetry in the BFS field makes that route "down" every run. G proves the up-route is fully learnable (curriculum mastered at 783k steps vs E's 690k) and STAYS the up-route from free starts (39-40/40 up). Route diversity today = two policies: E or F (down) + G (up).

Configs: `configs/rl/gen_[a-g]_*.yaml`. Best checkpoints: `storage_local/ant__20260821_1155__21048333__*gen_a*`, `...20260822_0035__21072278__*gen_d*`, `...20260822_1754__21107972__*gen_e*`, `...20260823_1120__21144812__*gen_f*`, `...20260824_1124__21243500__*gen_g*` (each under `checkpoints/best/`).
