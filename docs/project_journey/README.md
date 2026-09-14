# Project journey

Four chapters, in order. Each one states its question, its answer, the
evidence, what worked, and what stayed open. The project was closed on
2026-09-14; chapter 04 §16 is the final summary.

| chapter | question | answer |
|---|---|---|
| [01 Goal generalisation](01_goal_generalisation.md) | Can one ant learn random start and random goal? | Yes, with the geodesic reward: 100%. |
| [02 The self-learning limit](02_self_learning_limit.md) | Can it learn with sparse reward and no field? | No. Every method stalls at the second slit, x = 0.900. |
| [03 Imitation and the goal bug](03_imitation_and_the_goal_bug.md) | Can demos replace the field? | BC 88-95%, plus residual RL 97%. A frozen-goal bug in the cache was found and fixed. |
| [04 Multi-agent](04_multi_agent_and_the_cancelling_sum.md) | Can N ants, each seeing only its own row, do the same? | Yes: 5 ants 96-98%, 10 ants 94%, pure decentralised RL. One ant with equal data: 100%. Without the field: 0%. |

Numbers are held-out success on 200 fresh episodes (seeds 60000+ and 70000+)
unless a chapter says otherwise. State, checkpoints and cluster habits:
`ops/handoff_next_chat.md`.
