# RL concepts with nice names

Short, simple definitions. The last column says where it shows up in this project.

## Reward problems

| Name | Definition | In our project |
|---|---|---|
| Sparse reward | Reward comes only at rare events (e.g. task success). Most steps give 0, so there is almost no learning signal. | Reward 1.0 only when the T reaches the goal. |
| Dense / shaped reward | Extra reward every step to guide learning, e.g. based on distance to goal. | `reward_mode: shaped` adds progress reward. |
| Potential-based shaping | A safe form of shaping: reward = change of a potential function Φ. It provably does not change the optimal policy. | Our progress shaping is Φ = −distance. |
| Euclidean distance reward | Shaping with straight-line distance to the goal. Simple, but blind to walls: it pulls through obstacles. | Our current `shaped` mode. Fails at the barrier. |
| Geodesic distance reward | Shaping with the distance along the *real shortest feasible path* (around walls, through free space). Decreases monotonically along the true solution, so turning counts as progress. | Planned: BFS distance field over the free (x, y, θ) grid from the scanner. |
| Deceptive reward | The reward gradient points the wrong way: greedy progress leads into a trap, and the true path must first move "away". | Euclidean distance pulls the T straight into the wall; the turn moves it away from the goal. |
| Reward hacking | The agent finds a cheat that gets reward without doing the intended task. | The small-cap cheat: poking the small head through and "winning" without threading. |
| Local optimum | A behavior that is better than all nearby behaviors but far from the best one. Learning gets stuck there. | Pushing the T against the wall forever. |
| Credit assignment | The problem of knowing *which* earlier action caused a late reward. Long episodes make it hard. | The success bonus arrives ~80-400 steps after the key turn. |
| Success bonus / terminal reward | A single large reward at task completion, on top of any shaping. | `reward_success: 1.0`. |
| Discount factor (γ) | How much future reward is worth today. γ = 0.99 means reward n steps away is worth 0.99ⁿ. | We use 0.99 → "effective horizon" ≈ 100 steps. |
| Horizon | How many steps an episode can last. Long horizons weaken the discounted end reward and blur credit assignment. | `max_steps` cut 2000 → 500 for this reason. |

## Exploration problems

| Name | Definition | In our project |
|---|---|---|
| Exploration vs. exploitation | The core dilemma: try new things (explore) or repeat what already works (exploit). | All of training. |
| Hard exploration | Tasks where random behavior almost never finds any reward. | Random pushing never threads the slit. |
| Exploration bottleneck | A narrow "corridor" in state space that all successful episodes must pass through. | The tilted pass through each slit. |
| Dead-end state | A state from which success is impossible; the agent must go back (or the episode is wasted). | State b in the paper: small head poked through slit 1. |
| Action noise | Random perturbation added to actions to explore. The simplest exploration tool. | SAC's stochastic policy; PPO's Gaussian policy. |
| Entropy regularization | Reward the policy for staying random, so it keeps exploring instead of committing too early. | SAC's auto-entropy tuning; PPO's `ent_coef` (we anneal it). |
| gSDE (state-dependent exploration) | Noise that depends on the state and stays consistent within an episode — smoother exploration than fresh noise each step. | `use_sde` in our PPO config. |
| Intrinsic motivation / curiosity | Extra internal reward for visiting new or surprising states, to drive exploration. | Candidate tool if training stalls. |
| RND (random network distillation) | A curiosity method: a network tries to predict a fixed random network; where prediction fails, the state is novel → bonus. | Candidate tool. |
| Count-based exploration | Bonus for rarely visited states, computed from visit counts. | Alternative to RND. |

## Curriculum and transfer

| Name | Definition | In our project |
|---|---|---|
| Curriculum learning | Train on easier versions first, then harder ones. | Our `curriculum:` config section. |
| Negative transfer | Skills learned on task A make learning task B *worse*, because A rewarded the wrong behavior. | Wide-gap training teaches small-head-first, which is a dead end in the PNAS maze. |
| Reverse curriculum | Keep the task at full difficulty, but start episodes near the goal and move the start backward. | `curriculum.mode: reverse` — solved the old env with sparse reward. |
| Solution class / topology change | When simplifying a task creates *different kinds* of solutions, not just easier ones. This is what makes a curriculum unsafe. | Wide gap allows straight push; narrow gap does not. |
| Safe curriculum stage | An easier stage that keeps the same solution class — verified, not assumed. | Gap 0.17 → 0.16 on the PNAS maze, certified with the c-space scanner. |
| Automatic / self-paced curriculum | The difficulty advances by itself based on the agent's success rate, not a fixed schedule. | Our stages advance when rolling success ≥ 0.7. |
| Stall safety / force-advance | A cap on how long one stage may take before moving on, so a too-hard stage cannot block forever. | `max_steps_per_stage`. |
| Transfer learning | Reusing what was learned on one task to learn another faster. | Warm-starting via `run.init_from`. |
| Catastrophic forgetting | While learning a new task, the network destroys what it knew about the old one. | Risk when curriculum stages change too fast. |
| Domain randomization | Randomize the env (sizes, positions) during training so the policy generalizes instead of memorizing. | Future option: random wall positions (then obs must describe walls). |
| Sim-to-real gap | A policy trained in simulation fails on the real system because the simulator is not exact. | Our kinematic → dynamic step is a mini version of this. |

## Learning from data / demonstrations

| Name | Definition | In our project |
|---|---|---|
| Imitation learning | Learn a policy from expert example trajectories instead of reward. | Possible use of our saved success JSONs. |
| Behavior cloning (BC) | The simplest imitation: supervised learning of state → expert action. | Candidate warm-start from kinematic solutions. |
| Distributional shift | The states seen at run time differ from the training states, so the policy misbehaves. BC suffers this: one small error leads off the demo path, into states it never saw. | Why BC alone is fragile. |
| DAgger | Fix for BC's drift: run the learner, ask the expert to label the states the learner actually visits, retrain, repeat. | Possible if we use the BFS path as the "expert". |
| Learning from demonstrations (LfD) | Mix demonstrations into RL, e.g. put them in the replay buffer. | Candidate for SAC on the PNAS env. |
| Jump-start RL (JSRL) | Start episodes partway along a demonstration, letting RL finish the rest; move the start earlier over time. | Reverse curriculum + our BFS path = the same idea. |
| Hindsight experience replay (HER) | Relabel failed episodes as successes for the goal they *did* reach, creating signal from failure. | Considered; weak fit because our goal is fixed. |
| Goal-conditioned RL | The policy takes the goal as an input, so one policy can reach many goals. HER needs this framing. | Not used; our goal is fixed. |
| Offline RL | Learn a policy purely from a logged dataset, with no interaction with the env. | Our 38k saved successes could feed this someday. |

## Multi-agent (the end goal)

| Name | Definition | In our project |
|---|---|---|
| Multi-agent RL (MARL) | Several agents learn together in one env; each agent's world changes as the others learn. | The real ant experiment: many pullers on one load. |
| Centralized control | One brain sees everything and commands all agents. Simple but unrealistic. | Current SB3 setup: one network outputs all ants' actions. |
| Decentralized control | Each agent acts from its own local observation. What real ants do. | The per-ant obs rows already anticipate this. |
| CTDE (centralized training, decentralized execution) | Train with global information, but the final policies act on local observations only. | Standard recipe for the multi-agent phase. |
| Parameter sharing | All agents use the same network weights (with different inputs). Cheap and often strong. | Natural fit: ants are identical. |
| Emergent behavior | Group-level skill that no single agent has or plans. | The ants' collective memory / persistence in the paper. |
| Joint action space | The combined action of all agents. Grows exponentially with agent count — the core MARL difficulty. | 10 ants × [angle, magnitude] each. |

## Motion planning (the geometry side)

| Name | Definition | In our project |
|---|---|---|
| Piano-movers problem | The classic problem of moving a rigid object through obstacles; asks whether *any* collision-free path exists. | The experiment (and this repo) is literally named after it. |
| Configuration space (c-space) | The space of all poses of the object — here (x, y, θ). One point = one full pose; a path = a motion. | Our scanner works in this space. |
| Free space | The part of c-space with no collision. The task = find a path through it. | What `free_mask` computes. |
| Connectivity / reachability | Whether two regions of free space are joined by a path. If not, the task is impossible. | The flood-fill test behind "passable: True/False". |
| Narrow passage | A thin channel of free space; hard for both planners and RL. | The diagonal channel through each slit. |
| BFS (breadth-first search) | Graph search that expands from a start node layer by layer, like a wave. Finds shortest paths and labels every node with its distance. | Run from the goal over the free c-space grid → geodesic distance field. |
| Flood fill | BFS used only to mark what is reachable, without distances. | The scanner's "passable: True/False" test. |
| Rejection sampling | Draw random candidates and keep only the valid ones. | `sample_free_pose` for spawn poses. |

## General terms worth knowing

| Name | Definition | In our project |
|---|---|---|
| On-policy / off-policy | On-policy (PPO) learns only from fresh data of the current policy; off-policy (SAC) reuses old data from a replay buffer. | Why SAC is more sample-efficient here. |
| Replay buffer | Off-policy memory: past transitions stored and re-sampled for many updates. | SAC's buffer; also where demos could go. |
| Actor-critic | Two networks: the actor picks actions, the critic estimates how good states/actions are and guides the actor. | Both PPO and SAC are actor-critic. |
| Value function / critic | Predicts the future discounted reward from a state (or state-action pair). | The "V" and "Q" in the logs. |
| Policy gradient | Improve the policy directly by nudging action probabilities in the direction that increased reward. | PPO's core mechanism. |
| Deterministic vs. stochastic policy | Deterministic: same state → same action (good for eval). Stochastic: actions are sampled (good for exploration). | `deterministic=True` in our eval. |
| Sample efficiency | How much experience is needed to learn. | SAC solved in 422k steps; PPO needed 22.9M. |
| Partial observability | The agent cannot see the full state, so the world looks ambiguous. | Dynamic mode: linear velocity is not in the obs. |
| Markov property | The current observation contains everything needed to predict what happens next. | Why we add velocity/wall features to the obs. |
| Termination vs. truncation | Termination: the task itself ended (success/failure). Truncation: time ran out. Gymnasium separates them because bootstrapping differs. | `terminated` = goal reached; `truncated` = 500 steps. |
| Vectorized envs | Running n copies of the env in parallel to collect experience faster. | PPO's `n_envs: 8`. |
| Overfitting to the env | The policy memorizes one fixed layout instead of learning a general skill. | Fixed walls + fixed spawn make this likely. |
