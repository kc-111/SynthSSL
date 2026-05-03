# World Model Evaluation Environment — Core Specification

# Need add gravity/physics/etc. tasks. Moreover, we could see generalization to other textures etc.

## Purpose

A configurable simulation environment for studying self-supervised world models, with a primary focus on **offline trajectory optimization**: using a learned world model as a simulator to plan action sequences that are then executed in the real environment. The simulator-vs-execution gap is the principal measurement.

The environment is a **factorial benchmark with diagnostic intent**: each environment property is an independent knob, so holding others fixed and varying one isolates which world properties the world-model training signal can and cannot capture. Generalization, RL extensions, and representation-quality probes are first-class but optional layers on top of the same substrate.

## Design principles

- **Factorial, not bundled.** Every environment property is an independent knob. Vary one at a time.
- **Plan-then-execute is primary.** Action sequences are produced from the world model and executed in the ground-truth environment; the gap between simulated and executed outcomes is the main signal.
- **Train and evaluation share maps and objects by default.** Generalization is a separate, opt-in axis.
- **Goals are drawn from training trajectories**, not from held-out states. Deliberate methodological commitment: we are testing the model's ability to plan to known states, not to extrapolate.
- **Privileged ground-truth state is available to the evaluation harness but never to the trained model.**

---

## Environment characteristics

### View and topology

- **2D**: top-down with cone field-of-view, blocked by walls. Cone angle and depth are configurable. Topology is **flat** (walled) or **toroidal** (right edge wraps to left, top to bottom).
- **3D**: first-person view, flat ground only. Spherical/curved 3D adds confounds without testing anything new — the real world is flat to the eye.

### Memory regime

- **2D fog of war (on/off)**: off gives full top-down visibility within bounds; on restricts the observation to the cone field-of-view. Persistent memory of visited cells is not provided — it must be learned.
- **3D**: no explicit fog. Partial observability is intrinsic to first-person view; render distance bounds per-frame visibility.

(See open question 1 on whether to expose an optional visited-set channel in 3D for parity with 2D-fog-on.)

### Agent

- **Movement**: 4-directional, configurable speed (see below).
- **Action space**: 8 discrete actions, one-hot encoded, fixed across all configurations — 4 movement, pickup, put-down, throw (no-op when physics off), no-op.

### Movement speed

Movement speed is a knob with two layers:
- **Baseline speed**: constant within an episode, sampled from a configured set per episode. The agent must infer current speed from observation deltas (single frames don't reveal it).
- **Lasting-effect zones** (see Regions) can apply a persistent multiplier on top of the baseline.

Speed enables a study question: do learned representations factor speed cleanly, or does the model conflate "moved up at speed 1.0 for 2 steps" with "moved up at speed 2.0 for 1 step"? Train/test speed splits (train on {1.0, 1.5}, test on {2.0}) constitute a fourth generalization knob — extrapolation to unseen speeds is a different question from inference within the training distribution.

**Caveat for evaluation under speed variation**: at higher speeds, H steps of plan horizon cover more ground, confounding cross-speed comparisons of plan success. When varying speed, hold either *steps taken* or *distance covered* constant in eval, depending on the question.

### Objects

Always present and always interactable (pickup, put-down; throw when physics is on). Meshes drawn from a fixed asset pool per scene template (see "3D scenes" — applies to 2D as well, with simpler stylized assets).

### Stochasticity

- **Off**: fully deterministic.
- **Level 1**: ambient movers on scripted, defined paths. May collide with the agent.
- **Level 2**: ambient movers on patterns sampled per-environment — fixed within an episode, unknown to the agent, varying across environments.

Stochasticity affects the world but not the agent's own action consequences directly (except via collision). Episodes are seeded; planning and execution share a seed unless otherwise specified.

### Regions

- **Walls only**: standard impassable boundaries.
- **+ Uncontrollable zones**: inside the zone, the agent's actions have no effect. The episode does not terminate; the trajectory continues with the agent stuck.
- **+ Lasting-effect zones**: entering applies a persistent state modifier (e.g., movement-speed multiplier) that continues after exit. The modifier is part of environment state but is not directly observable.

Cumulative levels: each includes the previous.

### Physics

- **Off**: throw is a no-op. Object collisions are still resolved (objects don't pass through walls or each other).
- **On (3D only)**: throw launches the held object under gravity. Trajectory is deterministic given direction and force.

---

## Dataset and training data

### Data collection

Two policies generate offline (observation, action, next-observation) trajectories. No reward labels.
- **Random walk**: uniform random actions.
- **Scripted simple AI**: predefined exploration heuristic (e.g., wall-following, frontier-seeking). Not goal-directed.

### Map structure per dataset

- **2D**: multiple maps per dataset. Same generator, different wall layouts and object placements.
- **3D**: configurable. Default is **one map per dataset** to avoid implicit online world-building; multiple maps available for map-generalization studies.

### Train/test relationship

Default: identical maps, objects, and speed distribution between train and eval. **Generalization mode** (opt-in) exposes four independent knobs: held-out maps, held-out object configurations, held-out object identities, held-out speeds.

### 3D scenes

3D ships with multiple scene templates so findings aren't artifacts of a particular rendering style:
- **Indoor** (rooms, furniture)
- **Outdoor** (open terrain, trees, rocks)
- **Industrial** (corridors, machinery)
- **Abstract** (minimalist geometric, low-confound baseline)

Layouts are procedurally varied within a template. The template is a property of the dataset, not the episode.

---

## Evaluation protocol

### Primary: plan-then-execute

For each evaluation episode:
1. Sample a goal observation from a training trajectory.
2. Use the trained world model + a fixed search procedure to find an action sequence of length H minimizing latent-space distance between the simulated final observation and the goal.
3. Execute that exact action sequence in the ground-truth environment.
4. Record:
   - Simulated final state (model's predicted endpoint, in latent and in world coordinates if a decoder is available)
   - Executed final state (ground-truth endpoint)
   - **Goal-execution gap**: executed final state vs. goal, in ground-truth coordinates
   - **Simulator-execution gap**: simulated final state vs. executed final state

Search is held fixed across all configurations within a study. Default: discrete-action CEM with fixed sample budget and horizon H. Beam search and gradient-based variants are available; varying the search procedure constitutes a separate study axis.

### Secondary evaluations

- **Consecutive goals (sequential)**: A → B → C → ... → A. Each leg planned and executed separately, with leg *k+1* starting from the executed end of leg *k*. Measures error compounding and re-localization.
- **Consecutive goals (single long plan)**: one action sequence of length *T·k* searched at once for *k* waypoints. Measures long-horizon planning capacity.
- **Return-to-origin loop closure**: plan a trajectory that returns to start. Measures latent and world-coordinate closure error as a function of loop length — the diagnostic for representation drift when there is no observational landmark for closure.
- **Passive rollout under stochasticity**: no-op action sequence; environment evolves under stochastic dynamics. Compare predictor's rolled-forward latent against the encoded latent at each step.

### Probes

**During training:**
- **Coverage-vs-decodability**: under fog, what fraction of the visited map is decodable from the latent at each checkpoint.

**At evaluation:**
- **Position decoding**: linear probe from latent to ground-truth (x, y).
- **Topology fidelity**: RSA between latent dissimilarity and ground-truth geodesic distance on the correct manifold.
- **Trajectory smoothness**: latent displacement per environment-step, especially across topological boundaries (toroidal wraparound) and across speed changes.
- **Speed factorization**: under a speed sweep, does the latent admit a low-dimensional axis aligned with speed, or is speed entangled with position/heading?
- **Action-conditioned predictor accuracy**, decomposed by region type (controllable / uncontrollable / lasting-effect).
- **Subspace ablation**: variance explained by action-relevant vs. nuisance features after zeroing components.

Probes are held fixed across all configurations.

### Metrics

Primary, in ground-truth coordinates:
- **Plan success rate**: fraction of episodes where executed final state is within ε of goal.
- **Goal-execution gap** (continuous).
- **Simulator-execution gap** (continuous).
- **Loop-closure error** for return-to-origin.

Latent-space distances are secondary.

---

## Knob summary

| Knob | Values | Notes |
|---|---|---|
| View | 2D / 3D | |
| Topology | Flat / Toroidal | Toroidal is 2D only |
| Fog | Off / On | 2D only |
| Stochasticity | Off / L1 / L2 | |
| Regions | Walls / +Uncontrollable / +Lasting | Cumulative |
| Physics | Off / On | On is 3D only |
| Speed distribution | Single / Multi-speed / Held-out | Per-episode constant |
| Data policy | Random / Scripted | |
| Train/test split | Same / Diff. maps / Diff. object configs / Diff. object identities / Diff. speeds | Four independent generalization knobs |
| 3D scene template | Indoor / Outdoor / Industrial / Abstract | 3D only |

A full factorial is intractable. Studies pick **pivot configurations**: hold all but one knob fixed at a baseline, vary that knob across its values. Each pivot answers a single question.

---

## Extensions (opt-in)

- **Reinforcement learning**: a reward function can be specified; standard model-free and model-based baselines work without modification.
- **Generalization studies**: enable any of the four train/test difference knobs.
- **Online learning**: the environment supports online interaction.
- **Custom goal specifications**: image (default), object-configuration descriptor, or natural language (requires extra encoder).

---

## Implementation: Unity + ML-Agents

Both 2D and 3D are implemented in Unity. 2D is a top-down orthographic camera over a flat scene; 3D is a perspective first-person camera. Single codebase, identical physics and seeding semantics across views.

License: Unity Personal is free under the $200K revenue/funding threshold and is sufficient for this work.

Editor setup is a one-time skeleton scene with prefabs for `Agent`, `Wall`, `Object`, `UncontrollableZone`, `LastingEffectZone`, plus a `MapRoot` for spawned content and a camera rig (orthographic top-down for 2D, first-person for 3D, one enabled per run). All per-episode logic — map generation, agent control, region effects, ground-truth logging — lives in C# scripts and runs at episode start. Python drives training, planning, and evaluation via `mlagents-envs` against a headless Unity build.

### Common interface (Python side)

```
env.reset(seed) -> observation
env.step(action) -> (observation, info)
env.get_ground_truth_state() -> dict   # privileged, evaluation only
env.get_config() -> dict
```

Observations are RGB images of fixed resolution. `info` contains diagnostic fields (collision flags, zone entry, current speed) for evaluation logging. The trained model never sees `info` or `get_ground_truth_state`.

### Determinism and seeding

Environment seeds are explicit and reproducible. Stochasticity is seeded per-episode. Plan-then-execute uses the same seed for the simulator's training data and the evaluation execution, so divergence is attributable to the model rather than to stochastic mismatch.

### Data format

- Trajectory tensors: `(T, H, W, 3)` uint8 observations, `(T,)` int8 actions, `(T,)` episode boundaries.
- Per-trajectory metadata: map ID, scene template, speed, seed, data policy.
- Ground-truth state trajectory: included for evaluation episodes only.

---

## Open questions

1. Whether to expose persistent visited-set memory as an optional input channel in 3D, to make 2D-fog-on and 3D conditions more directly comparable. (Affects **Memory regime**.)
2. Observation resolution — model capacity vs. simulation throughput.
3. How object identity is represented across train/test under the object-generalization knob: visual similarity, semantic class, both.
4. Whether to expose privileged ground-truth state features as a baseline input for the RL extension's value-function comparisons.
5. Cone field-of-view parameters (angle, depth) for 2D.
6. Whether mid-episode speed changes (beyond the lasting-zone mechanism) are worth adding as a third speed regime.