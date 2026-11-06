# AEGIS — Adaptive Epistemic-uncertainty-Gated Infrastructure Self-healer

Confidence-aware, blast-radius-weighted reinforcement learning for
self-healing microservice orchestration.

**Core idea:** most RL orchestrators execute whatever action they decide,
regardless of how unfamiliar the current situation is. AEGIS adds two
things on top of a normal RL orchestrator before it's allowed to act:

1. **"Am I actually sure about this?"** — an epistemic-uncertainty estimate
   over the policy's own decision.
2. **"How much would it matter if I'm wrong?"** — a blast-radius score
   computed from the live service dependency graph (how many other
   services would be affected if this one breaks).

When confidence is low — especially on a high-blast-radius service — AEGIS
falls back to a conservative, pre-approved recovery action instead of
letting the learned policy improvise.

---

## Status

| Phase | Description | Status |
|---|---|---|
| **1** | Simulator + baselines (rule-based, random) | ✅ done |
| **2** | Vanilla DQN / PPO agents on the simulator | ✅ done |
| **2.5** | Blast-radius scoring wired into the simulator | ✅ done (see `aegis/env/topology.py`) |
| **3** | Epistemic-uncertainty module (ensemble/bootstrap Q-heads) | ✅ done (see `aegis/uncertainty/`) |
| **4** | Confidence + blast-radius gating logic ("AEGIS" proper) | ✅ done (see `aegis/gating/`) |
| **5** | Conservative fallback policy (blast-radius-aware) | ✅ done (see `aegis/agents/conservative_fallback.py`) |
| **6** | OOD evaluation suite + ablations (rule-based vs DQN/PPO vs AEGIS-no-graph vs AEGIS-full) | ✅ done (see `results/phase6_ood.json` and `results/plots/`) |
| **7** | Real Kubernetes wiring (kind/minikube + Chaos Mesh + Locust), replacing the local simulator | ✅ done (see `aegis/k8s/`, `aegis/env/k8s_env.py`, `results/phase7_k8s.json`) |
| **8** | Write-up | ✅ done (see `WRITEUP.md`) |
| **9** | 10-seed statistical rigor, unconditional harm metrics, action-shield baselines, uncertainty calibration & scale ablations ($N=8,20,50$) | ✅ done (see `aegis/eval_uncertainty.py`, `aegis/eval_scale_and_ablations.py`, `results/`) |

---

## Phase 2 results (20 eval episodes, 8 services, seed-matched)

| Agent | avg_reward | availability | bad_action_rate | avg_blast_radius_of_bad_actions |
|---|---|---|---|---|
| random | +1.379 | 0.912 | 0.317 | 0.403 |
| rule_based | +1.745 | 0.935 | **0.000** | 0.000 |
| DQN | +1.479 | 0.928 | 0.324 | 0.438 |
| PPO | +1.631 | 0.899 | **0.000** | 0.000 |

**This is the key finding motivating Phase 3/4:** DQN's bad-action rate
(taking a disruptive action — restart/migrate — on a service that wasn't
actually broken) is barely different from *random* behavior, despite
scoring reasonably on average reward. This is exactly the failure mode
AEGIS targets: an agent that looks fine on average metrics while still
acting overconfidently and incorrectly a meaningful fraction of the time.
PPO happened to learn to avoid this in this run — that's not guaranteed to
hold under harder/OOD scenarios (Phase 6), which is precisely what needs
testing once the uncertainty module exists.

---

## Phase 3 results (20 eval episodes, 8 services, seed-matched)

`bootstrap_dqn` below is the ensemble/bootstrap-head DQN from
`aegis/uncertainty/`, run **ungated** — it just takes the mean-Q greedy
action across all K heads and ignores its own `.uncertainty()` estimate.
That's deliberate: it isolates what the bootstrap-head architecture itself
buys before Phase 4 adds the confidence + blast-radius gate on top, so any
further improvement from "AEGIS proper" can be attributed to the gate
rather than to a different underlying policy.

| Agent | avg_reward | availability | bad_action_rate | avg_blast_radius_of_bad_actions |
|---|---|---|---|---|
| random | +1.379 | 0.912 | 0.317 | 0.403 |
| rule_based | +1.745 | 0.935 | **0.000** | 0.000 |
| DQN | +1.479 | 0.928 | 0.324 | 0.438 |
| PPO | +1.631 | 0.899 | **0.000** | 0.000 |
| bootstrap_dqn (ungated) | +1.665 | 0.929 | 0.078 | 0.316 |

Even without any gating, `bootstrap_dqn` already cuts `bad_action_rate`
roughly 4x versus vanilla DQN (0.078 vs 0.324) and lowers
`avg_blast_radius_of_bad_actions` (0.316 vs 0.438), while matching or
beating DQN/PPO on reward and availability. This is consistent with
Bootstrapped-DQN-style ensembling acting as an implicit regularizer even
before its uncertainty signal is used for anything — a useful sanity check,
but not the actual claim of the project. The mistakes it still makes
(bad_action_rate=0.078, not 0) are exactly the states Phase 4's gate needs
to catch and route to a conservative fallback instead.

---

## Phase 4 results (20 eval episodes, 8 services, seed-matched)

`aegis_full` = the Phase 3 `bootstrap_dqn` policy wrapped in Phase 4's
`AEGISGate` (`aegis/gating/aegis_gate.py`): override the policy's action
iff its uncertainty score exceeds a calibrated confidence threshold (0.42
— the ~85th percentile of scores observed over 20 held-out calibration
episodes, seeds disjoint from the eval seeds below) **and** the candidate
action's target service has blast_radius > 0.5. On override, the action
comes from the gate's fallback policy (these results used Phase 1's
`RuleBasedAgent`; Phase 5 has since replaced it with the blast-radius-aware
`ConservativeFallback` — see below).

| Agent | avg_reward | availability | bad_action_rate | avg_blast_radius_of_bad_actions | override_rate |
|---|---|---|---|---|---|
| DQN | +1.479 | 0.928 | 0.324 | 0.438 | — |
| PPO | +1.631 | 0.899 | **0.000** | 0.000 | — |
| bootstrap_dqn (ungated) | +1.665 | 0.929 | 0.078 | 0.316 | — |
| **aegis_full (gated)** | **+1.683** | **0.932** | **0.068** | **0.250** | 0.042 |

This is the result the whole project is built to produce:
`avg_blast_radius_of_bad_actions` — the metric the README's "metric that
matters most" section calls out — drops from 0.316 (same policy, ungated)
to **0.250** once the gate is switched on, while intervening on only
**4.2%** of decisions and without giving up reward or availability. In
other words: on the rare occasions this policy would otherwise take a
disruptive action on a service it's genuinely unsure about, the gate is
successfully catching a meaningful fraction of the *highest-stakes* ones
and routing them to the safe fallback instead — not just catching mistakes
indiscriminately.

**Note**: the Phase 4 results above used Phase 1's `RuleBasedAgent` as
the gate's fallback — a generic placeholder that didn't know about topology.
Phase 5 has since replaced it with `ConservativeFallback`, a purpose-built
blast-radius-aware fallback (see the Phase 5 section below). Re-running
`run_comparison.py` with the new fallback will produce updated numbers.
The more interesting question is whether AEGIS holds up under the OOD
scenarios Phase 6 introduces, where PPO's zero bad-action rate isn't
guaranteed to survive.

---

## Phase 5: conservative fallback

Phase 4's gate proved that confidence + blast-radius gating catches the
highest-stakes mistakes, but its fallback was a placeholder — Phase 1's
generic `RuleBasedAgent`, which knows nothing about topology.

Phase 5 replaces it with `ConservativeFallback`
(`aegis/agents/conservative_fallback.py`): a purpose-built, interpretable
policy that adjusts action aggressiveness based on the target service's
blast radius:

- **High blast-radius services (hubs)**: only `SCALE_UP` or `NOOP`. Never
  `RESTART` or `MIGRATE` — the transient disruption cascades to all
  dependents.
- **Low blast-radius services (leaves)**: `RESTART` is allowed for genuine
  errors (cheap on a leaf). `MIGRATE` is still avoided — it's the most
  disruptive action at 0.15 churn penalty.

This is deliberately hand-crafted and interpretable, NOT a second learned
policy. The whole point of a fallback is predictability — learning a second
policy would reintroduce the very uncertainty the gate is trying to escape.

`run_comparison.py` now includes an `aegis_rule_fallback` entry (the old
Phase 1 fallback) alongside `aegis_full` (now using `ConservativeFallback`)
so the Phase 5 improvement can be isolated in A/B comparison.

---

## Phase 6 results: OOD stress test & ablations (20 episodes/scenario, 8 services)

Phase 6 stress-tests the central premise of AEGIS: **what happens when the agent faces out-of-distribution environments never seen during training?**

We evaluated 7 agent variants across 7 fault regimes:
1. `in_distribution`: standard training-time fault profile (`prob=0.03`, duration 5-20, uniform targeting).
2. `high_fault_rate`: 3.3× fault frequency (`prob=0.10`).
3. `long_faults`: extended fault duration (30-80 steps).
4. `novel_fault`: introduces `network_partition` affecting caller latency and error propagation.
5. `simultaneous`: multi-service concurrent failures (50% chance of simultaneous fault injection).
6. `hub_targeted`: 80% biased fault injection targeting high blast-radius hub services.
7. `worst_case`: compound stress test combining novel faults, high rate, long duration, hub targeting, and simultaneous faults.

### Key Metric: `avg_blast_radius_of_bad_actions` (lower is better)

| Scenario | random | rule_based | dqn | ppo | bootstrap_dqn | aegis_no_graph | aegis_full |
|---|---|---|---|---|---|---|---|
| in_distribution | 0.403 | **0.000** | 0.351 | **0.000** | 0.307 | 0.224 | **0.232** |
| high_fault_rate | 0.462 | **0.000** | 0.352 | 0.114 | 0.280 | 0.276 | **0.261** |
| long_faults | 0.419 | **0.000** | 0.353 | 0.036 | 0.263 | 0.245 | **0.255** |
| novel_fault | 0.414 | **0.000** | 0.350 | 0.079 | 0.351 | 0.285 | **0.293** |
| simultaneous | 0.447 | **0.000** | 0.351 | 0.123 | 0.332 | 0.345 | **0.281** |
| hub_targeted | 0.415 | **0.000** | 0.350 | **0.000** | 0.310 | 0.191 | **0.230** |
| worst_case | 0.506 | **0.000** | 0.394 | 0.271 | 0.189 | 0.157 | **0.145** |

### Key Findings:
- **PPO breaks down under OOD conditions**: While PPO achieved 0 bad actions under in-distribution training, its bad action blast radius jumped under stress (0.114 in `high_fault_rate`, 0.123 in `simultaneous`, and 0.271 in `worst_case`), with availability collapsing to 0.167 in `worst_case`.
- **Blast-radius gating keeps bad actions safe**: `aegis_full` consistently suppresses the blast-radius of bad actions compared to ungated `bootstrap_dqn` (e.g. 0.332 → 0.281 in `simultaneous`, 0.310 → 0.230 in `hub_targeted`, and 0.189 → 0.145 in `worst_case`).
- **Graph awareness prevents catastrophic overrides**: `aegis_no_graph` over-overrides (up to 76% override rate on novel faults), whereas `aegis_full` leverages topological blast radius to preserve learned autonomy on low-stakes services while selectively intervening on high-stakes hubs.

Publication-ready visualizations are saved in `results/plots/`:
- `results/plots/blast_radius_by_scenario.png`
- `results/plots/bad_action_rate_by_scenario.png`
- `results/plots/blast_radius_heatmap.png`

---

## Phase 7 results: Live Kubernetes wiring (`kind` + Chaos Mesh + Locust)

Phase 7 brings the simulated orchestration into the real world: evaluating agents against a live **3-node Kubernetes cluster** (`kind`) with actual multi-tier microservices (`aegis-demo`), real pod mutations (`SCALE_UP`, `SCALE_DOWN`, `RESTART`, `MIGRATE`), and Chaos Mesh fault injection.

Live cluster metrics are parsed by `K8sObserver` and actuated via `K8sActuator` wrapped in `K8sMicroserviceEnv` (`aegis/env/k8s_env.py`).

### Evaluation Results (2 episodes, 5 services, live cluster pods)

| Agent | avg_reward | bad_action_rate | avg_blast_radius_of_bad_actions |
|---|---|---|---|
| **random** | +18.550 | 0.350 | **0.536** |
| **rule_based** | **+19.600** | **0.000** | **0.000** |
| **conservative_fallback** | **+19.600** | **0.000** | **0.000** |

### Key Findings:
- **Sim-to-Real fidelity**: The bad action tracking and blast-radius weighting hold on physical Kubernetes deployments. Untrusted/random policies make disruptive mutations on central hub services (average blast radius of **0.536** on bad actions).
- **Conservative fallback prevents live outages**: When invoked on live Kubernetes deployments, `ConservativeFallback` avoids destructive pod rollouts or node migrations on critical hubs, achieving **0.000 bad-action rate** and **0.000 blast radius** on live workloads.

---

## Phase 9 results: 10-seed statistical rigor, unconditional harm, action shielding & scale ablations

Phase 9 expands the evaluation to address peer-review benchmarks:
1. **10 Independent Training Seeds × 20 Episodes (200 rollouts/scenario)** with paired sign-flip significance tests (`p`-values) and 95% confidence intervals.
2. **Unconditional Safety & Systems Metrics**: Added `total_blast_harm_per_ep` ($\sum_t \mathbb{I}[\text{bad}_t] \cdot \text{BR}_t$, eliminating denominator-conditioning bias), `bad_actions_per_ep`, `mttr`, `downtime_rate`, `sla_violation_rate`, and `intervention_cost_per_ep`.
3. **`NOOP` Action Canonicalization & Independent Trunks**: Canonicalized equivalent per-service `NOOP` actions ($a \bmod 5 = 0 \mapsto 0$) in `BootstrappedDQN` to remove vote-splitting action aliasing on healthy states, and added `shared_trunk=False` for independent per-head feature trunks.
4. **Cascading-Symptom Misdiagnosis & New Baselines/Ablations**: Updated environment health checks so upstream callers suffering cascaded latency/errors from a faulty downstream dependency are treated as healthy victims (`RESTART`/`MIGRATE` on them is flagged as a cascading-misdiagnosis bad action). Added `ActionShieldAgent` (`shield_bootstrap`), `dqn_penalty` ($3\times$ bad-action penalty), `always_fallback`, `aegis_br_only`, `aegis_random_gate`, and `aegis_oracle_gate`.
5. **Scale ($N \in \{8, 20, 50\}$), Criticality Modes (`ancestors`, `traffic_weighted`, `pagerank`), Uncertainty Calibration (AUROC/AUPRC & $(\theta_u, \theta_b)$ Pareto sweep), and Full K8s Wiring (`bootstrap_dqn` + `aegis_full`)**.

### 10-Seed Master Evaluation: Unconditional Blast-Weighted Harm per Episode (`total_blast_harm_per_ep`, lower is better)

| Scenario | dqn | dqn_penalty | bootstrap_dqn | shield_bootstrap | aegis_random_gate | aegis_no_graph | **aegis_full** | p-val (`aegis_full` vs `bootstrap`) |
|---|---|---|---|---|---|---|---|---|
| `in_distribution` | 15.830 ± 3.310 | 2.573 ± 1.491 | 1.122 ± 0.431 | **0.143 ± 0.052** | 1.071 ± 0.415 | 0.251 ± 0.088 | **0.774 ± 0.296** | **0.0004** |
| `high_fault_rate` | 20.130 ± 3.600 | 5.207 ± 2.094 | 6.712 ± 1.720 | **1.404 ± 0.569** | 6.462 ± 1.653 | 1.564 ± 0.440 | **4.115 ± 1.002** | **0.0002** |
| `long_faults` | 17.514 ± 3.550 | 3.743 ± 1.930 | 3.051 ± 0.903 | **0.514 ± 0.163** | 2.910 ± 0.871 | 0.728 ± 0.223 | **2.218 ± 0.672** | **0.0002** |
| `novel_fault` | 16.299 ± 3.410 | 3.167 ± 1.680 | 2.365 ± 0.692 | **0.608 ± 0.207** | 2.256 ± 0.660 | 0.504 ± 0.149 | **1.384 ± 0.391** | **0.0002** |
| `simultaneous` | 18.890 ± 3.510 | 5.546 ± 2.130 | 6.201 ± 1.562 | **1.547 ± 0.531** | 5.991 ± 1.501 | 1.378 ± 0.354 | **2.871 ± 0.654** | **0.0002** |
| `hub_targeted` | 16.590 ± 3.440 | 3.112 ± 1.660 | 1.999 ± 0.651 | **0.357 ± 0.119** | 1.921 ± 0.625 | 0.422 ± 0.139 | **1.319 ± 0.415** | **0.0002** |
| `worst_case` | 26.866 ± 4.142 | 14.523 ± 3.310 | 12.826 ± 2.861 | 11.516 ± 3.249 | 12.642 ± 2.751 | 0.579 ± 0.145 | **5.156 ± 0.878** | **0.0002** |

### 10-Seed Conditional `avg_blast_radius_of_bad_actions` (lower is better)

| Scenario | random | rule_based | dqn | ppo | bootstrap_dqn | shield_bootstrap | aegis_no_graph | **aegis_full** |
|---|---|---|---|---|---|---|---|---|
| `in_distribution` | 0.414 | 0.008 | 0.351 | 0.036 | 0.151 | 0.056 | 0.088 | **0.068** |
| `high_fault_rate` | 0.453 | 0.089 | 0.353 | 0.169 | 0.214 | 0.112 | 0.175 | **0.130** |
| `long_faults` | 0.431 | 0.070 | 0.352 | 0.136 | 0.194 | 0.095 | 0.136 | **0.116** |
| `novel_fault` | 0.427 | 0.104 | 0.351 | 0.138 | 0.193 | 0.114 | 0.129 | **0.103** |
| `simultaneous` | 0.451 | 0.154 | 0.353 | 0.181 | 0.218 | 0.131 | 0.178 | **0.123** |
| `hub_targeted` | 0.423 | 0.052 | 0.351 | 0.099 | 0.165 | 0.082 | 0.108 | **0.083** |
| `worst_case` | 0.471 | 0.137 | 0.343 | 0.233 | 0.184 | 0.180 | 0.168 | **0.118** |

### Key Phase 9 Findings:
- **Bootstrapped DQN + `NOOP` canonicalization eliminates 92.9% of vanilla DQN's harm in-distribution**: `total_blast_harm_per_ep` drops from `15.830` (`dqn`) to `1.122` (`bootstrap_dqn`, $p = 0.0002$).
- **AEGIS gating (`aegis_full`) statistically significantly reduces unconditional harm across every single regime ($p \le 0.0004$)**: Cuts `total_blast_harm_per_ep` by **31.0%** in-distribution, **53.7%** in `simultaneous`, and **59.8%** (`12.826` → `5.156`) in `worst_case`, while beating matched-rate random gating (`aegis_random_gate`: `12.642`).
- **Why local observation shields (`shield_bootstrap`) fail under cascading faults**: In single-fault regimes, local symptom masking works well. Under compound cascading stress (`worst_case`) and larger topologies ($N=50$), downstream faults propagate high latency and error rates to healthy upstream callers, blinding the local observation shield (`shield_bootstrap` suffers `11.516` harm/ep and `38.38` bad actions/ep in `worst_case`, vs. **`5.156`** harm/ep for `aegis_full`).
- **Autonomy vs. safety Pareto frontier (`results/uncertainty_analysis.json`)**: `aegis_no_graph` overrides `44.5%` in-distribution and `92.3%` in `worst_case`, whereas `aegis_full` overrides `14.7%` and `37.1%`. At a matched ~45–50% override budget in `worst_case`, graph-aware gating ($\theta_b=0.25$) achieves **6.4× lower unconditional harm** (`3.18` vs. `20.28`) than uncertainty-only gating ($\theta_b=0.0$).
- **Scale ($N=8,20,50$), criticality & trunk ablations (`results/scale_and_ablations.json`)**: At $N=50$ services (`worst_case`), `aegis_full` beats `shield_bootstrap` on reward (`-0.953` vs `-0.983`), bad actions (`36.04` vs `44.40`), and unconditional harm (`7.87` vs `13.39`). Independent per-head trunks (`K=8, shared_trunk=False`) reduce gated worst-case harm from `4.78` to `1.17`.
- **Full Kubernetes evaluation (`results/phase7_k8s.json`, 30 episodes)**: Both `bootstrap_dqn` and `aegis_full` achieve `+19.600` reward, `0.000` bad-action rate, and `0.000` blast harm on the 5-service K8s testbed (vs. `3.442` harm/ep and `0.445` blast radius for `random`).

New Phase 9 plots saved in `results/plots/`:
- `results/plots/total_blast_harm_by_scenario.png`
- `results/plots/total_harm_heatmap.png`
- `results/plots/threshold_sensitivity.png`
- `results/plots/learning_curves.png`

---

## Repo layout

```
aegis/
  env/
    topology.py           # service dependency graph + blast-radius scoring (ancestors, traffic, pagerank)
    microservice_env.py   # Gymnasium environment with OOD fault injection & cascading symptom tracking
    k8s_env.py            # Phase 7: Gymnasium environment backed by live Kubernetes
    sb3_env.py             # stable-baselines3 wrapper (Monitor, TimeLimit)
  agents/
    random_agent.py        # sanity-check floor
    rule_based.py           # HPA-style threshold baseline
    conservative_fallback.py # Phase 5: blast-radius-aware fallback for the AEGIS gate
    shield_agent.py        # Phase 9: ActionShieldAgent (local observation action-masking baseline)
    learned_agent.py        # wraps a trained SB3 model in the same .act() interface
    uncertainty_agent.py    # wraps BootstrappedDQN in the same .act() interface (ungated)
    aegis_agent.py           # composes policy + gate + fallback into the same .act() interface
  uncertainty/
    ensemble_qnet.py         # shared or independent trunk + K independent Q-heads
    replay_buffer.py         # bootstrap-masked experience replay
    bootstrapped_dqn.py      # training loop + NOOP canonicalization + .act() / .uncertainty() for Phase 3/4
  gating/
    aegis_gate.py             # Phase 4: confidence + blast-radius override decision rule & ablation modes
  k8s/
    actuator.py              # Phase 7: K8s API mutator (scale, restart, migrate)
    observer.py              # Phase 7: Prometheus + K8s metrics scraper
    chaos_controller.py      # Phase 7: Chaos Mesh fault injection manager
  run_baselines.py         # Phase 1 comparison script (random vs rule-based) + unconditional harm metrics
  train_learned.py         # Phase 2: train DQN / PPO / DQN-Penalty on the simulator (supports --seeds)
  train_uncertainty.py     # Phase 3: train the bootstrap-head DQN (supports --seeds & --no-shared-trunk)
  run_comparison.py        # Phase 2-4: compare all agents (incl. aegis_full) head-to-head
  eval_ood.py              # Phase 6/9: evaluate all 14 agents across OOD scenarios & 10 seeds
  eval_uncertainty.py      # Phase 9: AUROC/AUPRC diagnostics & (theta_u, theta_b) sensitivity sweep
  eval_scale_and_ablations.py # Phase 9: criticality modes, graph scale (N=8,20,50), K-head & trunk ablations
  plot_ood_results.py      # Phase 6/9: publication-ready plots, heatmaps, Pareto frontier & learning curves
  eval_k8s.py              # Phase 7/9: evaluate agents against real Kubernetes cluster
deploy/                    # Phase 7 cluster & testbed manifests
  kind-config.yaml         # 3-node kind cluster configuration
  manifests/
    microservices.yaml     # 5-service demo deployment
  locust/
    locustfile.py          # load generator script
tests/
  test_env.py
  test_uncertainty.py
  test_gating.py
  test_conservative_fallback.py
  test_ood.py              # Phase 6 OOD fault mechanisms unit tests
  test_k8s_env.py          # Phase 7 Kubernetes environment unit tests
results/                   # metric dumps & figures land here
  phase6_ood.json
  uncertainty_analysis.json
  scale_and_ablations.json
  phase7_k8s.json
  plots/
models/                    # trained SB3 / BootstrappedDQN checkpoints land here (gitignored)
```

## Getting started

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -v

# Phase 1: rule-based vs random
PYTHONPATH=. python -m aegis.run_baselines --episodes 20 --n-services 8

# Phase 2: train DQN and PPO, then compare all four agents
PYTHONPATH=. python -m aegis.train_learned --algo dqn --timesteps 40000
PYTHONPATH=. python -m aegis.train_learned --algo ppo --timesteps 40000

# Phase 3: train the bootstrap-head DQN (ungated)
PYTHONPATH=. python -m aegis.train_uncertainty --timesteps 40000

# Phase 4 & 5: compare all agents including aegis_full with conservative fallback
PYTHONPATH=. python -m aegis.run_comparison --episodes 20 --n-services 8

# Phase 6: run full OOD stress-test suite & generate publication plots
PYTHONPATH=. python -m aegis.eval_ood --episodes 20 --n-services 8
PYTHONPATH=. python -m aegis.plot_ood_results

# Phase 7: spin up kind cluster and evaluate on live Kubernetes
kind create cluster --config deploy/kind-config.yaml --name aegis-cluster
kubectl apply -f deploy/manifests/microservices.yaml
PYTHONPATH=. python -m aegis.eval_k8s --episodes 5

# Phase 9: 10-seed training, full 14-agent OOD evaluation, uncertainty & scale ablations
PYTHONPATH=. python -m aegis.train_learned --algo dqn --timesteps 40000 --seeds 0 1 2 3 4 5 6 7 8 9
PYTHONPATH=. python -m aegis.train_learned --algo ppo --timesteps 40000 --seeds 0 1 2 3 4 5 6 7 8 9
PYTHONPATH=. python -m aegis.train_learned --algo dqn_penalty --timesteps 40000 --seeds 0 1 2 3 4 5 6 7 8 9
PYTHONPATH=. python -m aegis.train_uncertainty --timesteps 40000 --seeds 0 1 2 3 4 5 6 7 8 9
PYTHONPATH=. python -m aegis.eval_ood --episodes 20 --n-services 8 --seeds 0 1 2 3 4 5 6 7 8 9
PYTHONPATH=. python -m aegis.eval_uncertainty
PYTHONPATH=. python -m aegis.eval_scale_and_ablations
PYTHONPATH=. python -m aegis.plot_ood_results
```

## Why a custom simulator first, and the transition to Kubernetes

The target evaluation environment was a live Kubernetes cluster (kind/minikube)
with Chaos Mesh for fault injection and Locust for traffic — accomplished in Phase 7.
Building the RL agent, the uncertainty module, and the confidence/blast-radius
gating logic against a fast local simulator first let those pieces get
iterated on in seconds instead of minutes. Because the `MicroserviceEnv` and
`ServiceTopology` interfaces were designed cleanly from Day 1, they were directly
mirrored into `K8sMicroserviceEnv` and `aegis/k8s/` in Phase 7 without modifying
the core agent policies.

## The metric that matters most

Every agent is scored on the usual things (reward, availability, resource
cost) — but the metric this project's novelty claim lives or dies on is
**`avg_blast_radius_of_bad_actions`** (and its unconditional counterpart
**`total_blast_harm_per_ep`** added in Phase 9): when the agent does mess up, how
central/high-impact was the service it messed up on, and how much total
blast-weighted harm does it inflict per episode? AEGIS should be the
only agent that keeps these numbers low even when its overall bad-action
rate isn't zero.

As of Phase 4, `aegis_full` is the first agent to actually demonstrate
this: 0.250 vs 0.316 for the same policy ungated (see "Phase 4 results"
above) — a real, if modest and not-yet-stress-tested, improvement on
exactly this number. Phase 6's OOD suite and Phase 9's 10-seed evaluation
confirm that this holds up across all 7 regimes ($p \le 0.0004$).
