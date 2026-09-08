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
| **4** | Confidence + blast-radius gating logic ("AEGIS" proper) | ✅ done (see `aegis/gating/`) — fallback is Phase 1's RuleBasedAgent as a placeholder |
| **5** | Conservative fallback policy (blast-radius-aware) | ⬜ next |
| **6** | OOD evaluation suite + ablations (rule-based vs DQN/PPO vs AEGIS-no-graph vs AEGIS-full) | ⬜ |
| **7** | Real Kubernetes wiring (kind/minikube + Chaos Mesh + Locust), replacing the local simulator | ⬜ |
| **8** | Write-up | ⬜ |

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
comes from Phase 1's `RuleBasedAgent` — a deliberate placeholder fallback
until Phase 5 builds a purpose-built blast-radius-aware one (see
`aegis_gate.py`'s docstring for the reasoning).

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

Caveats worth being upfront about (this is exactly what Phase 6 is for):
this is one seed-matched eval run, not a significance test; the fallback
is a stand-in, not the real Phase 5 policy; and PPO already achieves
bad_action_rate=0.000 in this particular simulator, so the harder,
more interesting test is whether AEGIS holds up under the OOD scenarios
Phase 6 introduces, where PPO's zero rate isn't guaranteed to survive.

---

## Repo layout

```
aegis/
  env/
    topology.py           # service dependency graph + blast-radius scoring
    microservice_env.py   # Gymnasium environment (the "simulator")
    sb3_env.py             # stable-baselines3 wrapper (Monitor, TimeLimit)
  agents/
    random_agent.py        # sanity-check floor
    rule_based.py           # HPA-style threshold baseline; also Phase 4's placeholder gate fallback
    learned_agent.py        # wraps a trained SB3 model in the same .act() interface
    uncertainty_agent.py    # wraps BootstrappedDQN in the same .act() interface (ungated)
    aegis_agent.py           # composes policy + gate + fallback into the same .act() interface
  uncertainty/
    ensemble_qnet.py         # shared trunk + K independent Q-heads
    replay_buffer.py         # bootstrap-masked experience replay
    bootstrapped_dqn.py      # training loop + .act() / .uncertainty() for Phase 3/4
  gating/
    aegis_gate.py             # Phase 4: confidence + blast-radius override decision rule
  run_baselines.py         # Phase 1 comparison script (random vs rule-based)
  train_learned.py         # Phase 2: train DQN / PPO on the simulator
  train_uncertainty.py     # Phase 3: train the bootstrap-head DQN
  run_comparison.py        # Phase 2-4: compare all agents (incl. aegis_full) head-to-head
tests/
  test_env.py
  test_uncertainty.py
  test_gating.py
results/                   # metric dumps land here (gitignored except .gitkeep)
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

# Phase 4: no separate training step -- the gate wraps the Phase 3 checkpoint.
# This now compares all six agents, including aegis_full.
PYTHONPATH=. python -m aegis.run_comparison --episodes 20 --n-services 8
```

## Why a custom simulator instead of real Kubernetes right away

The real target evaluation environment is a Kubernetes cluster (kind/minikube)
with Chaos Mesh for fault injection and Locust for traffic — that's Phase 7.
Building the RL agent, the uncertainty module, and the confidence/blast-radius
gating logic against a fast local simulator first lets those pieces get
iterated on in seconds instead of minutes, and the `MicroserviceEnv` /
`ServiceTopology` interfaces are designed to be swapped for a real
K8s-backed environment later without touching the agent code.

## The metric that matters most

Every agent is scored on the usual things (reward, availability, resource
cost) — but the metric this project's novelty claim lives or dies on is
**`avg_blast_radius_of_bad_actions`**: when the agent does mess up, how
central/high-impact was the service it messed up on? AEGIS should be the
only agent that keeps this number low even when its overall bad-action
rate isn't zero.

As of Phase 4, `aegis_full` is the first agent to actually demonstrate
this: 0.250 vs 0.316 for the same policy ungated (see "Phase 4 results"
above) — a real, if modest and not-yet-stress-tested, improvement on
exactly this number. Phase 6's OOD suite is what will show whether that
holds up outside this simulator's in-distribution fault patterns.
