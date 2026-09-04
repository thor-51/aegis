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
| **3** | Epistemic-uncertainty module (ensemble/bootstrap Q-heads) | ⬜ next |
| **4** | Confidence + blast-radius gating logic ("AEGIS" proper) | ⬜ |
| **5** | Conservative fallback policy (blast-radius-aware) | ⬜ |
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

## Repo layout

```
aegis/
  env/
    topology.py           # service dependency graph + blast-radius scoring
    microservice_env.py   # Gymnasium environment (the "simulator")
    sb3_env.py             # stable-baselines3 wrapper (Monitor, TimeLimit)
  agents/
    random_agent.py       # sanity-check floor
    rule_based.py          # HPA-style threshold baseline
    learned_agent.py       # wraps a trained SB3 model in the same .act() interface
  run_baselines.py         # Phase 1 comparison script (random vs rule-based)
  train_learned.py         # Phase 2: train DQN / PPO on the simulator
  run_comparison.py        # Phase 2: compare all four agents head-to-head
tests/
  test_env.py
results/                   # metric dumps land here (gitignored except .gitkeep)
models/                    # trained SB3 models land here (gitignored)
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
