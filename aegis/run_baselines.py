"""
run_baselines.py — Phase 1 deliverable: run the random and rule-based agents
through MicroserviceEnv and report the metrics we'll track for every agent
for the rest of the project (availability, reward, bad-action rate, and
blast-radius of bad actions — the metric that motivates AEGIS's Phase 2.5
upgrade).

Usage:
    python -m aegis.run_baselines --episodes 20 --n-services 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aegis.agents.random_agent import RandomAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.env.microservice_env import MicroserviceEnv


def _compute_mttr(sla_violations_per_step: list[int]) -> float:
    """
    Mean Time To Recovery (MTTR) in steps across outage streaks in an episode.
    An outage streak starts when sla_violations > 0 and ends when it returns to 0
    (or at episode end). If no SLA violations occur, MTTR is 0.0.
    """
    streaks: list[int] = []
    cur = 0
    for v in sla_violations_per_step:
        if v > 0:
            cur += 1
        elif cur > 0:
            streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    return float(np.mean(streaks)) if streaks else 0.0


def run_episode(env: MicroserviceEnv, agent, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    rewards, availabilities, bad_actions, blast_radii = [], [], [], []
    blast_harms, downtimes, sla_rates, sla_counts, int_costs = [], [], [], [], []

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        availabilities.append(info["availability"])
        bad_actions.append(info["bad_action"])
        blast_harms.append(info.get("blast_weighted_harm", info["blast_radius_of_bad_action"]))
        downtimes.append(info.get("downtime", 1.0 if info["sla_violations"] > 0 else 0.0))
        sla_counts.append(int(info["sla_violations"]))
        sla_rates.append(info.get("sla_violation_rate", info["sla_violations"] / env.n_services))
        int_costs.append(info.get("intervention_cost", 0.0))
        if info["bad_action"]:
            blast_radii.append(info["blast_radius_of_bad_action"])

    result = {
        "avg_reward": float(np.mean(rewards)),
        "avg_availability": float(np.mean(availabilities)),
        "bad_action_rate": float(np.mean(bad_actions)),
        "bad_actions_per_ep": float(np.sum(bad_actions)),
        "avg_blast_radius_of_bad_actions": float(np.mean(blast_radii)) if blast_radii else 0.0,
        "total_blast_harm_per_ep": float(np.sum(blast_harms)),
        "mean_blast_harm_per_step": float(np.mean(blast_harms)),
        "downtime_rate": float(np.mean(downtimes)),
        "sla_violation_rate": float(np.mean(sla_rates)),
        "mttr": _compute_mttr(sla_counts),
        "intervention_cost_per_ep": float(np.sum(int_costs)),
    }
    # Gated agents (Phase 4's AEGISAgent, ActionShieldAgent) expose how often they overrode
    # the learned policy; plain agents don't have this attribute, so it's
    # only added to the result when present -- run_agent_suite aggregates
    # whatever keys results[0] happens to have, so this stays backward
    # compatible with every other agent.
    if hasattr(agent, "override_rate"):
        result["override_rate"] = agent.override_rate
    return result


def run_agent_suite(
    agent_name: str,
    agent_factory,
    env_factory,
    episodes: int,
    return_raw: bool = False,
) -> dict:
    results = []
    for ep in range(episodes):
        env = env_factory(seed=ep)
        agent = agent_factory(env)
        results.append(run_episode(env, agent, seed=ep))

    agg: dict[str, float | str | list] = {}
    n = max(1, len(results))
    for k in results[0].keys():
        vals = np.array([r[k] for r in results], dtype=np.float64)
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        agg[f"{k}_ci95"] = float(1.96 * np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    agg["agent"] = agent_name
    agg["n_episodes"] = n
    if return_raw:
        agg["episode_results"] = results
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--out", type=str, default="results/phase1_baselines.json")
    args = parser.parse_args()

    def env_factory(seed):
        return MicroserviceEnv(
            n_services=args.n_services, episode_length=args.episode_length, seed=seed
        )

    agents = {
        "random": lambda env: RandomAgent(env.action_space.n, seed=0),
        "rule_based": lambda env: RuleBasedAgent(n_services=args.n_services),
    }

    all_results = []
    for name, factory in agents.items():
        agg = run_agent_suite(name, factory, env_factory, args.episodes)
        all_results.append(agg)
        print(
            f"{name:>12s} | avg_reward={agg['avg_reward']:+.3f} "
            f"| availability={agg['avg_availability']:.3f} "
            f"| bad_action_rate={agg['bad_action_rate']:.3f} "
            f"| avg_blast_radius_of_bad_actions={agg['avg_blast_radius_of_bad_actions']:.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
