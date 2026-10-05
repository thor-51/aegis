"""
aegis/eval_k8s.py — Evaluates RL agents and AEGIS against a live Kubernetes cluster.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from aegis.agents.conservative_fallback import ConservativeFallback
from aegis.agents.random_agent import RandomAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.env.k8s_env import K8sMicroserviceEnv
from aegis.gating.aegis_gate import AEGISGate

logger = logging.getLogger(__name__)


def evaluate_agent_k8s(agent, env: K8sMicroserviceEnv, n_episodes: int = 3) -> dict:
    total_rewards = []
    bad_actions = []
    bad_action_blast_radii = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward

            if step_info.get("is_bad_action"):
                bad_actions.append(1)
                bad_action_blast_radii.append(step_info.get("target_blast_radius", 0.0))
            else:
                bad_actions.append(0)

        total_rewards.append(ep_reward)

    bad_action_rate = float(np.mean(bad_actions)) if bad_actions else 0.0
    avg_br = float(np.mean(bad_action_blast_radii)) if bad_action_blast_radii else 0.0

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "bad_action_rate": bad_action_rate,
        "avg_blast_radius_of_bad_actions": avg_br,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AEGIS on real/mock Kubernetes cluster")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per agent")
    parser.add_argument("--step-duration", type=float, default=2.0, help="Wait seconds per step")
    parser.add_argument("--out", type=str, default="results/phase7_k8s.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    env = K8sMicroserviceEnv(step_duration_sec=args.step_duration)

    results = {}

    print("\n--- Evaluating RandomAgent on K8s ---")
    random_agent = RandomAgent(int(env.action_space.n))
    results["random"] = evaluate_agent_k8s(random_agent, env, n_episodes=args.episodes)

    print("\n--- Evaluating RuleBasedAgent on K8s ---")
    rule_agent = RuleBasedAgent(env.n_services)
    results["rule_based"] = evaluate_agent_k8s(rule_agent, env, n_episodes=args.episodes)

    print("\n--- Evaluating ConservativeFallback on K8s ---")
    fallback_agent = ConservativeFallback(n_services=env.n_services, topology=env.topology)
    results["conservative_fallback"] = evaluate_agent_k8s(fallback_agent, env, n_episodes=args.episodes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.out}:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
