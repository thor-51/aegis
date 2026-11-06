"""
aegis/eval_k8s.py — Evaluates RL agents and AEGIS against a live Kubernetes cluster.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from aegis.agents.aegis_agent import AEGISAgent
from aegis.agents.conservative_fallback import ConservativeFallback
from aegis.agents.random_agent import RandomAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.agents.uncertainty_agent import UncertaintyAgent
from aegis.env.k8s_env import K8sMicroserviceEnv
from aegis.env.microservice_env import MicroserviceEnv
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN

logger = logging.getLogger(__name__)


def evaluate_agent_k8s(agent, env: K8sMicroserviceEnv, n_episodes: int = 3) -> dict:
    total_rewards = []
    bad_actions = []
    bad_action_blast_radii = []
    ep_bad_counts = []
    ep_blast_harms = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        ep_reward = 0.0
        ep_bad = 0
        ep_harm = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward

            if step_info.get("is_bad_action"):
                br = float(step_info.get("target_blast_radius", 0.0))
                bad_actions.append(1)
                bad_action_blast_radii.append(br)
                ep_bad += 1
                ep_harm += br
            else:
                bad_actions.append(0)

        total_rewards.append(ep_reward)
        ep_bad_counts.append(ep_bad)
        ep_blast_harms.append(ep_harm)

    bad_action_rate = float(np.mean(bad_actions)) if bad_actions else 0.0
    avg_br = float(np.mean(bad_action_blast_radii)) if bad_action_blast_radii else 0.0

    res = {
        "avg_reward": float(np.mean(total_rewards)),
        "bad_action_rate": bad_action_rate,
        "bad_actions_per_ep": float(np.mean(ep_bad_counts)),
        "total_blast_harm_per_ep": float(np.mean(ep_blast_harms)),
        "avg_blast_radius_of_bad_actions": avg_br,
    }
    if hasattr(agent, "override_rate"):
        res["override_rate"] = float(agent.override_rate)
    return res


def _load_or_train_k8s_bootstrap_model(n_services: int, model_dir: Path) -> BootstrappedDQN:
    ckpt_path = model_dir / f"bootstrapped_dqn_k8s_{n_services}svc.pt"
    if ckpt_path.exists():
        return BootstrappedDQN.load(ckpt_path)

    sim_env = MicroserviceEnv(n_services=n_services, episode_length=200, seed=0)
    agent = BootstrappedDQN(
        obs_dim=sim_env.observation_space.shape[0],
        n_actions=sim_env.action_space.n,
        n_heads=8,
        seed=0,
    )
    agent.learn(sim_env, total_timesteps=40_000, log_interval=10_000)
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save(ckpt_path)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate AEGIS on real/mock Kubernetes cluster")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per agent")
    parser.add_argument("--step-duration", type=float, default=2.0, help="Wait seconds per step")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--out", type=str, default="results/phase7_k8s.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    env = K8sMicroserviceEnv(step_duration_sec=args.step_duration)
    bootstrap_model = _load_or_train_k8s_bootstrap_model(env.n_services, Path(args.model_dir))

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

    print("\n--- Evaluating Ungated BootstrappedDQN on K8s ---")
    ungated_agent = UncertaintyAgent(bootstrap_model)
    results["bootstrap_dqn"] = evaluate_agent_k8s(ungated_agent, env, n_episodes=args.episodes)

    print("\n--- Evaluating Full AEGIS (BootstrappedDQN + Gate + Fallback) on K8s ---")
    aegis_agent = AEGISAgent(policy=bootstrap_model, topology=env.topology)
    results["aegis_full"] = evaluate_agent_k8s(aegis_agent, env, n_episodes=args.episodes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.out}:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
