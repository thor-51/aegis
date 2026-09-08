"""
run_comparison.py — evaluate all agents built so far (random, rule-based,
DQN, PPO, bootstrap_dqn, aegis_full) on identical episodes and report the
same metrics for each, so results are directly comparable.

Assumes DQN/PPO/bootstrap_dqn were already trained via train_learned.py /
train_uncertainty.py (models/*.zip, models/*.pt). Missing models are
skipped with a warning rather than failing the run.

Usage:
    python -m aegis.run_comparison --episodes 20 --n-services 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aegis.agents.aegis_agent import AEGISAgent
from aegis.agents.learned_agent import LearnedAgent
from aegis.agents.random_agent import RandomAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.agents.uncertainty_agent import UncertaintyAgent
from aegis.env.microservice_env import MicroserviceEnv
from aegis.run_baselines import run_agent_suite
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


def try_load_learned(algo: str, model_dir: Path):
    path = model_dir / f"{algo}_microservice.zip"
    if not path.exists():
        print(f"[skip] {algo.upper()}: no trained model found at {path} "
              f"(run `python -m aegis.train_learned --algo {algo}` first)")
        return None
    from stable_baselines3 import DQN, PPO

    cls = {"dqn": DQN, "ppo": PPO}[algo]
    model = cls.load(path)
    return model


def try_load_bootstrapped_dqn(model_dir: Path):
    path = model_dir / "bootstrapped_dqn_microservice.pt"
    if not path.exists():
        print(f"[skip] BOOTSTRAP_DQN: no trained model found at {path} "
              f"(run `python -m aegis.train_uncertainty` first)")
        return None
    return BootstrappedDQN.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--out", type=str, default="results/phase2_comparison.json")
    args = parser.parse_args()

    def env_factory(seed):
        return MicroserviceEnv(
            n_services=args.n_services, episode_length=args.episode_length, seed=seed
        )

    agents = {
        "random": lambda env: RandomAgent(env.action_space.n, seed=0),
        "rule_based": lambda env: RuleBasedAgent(n_services=args.n_services),
    }

    model_dir = Path(args.model_dir)
    for algo in ("dqn", "ppo"):
        model = try_load_learned(algo, model_dir)
        if model is not None:
            agents[algo] = (lambda env, m=model: LearnedAgent(m))

    # Phase 3: bootstrap-head DQN, *ungated* -- takes the mean-Q greedy
    # action across all heads and ignores its own uncertainty estimate.
    # This is the fair baseline for Phase 4's gated AEGIS to beat: same
    # underlying policy, no confidence/blast-radius override.
    bootstrap_model = try_load_bootstrapped_dqn(model_dir)
    if bootstrap_model is not None:
        agents["bootstrap_dqn"] = (lambda env, m=bootstrap_model: UncertaintyAgent(m))

        # Phase 4: the same policy, now gated. env.topology is a fresh
        # ServiceTopology per episode (env_factory reseeds it), so the
        # agent -- and its RuleBasedAgent fallback -- must be constructed
        # per episode too, same as every other agent here.
        agents["aegis_full"] = (
            lambda env, m=bootstrap_model: AEGISAgent(policy=m, topology=env.topology)
        )

    all_results = []
    for name, factory in agents.items():
        agg = run_agent_suite(name, factory, env_factory, args.episodes)
        all_results.append(agg)
        line = (
            f"{name:>12s} | avg_reward={agg['avg_reward']:+.3f} "
            f"| availability={agg['avg_availability']:.3f} "
            f"| bad_action_rate={agg['bad_action_rate']:.3f} "
            f"| avg_blast_radius_of_bad_actions={agg['avg_blast_radius_of_bad_actions']:.3f}"
        )
        if "override_rate" in agg:
            line += f" | override_rate={agg['override_rate']:.3f}"
        print(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
