"""
eval_ood.py — Phase 6 deliverable: evaluate all agent variants across
in-distribution and out-of-distribution (OOD) fault scenarios.

This is the stress test for AEGIS's core claim. Phases 1–5 proved the gate
works under training-time conditions — but PPO already hit bad_action_rate=0
there, making the gate look unnecessary. Phase 6 introduces scenarios the
agents have never trained on (novel fault types, higher fault rates,
correlated hub failures, simultaneous multi-service faults) to test whether
AEGIS's uncertainty-based gate matters when the policy is genuinely out of
its depth.

Usage:
    python -m aegis.eval_ood --episodes 20 --n-services 8

Assumes models were trained via train_learned.py / train_uncertainty.py.
Missing models are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from aegis.agents.aegis_agent import AEGISAgent
from aegis.agents.learned_agent import LearnedAgent
from aegis.agents.random_agent import RandomAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.agents.uncertainty_agent import UncertaintyAgent
from aegis.env.microservice_env import FaultConfig, MicroserviceEnv
from aegis.run_baselines import run_agent_suite
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


# -------------------------------------------------------------------- #
# OOD scenarios
# -------------------------------------------------------------------- #
# Training distribution: FaultConfig() — 4 fault types, prob=0.03,
# duration 5-20, uniform targeting, no simultaneous faults.

SCENARIOS: dict[str, FaultConfig] = OrderedDict([
    ("in_distribution", FaultConfig()),

    ("high_fault_rate", FaultConfig(prob_per_step=0.10)),

    ("long_faults", FaultConfig(duration_range=(30, 80))),

    ("novel_fault", FaultConfig(
        types=("cpu_spike", "latency_spike", "crash_loop", "mem_leak", "network_partition"),
    )),

    ("simultaneous", FaultConfig(
        simultaneous_fault_prob=0.5,
    )),

    ("hub_targeted", FaultConfig(
        fault_target_bias=0.8,
    )),

    ("worst_case", FaultConfig(
        types=("cpu_spike", "latency_spike", "crash_loop", "mem_leak", "network_partition"),
        prob_per_step=0.10,
        duration_range=(30, 80),
        fault_target_bias=0.6,
        simultaneous_fault_prob=0.3,
    )),
])


# -------------------------------------------------------------------- #
# Agent builders
# -------------------------------------------------------------------- #
def build_agent_factories(
    n_services: int,
    model_dir: Path,
) -> dict[str, object]:
    """Return a dict of agent_name -> factory(env) for all available agents."""
    agents = OrderedDict()

    agents["random"] = lambda env: RandomAgent(env.action_space.n, seed=0)
    agents["rule_based"] = lambda env: RuleBasedAgent(n_services=n_services)

    # Try loading learned models
    for algo in ("dqn", "ppo"):
        path = model_dir / f"{algo}_microservice.zip"
        if not path.exists():
            print(f"[skip] {algo.upper()}: no model at {path}")
            continue
        from stable_baselines3 import DQN, PPO
        cls = {"dqn": DQN, "ppo": PPO}[algo]
        model = cls.load(path)
        agents[algo] = (lambda env, m=model: LearnedAgent(m))

    # Try loading bootstrap DQN
    bootstrap_path = model_dir / "bootstrapped_dqn_microservice.pt"
    if not bootstrap_path.exists():
        print(f"[skip] BOOTSTRAP_DQN: no model at {bootstrap_path}")
        return agents

    bootstrap_model = BootstrappedDQN.load(bootstrap_path)

    # Ungated bootstrap DQN
    agents["bootstrap_dqn"] = (
        lambda env, m=bootstrap_model: UncertaintyAgent(m)
    )

    # AEGIS no-graph: confidence gating only, no blast-radius check
    agents["aegis_no_graph"] = (
        lambda env, m=bootstrap_model: AEGISAgent(
            policy=m, topology=env.topology, use_blast_radius=False,
        )
    )

    # AEGIS full: confidence + blast-radius + ConservativeFallback
    agents["aegis_full"] = (
        lambda env, m=bootstrap_model: AEGISAgent(
            policy=m, topology=env.topology,
        )
    )

    return agents


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Phase 6: OOD evaluation suite")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--out", type=str, default="results/phase6_ood.json")
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Run only these scenarios (default: all). Options: "
             + ", ".join(SCENARIOS.keys()),
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    agents = build_agent_factories(args.n_services, model_dir)
    print(f"Agents available: {list(agents.keys())}\n")

    scenarios_to_run = args.scenarios or list(SCENARIOS.keys())
    all_results: dict[str, list[dict]] = {}

    for scenario_name in scenarios_to_run:
        if scenario_name not in SCENARIOS:
            print(f"[warn] Unknown scenario '{scenario_name}', skipping")
            continue

        fault_config = SCENARIOS[scenario_name]
        print(f"{'=' * 70}")
        print(f"Scenario: {scenario_name}")
        print(f"  fault_types={fault_config.types}")
        print(f"  prob_per_step={fault_config.prob_per_step}")
        print(f"  duration_range={fault_config.duration_range}")
        print(f"  fault_target_bias={fault_config.fault_target_bias}")
        print(f"  simultaneous_fault_prob={fault_config.simultaneous_fault_prob}")
        print(f"{'=' * 70}")

        def env_factory(seed, fc=fault_config):
            return MicroserviceEnv(
                n_services=args.n_services,
                episode_length=args.episode_length,
                fault_config=fc,
                seed=seed,
            )

        scenario_results = []
        for name, factory in agents.items():
            agg = run_agent_suite(name, factory, env_factory, args.episodes)
            agg["scenario"] = scenario_name
            scenario_results.append(agg)

            line = (
                f"  {name:>18s} | avg_reward={agg['avg_reward']:+.3f} "
                f"| avail={agg['avg_availability']:.3f} "
                f"| bad_action_rate={agg['bad_action_rate']:.3f} "
                f"| blast_radius_bad={agg['avg_blast_radius_of_bad_actions']:.3f}"
            )
            if "override_rate" in agg:
                line += f" | override={agg['override_rate']:.3f}"
            print(line)

        all_results[scenario_name] = scenario_results
        print()

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {out_path}")

    # Print summary table
    _print_summary(all_results)


def _print_summary(all_results: dict[str, list[dict]]):
    """Print a compact scenario × agent summary table for the key metric."""
    print(f"\n{'=' * 70}")
    print("SUMMARY: avg_blast_radius_of_bad_actions (lower is better)")
    print(f"{'=' * 70}")

    # Collect all agent names across scenarios
    agent_names = []
    for results in all_results.values():
        for r in results:
            if r["agent"] not in agent_names:
                agent_names.append(r["agent"])

    header = f"{'scenario':>20s} | " + " | ".join(f"{a:>12s}" for a in agent_names)
    print(header)
    print("-" * len(header))

    for scenario_name, results in all_results.items():
        lookup = {r["agent"]: r for r in results}
        row = f"{scenario_name:>20s} | "
        row += " | ".join(
            f"{lookup[a]['avg_blast_radius_of_bad_actions']:12.3f}"
            if a in lookup else f"{'n/a':>12s}"
            for a in agent_names
        )
        print(row)

    print()
    print("SUMMARY: bad_action_rate (lower is better)")
    print("-" * len(header))
    for scenario_name, results in all_results.items():
        lookup = {r["agent"]: r for r in results}
        row = f"{scenario_name:>20s} | "
        row += " | ".join(
            f"{lookup[a]['bad_action_rate']:12.3f}"
            if a in lookup else f"{'n/a':>12s}"
            for a in agent_names
        )
        print(row)


if __name__ == "__main__":
    main()
