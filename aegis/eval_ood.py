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
from aegis.agents.conservative_fallback import ConservativeFallback
from aegis.agents.learned_agent import LearnedAgent
from aegis.agents.random_agent import RandomAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.agents.shield_agent import ActionShieldAgent
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
    model_seed: int | None = None,
    include_ablations: bool = True,
) -> dict[str, object]:
    """Return a dict of agent_name -> factory(env) for all available agents."""
    agents = OrderedDict()

    agents["random"] = lambda env: RandomAgent(env.action_space.n, seed=0)
    agents["rule_based"] = lambda env: RuleBasedAgent(n_services=n_services)
    if include_ablations:
        agents["always_fallback"] = (
            lambda env: ConservativeFallback(n_services=n_services, topology=env.topology)
        )

    suffix = f"_seed{model_seed}" if model_seed is not None else ""

    # Try loading learned models (including optional reward-penalty safe-RL baseline)
    for algo_name, algo_key in (("dqn", "dqn"), ("ppo", "ppo"), ("dqn_penalty", "dqn")):
        path = model_dir / f"{algo_name}_microservice{suffix}.zip"
        if not path.exists() and model_seed is not None:
            path = model_dir / f"{algo_name}_microservice.zip"
        if not path.exists():
            if algo_name in ("dqn", "ppo"):
                print(f"[skip] {algo_name.upper()}: no model at {path}")
            continue
        from stable_baselines3 import DQN, PPO
        cls = {"dqn": DQN, "ppo": PPO}[algo_key]
        model = cls.load(path)
        agents[algo_name] = (lambda env, m=model: LearnedAgent(m))

    # Try loading bootstrap DQN
    bootstrap_path = model_dir / f"bootstrapped_dqn_microservice{suffix}.pt"
    if not bootstrap_path.exists() and model_seed is not None:
        bootstrap_path = model_dir / "bootstrapped_dqn_microservice.pt"
    if not bootstrap_path.exists():
        print(f"[skip] BOOTSTRAP_DQN: no model at {bootstrap_path}")
        return agents

    bootstrap_model = BootstrappedDQN.load(bootstrap_path)

    # Ungated bootstrap DQN
    agents["bootstrap_dqn"] = (
        lambda env, m=bootstrap_model: UncertaintyAgent(m)
    )

    if include_ablations:
        # Action Shield / Mask on Bootstrap DQN (blocks RESTART/MIGRATE on healthy-looking obs)
        agents["shield_bootstrap"] = (
            lambda env, m=bootstrap_model: ActionShieldAgent(
                base_agent=UncertaintyAgent(m),
                n_services=n_services,
                fallback=ConservativeFallback(n_services=n_services, topology=env.topology),
            )
        )

        # Blast-radius-only gate (ignores uncertainty)
        agents["aegis_br_only"] = (
            lambda env, m=bootstrap_model: AEGISAgent(
                policy=m, topology=env.topology, gate_mode="blast_radius_only",
            )
        )

        # Random gate with matched override rate (~5%)
        agents["aegis_random_gate"] = (
            lambda env, m=bootstrap_model: AEGISAgent(
                policy=m, topology=env.topology, gate_mode="random", random_override_prob=0.05,
            )
        )

        # AEGIS with RuleBasedAgent fallback (ablates ConservativeFallback)
        agents["aegis_rule_fallback"] = (
            lambda env, m=bootstrap_model: AEGISAgent(
                policy=m,
                topology=env.topology,
                fallback=RuleBasedAgent(n_services=n_services),
            )
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

    if include_ablations:
        # Oracle gate upper bound: overrides iff the candidate action is actually bad
        agents["aegis_oracle_gate"] = (
            lambda env, m=bootstrap_model: AEGISAgent(
                policy=m,
                topology=env.topology,
                gate_mode="oracle",
                oracle_fn=env.is_candidate_action_bad,
            )
        )

    return agents


def paired_sign_flip_p_value(x: np.ndarray, y: np.ndarray, n_perms: int = 5000, seed: int = 0) -> float:
    """
    Two-sided paired permutation (sign-flip) test for H0: mean(x - y) == 0.
    Implemented in pure NumPy with no external scipy dependency.
    """
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    obs = abs(float(np.mean(diff)))
    if obs < 1e-12:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perms, len(diff)))
    perm_means = np.abs(np.mean(signs * diff[None, :], axis=1))
    return float((np.sum(perm_means >= obs) + 1) / (n_perms + 1))


# -------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Phase 6: OOD evaluation suite")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--criticality-mode", type=str, default="ancestors",
                        choices=["ancestors", "traffic_weighted", "pagerank"])
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Optional list of model training seeds to aggregate across (e.g. --seeds 0 1 2 3 4 5 6 7 8 9)")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--out", type=str, default="results/phase6_ood.json")
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Run only these scenarios (default: all). Options: "
             + ", ".join(SCENARIOS.keys()),
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_seeds = args.seeds if args.seeds is not None else [None]
    agents_by_seed = [
        build_agent_factories(args.n_services, model_dir, model_seed=ms)
        for ms in model_seeds
    ]
    agent_names = list(agents_by_seed[0].keys())
    print(f"Agents available: {agent_names} (model_seeds={model_seeds})\n")

    scenarios_to_run = args.scenarios or list(SCENARIOS.keys())
    all_results: dict[str, list[dict]] = {}

    for scenario_name in scenarios_to_run:
        if scenario_name not in SCENARIOS:
            print(f"[warn] Unknown scenario '{scenario_name}', skipping")
            continue

        fault_config = SCENARIOS[scenario_name]
        print(f"{'=' * 86}")
        print(f"Scenario: {scenario_name} (criticality={args.criticality_mode}, seeds={len(model_seeds)}, episodes={args.episodes})")
        print(f"  fault_types={fault_config.types}")
        print(f"  prob_per_step={fault_config.prob_per_step}, duration_range={fault_config.duration_range}")
        print(f"  fault_target_bias={fault_config.fault_target_bias}, simultaneous_fault_prob={fault_config.simultaneous_fault_prob}")
        print(f"{'=' * 86}")

        def env_factory(seed, fc=fault_config):
            return MicroserviceEnv(
                n_services=args.n_services,
                episode_length=args.episode_length,
                fault_config=fc,
                seed=seed,
                criticality_mode=args.criticality_mode,
            )

        scenario_results = []
        raw_harm_by_agent: dict[str, np.ndarray] = {}
        raw_bad_by_agent: dict[str, np.ndarray] = {}

        for name in agent_names:
            # Deterministic/non-learned agents only need 1 run over the evaluation episodes
            is_deterministic_baseline = name in ("random", "rule_based", "always_fallback")
            seed_runs = []
            for s_idx, seed_agents in enumerate(agents_by_seed):
                if name not in seed_agents:
                    continue
                if is_deterministic_baseline and s_idx > 0:
                    seed_runs.append(seed_runs[0])
                    continue
                seed_runs.append(
                    run_agent_suite(
                        name, seed_agents[name], env_factory, args.episodes, return_raw=True
                    )
                )

            # Collect flat per-episode arrays for paired significance testing
            raw_harm_by_agent[name] = np.array(
                [ep["total_blast_harm_per_ep"] for sr in seed_runs for ep in sr["episode_results"]],
                dtype=np.float64,
            )
            raw_bad_by_agent[name] = np.array(
                [ep["bad_actions_per_ep"] for sr in seed_runs for ep in sr["episode_results"]],
                dtype=np.float64,
            )

            agg = {k: v for k, v in seed_runs[0].items() if k != "episode_results"}
            if len(seed_runs) > 1:
                n_s = len(seed_runs)
                for k in list(agg.keys()):
                    if isinstance(agg[k], (int, float)) and not k.endswith(("_std", "_ci95")):
                        vals = np.array([sr[k] for sr in seed_runs], dtype=np.float64)
                        agg[k] = float(np.mean(vals))
                        agg[f"{k}_seed_std"] = float(np.std(vals, ddof=1))
                        agg[f"{k}_seed_ci95"] = float(1.96 * np.std(vals, ddof=1) / np.sqrt(n_s))
                agg["n_seeds"] = n_s

            agg["scenario"] = scenario_name
            scenario_results.append(agg)

        # Compute paired permutation p-values vs bootstrap_dqn
        if "bootstrap_dqn" in raw_harm_by_agent:
            ref_harm = raw_harm_by_agent["bootstrap_dqn"]
            ref_bad = raw_bad_by_agent["bootstrap_dqn"]
            for agg in scenario_results:
                a_name = agg["agent"]
                if a_name in raw_harm_by_agent and len(raw_harm_by_agent[a_name]) == len(ref_harm):
                    agg["p_val_harm_vs_bootstrap"] = paired_sign_flip_p_value(
                        raw_harm_by_agent[a_name], ref_harm
                    )
                    agg["p_val_bad_vs_bootstrap"] = paired_sign_flip_p_value(
                        raw_bad_by_agent[a_name], ref_bad
                    )

        for agg in scenario_results:
            name = agg["agent"]
            ci_key = "avg_reward_seed_ci95" if "avg_reward_seed_ci95" in agg else "avg_reward_ci95"
            line = (
                f"  {name:>18s} | rew={agg['avg_reward']:+.3f}±{agg.get(ci_key, 0.0):.3f} "
                f"| avail={agg['avg_availability']:.3f} "
                f"| bad_ep={agg['bad_actions_per_ep']:5.2f} "
                f"| harm_ep={agg['total_blast_harm_per_ep']:5.2f} "
                f"| br_bad={agg['avg_blast_radius_of_bad_actions']:.3f} "
                f"| mttr={agg['mttr']:4.1f}"
            )
            if "override_rate" in agg:
                line += f" | ovr={agg['override_rate']:.3f}"
            if "p_val_harm_vs_bootstrap" in agg and name != "bootstrap_dqn":
                line += f" | p={agg['p_val_harm_vs_bootstrap']:.4f}"
            print(line)

        all_results[scenario_name] = scenario_results
        print()

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {out_path}")

    # Print summary tables
    _print_summary(all_results)


def _print_metric_table(all_results: dict[str, list[dict]], metric: str, title: str):
    print(f"\n{'=' * 86}")
    print(f"SUMMARY: {title} ({metric})")
    print(f"{'=' * 86}")

    agent_names = []
    for results in all_results.values():
        for r in results:
            if r["agent"] not in agent_names:
                agent_names.append(r["agent"])

    header = f"{'scenario':>16s} | " + " | ".join(f"{a:>14s}" for a in agent_names)
    print(header)
    print("-" * len(header))

    for scenario_name, results in all_results.items():
        lookup = {r["agent"]: r for r in results}
        row = f"{scenario_name:>16s} | "
        row += " | ".join(
            f"{lookup[a][metric]:14.3f}"
            if (a in lookup and metric in lookup[a]) else f"{'n/a':>14s}"
            for a in agent_names
        )
        print(row)


def _print_summary(all_results: dict[str, list[dict]]):
    """Print compact scenario × agent summary tables across unconditional and conditional metrics."""
    _print_metric_table(
        all_results,
        "total_blast_harm_per_ep",
        "Unconditional Total Blast-Weighted Harm per Episode (lower is better)",
    )
    _print_metric_table(
        all_results,
        "bad_actions_per_ep",
        "Unconditional Bad Actions per Episode (lower is better)",
    )
    _print_metric_table(
        all_results,
        "avg_blast_radius_of_bad_actions",
        "Conditional Mean Blast Radius of Bad Actions BR_bad (lower is better)",
    )
    _print_metric_table(
        all_results,
        "avg_reward",
        "Average Step Reward (higher is better)",
    )
    _print_metric_table(
        all_results,
        "avg_availability",
        "Average Availability (higher is better)",
    )
    _print_metric_table(
        all_results,
        "mttr",
        "Mean Time To Recovery in Steps (lower is better)",
    )


if __name__ == "__main__":
    main()
