"""
eval_scale_and_ablations.py — Reviewer Tasks #8, #9, #10:
  1. Criticality definition generalization (`ancestors`, `traffic_weighted`, `pagerank`).
  2. Topology scale generalization across graph sizes (N = 8, 20, 50 services).
  3. Ensemble head count K in {2, 4, 8, 16} and shared vs independent trunk diversity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.agents.aegis_agent import AEGISAgent
from aegis.agents.conservative_fallback import ConservativeFallback
from aegis.agents.random_agent import RandomAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.agents.shield_agent import ActionShieldAgent
from aegis.agents.uncertainty_agent import UncertaintyAgent
from aegis.env.microservice_env import MicroserviceEnv
from aegis.eval_ood import SCENARIOS
from aegis.run_baselines import run_agent_suite
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


def evaluate_criticality_modes(
    bootstrap_model: BootstrappedDQN,
    n_services: int = 8,
    episodes: int = 50,
    episode_length: int = 200,
) -> dict:
    results = {}
    scenarios = ["in_distribution", "novel_fault", "simultaneous", "hub_targeted", "worst_case"]

    for mode in ("ancestors", "traffic_weighted", "pagerank"):
        mode_res = {}
        for scen in scenarios:
            fc = SCENARIOS[scen]

            def env_factory(seed, f=fc, m=mode):
                return MicroserviceEnv(
                    n_services=n_services,
                    episode_length=episode_length,
                    fault_config=f,
                    seed=seed,
                    criticality_mode=m,
                )

            agents = {
                "rule_based": lambda env: RuleBasedAgent(n_services=n_services),
                "bootstrap_dqn": lambda env: UncertaintyAgent(bootstrap_model),
                "shield_bootstrap": lambda env: ActionShieldAgent(
                    UncertaintyAgent(bootstrap_model),
                    n_services=n_services,
                    fallback=ConservativeFallback(n_services=n_services, topology=env.topology),
                ),
                "aegis_no_graph": lambda env: AEGISAgent(
                    policy=bootstrap_model, topology=env.topology, use_blast_radius=False
                ),
                "aegis_full": lambda env: AEGISAgent(
                    policy=bootstrap_model, topology=env.topology
                ),
            }
            scen_rows = []
            for name, factory in agents.items():
                agg = run_agent_suite(name, factory, env_factory, episodes)
                scen_rows.append(agg)
            mode_res[scen] = scen_rows
        results[mode] = mode_res
    return results


def evaluate_scale_generalization(
    scales: tuple[int, ...] = (8, 20, 50),
    train_steps: int = 12_000,
    episodes: int = 30,
    episode_length: int = 200,
) -> dict:
    """Train a BootstrappedDQN at each service graph scale N and evaluate on worst_case & simultaneous."""
    scale_results = {}
    for n_svc in scales:
        print(f"\n--- Scale N={n_svc} services (training BootstrappedDQN for {train_steps} steps) ---")
        train_env = MicroserviceEnv(n_services=n_svc, episode_length=episode_length, seed=0)
        model = BootstrappedDQN(
            obs_dim=train_env.observation_space.shape[0],
            n_actions=train_env.action_space.n,
            n_heads=8,
            seed=0,
        )
        model.learn(train_env, total_timesteps=train_steps, log_interval=max(1, train_steps // 2))

        scale_results[str(n_svc)] = {}
        for scen in ("in_distribution", "simultaneous", "worst_case"):
            fc = SCENARIOS[scen]

            def env_factory(seed, n=n_svc, f=fc):
                return MicroserviceEnv(
                    n_services=n, episode_length=episode_length, fault_config=f, seed=seed
                )

            agents = {
                "random": lambda env, n=n_svc: RandomAgent(env.action_space.n, seed=0),
                "rule_based": lambda env, n=n_svc: RuleBasedAgent(n_services=n),
                "always_fallback": lambda env, n=n_svc: ConservativeFallback(
                    n_services=n, topology=env.topology
                ),
                "bootstrap_dqn": lambda env, m=model: UncertaintyAgent(m),
                "shield_bootstrap": lambda env, m=model, n=n_svc: ActionShieldAgent(
                    UncertaintyAgent(m),
                    n_services=n,
                    fallback=ConservativeFallback(n_services=n, topology=env.topology),
                ),
                "aegis_no_graph": lambda env, m=model: AEGISAgent(
                    policy=m, topology=env.topology, use_blast_radius=False
                ),
                "aegis_full": lambda env, m=model: AEGISAgent(
                    policy=m, topology=env.topology
                ),
            }
            rows = []
            for name, factory in agents.items():
                agg = run_agent_suite(name, factory, env_factory, episodes)
                rows.append(agg)
                print(
                    f"  N={n_svc:2d} | {scen:15s} | {name:16s} | "
                    f"rew={agg['avg_reward']:+.3f} | harm/ep={agg['total_blast_harm_per_ep']:5.2f} | "
                    f"bad/ep={agg['bad_actions_per_ep']:5.2f} | ovr={agg.get('override_rate', 0.0):.3f}"
                )
            scale_results[str(n_svc)][scen] = rows
    return scale_results


def evaluate_head_and_trunk_ablations(
    n_services: int = 8,
    train_steps: int = 20_000,
    episodes: int = 30,
    episode_length: int = 200,
) -> list[dict]:
    configs = [
        {"n_heads": 2, "shared_trunk": True},
        {"n_heads": 4, "shared_trunk": True},
        {"n_heads": 8, "shared_trunk": True},
        {"n_heads": 16, "shared_trunk": True},
        {"n_heads": 8, "shared_trunk": False},
    ]
    rows = []
    fc = SCENARIOS["simultaneous"]

    def env_factory(seed):
        return MicroserviceEnv(
            n_services=n_services, episode_length=episode_length, fault_config=fc, seed=seed
        )

    for cfg in configs:
        k = cfg["n_heads"]
        st = cfg["shared_trunk"]
        label = f"K={k},shared_trunk={st}"
        print(f"\nTraining {label} ({train_steps} steps)...")
        train_env = MicroserviceEnv(n_services=n_services, episode_length=episode_length, seed=0)
        model = BootstrappedDQN(
            obs_dim=train_env.observation_space.shape[0],
            n_actions=train_env.action_space.n,
            n_heads=k,
            shared_trunk=st,
            seed=0,
        )
        model.learn(train_env, total_timesteps=train_steps, log_interval=max(1, train_steps // 2))

        ungated = run_agent_suite(
            f"ungated_{label}", lambda env, m=model: UncertaintyAgent(m), env_factory, episodes
        )
        gated = run_agent_suite(
            f"aegis_{label}",
            lambda env, m=model: AEGISAgent(policy=m, topology=env.topology),
            env_factory,
            episodes,
        )
        rows.append(
            {
                "n_heads": k,
                "shared_trunk": st,
                "ungated_harm_per_ep": ungated["total_blast_harm_per_ep"],
                "ungated_bad_per_ep": ungated["bad_actions_per_ep"],
                "ungated_reward": ungated["avg_reward"],
                "aegis_harm_per_ep": gated["total_blast_harm_per_ep"],
                "aegis_bad_per_ep": gated["bad_actions_per_ep"],
                "aegis_reward": gated["avg_reward"],
                "aegis_override_rate": gated.get("override_rate", 0.0),
            }
        )
        print(
            f"  {label:24s} | ungated_harm={ungated['total_blast_harm_per_ep']:.2f} "
            f"-> aegis_harm={gated['total_blast_harm_per_ep']:.2f} "
            f"(ovr={gated.get('override_rate', 0.0):.3f})"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/bootstrapped_dqn_microservice.pt")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--scale-train-steps", type=int, default=15_000)
    parser.add_argument("--ablation-train-steps", type=int, default=20_000)
    parser.add_argument("--out", type=str, default="results/scale_and_ablations.json")
    args = parser.parse_args()

    model = BootstrappedDQN.load(args.model_path)
    print("1. Evaluating alternative criticality definitions (ancestors, traffic_weighted, pagerank)...")
    crit = evaluate_criticality_modes(model, episodes=args.episodes)

    print("\n2. Evaluating scale generalization across N in {8, 20, 50} services...")
    scale = evaluate_scale_generalization(
        scales=(8, 20, 50), train_steps=args.scale_train_steps, episodes=args.episodes
    )

    print("\n3. Evaluating ensemble K in {2, 4, 8, 16} and shared vs independent trunk diversity...")
    k_abl = evaluate_head_and_trunk_ablations(
        train_steps=args.ablation_train_steps, episodes=args.episodes
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "criticality_modes": crit,
                "scale_generalization": scale,
                "head_and_trunk_ablations": k_abl,
            },
            indent=2,
        )
    )
    print(f"\nSaved scale and ablation results to {out_path}")


if __name__ == "__main__":
    main()

