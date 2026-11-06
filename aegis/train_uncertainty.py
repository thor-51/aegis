"""
train_uncertainty.py — Phase 3 deliverable: train the ensemble/bootstrap-head
BootstrappedDQN on MicroserviceEnv.

Unlike train_learned.py (which hands off to SB3), this drives
BootstrappedDQN's own training loop directly, since the per-head bootstrap
masking and cross-head disagreement scoring live outside what SB3 exposes.

Usage:
    python -m aegis.train_uncertainty --timesteps 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis.env.microservice_env import MicroserviceEnv
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


def train_single_seed(
    timesteps: int,
    n_services: int,
    episode_length: int,
    n_heads: int,
    mask_prob: float,
    seed: int,
    shared_trunk: bool,
    out_dir: Path,
    save_default_name: bool = True,
    tag: str = "",
):
    env = MicroserviceEnv(
        n_services=n_services, episode_length=episode_length, seed=seed
    )
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = BootstrappedDQN(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_heads=n_heads,
        mask_prob=mask_prob,
        seed=seed,
        shared_trunk=shared_trunk,
    )
    history = agent.learn(env, total_timesteps=timesteps, log_interval=10_000)

    out_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = f"bootstrapped_dqn{tag}_microservice"
    seeded_path = out_dir / f"{name_prefix}_seed{seed}.pt"
    agent.save(seeded_path)
    if save_default_name and seed == 0 and not tag:
        default_path = out_dir / "bootstrapped_dqn_microservice.pt"
        agent.save(default_path)

    print(f"Saved BootstrappedDQN (seed={seed}, heads={n_heads}, shared_trunk={shared_trunk}) -> {seeded_path}")
    return {
        "seed": seed,
        "n_heads": n_heads,
        "shared_trunk": shared_trunk,
        "episode_rewards": history["episode_rewards"],
        "episode_bad_action_rates": history["episode_bad_action_rates"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mask-prob", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Optional list of seeds to train sequentially")
    parser.add_argument("--no-shared-trunk", action="store_true",
                        help="Use independent per-head feature trunks instead of a shared trunk")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional filename tag (e.g. _k4 or _indep)")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--curves-out", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    shared_trunk = not args.no_shared_trunk
    curves = []
    for s in seeds:
        c = train_single_seed(
            timesteps=args.timesteps,
            n_services=args.n_services,
            episode_length=args.episode_length,
            n_heads=args.n_heads,
            mask_prob=args.mask_prob,
            seed=s,
            shared_trunk=shared_trunk,
            out_dir=out_dir,
            save_default_name=(s == seeds[0]),
            tag=args.tag,
        )
        curves.append(c)

    curves_path = Path(args.curves_out or f"results/learning_curves_bootstrapped_dqn{args.tag}.json")
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.write_text(json.dumps(curves, indent=2))
    print(f"Saved learning curves to {curves_path}")


if __name__ == "__main__":
    main()
