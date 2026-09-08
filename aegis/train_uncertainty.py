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
from pathlib import Path

from aegis.env.microservice_env import MicroserviceEnv
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mask-prob", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    env = MicroserviceEnv(
        n_services=args.n_services, episode_length=args.episode_length, seed=args.seed
    )
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = BootstrappedDQN(
        obs_dim=obs_dim,
        n_actions=n_actions,
        n_heads=args.n_heads,
        mask_prob=args.mask_prob,
        seed=args.seed,
    )
    agent.learn(env, total_timesteps=args.timesteps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrapped_dqn_microservice.pt"
    agent.save(out_path)
    print(f"\nSaved trained BootstrappedDQN to {out_path}")


if __name__ == "__main__":
    main()
