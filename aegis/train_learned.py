"""
train_learned.py — Phase 2 deliverable: train vanilla DQN and PPO agents
on MicroserviceEnv. These are the "trusts itself always" baselines that
AEGIS (Phase 3/4) is measured against — same architecture-class of agent,
minus any confidence/blast-radius gating.

Usage:
    python -m aegis.train_learned --algo dqn --timesteps 50000
    python -m aegis.train_learned --algo ppo --timesteps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from aegis.env.sb3_env import make_env

ALGOS = {"dqn": DQN, "ppo": PPO}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS.keys()), required=True)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="models")
    args = parser.parse_args()

    env = DummyVecEnv(
        [make_env(n_services=args.n_services, episode_length=args.episode_length, seed=args.seed)]
    )

    algo_cls = ALGOS[args.algo]
    kwargs = dict(policy="MlpPolicy", env=env, verbose=1, seed=args.seed)
    if args.algo == "dqn":
        # Discrete action space, small-ish state -> a lean MLP + higher exploration
        # fraction, since faults are sparse and the agent needs to see enough of them.
        kwargs.update(
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            train_freq=4,
            target_update_interval=500,
        )
    else:  # ppo
        kwargs.update(
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01,
        )

    model = algo_cls(**kwargs)
    model.learn(total_timesteps=args.timesteps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.algo}_microservice.zip"
    model.save(out_path)
    print(f"\nSaved trained {args.algo.upper()} model to {out_path}")


if __name__ == "__main__":
    main()
