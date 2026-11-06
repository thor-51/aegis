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
import json
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from aegis.env.sb3_env import make_env

ALGOS = {"dqn": DQN, "ppo": PPO, "dqn_penalty": DQN}


def train_single_seed(
    algo: str,
    timesteps: int,
    n_services: int,
    episode_length: int,
    seed: int,
    out_dir: Path,
    penalty_scale: float = 1.0,
    save_default_name: bool = True,
):
    if algo == "dqn_penalty" and penalty_scale == 1.0:
        penalty_scale = 3.0

    env_fn = make_env(
        n_services=n_services,
        episode_length=episode_length,
        seed=seed,
        bad_action_penalty_scale=penalty_scale,
    )
    monitor_env = env_fn()
    env = DummyVecEnv([lambda: monitor_env])

    algo_cls = ALGOS[algo]
    kwargs = dict(policy="MlpPolicy", env=env, verbose=0, seed=seed)
    if algo in ("dqn", "dqn_penalty"):
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
    model.learn(total_timesteps=timesteps)

    out_dir.mkdir(parents=True, exist_ok=True)
    seeded_path = out_dir / f"{algo}_microservice_seed{seed}.zip"
    model.save(seeded_path)
    if save_default_name and seed == 0:
        default_path = out_dir / f"{algo}_microservice.zip"
        model.save(default_path)

    ep_rewards = [float(r) for r in monitor_env.get_episode_rewards()]
    print(f"Saved {algo.upper()} (seed={seed}, episodes={len(ep_rewards)}, final_20_rew={sum(ep_rewards[-20:])/max(1,len(ep_rewards[-20:])):+.3f}) -> {seeded_path}")
    return {"algo": algo, "seed": seed, "episode_rewards": ep_rewards}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS.keys()), required=True)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Optional list of seeds to train sequentially (e.g. --seeds 0 1 2 3 4)")
    parser.add_argument("--penalty-scale", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--curves-out", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    seeds = args.seeds if args.seeds is not None else [args.seed]
    curves = []
    for s in seeds:
        c = train_single_seed(
            algo=args.algo,
            timesteps=args.timesteps,
            n_services=args.n_services,
            episode_length=args.episode_length,
            seed=s,
            out_dir=out_dir,
            penalty_scale=args.penalty_scale,
            save_default_name=(s == seeds[0]),
        )
        curves.append(c)

    curves_path = Path(args.curves_out or f"results/learning_curves_{args.algo}.json")
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.write_text(json.dumps(curves, indent=2))
    print(f"Saved learning curves to {curves_path}")


if __name__ == "__main__":
    main()
