"""
sb3_env.py — small helpers to make MicroserviceEnv play nicely with
stable-baselines3 (which wants a Gymnasium env factory + Monitor wrapper).

Phase: 2 (vanilla DQN / PPO baselines)
"""

from __future__ import annotations

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from aegis.env.microservice_env import MicroserviceEnv


def make_env(n_services: int = 8, episode_length: int = 200, seed: int | None = None):
    def _init():
        env = MicroserviceEnv(n_services=n_services, episode_length=episode_length, seed=seed)
        env = TimeLimit(env, max_episode_steps=episode_length)
        env = Monitor(env)
        return env

    return _init
