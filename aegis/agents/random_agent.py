"""random_agent.py — pure random baseline (sanity-check floor, not a real comparison point)."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, action_space_n: int, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.n = action_space_n

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n))
