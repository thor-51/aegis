"""
replay_buffer.py — bootstrap replay buffer for the ensemble Q-network.

Phase: 3 (epistemic-uncertainty module)

Standard experience replay, plus a per-transition, per-head boolean mask:
each of the K heads independently "sees" a given transition with
probability `mask_prob`, decided once at insertion time and stored
alongside it. This is the bootstrap-masking scheme from Osband et al.,
"Deep Exploration via Bootstrapped DQN" (2016). Training each head on a
different resampled subset of the data is what makes the heads actually
disagree on genuinely unfamiliar states, instead of just converging to
(near-)identical functions because they all saw the same data.
"""

from __future__ import annotations

import numpy as np


class BootstrapReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        n_heads: int,
        mask_prob: float = 0.8,
        seed: int | None = None,
    ):
        self.capacity = capacity
        self.n_heads = n_heads
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, n_heads), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.masks[i] = (self.rng.random(self.n_heads) < self.mask_prob).astype(np.float32)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
            self.masks[idx],
        )

    def __len__(self) -> int:
        return self.size
