"""
ensemble_qnet.py — the "K Q-heads on a shared trunk" network AEGIS's
epistemic-uncertainty estimate is built on.

Phase: 3 (epistemic-uncertainty module)

Design follows Osband et al., "Deep Exploration via Bootstrapped DQN"
(2016): a shared feature trunk feeds K independent heads, each producing
its own Q(s, ·) estimate. The heads are decorrelated by (a) random
initialization and (b) each seeing a different bootstrap-masked subset of
the replay buffer during training (see replay_buffer.py) — not by
architecture alone. Disagreement across the K heads on a given state is
the raw signal that Phase 4's confidence/blast-radius gate will consume.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnsembleQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_heads: int = 8,
        trunk_hidden: int = 128,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_actions = n_actions

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(trunk_hidden, head_hidden),
                    nn.ReLU(),
                    nn.Linear(head_hidden, n_actions),
                )
                for _ in range(n_heads)
            ]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q-values of shape (n_heads, batch, n_actions)."""
        feat = self.trunk(obs)
        return torch.stack([head(feat) for head in self.heads], dim=0)
