"""learned_agent.py — wraps a trained SB3 model to match the .act(obs) interface
used by RuleBasedAgent / RandomAgent, so run_baselines.py can compare all of them
the same way.
"""

from __future__ import annotations

import numpy as np


class LearnedAgent:
    def __init__(self, model):
        self.model = model

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
