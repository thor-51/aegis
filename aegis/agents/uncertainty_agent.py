"""
uncertainty_agent.py — wraps a trained BootstrappedDQN to match the
.act(obs) interface used by RandomAgent / RuleBasedAgent / LearnedAgent, so
run_baselines.py / run_comparison.py can score it the same way as everyone
else.

Deliberately *not* gated: .act() here just takes the mean-Q greedy action
across all K heads, ignoring the uncertainty estimate entirely. This is the
"same policy quality as DQN/PPO, minus any confidence/blast-radius gating"
control condition described in bootstrapped_dqn.py -- it exists so that any
improvement AEGIS shows once Phase 4's gate is added can be attributed to
the gate, not to a different underlying policy. The gate itself (which
*does* use .uncertainty()) is Phase 4.
"""

from __future__ import annotations

import numpy as np

from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN, UncertaintyEstimate


class UncertaintyAgent:
    def __init__(self, model: BootstrappedDQN):
        self.model = model

    def act(self, obs: np.ndarray) -> int:
        return self.model.act(obs)

    def uncertainty(self, obs: np.ndarray) -> UncertaintyEstimate:
        """Exposed for Phase 4's gate and for inspection/plots; unused by .act()."""
        return self.model.uncertainty(obs)
