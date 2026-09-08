"""
aegis_agent.py — "AEGIS proper" as a single agent matching the .act(obs)
interface used by every other agent in this repo, so run_baselines.py /
run_comparison.py can score it identically to random / rule_based / DQN /
PPO / bootstrap_dqn (ungated).

This is the composition point: Phase 3's BootstrappedDQN provides the
uncertainty estimate, Phase 4's AEGISGate decides whether to trust it, and
Phase 5's ConservativeFallback provides the blast-radius-aware fallback
action when the gate fires. (Before Phase 5, the Phase 1 RuleBasedAgent
was used as a placeholder — see aegis_gate.py's module docstring.)
"""

from __future__ import annotations

import numpy as np

from aegis.agents.conservative_fallback import ConservativeFallback
from aegis.env.topology import ServiceTopology
from aegis.gating.aegis_gate import (
    DEFAULT_BLAST_RADIUS_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    AEGISGate,
)
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


class AEGISAgent:
    def __init__(
        self,
        policy: BootstrappedDQN,
        topology: ServiceTopology,
        fallback=None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        blast_radius_threshold: float = DEFAULT_BLAST_RADIUS_THRESHOLD,
        use_blast_radius: bool = True,
    ):
        self.fallback = fallback or ConservativeFallback(
            n_services=topology.n_services, topology=topology,
        )
        self.gate = AEGISGate(
            policy=policy,
            fallback=self.fallback,
            topology=topology,
            confidence_threshold=confidence_threshold,
            blast_radius_threshold=blast_radius_threshold,
            use_blast_radius=use_blast_radius,
        )

    def act(self, obs: np.ndarray) -> int:
        return self.gate.act(obs)

    @property
    def override_rate(self) -> float:
        return self.gate.override_rate
