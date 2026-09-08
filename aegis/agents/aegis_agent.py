"""
aegis_agent.py — Phase 4 deliverable: "AEGIS proper" as a single agent
matching the .act(obs) interface used by every other agent in this repo, so
run_baselines.py / run_comparison.py can score it identically to
random / rule_based / DQN / PPO / bootstrap_dqn (ungated).

This is the composition point: Phase 3's BootstrappedDQN provides the
uncertainty estimate, Phase 4's AEGISGate decides whether to trust it, and
(until Phase 5 lands) the Phase 1 RuleBasedAgent stands in as the
conservative fallback. See aegis_gate.py's module docstring for why
RuleBasedAgent specifically was picked as the placeholder.
"""

from __future__ import annotations

import numpy as np

from aegis.agents.rule_based import RuleBasedAgent
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
        self.fallback = fallback or RuleBasedAgent(n_services=topology.n_services)
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
