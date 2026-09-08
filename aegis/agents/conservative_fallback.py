"""
conservative_fallback.py — Phase 5 deliverable: a blast-radius-aware
conservative fallback policy for the AEGIS gate.

Phase 4's gate overrides the learned policy when confidence is low on a
high-blast-radius service, but until now it handed off to the generic Phase 1
RuleBasedAgent, which knows nothing about topology. This agent replaces that
placeholder with a purpose-built fallback that adjusts action aggressiveness
based on the target service's blast radius.

Design rationale: this is deliberately a hand-crafted, interpretable policy —
NOT a second learned agent. The whole point of a fallback is predictability
("pre-approved recovery action" in the README's framing). Learning a second
policy would reintroduce the very uncertainty the gate is trying to escape.

Action-selection strategy:
    HIGH blast radius (> threshold):
        - CPU saturated → SCALE_UP   (adds capacity, no disruption)
        - Error/latency → SCALE_UP   (dilutes bad replicas, avoids restart churn)
        - Nothing broken → NOOP
        - NEVER restart or migrate a hub — the transient disruption cascades
          to all dependents (churn penalties: restart=0.05, migrate=0.15).

    LOW blast radius (≤ threshold):
        - High error rate → RESTART   (cheap on a leaf, clears errors fast)
        - CPU saturated  → SCALE_UP
        - Nothing broken → NOOP
        - Still avoids MIGRATE — it's the most disruptive action (0.15 churn)
          and requires multi-step fault detection we can't do from a single obs.
"""

from __future__ import annotations

import numpy as np

from aegis.env.microservice_env import N_ACTION_TYPES, ActionType
from aegis.env.topology import ServiceTopology


class ConservativeFallback:
    """Blast-radius-aware conservative fallback for the AEGIS gate."""

    def __init__(
        self,
        n_services: int,
        topology: ServiceTopology,
        cpu_scale_up_threshold: float = 0.75,
        error_restart_threshold: float = 0.3,
        latency_threshold: float = 0.15,  # normalized: 150ms / 1000
        blast_radius_threshold: float = 0.5,
    ):
        self.n_services = n_services
        self.topology = topology
        self.cpu_up = cpu_scale_up_threshold
        self.err_restart = error_restart_threshold
        self.lat_threshold = latency_threshold
        self.br_threshold = blast_radius_threshold

    def act(self, obs: np.ndarray) -> int:
        # obs layout: [cpu, mem, lat, err, replicas] * n_services, then 1 extra scalar
        per_service = obs[: self.n_services * 5].reshape(self.n_services, 5)

        # Identify the service most in need of attention.
        # Scoring: weighted combination of error rate and CPU saturation.
        urgency = per_service[:, 3] * 2.0 + per_service[:, 0]  # errors weighted 2x
        target_service = int(np.argmax(urgency))
        target_feats = per_service[target_service]

        cpu = target_feats[0]
        lat = target_feats[2]  # already normalized to [0, 1] (latency / 1000)
        err = target_feats[3]
        br = self.topology.blast_radius(target_service)

        has_error = err > self.err_restart
        has_cpu_pressure = cpu > self.cpu_up
        has_latency_issue = lat > self.lat_threshold

        if br > self.br_threshold:
            # HIGH blast radius — conservative path: never restart/migrate.
            if has_cpu_pressure or has_error or has_latency_issue:
                return target_service * N_ACTION_TYPES + ActionType.SCALE_UP
            return 0  # noop on service 0

        else:
            # LOW blast radius — allow restart for genuine errors.
            if has_error:
                return target_service * N_ACTION_TYPES + ActionType.RESTART
            if has_cpu_pressure or has_latency_issue:
                return target_service * N_ACTION_TYPES + ActionType.SCALE_UP
            return 0  # noop on service 0
