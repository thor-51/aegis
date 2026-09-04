"""
rule_based.py — the "old-school" baseline: static threshold rules,
similar in spirit to Kubernetes' native HPA + simple restart-on-error logic.

Phase: 1 (baselines)

This is deliberately simple and deterministic — it's the floor every
learned approach (DQN, PPO, AEGIS) needs to beat.
"""

from __future__ import annotations

import numpy as np

from aegis.env.microservice_env import N_ACTION_TYPES, ActionType


class RuleBasedAgent:
    def __init__(
        self,
        n_services: int,
        cpu_scale_up_threshold: float = 0.75,
        cpu_scale_down_threshold: float = 0.25,
        error_restart_threshold: float = 0.3,
    ):
        self.n_services = n_services
        self.cpu_up = cpu_scale_up_threshold
        self.cpu_down = cpu_scale_down_threshold
        self.err_restart = error_restart_threshold

    def act(self, obs: np.ndarray) -> int:
        # obs layout: [cpu, mem, lat, err, replicas] * n_services, then 1 extra scalar
        per_service = obs[: self.n_services * 5].reshape(self.n_services, 5)

        # Priority 1: restart the worst error offender, if any is bad enough.
        err = per_service[:, 3]
        worst_err_idx = int(np.argmax(err))
        if err[worst_err_idx] > self.err_restart:
            return worst_err_idx * N_ACTION_TYPES + ActionType.RESTART

        # Priority 2: scale up the most CPU-saturated service, if over threshold.
        cpu = per_service[:, 0]
        worst_cpu_idx = int(np.argmax(cpu))
        if cpu[worst_cpu_idx] > self.cpu_up:
            return worst_cpu_idx * N_ACTION_TYPES + ActionType.SCALE_UP

        # Priority 3: scale down the most idle service, if very underutilized.
        best_cpu_idx = int(np.argmin(cpu))
        if cpu[best_cpu_idx] < self.cpu_down and per_service[best_cpu_idx, 4] > 1 / 10:
            return best_cpu_idx * N_ACTION_TYPES + ActionType.SCALE_DOWN

        return 0  # noop on service 0
