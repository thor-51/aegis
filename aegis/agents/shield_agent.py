"""
shield_agent.py — Observation-threshold Action Shield / Mask baseline.

Blocks RESTART and MIGRATE actions targeting a service whose currently observed
latency is below 150ms (normalized < 0.15) and error rate is below 0.10,
replacing blocked actions with a fallback (default: NOOP or ConservativeFallback).

Used to compare AEGIS against a classic model-free runtime shield that requires
neither an ensemble nor a dependency graph.
"""

from __future__ import annotations

import numpy as np

from aegis.env.microservice_env import ActionType, N_ACTION_TYPES, N_FEATURES_PER_SERVICE


class ActionShieldAgent:
    def __init__(
        self,
        base_agent,
        n_services: int,
        fallback=None,
        latency_threshold_norm: float = 0.15,  # 150ms / 1000ms
        error_threshold: float = 0.10,
    ):
        self.base_agent = base_agent
        self.n_services = n_services
        self.fallback = fallback
        self.latency_threshold_norm = latency_threshold_norm
        self.error_threshold = error_threshold

        self.n_calls = 0
        self.n_overrides = 0

    def is_masked(self, obs: np.ndarray, action: int) -> bool:
        service_id, action_type_int = divmod(int(action), N_ACTION_TYPES)
        action_type = ActionType(action_type_int)
        if action_type not in (ActionType.RESTART, ActionType.MIGRATE):
            return False

        base_idx = service_id * N_FEATURES_PER_SERVICE
        lat_norm = float(obs[base_idx + 2])
        err_rate = float(obs[base_idx + 3])
        return bool(lat_norm < self.latency_threshold_norm and err_rate < self.error_threshold)

    def act(self, obs: np.ndarray) -> int:
        candidate = int(self.base_agent.act(obs))
        self.n_calls += 1

        if self.is_masked(obs, candidate):
            self.n_overrides += 1
            if self.fallback is not None:
                return int(self.fallback.act(obs))
            return 0  # canonical NOOP
        return candidate

    @property
    def override_rate(self) -> float:
        return self.n_overrides / self.n_calls if self.n_calls else 0.0

