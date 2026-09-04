"""
microservice_env.py — a lightweight, dependency-graph-aware microservice
cluster simulator, wrapped as a Gymnasium environment.

Phase: 1 (baseline environment)

This is a stand-in for the real Kubernetes-based simulator (kind/minikube +
Chaos Mesh + Locust). It runs the same *logic* — services with load,
resource usage, faults, and a dependency graph — cheaply and locally, so
Phases 1-3 (baseline agents, uncertainty module, gating logic) can be built
and iterated on fast, before wiring in the real K8s stack in a later phase.

State per service (5 features):
    [cpu_util, mem_util, latency_norm, error_rate, replica_count_norm]

Global observation = concatenation across all services + a scalar
"time since last fault" feature.

Action space (Discrete):
    For N services and 5 action types {noop, scale_up, scale_down, restart, migrate}
    action = service_id * 5 + action_type
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aegis.env.topology import ServiceTopology


class ActionType(enum.IntEnum):
    NOOP = 0
    SCALE_UP = 1
    SCALE_DOWN = 2
    RESTART = 3
    MIGRATE = 4


N_ACTION_TYPES = len(ActionType)
N_FEATURES_PER_SERVICE = 5
MAX_REPLICAS = 10


@dataclass
class ServiceState:
    cpu_util: float = 0.3
    mem_util: float = 0.3
    latency: float = 50.0       # ms
    error_rate: float = 0.0     # 0..1
    replicas: int = 2
    fault_active: str | None = None
    fault_ttl: int = 0


@dataclass
class FaultConfig:
    # Fault types this environment can inject. Kept simple in Phase 1;
    # Phase 4 (OOD evaluation) will add *novel combinations* not seen here.
    types: tuple[str, ...] = ("cpu_spike", "latency_spike", "crash_loop", "mem_leak")
    prob_per_step: float = 0.03
    duration_range: tuple[int, int] = (5, 20)


class MicroserviceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_services: int = 8,
        episode_length: int = 200,
        edge_prob: float = 0.25,
        fault_config: FaultConfig | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_services = n_services
        self.episode_length = episode_length
        self.fault_config = fault_config or FaultConfig()
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        self.topology = ServiceTopology(n_services, edge_prob=edge_prob, seed=seed)

        self.action_space = spaces.Discrete(n_services * N_ACTION_TYPES)
        obs_dim = n_services * N_FEATURES_PER_SERVICE + 1
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.services: list[ServiceState] = []
        self.t = 0
        self.steps_since_fault = 0
        self.last_action_service: int | None = None
        self.episode_log: list[dict] = []

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.services = [ServiceState() for _ in range(self.n_services)]
        self.t = 0
        self.steps_since_fault = 0
        self.last_action_service = None
        self.episode_log = []
        return self._get_obs(), {}

    def step(self, action: int):
        service_id, action_type = divmod(int(action), N_ACTION_TYPES)
        action_type = ActionType(action_type)

        self._apply_traffic_and_faults()

        pre = self.services[service_id]
        was_healthy = pre.latency < 150 and pre.error_rate < 0.1 and pre.fault_active is None

        churn_penalty = self._apply_action(service_id, action_type)
        self._propagate_load()

        reward, info = self._compute_reward(churn_penalty, service_id, action_type, was_healthy)

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_length
        obs = self._get_obs()

        info.update(
            {
                "service_id": service_id,
                "action_type": action_type.name,
                "blast_radius": self.topology.blast_radius(service_id),
            }
        )
        self.episode_log.append(info)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Internal dynamics
    # ------------------------------------------------------------------ #
    def _apply_traffic_and_faults(self):
        # Baseline traffic fluctuation (diurnal-ish noise)
        for s in self.services:
            drift = self.rng.normal(0, 0.02)
            s.cpu_util = float(np.clip(s.cpu_util + drift, 0.05, 1.0))
            s.mem_util = float(np.clip(s.mem_util + self.rng.normal(0, 0.01), 0.05, 1.0))

        # Random fault injection
        self.steps_since_fault += 1
        if self.rng.random() < self.fault_config.prob_per_step:
            target = int(self.rng.integers(self.n_services))
            fault_type = self.rng.choice(self.fault_config.types)
            duration = int(self.rng.integers(*self.fault_config.duration_range))
            self.services[target].fault_active = str(fault_type)
            self.services[target].fault_ttl = duration
            self.steps_since_fault = 0

        # Apply active faults
        for s in self.services:
            if s.fault_active and s.fault_ttl > 0:
                if s.fault_active == "cpu_spike":
                    s.cpu_util = min(1.0, s.cpu_util + 0.4)
                elif s.fault_active == "latency_spike":
                    s.latency += 300
                elif s.fault_active == "crash_loop":
                    s.error_rate = min(1.0, s.error_rate + 0.5)
                elif s.fault_active == "mem_leak":
                    s.mem_util = min(1.0, s.mem_util + 0.05)
                s.fault_ttl -= 1
                if s.fault_ttl <= 0:
                    s.fault_active = None
            else:
                # natural decay back toward baseline when no fault
                s.latency = max(20.0, s.latency * 0.9 + 50.0 * 0.1)
                s.error_rate = max(0.0, s.error_rate * 0.85)

    def _apply_action(self, service_id: int, action_type: ActionType) -> float:
        """Apply the chosen action; return a small 'churn' penalty for costly actions."""
        s = self.services[service_id]
        churn_penalty = 0.0

        if action_type == ActionType.NOOP:
            pass
        elif action_type == ActionType.SCALE_UP:
            s.replicas = min(MAX_REPLICAS, s.replicas + 1)
            s.cpu_util = max(0.05, s.cpu_util * 0.7)
            churn_penalty = 0.02
        elif action_type == ActionType.SCALE_DOWN:
            s.replicas = max(1, s.replicas - 1)
            s.cpu_util = min(1.0, s.cpu_util * 1.2)
            churn_penalty = 0.01
        elif action_type == ActionType.RESTART:
            s.error_rate = max(0.0, s.error_rate * 0.2)
            s.latency = max(20.0, s.latency * 0.6)
            churn_penalty = 0.05
        elif action_type == ActionType.MIGRATE:
            # Migration briefly increases latency/error (cold start) but clears faults.
            s.fault_active = None
            s.fault_ttl = 0
            s.latency += 100
            s.error_rate = min(1.0, s.error_rate + 0.1)
            churn_penalty = 0.15  # migration is the most disruptive, costly action

        self.last_action_service = service_id
        return churn_penalty

    def _propagate_load(self):
        """A struggling service's callers feel some pressure too (cascading effect)."""
        for s_id, s in enumerate(self.services):
            if s.error_rate > 0.3 or s.latency > 250:
                for caller_id in self.topology.dependents(s_id):
                    caller = self.services[caller_id]
                    caller.latency = min(2000.0, caller.latency + 20)
                    caller.error_rate = min(1.0, caller.error_rate + 0.03)

    def _compute_reward(
        self,
        churn_penalty: float,
        acted_service: int,
        action_type: ActionType,
        was_healthy: bool,
    ):
        sla_violations = sum(1 for s in self.services if s.latency > 200 or s.error_rate > 0.2)
        avg_util = np.mean([s.cpu_util for s in self.services])
        resource_cost = np.mean([s.replicas for s in self.services]) / MAX_REPLICAS

        availability = 1.0 - (sla_violations / self.n_services)
        reward = (
            2.0 * availability
            - 1.0 * (sla_violations / self.n_services)
            - 0.3 * resource_cost
            - churn_penalty
        )

        bad_action = False
        blast_radius = self.topology.blast_radius(acted_service)
        if action_type != ActionType.NOOP:
            if was_healthy and action_type in (ActionType.MIGRATE, ActionType.RESTART):
                # Acted aggressively on something that wasn't actually broken.
                reward -= 0.5 * (1.0 + blast_radius)  # costlier if it was a high-impact service
                bad_action = True

        info = {
            "availability": availability,
            "sla_violations": sla_violations,
            "avg_cpu_util": float(avg_util),
            "resource_cost": float(resource_cost),
            "bad_action": bad_action,
            "blast_radius_of_bad_action": blast_radius if bad_action else 0.0,
        }
        return float(reward), info

    def _get_obs(self) -> np.ndarray:
        feats = []
        for s in self.services:
            feats.extend(
                [
                    s.cpu_util,
                    s.mem_util,
                    min(1.0, s.latency / 1000.0),
                    s.error_rate,
                    s.replicas / MAX_REPLICAS,
                ]
            )
        feats.append(min(1.0, self.steps_since_fault / 50.0))
        return np.array(feats, dtype=np.float32)

    def render(self):
        for i, s in enumerate(self.services):
            flag = f" FAULT:{s.fault_active}" if s.fault_active else ""
            print(
                f"svc_{i:02d} cpu={s.cpu_util:.2f} mem={s.mem_util:.2f} "
                f"lat={s.latency:6.1f} err={s.error_rate:.2f} replicas={s.replicas}{flag}"
            )


if __name__ == "__main__":
    env = MicroserviceEnv(n_services=6, seed=0)
    obs, _ = env.reset()
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"reward={r:.3f} info={info}")
