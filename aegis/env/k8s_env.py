"""
aegis/env/k8s_env.py — Gymnasium environment for real Kubernetes cluster orchestration.
Plugs into the identical agent API as MicroserviceEnv.
"""

from __future__ import annotations

import time
from typing import Sequence
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aegis.env.microservice_env import ActionType, N_ACTION_TYPES, N_FEATURES_PER_SERVICE
from aegis.env.topology import ServiceTopology
from aegis.k8s.actuator import K8sActuator
from aegis.k8s.observer import K8sObserver
from aegis.k8s.chaos_controller import ChaosController


class K8sMicroserviceEnv(gym.Env):
    """
    Gymnasium environment backed by a real Kubernetes cluster, Prometheus, and Chaos Mesh.
    Observes live cluster metrics, executes actions via K8s API, and tracks blast-radius metrics.
    """

    def __init__(
        self,
        service_names: Sequence[str] = (
            "service-auth",
            "service-db",
            "service-api",
            "service-worker",
            "service-cache",
        ),
        step_duration_sec: float = 5.0,
        max_steps_per_episode: int = 20,
        prometheus_url: str = "http://localhost:9090",
        namespace: str = "aegis-demo",
        seed: int | None = None,
    ):
        super().__init__()
        self.service_names = list(service_names)
        self.n_services = len(self.service_names)
        self.step_duration_sec = step_duration_sec
        self.max_steps = max_steps_per_episode
        self.current_step = 0

        # Topology and Blast Radius
        self.topology = ServiceTopology(self.n_services, edge_prob=0.3, seed=seed)

        # K8s components
        self.actuator = K8sActuator(self.service_names, namespace=namespace)
        self.observer = K8sObserver(self.service_names, prometheus_url=prometheus_url, namespace=namespace)
        self.chaos = ChaosController(self.service_names, namespace=namespace)

        # Observation & Action space
        obs_dim = self.n_services * N_FEATURES_PER_SERVICE + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_services * N_ACTION_TYPES)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.chaos.clear_all()

        obs = self.observer.get_full_observation(time_since_last_fault=1.0)
        info = {
            "step": self.current_step,
            "blast_radius_vector": self.topology.blast_radius_vector(),
        }
        return obs, info

    def step(self, action: int):
        self.current_step += 1

        service_id = action // N_ACTION_TYPES
        action_type = ActionType(action % N_ACTION_TYPES)

        # Execute on cluster
        exec_info = self.actuator.execute_action(service_id, action_type)

        # Wait for cluster / metric reconciliation if step_duration > 0
        if self.step_duration_sec > 0:
            time.sleep(self.step_duration_sec)

        # Scrape state
        obs = self.observer.get_full_observation()

        # Extract metrics to compute reward
        svc_obs = self.observer.get_service_observation(self.service_names[service_id])
        error_rate = float(svc_obs[3])
        latency_norm = float(svc_obs[2])

        # Bad action check: disruptive action taken while service was healthy
        is_disruptive = action_type in (ActionType.RESTART, ActionType.MIGRATE)
        is_healthy = (error_rate < 0.05) and (latency_norm < 0.4)
        is_bad_action = is_disruptive and is_healthy

        # Blast radius
        target_blast_radius = self.topology.blast_radius(service_id)

        # Simple reward formulation: availability + low latency - churn
        churn_penalty = 0.15 if is_bad_action else 0.0
        availability = 1.0 - error_rate
        reward = float(availability - 0.2 * latency_norm - churn_penalty)

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "step": self.current_step,
            "is_bad_action": is_bad_action,
            "target_blast_radius": target_blast_radius,
            "exec_info": exec_info,
            "availability": availability,
        }

        return obs, reward, terminated, truncated, info
