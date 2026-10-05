import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from aegis.env.k8s_env import K8sMicroserviceEnv
from aegis.k8s.actuator import K8sActuator, ActionType
from aegis.k8s.observer import K8sObserver
from aegis.k8s.chaos_controller import ChaosController


def test_actuator_noop():
    actuator = K8sActuator(service_names=["service-auth", "service-db"])
    res = actuator.execute_action(0, ActionType.NOOP)
    assert res["status"] == "ok"
    assert res["action"] == "NOOP"


def test_actuator_mock_execution():
    actuator = K8sActuator(service_names=["service-auth", "service-db"])
    res = actuator.execute_action(1, ActionType.SCALE_UP)
    assert res["action"] == "SCALE_UP"


def test_observer_dimensions():
    services = ["service-auth", "service-db", "service-api"]
    observer = K8sObserver(service_names=services)
    obs = observer.get_full_observation()
    # 3 services * 5 features + 1 time feature = 16
    assert obs.shape == (16,)
    assert obs.dtype == np.float32


def test_k8s_env_step():
    services = ["service-auth", "service-db", "service-api"]
    env = K8sMicroserviceEnv(service_names=services, step_duration_sec=0.0, max_steps_per_episode=2)
    obs, info = env.reset()
    assert obs.shape == (16,)

    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (16,)
    assert "is_bad_action" in info
    assert "target_blast_radius" in info
    assert not terminated

    obs, reward, terminated, truncated, info = env.step(1)
    assert terminated
