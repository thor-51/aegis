import numpy as np

from aegis.env.microservice_env import MicroserviceEnv
from aegis.env.topology import ServiceTopology


def test_topology_blast_radius_range():
    topo = ServiceTopology(n_services=10, seed=1)
    for i in range(10):
        br = topo.blast_radius(i)
        assert 0.0 <= br <= 1.0


def test_topology_has_a_hub_and_a_leaf():
    topo = ServiceTopology(n_services=10, seed=1)
    scores = topo.blast_radius_vector()
    assert scores.max() > scores.min()  # not flat -- some services matter more


def test_env_reset_obs_shape():
    env = MicroserviceEnv(n_services=6, seed=0)
    obs, _ = env.reset()
    assert obs.shape == (6 * 5 + 1,)
    assert env.observation_space.contains(obs)


def test_env_step_runs_full_episode():
    env = MicroserviceEnv(n_services=6, episode_length=20, seed=0)
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        assert isinstance(reward, float)
        assert "availability" in info
        assert "blast_radius" in info
    assert steps == 20


def test_bad_action_flagged_when_healthy_service_migrated():
    env = MicroserviceEnv(n_services=4, episode_length=5, seed=2)
    env.reset(seed=2)
    from aegis.env.microservice_env import ActionType, N_ACTION_TYPES

    # Force service 0 to look healthy, then migrate it -- should be flagged bad.
    env.services[0].latency = 30.0
    env.services[0].error_rate = 0.0
    env.services[0].fault_active = None
    action = 0 * N_ACTION_TYPES + ActionType.MIGRATE
    _, _, _, _, info = env.step(action)
    assert info["bad_action"] is True
