"""
Tests for ConservativeFallback — Phase 5's blast-radius-aware fallback agent.
"""

import numpy as np
import pytest

from aegis.agents.conservative_fallback import ConservativeFallback
from aegis.env.microservice_env import (
    ActionType,
    MicroserviceEnv,
    N_ACTION_TYPES,
    N_FEATURES_PER_SERVICE,
)
from aegis.env.topology import ServiceTopology


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #
def _topology_with_fixed_blast_radius(br: float, n_services: int = 4):
    """Return a topology where every service has the same blast radius."""
    topo = ServiceTopology(n_services=n_services, seed=0)
    topo._blast_radius_cache = {i: br for i in range(n_services)}
    return topo


def _topology_with_per_service_blast_radius(br_values: list[float]):
    """Return a topology where each service has a specific blast radius."""
    n = len(br_values)
    topo = ServiceTopology(n_services=n, seed=0)
    topo._blast_radius_cache = {i: br for i, br in enumerate(br_values)}
    return topo


def _make_obs(
    n_services: int = 4,
    service_overrides: dict[int, dict] | None = None,
) -> np.ndarray:
    """Build a synthetic observation vector.

    All services start at baseline (cpu=0.3, mem=0.3, lat_norm=0.05,
    err=0.0, replicas_norm=0.2). Override individual services via
    service_overrides = {service_id: {"cpu": 0.9, "err": 0.5, ...}}.
    """
    defaults = {"cpu": 0.3, "mem": 0.3, "lat": 0.05, "err": 0.0, "rep": 0.2}
    feats = []
    for i in range(n_services):
        vals = dict(defaults)
        if service_overrides and i in service_overrides:
            vals.update(service_overrides[i])
        feats.extend([vals["cpu"], vals["mem"], vals["lat"], vals["err"], vals["rep"]])
    feats.append(0.5)  # steps_since_fault / 50
    return np.array(feats, dtype=np.float32)


def _decode_action(action: int):
    """Return (service_id, ActionType) from a flat action int."""
    service_id, action_type = divmod(action, N_ACTION_TYPES)
    return service_id, ActionType(action_type)


# -------------------------------------------------------------------- #
# High blast radius: should NEVER restart or migrate
# -------------------------------------------------------------------- #
class TestHighBlastRadius:
    def test_avoids_restart_on_high_br_with_errors(self):
        """High-BR service with high error rate → SCALE_UP, not RESTART."""
        topo = _topology_with_fixed_blast_radius(br=0.9, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4, service_overrides={2: {"err": 0.8}})
        action = agent.act(obs)
        sid, atype = _decode_action(action)

        assert sid == 2, f"Should target the erroring service, got service {sid}"
        assert atype == ActionType.SCALE_UP, (
            f"High-BR fallback should SCALE_UP, not {atype.name}"
        )

    def test_avoids_migrate_on_high_br(self):
        """High-BR service: MIGRATE should never be selected."""
        topo = _topology_with_fixed_blast_radius(br=0.9, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        # Simulate a service with every problem at once
        obs = _make_obs(
            n_services=4,
            service_overrides={1: {"cpu": 0.95, "err": 0.9, "lat": 0.5}},
        )
        action = agent.act(obs)
        _, atype = _decode_action(action)
        assert atype != ActionType.MIGRATE

    def test_scale_up_for_cpu_saturation_on_high_br(self):
        topo = _topology_with_fixed_blast_radius(br=0.8, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4, service_overrides={3: {"cpu": 0.9}})
        action = agent.act(obs)
        sid, atype = _decode_action(action)

        assert sid == 3
        assert atype == ActionType.SCALE_UP

    def test_scale_up_for_latency_on_high_br(self):
        topo = _topology_with_fixed_blast_radius(br=0.7, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4, service_overrides={0: {"lat": 0.3}})
        action = agent.act(obs)
        sid, atype = _decode_action(action)

        assert sid == 0
        assert atype == ActionType.SCALE_UP

    def test_noop_when_healthy_on_high_br(self):
        topo = _topology_with_fixed_blast_radius(br=0.9, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4)  # all healthy
        action = agent.act(obs)
        assert action == 0  # noop on service 0


# -------------------------------------------------------------------- #
# Low blast radius: allowed to restart, but never migrate
# -------------------------------------------------------------------- #
class TestLowBlastRadius:
    def test_restart_allowed_on_low_br_with_errors(self):
        """Low-BR service with high error rate → RESTART is acceptable."""
        topo = _topology_with_fixed_blast_radius(br=0.1, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4, service_overrides={2: {"err": 0.8}})
        action = agent.act(obs)
        sid, atype = _decode_action(action)

        assert sid == 2
        assert atype == ActionType.RESTART

    def test_avoids_migrate_on_low_br(self):
        """Even low-BR services should never get MIGRATE from the fallback."""
        topo = _topology_with_fixed_blast_radius(br=0.1, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(
            n_services=4,
            service_overrides={1: {"cpu": 0.95, "err": 0.9, "lat": 0.5}},
        )
        action = agent.act(obs)
        _, atype = _decode_action(action)
        assert atype != ActionType.MIGRATE

    def test_scale_up_for_cpu_on_low_br(self):
        topo = _topology_with_fixed_blast_radius(br=0.2, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4, service_overrides={0: {"cpu": 0.85}})
        action = agent.act(obs)
        sid, atype = _decode_action(action)

        assert sid == 0
        assert atype == ActionType.SCALE_UP

    def test_noop_when_healthy_on_low_br(self):
        topo = _topology_with_fixed_blast_radius(br=0.1, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4)
        action = agent.act(obs)
        assert action == 0


# -------------------------------------------------------------------- #
# Mixed topology: services have different blast radii
# -------------------------------------------------------------------- #
class TestMixedTopology:
    def test_targets_most_urgent_service(self):
        """With multiple issues, targets the one with highest urgency score."""
        # Service 0: low BR, mild CPU issue
        # Service 1: high BR, severe error
        topo = _topology_with_per_service_blast_radius([0.1, 0.9, 0.2, 0.3])
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(
            n_services=4,
            service_overrides={
                0: {"cpu": 0.8},
                1: {"err": 0.7},
            },
        )
        action = agent.act(obs)
        sid, atype = _decode_action(action)

        # Service 1 has higher urgency (err=0.7 * 2 = 1.4 vs cpu=0.8)
        assert sid == 1
        # But it's high-BR, so should get SCALE_UP, not RESTART
        assert atype == ActionType.SCALE_UP

    def test_low_br_service_gets_restart_for_errors(self):
        """When the most urgent service is low-BR, errors → restart."""
        topo = _topology_with_per_service_blast_radius([0.1, 0.1, 0.9, 0.1])
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(
            n_services=4,
            service_overrides={1: {"err": 0.5}},
        )
        action = agent.act(obs)
        sid, atype = _decode_action(action)

        assert sid == 1
        assert atype == ActionType.RESTART


# -------------------------------------------------------------------- #
# Threshold edge cases
# -------------------------------------------------------------------- #
class TestThresholds:
    def test_blast_radius_exactly_at_threshold_is_low(self):
        """BR == 0.5 should be treated as LOW (threshold is strict >)."""
        topo = _topology_with_fixed_blast_radius(br=0.5, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4, service_overrides={0: {"err": 0.8}})
        action = agent.act(obs)
        _, atype = _decode_action(action)

        # BR == threshold → low path → restart is allowed
        assert atype == ActionType.RESTART

    def test_error_exactly_at_threshold_is_not_triggered(self):
        """Error rate == 0.3 should NOT trigger restart (threshold is strict >)."""
        topo = _topology_with_fixed_blast_radius(br=0.1, n_services=4)
        agent = ConservativeFallback(n_services=4, topology=topo)

        obs = _make_obs(n_services=4, service_overrides={0: {"err": 0.3}})
        action = agent.act(obs)
        assert action == 0  # noop — error rate not above threshold

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        topo = _topology_with_fixed_blast_radius(br=0.3, n_services=4)
        agent = ConservativeFallback(
            n_services=4,
            topology=topo,
            blast_radius_threshold=0.2,  # 0.3 > 0.2 → high-BR path
            error_restart_threshold=0.1,
        )

        obs = _make_obs(n_services=4, service_overrides={0: {"err": 0.5}})
        action = agent.act(obs)
        _, atype = _decode_action(action)

        # BR=0.3 > threshold=0.2, so high-BR path → SCALE_UP, not RESTART
        assert atype == ActionType.SCALE_UP


# -------------------------------------------------------------------- #
# Integration: full episode
# -------------------------------------------------------------------- #
class TestIntegration:
    def test_runs_full_episode_without_crashing(self):
        env = MicroserviceEnv(n_services=4, episode_length=50, seed=42)
        agent = ConservativeFallback(n_services=4, topology=env.topology)

        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            action = agent.act(obs)
            assert 0 <= action < env.action_space.n
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == 50

    def test_never_produces_migrate_action_over_full_episode(self):
        """Over an entire episode, ConservativeFallback should never MIGRATE."""
        env = MicroserviceEnv(n_services=8, episode_length=200, seed=7)
        agent = ConservativeFallback(n_services=8, topology=env.topology)

        obs, _ = env.reset(seed=7)
        done = False
        while not done:
            action = agent.act(obs)
            _, atype = _decode_action(action)
            assert atype != ActionType.MIGRATE, "ConservativeFallback should never MIGRATE"
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    def test_never_restarts_high_br_service_over_full_episode(self):
        """Over an entire episode, high-BR services should never get RESTART."""
        env = MicroserviceEnv(n_services=8, episode_length=200, seed=13)
        agent = ConservativeFallback(n_services=8, topology=env.topology)

        obs, _ = env.reset(seed=13)
        done = False
        while not done:
            action = agent.act(obs)
            sid, atype = _decode_action(action)
            br = env.topology.blast_radius(sid)
            if br > 0.5:
                assert atype != ActionType.RESTART, (
                    f"Restarted service {sid} with blast_radius={br:.2f}"
                )
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
