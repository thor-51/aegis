"""
test_ood.py — Unit and integration tests for Phase 6 OOD scenarios and mechanics.
"""

import numpy as np
import pytest

from aegis.env.microservice_env import FaultConfig, MicroserviceEnv
from aegis.eval_ood import SCENARIOS


def test_ood_scenarios_presence():
    """Verify that standard Phase 6 scenarios are defined."""
    expected = [
        "in_distribution",
        "high_fault_rate",
        "long_faults",
        "novel_fault",
        "simultaneous",
        "hub_targeted",
        "worst_case",
    ]
    for name in expected:
        assert name in SCENARIOS, f"Missing scenario: {name}"


def test_network_partition_fault_dynamics():
    """Verify that network_partition affects caller services' latency and error rate."""
    env = MicroserviceEnv(n_services=8, seed=42)
    env.reset()

    # Find a service with at least one caller/dependent
    target_service = None
    for s_id in range(env.n_services):
        if len(env.topology.dependents(s_id)) > 0:
            target_service = s_id
            break
    assert target_service is not None

    caller_id = next(iter(env.topology.dependents(target_service)))
    initial_caller_latency = env.services[caller_id].latency
    initial_caller_errors = env.services[caller_id].error_rate

    # Inject network partition directly
    env.services[target_service].fault_active = "network_partition"
    env.services[target_service].fault_ttl = 5

    # Run one step
    env.step(0)  # No-op action

    caller = env.services[caller_id]
    assert caller.latency >= initial_caller_latency
    assert caller.error_rate >= initial_caller_errors


def test_fault_target_bias():
    """Verify that fault_target_bias biases target selection towards high-blast-radius services."""
    fc = FaultConfig(prob_per_step=1.0, fault_target_bias=1.0)
    env = MicroserviceEnv(n_services=8, fault_config=fc, seed=123)
    env.reset()

    br = env.topology.blast_radius_vector()
    highest_br_services = np.where(br == br.max())[0]

    # Pick targets multiple times
    targets = [env._pick_fault_target() for _ in range(50)]
    # With bias=1.0, choices should disproportionately favor higher blast radius
    avg_chosen_br = float(np.mean([br[t] for t in targets]))
    assert avg_chosen_br > float(np.mean(br))


def test_simultaneous_fault_prob():
    """Verify simultaneous fault injection when simultaneous_fault_prob > 0."""
    fc = FaultConfig(
        prob_per_step=1.0,
        simultaneous_fault_prob=1.0,
        duration_range=(10, 20),
    )
    env = MicroserviceEnv(n_services=6, fault_config=fc, seed=99)
    env.reset()
    env.step(0)

    active_faults = [s for s in env.services if s.fault_active is not None]
    assert len(active_faults) > 1
