import numpy as np

from aegis.agents.aegis_agent import AEGISAgent
from aegis.agents.rule_based import RuleBasedAgent
from aegis.env.microservice_env import MicroserviceEnv, N_ACTION_TYPES
from aegis.env.topology import ServiceTopology
from aegis.gating.aegis_gate import AEGISGate, calibrate_confidence_threshold
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


class _StubFallback:
    """Deterministic fallback so tests can assert exactly which action came from where."""

    def act(self, obs):
        return 999


def _agent_with_fixed_uncertainty(score: float, action: int, n_actions: int = 10):
    """A BootstrappedDQN-shaped stub whose .uncertainty() always returns a
    fixed score/action, so gate behavior can be tested without depending on
    what a real trained network happens to output."""
    from aegis.uncertainty.bootstrapped_dqn import UncertaintyEstimate

    class _StubPolicy:
        def uncertainty(self, obs):
            return UncertaintyEstimate(
                action=action,
                q_mean=np.zeros(n_actions),
                q_std_per_action=np.zeros(n_actions),
                vote_disagreement=score,
                score=score,
            )

    return _StubPolicy()


def _topology_with_fixed_blast_radius(br: float, n_services: int = 4):
    topo = ServiceTopology(n_services=n_services, seed=0)
    topo._blast_radius_cache = {i: br for i in range(n_services)}
    return topo


# --------------------------------------------------------------------- #
# Core gate decision rule
# --------------------------------------------------------------------- #
def test_gate_does_not_override_when_confident():
    policy = _agent_with_fixed_uncertainty(score=0.1, action=3)  # service 0, action_type 3
    topo = _topology_with_fixed_blast_radius(br=0.9)  # high stakes, but confident
    gate = AEGISGate(policy, _StubFallback(), topo, confidence_threshold=0.4, blast_radius_threshold=0.5)

    decision = gate.decide(np.zeros(1))
    assert decision.overridden is False
    assert decision.action == 3


def test_gate_does_not_override_when_low_blast_radius():
    policy = _agent_with_fixed_uncertainty(score=0.9, action=3)  # very uncertain
    topo = _topology_with_fixed_blast_radius(br=0.1)  # but low stakes
    gate = AEGISGate(policy, _StubFallback(), topo, confidence_threshold=0.4, blast_radius_threshold=0.5)

    decision = gate.decide(np.zeros(1))
    assert decision.overridden is False
    assert decision.action == 3


def test_gate_overrides_when_uncertain_and_high_stakes():
    policy = _agent_with_fixed_uncertainty(score=0.9, action=3)
    topo = _topology_with_fixed_blast_radius(br=0.9)
    gate = AEGISGate(policy, _StubFallback(), topo, confidence_threshold=0.4, blast_radius_threshold=0.5)

    decision = gate.decide(np.zeros(1))
    assert decision.overridden is True
    assert decision.action == 999  # came from the fallback, not the policy


def test_gate_thresholds_are_strict_inequalities():
    # Exactly at the threshold should NOT trigger an override (> not >=).
    policy = _agent_with_fixed_uncertainty(score=0.4, action=3)
    topo = _topology_with_fixed_blast_radius(br=0.5)
    gate = AEGISGate(policy, _StubFallback(), topo, confidence_threshold=0.4, blast_radius_threshold=0.5)

    decision = gate.decide(np.zeros(1))
    assert decision.overridden is False


# --------------------------------------------------------------------- #
# Ablation toggle (Phase 6's "AEGIS-no-graph" needs this to already work)
# --------------------------------------------------------------------- #
def test_gate_with_blast_radius_disabled_ignores_topology():
    policy = _agent_with_fixed_uncertainty(score=0.9, action=3)
    topo = _topology_with_fixed_blast_radius(br=0.0)  # would normally block override
    gate = AEGISGate(
        policy, _StubFallback(), topo,
        confidence_threshold=0.4, blast_radius_threshold=0.5, use_blast_radius=False,
    )

    decision = gate.decide(np.zeros(1))
    assert decision.overridden is True  # confidence alone was enough


# --------------------------------------------------------------------- #
# override_rate bookkeeping
# --------------------------------------------------------------------- #
def test_override_rate_tracks_calls():
    policy = _agent_with_fixed_uncertainty(score=0.9, action=3)
    topo = _topology_with_fixed_blast_radius(br=0.9)
    gate = AEGISGate(policy, _StubFallback(), topo, confidence_threshold=0.4, blast_radius_threshold=0.5)

    assert gate.override_rate == 0.0  # no calls yet
    for _ in range(5):
        gate.act(np.zeros(1))
    assert gate.n_calls == 5
    assert gate.override_rate == 1.0  # every call was uncertain + high-stakes


def test_override_rate_mixed():
    calls = {"n": 0}

    class _AlternatingPolicy:
        def uncertainty(self, obs):
            from aegis.uncertainty.bootstrapped_dqn import UncertaintyEstimate

            calls["n"] += 1
            score = 0.9 if calls["n"] % 2 == 0 else 0.1  # alternate confident/uncertain
            return UncertaintyEstimate(
                action=0, q_mean=np.zeros(5), q_std_per_action=np.zeros(5),
                vote_disagreement=score, score=score,
            )

    topo = _topology_with_fixed_blast_radius(br=0.9)
    gate = AEGISGate(_AlternatingPolicy(), _StubFallback(), topo, confidence_threshold=0.4, blast_radius_threshold=0.5)
    for _ in range(10):
        gate.act(np.zeros(1))
    assert gate.override_rate == 0.5


# --------------------------------------------------------------------- #
# calibrate_confidence_threshold
# --------------------------------------------------------------------- #
def test_calibrate_confidence_threshold_returns_percentile_of_observed_scores():
    n_services = 4
    obs_dim = n_services * 5 + 1
    agent = BootstrappedDQN(obs_dim=obs_dim, n_actions=n_services * N_ACTION_TYPES, n_heads=4, seed=0, device="cpu")

    def env_factory(seed):
        return MicroserviceEnv(n_services=n_services, episode_length=10, seed=seed)

    tau = calibrate_confidence_threshold(agent, env_factory, seeds=[0, 1], percentile=50.0)
    assert 0.0 <= tau <= 1.0


# --------------------------------------------------------------------- #
# AEGISAgent (the composed, .act(obs)-compatible version)
# --------------------------------------------------------------------- #
def test_aegis_agent_defaults_to_rule_based_fallback():
    topo = ServiceTopology(n_services=4, seed=0)
    policy = _agent_with_fixed_uncertainty(score=0.99, action=3, n_actions=20)
    agent = AEGISAgent(policy=policy, topology=topo, confidence_threshold=0.1, blast_radius_threshold=-1.0)
    assert isinstance(agent.fallback, RuleBasedAgent)


def test_aegis_agent_runs_full_episode_without_crashing():
    env = MicroserviceEnv(n_services=4, episode_length=15, seed=0)
    real_policy = BootstrappedDQN(
        obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n,
        n_heads=3, seed=0, device="cpu",
    )
    agent = AEGISAgent(policy=real_policy, topology=env.topology)

    obs, _ = env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        action = agent.act(obs)
        assert 0 <= action < env.action_space.n
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps == 15
    assert 0.0 <= agent.override_rate <= 1.0
