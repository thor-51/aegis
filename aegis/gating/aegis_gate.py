"""
aegis_gate.py — Phase 4 deliverable: the confidence + blast-radius gate.
This is "AEGIS proper" -- everything before this phase (the uncertainty
module, blast-radius scoring) was building the two signals this file
combines into a single act-or-fallback decision.

Gate rule (see README "Core idea"):
    override the learned policy's action iff
        epistemic_uncertainty(state) > confidence_threshold
        AND
        blast_radius(candidate action's service) > blast_radius_threshold

Both conditions are required by design, not just uncertainty alone: a
low-blast-radius service is exactly the case where it's fine to let the
policy improvise even if it's unsure, because a wrong guess is cheap.
Gating on confidence alone (`use_blast_radius=False`) is kept as a toggle
for the "AEGIS-no-graph" ablation the README earmarks for Phase 6 --
Phase 4 does not draw conclusions from that mode, it just makes the
comparison possible later without new code.

Fallback policy: Phase 5 owns building a purpose-built, blast-radius-aware
conservative fallback. Until then this gate accepts *any* object with an
.act(obs) method as its fallback -- run_comparison.py currently plugs in
the Phase 1 RuleBasedAgent, since it's the only agent that already
achieves bad_action_rate=0.000 in the Phase 2/3 results table, making it a
reasonable interim "pre-approved recovery action" while Phase 5 is being
built. This is a placeholder, not a design claim -- see agents/aegis_agent.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aegis.env.microservice_env import N_ACTION_TYPES
from aegis.env.topology import ServiceTopology
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN, UncertaintyEstimate

# Defaults chosen by calibration (see calibrate_confidence_threshold below),
# not hand-tuned to the eval seeds: confidence_threshold is the ~85th
# percentile of uncertainty scores observed over 20 held-out episodes
# (seeds 20-39, disjoint from the eval seeds 0-19 used everywhere else in
# this repo) with the Phase 3 bootstrap_dqn checkpoint. blast_radius_threshold
# is a semantic choice, not a fitted one: 0.5 means "more than half of the
# other services in the cluster transitively depend on this one." Together
# these gave a ~5% override rate on the calibration episodes -- intervening
# only on the states that are both genuinely uncertain and genuinely
# high-stakes, not on every mildly-ambiguous state.
DEFAULT_CONFIDENCE_THRESHOLD = 0.42
DEFAULT_BLAST_RADIUS_THRESHOLD = 0.5


@dataclass
class GateDecision:
    action: int
    overridden: bool
    uncertainty: UncertaintyEstimate
    blast_radius: float


class AEGISGate:
    """
    Wraps a BootstrappedDQN policy + a fallback agent + a ServiceTopology
    into a single confidence/blast-radius-gated decision rule. Call
    .decide(obs) for the full breakdown (used by tests and diagnostics) or
    .act(obs) for just the action (matches every other agent's interface).
    """

    def __init__(
        self,
        policy: BootstrappedDQN,
        fallback,
        topology: ServiceTopology,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        blast_radius_threshold: float = DEFAULT_BLAST_RADIUS_THRESHOLD,
        use_blast_radius: bool = True,
    ):
        self.policy = policy
        self.fallback = fallback
        self.topology = topology
        self.confidence_threshold = confidence_threshold
        self.blast_radius_threshold = blast_radius_threshold
        self.use_blast_radius = use_blast_radius

        self.n_calls = 0
        self.n_overrides = 0

    def decide(self, obs: np.ndarray) -> GateDecision:
        est = self.policy.uncertainty(obs)
        candidate_service = est.action // N_ACTION_TYPES
        br = self.topology.blast_radius(candidate_service)

        low_confidence = est.score > self.confidence_threshold
        high_stakes = (not self.use_blast_radius) or (br > self.blast_radius_threshold)
        overridden = bool(low_confidence and high_stakes)

        self.n_calls += 1
        if overridden:
            self.n_overrides += 1
            action = self.fallback.act(obs)
        else:
            action = est.action

        return GateDecision(action=action, overridden=overridden, uncertainty=est, blast_radius=br)

    def act(self, obs: np.ndarray) -> int:
        return self.decide(obs).action

    @property
    def override_rate(self) -> float:
        return self.n_overrides / self.n_calls if self.n_calls else 0.0


def calibrate_confidence_threshold(
    policy: BootstrappedDQN,
    env_factory,
    seeds,
    percentile: float = 85.0,
) -> float:
    """
    Run `policy` (ungated) over the given episodes and return the given
    percentile of its observed uncertainty scores. Meant to be run on
    seeds disjoint from whatever seeds are later used for evaluation, so
    the threshold reflects "what does low-confidence look like for this
    policy in general" rather than being fit to the test set.
    """
    scores: list[float] = []
    for seed in seeds:
        env = env_factory(seed)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            est = policy.uncertainty(obs)
            scores.append(est.score)
            obs, _reward, terminated, truncated, _info = env.step(est.action)
            done = terminated or truncated
    return float(np.percentile(scores, percentile))
