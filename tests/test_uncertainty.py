import numpy as np
import torch

from aegis.env.microservice_env import MicroserviceEnv
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN, UncertaintyEstimate
from aegis.uncertainty.ensemble_qnet import EnsembleQNetwork
from aegis.uncertainty.replay_buffer import BootstrapReplayBuffer


# --------------------------------------------------------------------- #
# EnsembleQNetwork
# --------------------------------------------------------------------- #
def test_ensemble_qnet_output_shape():
    net = EnsembleQNetwork(obs_dim=10, n_actions=4, n_heads=5)
    obs = torch.randn(3, 10)
    q = net(obs)
    assert q.shape == (5, 3, 4)


def test_ensemble_qnet_heads_are_decorrelated_at_init():
    # Different heads should not produce identical outputs -- if they did,
    # bootstrapping would be pointless and uncertainty would always read 0.
    net = EnsembleQNetwork(obs_dim=8, n_actions=4, n_heads=8)
    obs = torch.randn(1, 8)
    q = net(obs).squeeze(1)  # (K, A)
    assert not torch.allclose(q[0], q[1])


# --------------------------------------------------------------------- #
# BootstrapReplayBuffer
# --------------------------------------------------------------------- #
def test_replay_buffer_add_and_sample_shapes():
    buf = BootstrapReplayBuffer(capacity=100, obs_dim=6, n_heads=4, mask_prob=0.8, seed=0)
    for i in range(50):
        buf.add(
            obs=np.full(6, float(i)),
            action=i % 3,
            reward=1.0,
            next_obs=np.full(6, float(i + 1)),
            done=False,
        )
    assert len(buf) == 50

    obs, actions, rewards, next_obs, dones, masks = buf.sample(16)
    assert obs.shape == (16, 6)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_obs.shape == (16, 6)
    assert dones.shape == (16,)
    assert masks.shape == (16, 4)
    assert set(np.unique(masks)).issubset({0.0, 1.0})


def test_replay_buffer_wraps_around_capacity():
    buf = BootstrapReplayBuffer(capacity=10, obs_dim=2, n_heads=2, seed=0)
    for i in range(25):
        buf.add(np.zeros(2), 0, 0.0, np.zeros(2), False)
    assert len(buf) == 10  # capped, not 25


def test_replay_buffer_mask_prob_roughly_respected():
    # With mask_prob=1.0 every head should see every transition.
    buf = BootstrapReplayBuffer(capacity=200, obs_dim=2, n_heads=4, mask_prob=1.0, seed=0)
    for _ in range(200):
        buf.add(np.zeros(2), 0, 0.0, np.zeros(2), False)
    assert buf.masks.mean() == 1.0

    buf0 = BootstrapReplayBuffer(capacity=200, obs_dim=2, n_heads=4, mask_prob=0.0, seed=0)
    for _ in range(200):
        buf0.add(np.zeros(2), 0, 0.0, np.zeros(2), False)
    assert buf0.masks.mean() == 0.0


# --------------------------------------------------------------------- #
# BootstrappedDQN
# --------------------------------------------------------------------- #
def _make_agent(seed=0, n_heads=6):
    return BootstrappedDQN(obs_dim=9, n_actions=5, n_heads=n_heads, seed=seed, device="cpu")


def test_bootstrapped_dqn_act_returns_valid_action():
    agent = _make_agent()
    obs = np.random.default_rng(0).normal(size=9).astype(np.float32)
    a = agent.act(obs)
    assert isinstance(a, int)
    assert 0 <= a < 5


def test_bootstrapped_dqn_act_epsilon_greedy_explores():
    agent = _make_agent(seed=1)
    obs = np.zeros(9, dtype=np.float32)
    actions = {agent.act(obs, epsilon=1.0) for _ in range(50)}
    # epsilon=1.0 -> pure random -> should see more than one action across 50 draws
    assert len(actions) > 1


def test_bootstrapped_dqn_act_per_head_selection():
    agent = _make_agent(seed=2, n_heads=6)
    obs = np.random.default_rng(2).normal(size=9).astype(np.float32)
    for head in range(6):
        a = agent.act(obs, head=head)
        assert 0 <= a < 5


def test_uncertainty_estimate_shape_and_range():
    agent = _make_agent(seed=3, n_heads=6)
    obs = np.random.default_rng(3).normal(size=9).astype(np.float32)
    est = agent.uncertainty(obs)
    assert isinstance(est, UncertaintyEstimate)
    assert est.q_mean.shape == (5,)
    assert est.q_std_per_action.shape == (5,)
    assert 0.0 <= est.vote_disagreement <= 1.0
    assert 0.0 <= est.score <= 1.0
    assert 0 <= est.action < 5


def test_uncertainty_score_is_zero_for_identical_heads():
    # Force all heads to agree perfectly by zeroing out all but one head's
    # weights' contribution -- simpler: monkeypatch _q_all_heads to return
    # K identical rows and check the score collapses to 0.
    agent = _make_agent(seed=4, n_heads=4)
    identical = np.tile(np.array([[1.0, 2.0, 0.5, -1.0, 3.0]]), (4, 1))
    agent._q_all_heads = lambda obs: identical  # type: ignore[method-assign]
    est = agent.uncertainty(np.zeros(9, dtype=np.float32))
    assert est.vote_disagreement == 0.0
    assert est.score == 0.0
    assert est.action == 4  # argmax of [1, 2, 0.5, -1, 3]


def test_uncertainty_score_is_high_for_maximally_disagreeing_heads():
    agent = _make_agent(seed=5, n_heads=4)
    # Each head strongly prefers a different action -> high vote disagreement.
    disagreeing = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
        ]
    )
    agent._q_all_heads = lambda obs: disagreeing  # type: ignore[method-assign]
    est = agent.uncertainty(np.zeros(9, dtype=np.float32))
    assert est.vote_disagreement >= 0.5
    assert est.score > 0.3


def test_bootstrapped_dqn_learns_on_env_without_crashing():
    # Short smoke run against the real env -- not enough steps to converge,
    # just confirms the training loop (buffer -> loss -> target update) runs
    # end to end without shape errors.
    env = MicroserviceEnv(n_services=4, episode_length=20, seed=0)
    agent = BootstrappedDQN(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        n_heads=3,
        seed=0,
        device="cpu",
    )
    agent.learn(env, total_timesteps=200, learning_starts=32, train_freq=4, log_interval=1_000_000)
    assert agent._train_steps > 0

    obs, _ = env.reset(seed=0)
    est = agent.uncertainty(obs)
    assert 0.0 <= est.score <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    agent = _make_agent(seed=6, n_heads=3)
    obs = np.random.default_rng(6).normal(size=9).astype(np.float32)
    action_before = agent.act(obs)

    path = tmp_path / "agent.pt"
    agent.save(path)
    loaded = BootstrappedDQN.load(path, device="cpu")

    assert loaded.obs_dim == agent.obs_dim
    assert loaded.n_actions == agent.n_actions
    assert loaded.n_heads == agent.n_heads
    assert loaded.act(obs) == action_before
