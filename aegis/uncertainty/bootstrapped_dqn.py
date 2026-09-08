"""
bootstrapped_dqn.py — Phase 3 deliverable: an ensemble/bootstrap-head DQN
that produces both an action and an epistemic-uncertainty estimate for
that action, on top of MicroserviceEnv.

This is a self-contained training loop rather than an SB3 wrapper (unlike
train_learned.py's DQN/PPO), because SB3 doesn't expose the internals
needed for per-head bootstrap masking and cross-head disagreement scoring.

Phase 4 will consume `.uncertainty(obs)` — together with
ServiceTopology.blast_radius — to decide when to override this agent's
action with a conservative fallback. This file only produces the estimate
that gate will act on; it does not implement the gating itself, and
`.act()` alone (no gating) is meant to be benchmarked against DQN/PPO
un-gated, so any later improvement from AEGIS can be attributed to the
gate rather than to a different underlying policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from aegis.uncertainty.ensemble_qnet import EnsembleQNetwork
from aegis.uncertainty.replay_buffer import BootstrapReplayBuffer


@dataclass
class UncertaintyEstimate:
    action: int                     # greedy action w.r.t. mean-Q across heads
    q_mean: np.ndarray              # (n_actions,) mean Q across heads
    q_std_per_action: np.ndarray    # (n_actions,) std across heads, per action
    vote_disagreement: float        # fraction of heads whose greedy action != majority action
    score: float                    # single scalar in [0, 1] combining the two signals above


class BootstrappedDQN:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_heads: int = 8,
        mask_prob: float = 0.8,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_interval: int = 500,
        seed: int | None = None,
        device: str | None = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q_net = EnsembleQNetwork(obs_dim, n_actions, n_heads=n_heads).to(self.device)
        self.target_net = EnsembleQNetwork(obs_dim, n_actions, n_heads=n_heads).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = BootstrapReplayBuffer(
            buffer_size, obs_dim, n_heads, mask_prob=mask_prob, seed=seed
        )

        self._train_steps = 0

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _q_all_heads(self, obs: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        q = self.q_net(t)  # (K, 1, A)
        return q.squeeze(1).cpu().numpy()  # (K, A)

    def act(self, obs: np.ndarray, head: int | None = None, epsilon: float = 0.0) -> int:
        """
        head=None -> act greedily w.r.t. the mean Q across all heads (used
        at eval time, and by run_comparison.py via UncertaintyAgent).
        head=k -> act greedily w.r.t. only head k (used during training,
        for Bootstrapped-DQN-style "deep exploration": one randomly chosen
        head governs behavior for a whole episode, so exploration follows
        genuine model disagreement rather than pure noise).
        """
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        q = self._q_all_heads(obs)  # (K, A)
        q_for_action = q[head] if head is not None else q.mean(axis=0)
        return int(np.argmax(q_for_action))

    def uncertainty(self, obs: np.ndarray) -> UncertaintyEstimate:
        """
        The core Phase 3 output: "how much do my K heads disagree about
        what to do here?" Two complementary signals, combined into one
        score in [0, 1]:
          - q_std: how much the heads' Q-value *for the chosen action*
            spreads out (disagreement about magnitude)
          - vote_disagreement: what fraction of heads would have picked a
            *different* action entirely (disagreement about the decision)
        """
        q = self._q_all_heads(obs)  # (K, A)
        q_mean = q.mean(axis=0)
        q_std_per_action = q.std(axis=0)

        action = int(np.argmax(q_mean))
        greedy_per_head = np.argmax(q, axis=1)  # (K,)
        vote_disagreement = float(np.mean(greedy_per_head != action))

        # Normalize q_std for the chosen action against the spread of
        # mean-Q across actions, so the score is roughly scale-free
        # instead of tracking this env's particular reward magnitude.
        q_range = float(q_mean.max() - q_mean.min()) + 1e-6
        norm_q_std = float(np.clip(q_std_per_action[action] / q_range, 0.0, 1.0))

        score = float(np.clip(0.5 * norm_q_std + 0.5 * vote_disagreement, 0.0, 1.0))

        return UncertaintyEstimate(
            action=action,
            q_mean=q_mean,
            q_std_per_action=q_std_per_action,
            vote_disagreement=vote_disagreement,
            score=score,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones, masks = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)
        masks_t = torch.as_tensor(masks, device=self.device)  # (B, K)

        q_all = self.q_net(obs_t)  # (K, B, A)
        with torch.no_grad():
            next_q_all = self.target_net(next_obs_t)  # (K, B, A)
            next_q_max = next_q_all.max(dim=2).values  # (K, B)
            target = rewards_t.unsqueeze(0) + self.gamma * (1.0 - dones_t.unsqueeze(0)) * next_q_max

        idx = actions_t.view(1, -1, 1).expand(self.n_heads, -1, 1)
        q_taken = q_all.gather(2, idx).squeeze(2)  # (K, B)

        td_error_sq = (q_taken - target) ** 2  # (K, B)
        masked_loss = (td_error_sq * masks_t.T).sum() / masks_t.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        masked_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(masked_loss.item())

    def learn(
        self,
        env,
        total_timesteps: int,
        learning_starts: int = 1_000,
        train_freq: int = 4,
        exploration_fraction: float = 0.3,
        exploration_final_eps: float = 0.05,
        log_interval: int = 2_000,
    ):
        obs, _ = env.reset()
        current_head = int(self.rng.integers(self.n_heads))
        losses: list[float] = []
        episode_rewards: list[float] = []
        ep_reward = 0.0

        for t in range(total_timesteps):
            frac = min(1.0, t / max(1, int(exploration_fraction * total_timesteps)))
            epsilon = 1.0 + frac * (exploration_final_eps - 1.0)

            action = self.act(obs, head=current_head, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            self.buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

            if t >= learning_starts and t % train_freq == 0:
                loss = self._train_step()
                if loss is not None:
                    losses.append(loss)

            if done:
                episode_rewards.append(ep_reward)
                obs, _ = env.reset()
                current_head = int(self.rng.integers(self.n_heads))  # new episode, new "personality"
                ep_reward = 0.0

            if (t + 1) % log_interval == 0:
                avg_loss = float(np.mean(losses[-200:])) if losses else float("nan")
                avg_ep_reward = float(np.mean(episode_rewards[-20:])) if episode_rewards else float("nan")
                print(
                    f"step={t + 1}/{total_timesteps} eps={epsilon:.3f} "
                    f"recent_loss={avg_loss:.4f} recent_ep_reward={avg_ep_reward:+.3f}"
                )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        torch.save(
            {
                # Cast to plain python int: obs_dim/n_actions can arrive as
                # numpy scalars (e.g. from env.action_space.n on some
                # gymnasium/numpy versions), which newer torch's default
                # weights_only=True loader refuses to unpickle.
                "obs_dim": int(self.obs_dim),
                "n_actions": int(self.n_actions),
                "n_heads": int(self.n_heads),
                "state_dict": self.q_net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "BootstrappedDQN":
        # weights_only=False: this checkpoint is our own format (a dict of
        # plain ints + a state_dict), produced by .save() above, not an
        # arbitrary/untrusted file, so the extra pickle safety isn't needed.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            obs_dim=ckpt["obs_dim"],
            n_actions=ckpt["n_actions"],
            n_heads=ckpt["n_heads"],
            device=device,
        )
        agent.q_net.load_state_dict(ckpt["state_dict"])
        agent.target_net.load_state_dict(ckpt["state_dict"])
        agent.q_net.eval()
        return agent
