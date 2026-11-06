"""
eval_uncertainty.py — Diagnostic evaluation of the BootstrappedDQN uncertainty
signal (Reviewer Points 6 & 9):
  1. AUROC and AUPRC for detecting OOD states (in-distribution vs OOD fault scenarios).
  2. AUROC and AUPRC for predicting whether the ungated ensemble action is a bad action.
  3. Ablation of NOOP action canonicalization (canonicalized vs raw vote disagreement).
  4. Threshold sensitivity sweep over (theta_u, theta_b) trading off override rate
     against unconditional blast-weighted harm and reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aegis.agents.aegis_agent import AEGISAgent
from aegis.env.microservice_env import MicroserviceEnv
from aegis.eval_ood import SCENARIOS
from aegis.run_baselines import run_agent_suite
from aegis.uncertainty.bootstrapped_dqn import BootstrappedDQN


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U statistic) implemented in pure NumPy."""
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    # Handle tied scores by assigning average rank
    sorted_s = s[order]
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    u_stat = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (AUPRC) implemented in pure NumPy."""
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / n_pos

    # Step-wise integration over recall changes
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - recall_prev) * precision))
    return ap


def evaluate_uncertainty_discrimination(
    model: BootstrappedDQN,
    n_services: int = 8,
    episodes: int = 30,
    episode_length: int = 200,
) -> dict:
    ood_labels: list[int] = []
    bad_action_labels: list[int] = []
    canon_scores: list[float] = []
    raw_scores: list[float] = []
    canon_votes: list[float] = []
    raw_votes: list[float] = []
    q_stds: list[float] = []

    for scenario_name, fc in SCENARIOS.items():
        is_ood = 0 if scenario_name == "in_distribution" else 1
        for ep in range(episodes):
            env = MicroserviceEnv(
                n_services=n_services,
                episode_length=episode_length,
                fault_config=fc,
                seed=1000 + ep,
            )
            obs, _ = env.reset(seed=1000 + ep)
            done = False
            while not done:
                est = model.uncertainty(obs)
                q_range = float(est.q_mean.max() - est.q_mean.min()) + 1e-6
                norm_q_std = float(np.clip(est.q_std_per_action[est.action] / q_range, 0.0, 1.0))
                raw_score = float(np.clip(0.5 * norm_q_std + 0.5 * est.raw_vote_disagreement, 0.0, 1.0))

                obs, _r, term, trunc, info = env.step(est.action)
                done = term or trunc

                ood_labels.append(is_ood)
                bad_action_labels.append(1 if info["bad_action"] else 0)
                canon_scores.append(est.score)
                raw_scores.append(raw_score)
                canon_votes.append(est.vote_disagreement)
                raw_votes.append(est.raw_vote_disagreement)
                q_stds.append(norm_q_std)

    y_ood = np.array(ood_labels)
    y_bad = np.array(bad_action_labels)

    return {
        "n_samples": len(y_ood),
        "ood_positive_rate": float(np.mean(y_ood)),
        "bad_action_positive_rate": float(np.mean(y_bad)),
        "mean_vote_disagreement_canonical": float(np.mean(canon_votes)),
        "mean_vote_disagreement_raw": float(np.mean(raw_votes)),
        "ood_detection": {
            "score_canonical_auroc": compute_auroc(y_ood, np.array(canon_scores)),
            "score_canonical_auprc": compute_auprc(y_ood, np.array(canon_scores)),
            "score_raw_auroc": compute_auroc(y_ood, np.array(raw_scores)),
            "score_raw_auprc": compute_auprc(y_ood, np.array(raw_scores)),
            "q_std_auroc": compute_auroc(y_ood, np.array(q_stds)),
            "q_std_auprc": compute_auprc(y_ood, np.array(q_stds)),
        },
        "bad_action_prediction": {
            "score_canonical_auroc": compute_auroc(y_bad, np.array(canon_scores)),
            "score_canonical_auprc": compute_auprc(y_bad, np.array(canon_scores)),
            "score_raw_auroc": compute_auroc(y_bad, np.array(raw_scores)),
            "score_raw_auprc": compute_auprc(y_bad, np.array(raw_scores)),
            "q_std_auroc": compute_auroc(y_bad, np.array(q_stds)),
            "q_std_auprc": compute_auprc(y_bad, np.array(q_stds)),
        },
    }


def sweep_thresholds(
    model: BootstrappedDQN,
    scenario_name: str = "worst_case",
    n_services: int = 8,
    episodes: int = 20,
    episode_length: int = 200,
) -> list[dict]:
    fc = SCENARIOS[scenario_name]

    def env_factory(seed):
        return MicroserviceEnv(
            n_services=n_services, episode_length=episode_length, fault_config=fc, seed=seed
        )

    theta_u_list = [0.20, 0.30, 0.42, 0.55, 0.70]
    theta_b_list = [0.0, 0.25, 0.50, 0.75]
    sweep_rows: list[dict] = []

    for tu in theta_u_list:
        for tb in theta_b_list:
            factory = lambda env, u=tu, b=tb: AEGISAgent(
                policy=model,
                topology=env.topology,
                confidence_threshold=u,
                blast_radius_threshold=b,
                use_blast_radius=(b > 0.0),
            )
            agg = run_agent_suite(f"tu={tu:.2f},tb={tb:.2f}", factory, env_factory, episodes)
            sweep_rows.append(
                {
                    "theta_u": tu,
                    "theta_b": tb,
                    "override_rate": agg.get("override_rate", 0.0),
                    "total_blast_harm_per_ep": agg["total_blast_harm_per_ep"],
                    "bad_actions_per_ep": agg["bad_actions_per_ep"],
                    "avg_reward": agg["avg_reward"],
                    "avg_availability": agg["avg_availability"],
                }
            )
    return sweep_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate uncertainty calibration and threshold sensitivity")
    parser.add_argument("--model-path", type=str, default="models/bootstrapped_dqn_microservice.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n-services", type=int, default=8)
    parser.add_argument("--out", type=str, default="results/uncertainty_analysis.json")
    args = parser.parse_args()

    model = BootstrappedDQN.load(args.model_path)
    print("Running uncertainty discrimination (AUROC / AUPRC)...")
    disc = evaluate_uncertainty_discrimination(
        model, n_services=args.n_services, episodes=args.episodes
    )
    print(json.dumps(disc, indent=2))

    print("\nRunning threshold sensitivity sweep on 'worst_case'...")
    sweep = sweep_thresholds(
        model, scenario_name="worst_case", n_services=args.n_services, episodes=args.episodes
    )
    for row in sweep:
        print(
            f"  theta_u={row['theta_u']:.2f} theta_b={row['theta_b']:.2f} | "
            f"override={row['override_rate']:.3f} | harm/ep={row['total_blast_harm_per_ep']:.3f} | "
            f"bad/ep={row['bad_actions_per_ep']:.2f} | reward={row['avg_reward']:+.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"discrimination": disc, "threshold_sweep": sweep}, indent=2))
    print(f"\nSaved uncertainty analysis to {out_path}")


if __name__ == "__main__":
    main()

