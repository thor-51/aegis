"""
plot_ood_results.py — Generate publication-ready plots with 95% confidence
intervals for the revised AEGIS evaluation:
  1. Unconditional Total Blast-Weighted Harm per Episode (grouped bar + CI)
  2. Conditional Mean Blast Radius of Bad Actions (grouped bar + CI)
  3. Bad Action Rate (grouped bar + CI)
  4. Scenario × Agent Heatmaps (Unconditional Harm and Conditional BR_bad)
  5. Threshold Sensitivity Pareto Frontier (Override Rate vs Unconditional Harm)
  6. Multi-Seed Learning Curves (Mean ± 95% CI Episode Reward over training)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _plot_grouped_bars(
    plt,
    scenarios: list[str],
    agent_names: list[str],
    val_mat: np.ndarray,
    ci_mat: np.ndarray,
    ylabel: str,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(scenarios))
    width = min(0.11, 0.82 / max(1, len(agent_names)))
    for j, agent in enumerate(agent_names):
        offset = (j - len(agent_names) / 2 + 0.5) * width
        yerr = np.nan_to_num(ci_mat[:, j], nan=0.0)
        ax.bar(
            x + offset,
            val_mat[:, j],
            width,
            yerr=yerr,
            capsize=2,
            label=agent,
            alpha=0.88,
            error_kw={"elinewidth": 0.8, "alpha": 0.7},
        )
    ax.set_xlabel("Fault Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def _plot_heatmap(
    plt,
    scenarios: list[str],
    agent_names: list[str],
    mat: np.ndarray,
    cbar_label: str,
    title: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=40, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)

    vmax = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 1.0
    for i in range(len(scenarios)):
        for j in range(len(agent_names)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val > 0.55 * vmax else "black",
                )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def _plot_threshold_sensitivity(plt, unc_path: Path, out_dir: Path):
    if not unc_path.exists():
        return
    data = json.loads(unc_path.read_text())
    sweep = data.get("threshold_sweep", [])
    if not sweep:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    tb_values = sorted({row["theta_b"] for row in sweep})
    for tb in tb_values:
        pts = sorted([r for r in sweep if r["theta_b"] == tb], key=lambda r: r["override_rate"])
        xs = [r["override_rate"] * 100.0 for r in pts]
        ys = [r["total_blast_harm_per_ep"] for r in pts]
        label = "No-Graph (theta_b=0.0)" if tb == 0.0 else f"Graph-Gated (theta_b={tb:.2f})"
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)

    ax.set_xlabel("Override Rate (%)")
    ax.set_ylabel("Unconditional Blast-Weighted Harm / Episode")
    ax.set_title("Intervention vs. Safety Trade-off across (theta_u, theta_b)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "threshold_sensitivity.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def _plot_learning_curves(plt, results_dir: Path, out_dir: Path):
    curve_files = sorted(results_dir.glob("learning_curves_*.json"))
    if not curve_files:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for cf in curve_files:
        runs = json.loads(cf.read_text())
        if not runs:
            continue
        label = cf.stem.replace("learning_curves_", "").upper()
        min_len = min(len(r.get("episode_rewards", [])) for r in runs)
        if min_len < 5:
            continue
        mat = np.array([r["episode_rewards"][:min_len] for r in runs], dtype=np.float64)
        # Smooth with a 10-episode moving window
        window = min(10, min_len)
        kernel = np.ones(window) / window
        smoothed = np.array([np.convolve(row, kernel, mode="valid") for row in mat])
        mean = smoothed.mean(axis=0)
        ci = (
            1.96 * smoothed.std(axis=0, ddof=1) / np.sqrt(len(runs))
            if len(runs) > 1
            else np.zeros_like(mean)
        )
        xs = np.arange(len(mean)) + window
        ax.plot(xs, mean, linewidth=2, label=f"{label} (n={len(runs)} seeds)")
        if len(runs) > 1:
            ax.fill_between(xs, mean - ci, mean + ci, alpha=0.2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Return (10-ep moving average)")
    ax.set_title("Multi-Seed Training Learning Curves (Mean ± 95% CI)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "learning_curves.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot OOD evaluation, sensitivity, and learning curves")
    parser.add_argument("--results", type=str, default="results/phase6_ood.json")
    parser.add_argument("--uncertainty-results", type=str, default="results/uncertainty_analysis.json")
    parser.add_argument("--out-dir", type=str, default="results/plots")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib")
        return

    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        all_results = json.loads(results_path.read_text())
        scenarios = list(all_results.keys())
        agent_names = []
        for results in all_results.values():
            for r in results:
                if r["agent"] not in agent_names:
                    agent_names.append(r["agent"])

        def _build_matrices(metric: str) -> tuple[np.ndarray, np.ndarray]:
            val_mat = np.full((len(scenarios), len(agent_names)), np.nan)
            ci_mat = np.full((len(scenarios), len(agent_names)), 0.0)
            for i, scenario in enumerate(scenarios):
                lookup = {r["agent"]: r for r in all_results[scenario]}
                for j, agent in enumerate(agent_names):
                    if agent in lookup and metric in lookup[agent]:
                        val_mat[i, j] = lookup[agent][metric]
                        ci_key = f"{metric}_seed_ci95" if f"{metric}_seed_ci95" in lookup[agent] else f"{metric}_ci95"
                        ci_mat[i, j] = lookup[agent].get(ci_key, 0.0)
            return val_mat, ci_mat

        harm_mat, harm_ci = _build_matrices("total_blast_harm_per_ep")
        blast_mat, blast_ci = _build_matrices("avg_blast_radius_of_bad_actions")
        bad_rate_mat, bad_rate_ci = _build_matrices("bad_action_rate")

        _plot_grouped_bars(
            plt,
            scenarios,
            agent_names,
            harm_mat,
            harm_ci,
            "total_blast_harm_per_ep",
            "Unconditional Total Blast-Weighted Harm per Episode (Mean ± 95% CI, lower is better)",
            out_dir / "total_blast_harm_by_scenario.png",
        )
        _plot_grouped_bars(
            plt,
            scenarios,
            agent_names,
            blast_mat,
            blast_ci,
            "avg_blast_radius_of_bad_actions",
            "Conditional Blast Radius of Bad Actions BR_bad (Mean ± 95% CI, lower is better)",
            out_dir / "blast_radius_by_scenario.png",
        )
        _plot_grouped_bars(
            plt,
            scenarios,
            agent_names,
            bad_rate_mat,
            bad_rate_ci,
            "bad_action_rate",
            "Bad Action Rate (Mean ± 95% CI, lower is better)",
            out_dir / "bad_action_rate_by_scenario.png",
        )
        _plot_heatmap(
            plt,
            scenarios,
            agent_names,
            harm_mat,
            "total_blast_harm_per_ep",
            "Unconditional Blast-Weighted Harm per Episode: Scenario × Agent",
            out_dir / "total_harm_heatmap.png",
        )
        _plot_heatmap(
            plt,
            scenarios,
            agent_names,
            blast_mat,
            "avg_blast_radius_of_bad_actions",
            "Conditional Blast Radius of Bad Actions: Scenario × Agent",
            out_dir / "blast_radius_heatmap.png",
        )

    _plot_threshold_sensitivity(plt, Path(args.uncertainty_results), out_dir)
    _plot_learning_curves(plt, results_path.parent, out_dir)
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
