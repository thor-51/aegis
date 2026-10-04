"""
plot_ood_results.py — Phase 6: visualize OOD evaluation results.

Reads the JSON output from eval_ood.py and generates publication-ready plots:
  1. Bar chart: avg_blast_radius_of_bad_actions per agent, grouped by scenario
  2. Bar chart: bad_action_rate per agent, grouped by scenario
  3. Heatmap: agent × scenario for the key metric

Requires matplotlib (optional dependency, listed in requirements.txt).

Usage:
    python -m aegis.plot_ood_results [--results results/phase6_ood.json] [--out-dir results/plots/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot Phase 6 OOD results")
    parser.add_argument("--results", type=str, default="results/phase6_ood.json")
    parser.add_argument("--out-dir", type=str, default="results/plots")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with:")
        print("  pip install matplotlib")
        return

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"No results file found at {results_path}")
        print("Run `python -m aegis.eval_ood` first.")
        return

    all_results = json.loads(results_path.read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = list(all_results.keys())

    # Collect agent names (ordered consistently)
    agent_names = []
    for results in all_results.values():
        for r in results:
            if r["agent"] not in agent_names:
                agent_names.append(r["agent"])

    # Build data matrices: scenario × agent
    def _build_matrix(metric: str) -> np.ndarray:
        mat = np.full((len(scenarios), len(agent_names)), np.nan)
        for i, scenario in enumerate(scenarios):
            lookup = {r["agent"]: r for r in all_results[scenario]}
            for j, agent in enumerate(agent_names):
                if agent in lookup:
                    mat[i, j] = lookup[agent][metric]
        return mat

    blast_mat = _build_matrix("avg_blast_radius_of_bad_actions")
    bad_rate_mat = _build_matrix("bad_action_rate")

    # ---------------------------------------------------------------- #
    # Plot 1: Grouped bar chart — blast radius of bad actions
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    width = 0.11
    for j, agent in enumerate(agent_names):
        offset = (j - len(agent_names) / 2 + 0.5) * width
        ax.bar(x + offset, blast_mat[:, j], width, label=agent, alpha=0.85)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("avg_blast_radius_of_bad_actions")
    ax.set_title("Phase 6: Blast Radius of Bad Actions (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "blast_radius_by_scenario.png", dpi=150)
    print(f"Saved {out_dir / 'blast_radius_by_scenario.png'}")
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Plot 2: Grouped bar chart — bad action rate
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(14, 6))
    for j, agent in enumerate(agent_names):
        offset = (j - len(agent_names) / 2 + 0.5) * width
        ax.bar(x + offset, bad_rate_mat[:, j], width, label=agent, alpha=0.85)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("bad_action_rate")
    ax.set_title("Phase 6: Bad Action Rate (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bad_action_rate_by_scenario.png", dpi=150)
    print(f"Saved {out_dir / 'bad_action_rate_by_scenario.png'}")
    plt.close(fig)

    # ---------------------------------------------------------------- #
    # Plot 3: Heatmap — agent × scenario for blast radius
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(blast_mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=45, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title("Blast Radius of Bad Actions: Scenario × Agent")
    fig.colorbar(im, ax=ax, label="avg_blast_radius_of_bad_actions")

    # Annotate cells with values
    for i in range(len(scenarios)):
        for j in range(len(agent_names)):
            val = blast_mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if val > 0.3 else "black")

    fig.tight_layout()
    fig.savefig(out_dir / "blast_radius_heatmap.png", dpi=150)
    print(f"Saved {out_dir / 'blast_radius_heatmap.png'}")
    plt.close(fig)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
