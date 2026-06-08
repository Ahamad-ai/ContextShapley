"""
Full analysis: cross-benchmark Shapley comparison + paper-ready figures.

Usage:
    python3 experiments/analyze_full.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contextshapley.assembler import ALL_COMPONENT_IDS, COMPONENT_LABELS

RESULTS_DIR = PROJECT_ROOT / "results" / "raw"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_all_results() -> dict[str, list[dict]]:
    """Load results from all benchmarks."""
    benchmarks = {}
    for bm in ["ifeval", "hotpotqa", "gsm8k"]:
        path = RESULTS_DIR / bm / f"{bm}_results.json"
        if path.exists():
            with open(path) as f:
                benchmarks[bm] = json.load(f)
            print(f"  Loaded {bm}: {len(benchmarks[bm])} instances")
        else:
            print(f"  {bm}: not found, skipping")
    return benchmarks


def compute_shapley_table(all_results: dict[str, list]) -> pd.DataFrame:
    """Build a cross-benchmark Shapley value table."""
    rows = []
    for bm, results in all_results.items():
        for c in ALL_COMPONENT_IDS:
            values = [r["shapley_values"][c] for r in results]
            rows.append({
                "benchmark": bm,
                "component": c,
                "label": COMPONENT_LABELS[c],
                "mean_sv": np.mean(values),
                "std_sv": np.std(values),
                "min_sv": np.min(values),
                "max_sv": np.max(values),
            })
    return pd.DataFrame(rows)


def compute_interaction_table(all_results: dict[str, list]) -> pd.DataFrame:
    """Build a cross-benchmark interaction table."""
    rows = []
    for bm, results in all_results.items():
        pair_keys = set()
        for r in results:
            pair_keys.update(r["interactions"].keys())

        for pk in sorted(pair_keys):
            values = [r["interactions"].get(pk, 0.0) for r in results]
            ci, cj = pk.split(",")
            rows.append({
                "benchmark": bm,
                "pair": pk,
                "ci": ci,
                "cj": cj,
                "mean_interaction": np.mean(values),
                "std_interaction": np.std(values),
            })
    return pd.DataFrame(rows)


def plot_cross_benchmark_shapley(sv_df: pd.DataFrame, save_dir: Path):
    """Paper Figure 1: Shapley values across benchmarks (grouped bar chart)."""
    fig, ax = plt.subplots(figsize=(12, 5))

    benchmarks = sv_df["benchmark"].unique()
    n_bm = len(benchmarks)
    components = ALL_COMPONENT_IDS
    n_comp = len(components)
    x = np.arange(n_comp)
    width = 0.8 / n_bm

    colors = {"ifeval": "#3498db", "hotpotqa": "#e67e22", "gsm8k": "#2ecc71"}

    for i, bm in enumerate(benchmarks):
        bm_data = sv_df[sv_df["benchmark"] == bm]
        means = [bm_data[bm_data["component"] == c]["mean_sv"].values[0] for c in components]
        stds = [bm_data[bm_data["component"] == c]["std_sv"].values[0] for c in components]
        ax.bar(
            x + i * width - width * (n_bm - 1) / 2,
            means, width,
            yerr=stds,
            label=bm.upper(),
            color=colors.get(bm, "#95a5a6"),
            edgecolor="black", linewidth=0.5, capsize=3,
        )

    ax.set_xlabel("Context Component", fontsize=12)
    ax.set_ylabel("Mean Shapley Value", fontsize=12)
    ax.set_title("Shapley Values by Component Across Benchmarks", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([COMPONENT_LABELS[c] for c in components], fontsize=10)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(save_dir / "fig1_cross_benchmark_shapley.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [fig1] Cross-benchmark Shapley values")


def plot_interaction_heatmaps(int_df: pd.DataFrame, save_dir: Path):
    """Paper Figure 2: Interaction heatmap per benchmark."""
    benchmarks = int_df["benchmark"].unique()
    n_bm = len(benchmarks)

    fig, axes = plt.subplots(1, n_bm, figsize=(6 * n_bm, 5), squeeze=False)

    for idx, bm in enumerate(benchmarks):
        ax = axes[0, idx]
        bm_data = int_df[int_df["benchmark"] == bm]

        components = ALL_COMPONENT_IDS
        n = len(components)
        matrix = np.zeros((n, n))

        for _, row in bm_data.iterrows():
            i = components.index(row["ci"])
            j = components.index(row["cj"])
            matrix[i][j] = row["mean_interaction"]
            matrix[j][i] = row["mean_interaction"]

        labels = [COMPONENT_LABELS[c].split()[0] for c in components]  # short labels
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)

        sns.heatmap(
            matrix, xticklabels=labels, yticklabels=labels,
            annot=True, fmt=".3f", cmap="RdBu", center=0,
            vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax,
            annot_kws={"size": 8},
        )
        ax.set_title(bm.upper(), fontsize=12)

    plt.suptitle("Pairwise Interaction Matrices by Benchmark", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_dir / "fig2_interaction_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [fig2] Interaction heatmaps")


def plot_component_ranking(sv_df: pd.DataFrame, save_dir: Path):
    """Paper Figure 3: Component importance ranking shifts across task types."""
    benchmarks = sv_df["benchmark"].unique()

    fig, axes = plt.subplots(1, len(benchmarks), figsize=(5 * len(benchmarks), 4), squeeze=False)

    for idx, bm in enumerate(benchmarks):
        ax = axes[0, idx]
        bm_data = sv_df[sv_df["benchmark"] == bm].sort_values("mean_sv", ascending=True)

        colors = []
        for v in bm_data["mean_sv"]:
            if v > 0.01:
                colors.append("#2ecc71")
            elif v < -0.01:
                colors.append("#e74c3c")
            else:
                colors.append("#bdc3c7")

        ax.barh(bm_data["label"], bm_data["mean_sv"], xerr=bm_data["std_sv"],
                color=colors, edgecolor="black", linewidth=0.5, capsize=3)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title(bm.upper(), fontsize=12)
        ax.set_xlabel("Shapley Value")

    plt.suptitle("Component Importance Rankings by Task Type", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / "fig3_component_rankings.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [fig3] Component rankings")


def print_findings(sv_df: pd.DataFrame, int_df: pd.DataFrame, all_results: dict):
    """Print key findings in a structured format."""
    print("\n" + "#" * 70)
    print("  KEY FINDINGS")
    print("#" * 70)

    for bm in sv_df["benchmark"].unique():
        bm_sv = sv_df[sv_df["benchmark"] == bm].sort_values("mean_sv", ascending=False)
        bm_int = int_df[int_df["benchmark"] == bm].sort_values("mean_interaction", key=abs, ascending=False)
        n_instances = len(all_results.get(bm, []))

        print(f"\n  --- {bm.upper()} ({n_instances} instances) ---")

        # Top contributor
        top = bm_sv.iloc[0]
        print(f"  Highest contributor: {top['label']} (phi={top['mean_sv']:.4f})")

        # Lowest
        bot = bm_sv.iloc[-1]
        print(f"  Lowest contributor:  {bot['label']} (phi={bot['mean_sv']:.4f})")

        # Strongest interaction
        if not bm_int.empty:
            strongest = bm_int.iloc[0]
            if abs(strongest["mean_interaction"]) > 0.001:
                ci_l = COMPONENT_LABELS.get(strongest["ci"], strongest["ci"])
                cj_l = COMPONENT_LABELS.get(strongest["cj"], strongest["cj"])
                kind = "synergy" if strongest["mean_interaction"] > 0 else "interference"
                print(f"  Strongest interaction: {ci_l} x {cj_l} = {strongest['mean_interaction']:+.4f} ({kind})")
            else:
                print(f"  No significant interactions detected")

        # Efficiency axiom
        all_pass = all(r.get("efficiency_axiom_passes", False) for r in all_results.get(bm, []))
        print(f"  Efficiency axiom: {'ALL PASS' if all_pass else 'SOME FAIL'}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ContextShapley Full Analysis")
    print("=" * 70)

    all_results = load_all_results()
    if not all_results:
        print("No results found. Run experiments first.")
        return

    print("\nComputing tables...")
    sv_df = compute_shapley_table(all_results)
    int_df = compute_interaction_table(all_results)

    print("\nGenerating figures...")
    plot_cross_benchmark_shapley(sv_df, FIGURES_DIR)
    plot_interaction_heatmaps(int_df, FIGURES_DIR)
    plot_component_ranking(sv_df, FIGURES_DIR)

    # Save tables
    sv_df.to_csv(FIGURES_DIR / "shapley_values_full.csv", index=False)
    int_df.to_csv(FIGURES_DIR / "interactions_full.csv", index=False)
    print(f"\n  Tables saved to {FIGURES_DIR}")

    print_findings(sv_df, int_df, all_results)

    print(f"\n{'='*70}")
    print(f"  Analysis complete. Figures in {FIGURES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
