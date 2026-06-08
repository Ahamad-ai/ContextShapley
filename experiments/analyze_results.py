"""
Analyze pilot/experiment results: compute aggregate Shapley values,
interaction matrices, and generate visualizations.
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


def load_results(results_path: str | Path) -> list[dict]:
    """Load experiment results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def aggregate_shapley_values(results: list[dict]) -> pd.DataFrame:
    """
    Aggregate Shapley values across all task instances.

    Returns a DataFrame with columns: component, label, mean_sv, std_sv, min_sv, max_sv
    """
    rows = []
    for c in ALL_COMPONENT_IDS:
        values = [r["shapley_values"][c] for r in results]
        rows.append({
            "component": c,
            "label": COMPONENT_LABELS[c],
            "mean_sv": np.mean(values),
            "std_sv": np.std(values),
            "min_sv": np.min(values),
            "max_sv": np.max(values),
            "n": len(values),
        })
    return pd.DataFrame(rows)


def aggregate_interactions(results: list[dict]) -> pd.DataFrame:
    """
    Aggregate interaction indices across all task instances.

    Returns a DataFrame with columns: pair, mean_interaction, std, label
    """
    # Collect all pair keys
    pair_keys = set()
    for r in results:
        pair_keys.update(r["interactions"].keys())

    rows = []
    for pk in sorted(pair_keys):
        values = [r["interactions"].get(pk, 0.0) for r in results]
        ci, cj = pk.split(",")
        rows.append({
            "pair": pk,
            "ci": ci,
            "cj": cj,
            "mean_interaction": np.mean(values),
            "std_interaction": np.std(values),
            "label": "synergy" if np.mean(values) > 0.01 else "interference" if np.mean(values) < -0.01 else "neutral",
        })
    return pd.DataFrame(rows)


def plot_shapley_bar(sv_df: pd.DataFrame, title: str, save_path: Path):
    """Bar chart of mean Shapley values with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = []
    for v in sv_df["mean_sv"]:
        if v > 0.01:
            colors.append("#2ecc71")  # green = helps
        elif v < -0.01:
            colors.append("#e74c3c")  # red = hurts
        else:
            colors.append("#95a5a6")  # gray = neutral

    bars = ax.bar(
        sv_df["label"],
        sv_df["mean_sv"],
        yerr=sv_df["std_sv"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
    )

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean Shapley Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=15)

    # Annotate bars with values
    for bar, val in zip(bars, sv_df["mean_sv"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def plot_interaction_heatmap(interactions_df: pd.DataFrame, title: str, save_path: Path):
    """Heatmap of pairwise interaction indices."""
    # Build symmetric matrix
    components = ALL_COMPONENT_IDS
    n = len(components)
    matrix = np.zeros((n, n))

    for _, row in interactions_df.iterrows():
        i = components.index(row["ci"])
        j = components.index(row["cj"])
        matrix[i][j] = row["mean_interaction"]
        matrix[j][i] = row["mean_interaction"]

    labels = [COMPONENT_LABELS[c] for c in components]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use diverging colormap: red = interference, blue = synergy
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def plot_per_instance_shapley(results: list[dict], title: str, save_path: Path):
    """Grouped bar chart showing Shapley values per task instance."""
    n_tasks = len(results)
    if n_tasks > 20:
        results = results[:20]  # cap for readability
        n_tasks = 20

    components = ALL_COMPONENT_IDS
    x = np.arange(n_tasks)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(10, n_tasks * 1.2), 5))

    colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]
    for i, c in enumerate(components):
        values = [r["shapley_values"][c] for r in results]
        ax.bar(x + i * width, values, width, label=COMPONENT_LABELS[c], color=colors[i])

    ax.set_xlabel("Task Instance")
    ax.set_ylabel("Shapley Value")
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([r["task_id"] for r in results], rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {save_path}")


def print_summary(sv_df: pd.DataFrame, interactions_df: pd.DataFrame, results: list[dict]):
    """Print a text summary of the analysis."""
    print("\n" + "=" * 60)
    print("CONTEXT SHAPLEY ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nInstances analyzed: {len(results)}")

    # Efficiency axiom check
    all_pass = all(r.get("efficiency_axiom_passes", False) for r in results)
    print(f"Efficiency axiom: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    print("\n--- Mean Shapley Values (Component Contributions) ---")
    for _, row in sv_df.sort_values("mean_sv", ascending=False).iterrows():
        sign = "+" if row["mean_sv"] >= 0 else ""
        bar = "#" * int(abs(row["mean_sv"]) * 40)
        print(f"  {row['label']:22s}: {sign}{row['mean_sv']:.4f} (std={row['std_sv']:.4f})  {bar}")

    print("\n--- Top Interaction Effects ---")
    if not interactions_df.empty:
        top = interactions_df.sort_values("mean_interaction", key=abs, ascending=False).head(10)
        for _, row in top.iterrows():
            ci_label = COMPONENT_LABELS.get(row["ci"], row["ci"])
            cj_label = COMPONENT_LABELS.get(row["cj"], row["cj"])
            print(f"  {ci_label} x {cj_label}: {row['mean_interaction']:+.4f} ({row['label']})")


def analyze(results_path: str, output_dir: str | None = None):
    """Full analysis pipeline."""
    results_path = Path(results_path)
    results = load_results(results_path)

    if output_dir:
        out = Path(output_dir)
    else:
        out = PROJECT_ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    sv_df = aggregate_shapley_values(results)
    interactions_df = aggregate_interactions(results)

    print_summary(sv_df, interactions_df, results)

    # Generate plots
    plot_shapley_bar(sv_df, "Mean Shapley Values by Context Component", out / "shapley_values.png")
    plot_interaction_heatmap(interactions_df, "Pairwise Interaction Matrix", out / "interaction_heatmap.png")
    plot_per_instance_shapley(results, "Per-Instance Shapley Values", out / "per_instance_shapley.png")

    # Save tables
    sv_df.to_csv(out / "shapley_values.csv", index=False)
    interactions_df.to_csv(out / "interactions.csv", index=False)
    print(f"\n[saved] Tables -> {out}")

    return sv_df, interactions_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze ContextShapley results")
    parser.add_argument("results_path", help="Path to results JSON file")
    parser.add_argument("--output-dir", help="Output directory for figures/tables")
    args = parser.parse_args()
    analyze(args.results_path, args.output_dir)
