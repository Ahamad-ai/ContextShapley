"""
Cross-model analysis: GPT-5 vs Qwen3-8B comparison figures.
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

ROOT = PROJECT_ROOT / "results" / "raw"
FIGURES = PROJECT_ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = {
    "GPT-5": {
        "ifeval": ROOT / "ifeval" / "ifeval_results.json",
        "hotpotqa": ROOT / "hotpotqa" / "hotpotqa_results.json",
        "gsm8k": ROOT / "gsm8k" / "gsm8k_results.json",
    },
    "Qwen3-8B": {
        "ifeval": ROOT / "ifeval_qwen3_8b" / "ifeval_qwen3_8b_results.json",
        "hotpotqa": ROOT / "hotpotqa_qwen3_8b" / "hotpotqa_qwen3_8b_results.json",
        "gsm8k": ROOT / "gsm8k_qwen3_8b" / "gsm8k_qwen3_8b_results.json",
    },
}

BENCHMARKS = ["ifeval", "hotpotqa", "gsm8k"]
COMP_SHORT = {c: COMPONENT_LABELS[c].split()[0] for c in ALL_COMPONENT_IDS}


def load_all():
    data = {}
    for model, paths in MODELS.items():
        data[model] = {}
        for bm, path in paths.items():
            with open(path) as f:
                data[model][bm] = json.load(f)
    return data


def fig_cross_model_grouped(data):
    """Figure: Side-by-side Shapley values per benchmark, per model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    colors = {"GPT-5": "#3498db", "Qwen3-8B": "#e67e22"}
    width = 0.35
    x = np.arange(len(ALL_COMPONENT_IDS))

    for idx, bm in enumerate(BENCHMARKS):
        ax = axes[idx]
        for i, model in enumerate(MODELS):
            means = [np.mean([r["shapley_values"][c] for r in data[model][bm]]) for c in ALL_COMPONENT_IDS]
            stds = [np.std([r["shapley_values"][c] for r in data[model][bm]]) for c in ALL_COMPONENT_IDS]
            ax.bar(x + i * width - width / 2, means, width, yerr=stds,
                   label=model, color=colors[model], edgecolor="black", linewidth=0.5, capsize=3)

        ax.set_title(bm.upper(), fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([COMP_SHORT[c] for c in ALL_COMPONENT_IDS], fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Mean Shapley Value" if idx == 0 else "")
        if idx == 0:
            ax.legend(fontsize=9)

    plt.suptitle("Cross-Model Shapley Values: GPT-5 vs Qwen3-8B", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "crossmodel_shapley.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [fig] crossmodel_shapley.png")


def fig_s_dependence(data):
    """Figure: S-dependence comparison across benchmarks."""
    fig, ax = plt.subplots(figsize=(8, 5))

    bm_labels = [bm.upper() for bm in BENCHMARKS]
    x = np.arange(len(BENCHMARKS))
    width = 0.35
    colors = {"GPT-5": "#3498db", "Qwen3-8B": "#e67e22"}

    for i, model in enumerate(MODELS):
        s_vals = [np.mean([r["shapley_values"]["S"] for r in data[model][bm]]) for bm in BENCHMARKS]
        ax.bar(x + i * width - width / 2, s_vals, width,
               label=model, color=colors[model], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean Shapley Value of S (System Instructions)", fontsize=11)
    ax.set_title("System Instruction Dependence: GPT-5 vs Qwen3-8B", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(bm_labels, fontsize=11)
    ax.legend(fontsize=10)

    # Annotate
    for i, model in enumerate(MODELS):
        s_vals = [np.mean([r["shapley_values"]["S"] for r in data[model][bm]]) for bm in BENCHMARKS]
        for j, v in enumerate(s_vals):
            ax.text(j + i * width - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES / "crossmodel_s_dependence.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [fig] crossmodel_s_dependence.png")


def fig_interaction_comparison(data):
    """Figure: Interaction heatmaps side by side per benchmark."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row, model in enumerate(MODELS):
        for col, bm in enumerate(BENCHMARKS):
            ax = axes[row, col]
            results = data[model][bm]

            pair_vals = {}
            for r in results:
                for pk, v in r["interactions"].items():
                    pair_vals.setdefault(pk, []).append(v)

            n = len(ALL_COMPONENT_IDS)
            matrix = np.zeros((n, n))
            for pk, vs in pair_vals.items():
                ci, cj = pk.split(",")
                i, j = ALL_COMPONENT_IDS.index(ci), ALL_COMPONENT_IDS.index(cj)
                matrix[i][j] = np.mean(vs)
                matrix[j][i] = np.mean(vs)

            labels = [COMP_SHORT[c] for c in ALL_COMPONENT_IDS]
            vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
            sns.heatmap(matrix, xticklabels=labels, yticklabels=labels,
                        annot=True, fmt=".2f", cmap="RdBu", center=0,
                        vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax,
                        annot_kws={"size": 8})
            ax.set_title(f"{model} - {bm.upper()}", fontsize=10)

    plt.suptitle("Interaction Matrices: GPT-5 (top) vs Qwen3-8B (bottom)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES / "crossmodel_interactions.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [fig] crossmodel_interactions.png")


def fig_universal_vs_modelspecific(data):
    """Figure: Which findings are universal vs model-specific."""
    fig, ax = plt.subplots(figsize=(10, 6))

    findings = []
    for bm in BENCHMARKS:
        for c in ALL_COMPONENT_IDS:
            g = np.mean([r["shapley_values"][c] for r in data["GPT-5"][bm]])
            q = np.mean([r["shapley_values"][c] for r in data["Qwen3-8B"][bm]])
            findings.append({
                "benchmark": bm.upper(),
                "component": COMPONENT_LABELS[c],
                "GPT-5": g,
                "Qwen3-8B": q,
                "same_sign": (g > 0.01 and q > 0.01) or (g < -0.01 and q < -0.01) or (abs(g) <= 0.01 and abs(q) <= 0.01),
            })

    df = pd.DataFrame(findings)
    universal = df[df["same_sign"]].shape[0]
    divergent = df[~df["same_sign"]].shape[0]

    ax.scatter(df["GPT-5"], df["Qwen3-8B"], c=df["same_sign"].map({True: "#2ecc71", False: "#e74c3c"}),
               s=80, edgecolors="black", linewidth=0.5, zorder=3)

    # Diagonal line
    lim = max(abs(df["GPT-5"].max()), abs(df["Qwen3-8B"].max()), abs(df["GPT-5"].min()), abs(df["Qwen3-8B"].min())) + 0.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    ax.set_xlabel("GPT-5 Shapley Value", fontsize=12)
    ax.set_ylabel("Qwen3-8B Shapley Value", fontsize=12)
    ax.set_title(f"Universal vs Model-Specific Findings ({universal} agree, {divergent} diverge)", fontsize=13)

    # Annotate outliers
    for _, row in df.iterrows():
        if abs(row["GPT-5"] - row["Qwen3-8B"]) > 0.15:
            ax.annotate(f"{row['component'][:6]}\n{row['benchmark']}",
                        (row["GPT-5"], row["Qwen3-8B"]),
                        fontsize=7, ha="center", va="bottom")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ecc71", label=f"Universal ({universal})"),
        Patch(color="#e74c3c", label=f"Model-specific ({divergent})"),
    ], fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES / "crossmodel_universal_vs_specific.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [fig] crossmodel_universal_vs_specific.png")


def print_summary(data):
    print("\n" + "#" * 70)
    print("  CROSS-MODEL SUMMARY")
    print("#" * 70)

    for bm in BENCHMARKS:
        print(f"\n  --- {bm.upper()} ---")
        print(f"  {'Comp':<5} {'GPT-5':>8} {'Qwen3':>8} {'Delta':>8}")
        for c in ALL_COMPONENT_IDS:
            g = np.mean([r["shapley_values"][c] for r in data["GPT-5"][bm]])
            q = np.mean([r["shapley_values"][c] for r in data["Qwen3-8B"][bm]])
            print(f"  {c:<5} {g:+.4f} {q:+.4f} {q-g:+.4f}")


if __name__ == "__main__":
    print("Loading all results...")
    data = load_all()
    print("Generating cross-model figures...")
    fig_cross_model_grouped(data)
    fig_s_dependence(data)
    fig_interaction_comparison(data)
    fig_universal_vs_modelspecific(data)
    print_summary(data)
    print(f"\nAll figures saved to {FIGURES}")
