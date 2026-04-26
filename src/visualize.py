"""
visualize.py
============
Generate all plots for the CGS410 project report.

Figures produced:
    1. correlation_by_language.png  — bar chart of Spearman r per language
    2. correlation_by_typology.png  — boxplot of r by typological group
    3. dd_by_pmi_quartile.png       — mean DD per PMI quartile, all languages
    4. dd_distribution_comparison.png — DD distribution: high vs low PMI pairs

Usage:
    python src/visualize.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

# Colour scheme: consistent across all plots
COLORS = {
    "SVO":   "#2196F3",   # blue
    "SOV":   "#F44336",   # red
    "Mixed": "#4CAF50",   # green
}


def plot_correlation_by_language(results_df: pd.DataFrame,
                                  save_path: str = "results/correlation_by_language.png"):
    """
    Bar chart: Spearman r for each language, colour-coded by typology.
    All bars are negative — lower = stronger information locality effect.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    bar_colors = [COLORS[t] for t in results_df["typology"]]
    bars = ax.bar(
        results_df["language"],
        results_df["spearman_r"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.7,
    )

    # Annotate each bar with its r value
    for bar, r in zip(bars, results_df["spearman_r"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.01,
            f"{r:.3f}",
            ha="center", va="top",
            fontsize=8, color="white", fontweight="bold"
        )

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_title(
        "Spearman Correlation: PMI vs Dependency Distance\n"
        "(Negative = High-MI pairs are placed closer together)",
        fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xlabel("Language", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(min(results_df["spearman_r"]) - 0.05, 0.05)

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=t)
        for t, c in COLORS.items()
    ]
    ax.legend(handles=legend_handles, title="Typology", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")


def plot_correlation_by_typology(results_df: pd.DataFrame,
                                  save_path: str = "results/correlation_by_typology.png"):
    """
    Boxplot: distribution of Spearman r values within each typological group.
    Shows that SOV languages have stronger (more negative) correlations on average.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        data=results_df,
        x="typology", y="spearman_r",
        order=["SVO", "SOV", "Mixed"],
        palette=COLORS,
        ax=ax,
        width=0.5,
        linewidth=1.2,
    )
    # Overlay individual data points
    sns.stripplot(
        data=results_df,
        x="typology", y="spearman_r",
        order=["SVO", "SOV", "Mixed"],
        color="black", size=6, alpha=0.7, ax=ax
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(
        "PMI–Distance Correlation by Typological Group",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Word Order Typology", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")


def plot_dd_by_pmi_quartile(all_dfs: dict,
                             typology_map: dict,
                             save_path: str = "results/dd_by_pmi_quartile.png"):
    """
    3×3 grid: for each language, show mean DD across 4 PMI quartiles.
    A downward trend (Q1→Q4) confirms information locality.
    """
    languages = list(all_dfs.keys())
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=False)
    axes = axes.flatten()

    for i, lang in enumerate(languages):
        df = all_dfs[lang].copy()
        df["pmi_quartile"] = pd.qcut(
            df["PMI"], q=4,
            labels=["Q1\n(Low PMI)", "Q2", "Q3", "Q4\n(High PMI)"]
        )
        means = df.groupby("pmi_quartile", observed=True)["DD"].mean()
        typ   = typology_map[lang]

        axes[i].bar(
            means.index, means.values,
            color=COLORS[typ], edgecolor="black", linewidth=0.6
        )
        axes[i].set_title(f"{lang}  ({typ})", fontsize=10, fontweight="bold")
        axes[i].set_xlabel("PMI Quartile", fontsize=8)
        axes[i].set_ylabel("Mean DD", fontsize=8)
        axes[i].tick_params(labelsize=7)

    # Hide unused subplots if fewer than 9 languages
    for j in range(len(languages), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Mean Dependency Distance by PMI Quartile — All Languages\n"
        "(Decreasing trend Q1→Q4 confirms Information Locality)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def plot_dd_distribution(all_dfs: dict,
                          languages: list = None,
                          save_path: str = "results/dd_distribution_comparison.png"):
    """
    For selected languages: overlay DD histograms for high-PMI vs low-PMI pairs.
    Shift in distribution confirms that high-PMI pairs have shorter distances.
    """
    if languages is None:
        languages = ["English", "Hindi", "Russian"]

    fig, axes = plt.subplots(1, len(languages), figsize=(13, 4))

    for ax, lang in zip(axes, languages):
        df     = all_dfs[lang]
        median = df["PMI"].median()
        high   = df[df["PMI"] >  median]["DD"]
        low    = df[df["PMI"] <= median]["DD"]

        ax.hist(low,  bins=30, alpha=0.65, label="Low PMI",
                color="salmon",    density=True, edgecolor="white")
        ax.hist(high, bins=30, alpha=0.65, label="High PMI",
                color="steelblue", density=True, edgecolor="white")

        ax.axvline(low.mean(),  color="darkred",  linestyle="--",
                   linewidth=1.2, label=f"Mean DD (Low)={low.mean():.2f}")
        ax.axvline(high.mean(), color="darkblue", linestyle="--",
                   linewidth=1.2, label=f"Mean DD (High)={high.mean():.2f}")

        ax.set_title(lang, fontsize=11, fontweight="bold")
        ax.set_xlabel("Dependency Distance", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(0, 20)
        ax.legend(fontsize=7)

    plt.suptitle(
        "DD Distribution: High-PMI vs Low-PMI Pairs\n"
        "(High-PMI pairs shift distribution leftward — shorter distances)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    # Load pre-computed results CSV
    results_df = pd.read_csv("results/correlation_results.csv")

    plot_correlation_by_language(results_df)
    plot_correlation_by_typology(results_df)

    print("\nFor quartile and distribution plots, run correlation.py first")
    print("to generate the per-language DataFrames (all_dfs dict).")
