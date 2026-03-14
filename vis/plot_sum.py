from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot supplementary algorithm-stratified reporting figures from "
            "algo_effect_overall.tsv and algo_effect_by_year.tsv."
        )
    )
    parser.add_argument(
        "--overall",
        default="algo_effect_overall.tsv",
        help="Path to algo_effect_overall.tsv (default: %(default)s)",
    )
    parser.add_argument(
        "--by-year",
        dest="by_year",
        default="algo_effect_by_year.tsv",
        help="Path to algo_effect_by_year.tsv (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="supplement_algorithm_effects.png",
        help="Output figure path (default: %(default)s)",
    )
    parser.add_argument(
        "--top-n-bars",
        type=int,
        default=15,
        help="Number of algorithms to show in the bar chart and heatmap (default: %(default)s)",
    )
    parser.add_argument(
        "--top-n-lines",
        type=int,
        default=8,
        help="Number of algorithms to show in the year-wise line plot (default: %(default)s)",
    )
    parser.add_argument(
        "--min-yearly-n",
        type=int,
        default=50,
        help="Minimum n_articles to keep a year-specific point (default: %(default)s)",
    )
    parser.add_argument(
        "--include-generic",
        action="store_true",
        help="Include 'generic cluster*' in top-algorithm selections",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: %(default)s)",
    )
    return parser.parse_args()


def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")
    return df


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def maybe_filter_generic(df: pd.DataFrame, include_generic: bool) -> pd.DataFrame:
    if include_generic:
        return df.copy()
    return df.loc[df["algorithm"] != "generic cluster*"].copy()


def get_top_algorithms(
    overall_df: pd.DataFrame,
    top_n: int,
) -> list[str]:
    top = (
        overall_df.sort_values(["n_articles", "algorithm"], ascending=[False, True])
        .head(top_n)["algorithm"]
        .tolist()
    )
    return top


def shorten_algorithm_name(name: str) -> str:
    mapping = {
        "mini-batch k-means": "mini-batch k-means",
        "c-means/fuzzy c-means": "fuzzy c-means",
        "k-medoids/PAM": "k-medoids/PAM",
        "GMM/mixture": "GMM/mixture",
        "community detection": "community detection",
        "generic cluster*": "generic cluster",
    }
    return mapping.get(name, name)


def make_bar_panel(ax: plt.Axes, overall_df: pd.DataFrame, top_algorithms: list[str]) -> None:
    plot_df = (
        overall_df.loc[overall_df["algorithm"].isin(top_algorithms), ["algorithm", "n_articles"]]
        .copy()
        .sort_values("n_articles", ascending=True)
    )
    labels = [shorten_algorithm_name(x) for x in plot_df["algorithm"]]
    values = plot_df["n_articles"].to_numpy()

    ax.barh(labels, values)
    ax.set_title("Top algorithms by article count")
    ax.set_xlabel("Articles")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmax = values.max()
    ax.set_xlim(0, xmax * 1.12)
    for idx, value in enumerate(values):
        ax.text(value + xmax * 0.01, idx, f"{value:,}", va="center", fontsize=7)


def make_heatmap_panel(
    ax: plt.Axes,
    overall_df: pd.DataFrame,
    top_algorithms: list[str],
) -> None:
    metric_cols = [
        "pct_missing_params",
        "pct_missing_justification",
        "pct_missing_evaluation",
        "pct_missing_tuning",
        "pct_missing_reporting_signals",
    ]
    metric_labels = [
        "Missing\nparams",
        "Missing\njustification",
        "Missing\nevaluation",
        "Missing\ntuning",
        "Missing overall",
    ]

    plot_df = overall_df.loc[
        overall_df["algorithm"].isin(top_algorithms),
        ["algorithm", "n_articles"] + metric_cols,
    ].copy()

    plot_df["algorithm"] = pd.Categorical(
        plot_df["algorithm"],
        categories=top_algorithms,
        ordered=True,
    )
    plot_df = plot_df.sort_values("algorithm")

    mat = plot_df[metric_cols].to_numpy()
    ylabels = [shorten_algorithm_name(x) for x in plot_df["algorithm"].astype(str)]

    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title("Reporting percentages by algorithm")
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6.5,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent")

    ax.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

def make_lines_panel(
    ax: plt.Axes,
    by_year_df: pd.DataFrame,
    top_algorithms: list[str],
    min_yearly_n: int,
) -> None:
    plot_df = by_year_df.loc[
        by_year_df["algorithm"].isin(top_algorithms) & (by_year_df["n_articles"] >= min_yearly_n)
    ].copy()

    for algorithm in top_algorithms:
        sub = plot_df.loc[plot_df["algorithm"] == algorithm].sort_values("year")
        if sub.empty:
            continue
        ax.plot(
            sub["year"].to_numpy(),
            sub["pct_missing_reporting_signals"].to_numpy(),
            linewidth=1.8,
            label=shorten_algorithm_name(algorithm),
        )

    ax.set_title("Year-wise missing-reporting rates")
    ax.set_xlabel("Year")
    ax.set_ylabel("Articles (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
    )


def make_delta_panel(ax: plt.Axes, overall_df: pd.DataFrame, top_algorithms: list[str]) -> None:
    plot_df = overall_df.loc[
        overall_df["algorithm"].isin(top_algorithms),
        ["algorithm", "n_articles", "delta_missing_reporting_vs_overall_pp", "q_fdr_bh"],
    ].copy()

    x = plot_df["n_articles"].to_numpy()
    y = plot_df["delta_missing_reporting_vs_overall_pp"].to_numpy()
    labels = plot_df["algorithm"].astype(str).tolist()

    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.scatter(x, y, s=40)

    ax.set_xscale("log")
    ax.set_title("Deviation from overall missing-reporting baseline")
    ax.set_xlabel("Articles (log scale)")
    ax.set_ylabel("Delta vs overall (percentage points)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for xi, yi, label, qval in zip(x, y, labels, plot_df["q_fdr_bh"].to_numpy()):
        suffix = "*" if pd.notna(qval) and qval < 0.05 else ""
        ax.annotate(
            shorten_algorithm_name(label) + suffix,
            xy=(xi, yi),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
        )


def save_figure(fig: plt.Figure, output_path: str, dpi: int) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")

    pdf_path = out.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight")


def main() -> int:
    args = parse_args()
    style_matplotlib()

    overall_df = load_tsv(args.overall)
    by_year_df = load_tsv(args.by_year)

    required_overall_cols = {
        "algorithm",
        "n_articles",
        "pct_missing_params",
        "pct_missing_justification",
        "pct_missing_evaluation",
        "pct_missing_tuning",
        "pct_missing_reporting_signals",
        "delta_missing_reporting_vs_overall_pp",
        "q_fdr_bh",
    }
    required_by_year_cols = {
        "year",
        "algorithm",
        "n_articles",
        "pct_missing_reporting_signals",
    }

    missing_overall = required_overall_cols.difference(overall_df.columns)
    missing_by_year = required_by_year_cols.difference(by_year_df.columns)

    if missing_overall:
        raise ValueError(
            "Missing columns in overall TSV: " + ", ".join(sorted(missing_overall))
        )
    if missing_by_year:
        raise ValueError(
            "Missing columns in by-year TSV: " + ", ".join(sorted(missing_by_year))
        )

    overall_df = maybe_filter_generic(overall_df, args.include_generic)
    by_year_df = maybe_filter_generic(by_year_df, args.include_generic)

    top_algorithms_for_bars = get_top_algorithms(overall_df, args.top_n_bars)
    top_algorithms_for_lines = get_top_algorithms(overall_df, args.top_n_lines)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(hspace=0.08, wspace=0.05)

    make_bar_panel(axes[0, 0], overall_df, top_algorithms_for_bars)
    make_heatmap_panel(axes[0, 1], overall_df, top_algorithms_for_bars)
    make_lines_panel(axes[1, 0], by_year_df, top_algorithms_for_lines, args.min_yearly_n)
    make_delta_panel(axes[1, 1], overall_df, top_algorithms_for_bars)

    panel_labels = ["A", "B", "C", "D"]
    for ax, label in zip(axes.flat, panel_labels):
        ax.text(
            0.0,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    save_figure(fig, args.output, args.dpi)
    plt.close(fig)

    print(f"Saved figure to: {args.output}")
    print(f"Saved figure to: {Path(args.output).with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())