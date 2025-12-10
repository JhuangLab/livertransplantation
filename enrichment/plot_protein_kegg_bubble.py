#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from matplotlib.lines import Line2D

# Basic plotting configuration
plt.rcParams.update({
    "font.family": "Arial",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "legend.frameon": False,
    "font.size": 20
})


def kegg_bubble_final(
    df,
    title="KEGG Enrichment",
    outfile="kegg_bubble.png",
    top_n=15
):
    # 1. Data cleaning
    df = df[["Description", "p.adjust", "Count", "GeneRatio"]].copy()
    df = df.dropna()

    df["GeneRatio"] = df["GeneRatio"].apply(
        lambda x: float(x.split('/')[0]) / float(x.split('/')[1])
        if '/' in str(x) and len(x.split('/')) == 2 else np.nan
    )
    df["p.adjust"] = pd.to_numeric(df["p.adjust"], errors="coerce")
    df["p.adjust"] = df["p.adjust"].clip(lower=1e-5, upper=1.0)
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").astype(int)
    df = df[
        (df["Count"] >= 1)
        & (df["GeneRatio"].notna())
        & (df["p.adjust"] > 0)
    ]

    df = df.sort_values("p.adjust").head(top_n).iloc[::-1].reset_index(drop=True)
    if len(df) < 1:
        print("No valid data to plot.")
        return

    # 2. Bubble size
    base_size = 200
    size_step = 100
    df["size"] = base_size + (df["Count"] - 1) * size_step
    df["size"] = df["size"].clip(upper=2000)

    # 3. Color mapping
    p_min = df["p.adjust"].min()
    p_max = df["p.adjust"].max()
    norm = mcolors.Normalize(vmin=p_min, vmax=p_max)
    cmap = plt.cm.coolwarm_r  # red → blue gradient

    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 13))

    # Horizontal dashed lines
    for i in range(len(df)):
        ax.axhline(
            y=i,
            color="gray",
            linestyle="--",
            alpha=0.3,
            linewidth=0.8
        )

    scatter = ax.scatter(
        df["GeneRatio"], df.index,
        s=df["size"],
        c=df["p.adjust"],
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidth=1.0,
        alpha=0.85
    )

    # 5. Axes and labels
    ax.set_xlabel("GeneRatio", fontsize=24, fontweight="bold", labelpad=15)
    ax.set_yticks(df.index)
    ax.set_yticklabels(df["Description"], fontsize=28, fontweight="medium")
    ax.set_ylim(-1.8, len(df) + 0.8)
    ax.set_xlim(0, df["GeneRatio"].max() * 1.4)
    ax.tick_params(axis="x", labelsize=24)
    ax.tick_params(axis="y", labelsize=28, length=0)
    ax.grid(axis="x", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0)

    # 6. Colorbar (p.adjust) at top-right
    cbar_ax = fig.add_axes([0.92, 0.55, 0.04, 0.35])
    scatter = ax.scatter(
        df["GeneRatio"], df.index,
        s=df["size"],
        c=df["p.adjust"],
        cmap=plt.cm.coolwarm_r,
        edgecolors="black",
        linewidth=1.0,
        alpha=0.85
    )

    cbar = plt.colorbar(scatter, cax=cbar_ax, orientation="vertical")
    cbar.ax.invert_yaxis()  # red (small p.adjust) at the top
    cbar.ax.tick_params(labelsize=24)

    # Custom label on top of colorbar
    cbar.ax.text(
        0.5, 1.01, "p.adjust",
        ha="center", va="bottom",
        fontsize=24, fontweight="bold",
        rotation=0,
        transform=cbar.ax.transAxes
    )

    # 7. Count legend (bubble size)
    size_ax = fig.add_axes([0.92, 0.12, 0.04, 0.48])
    size_ax.axis("off")

    count_legend = sorted(df["Count"].unique(), reverse=True)
    size_legend = [base_size + (cnt - 1) * size_step for cnt in count_legend]
    size_legend = [min(s, 2000) for s in size_legend]

    proxies = []
    labels = []
    for cnt, s in zip(count_legend, size_legend):
        markersize = 2 * np.sqrt(s / np.pi)
        proxy = Line2D(
            [0], [0],
            marker="o", linestyle="none",
            markersize=markersize,
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=1.0,
            alpha=0.9
        )
        proxies.append(proxy)
        labels.append(str(cnt))

    size_ax.legend(
        proxies, labels,
        title="Count",
        title_fontsize=24,
        fontsize=24,
        loc="center",
        handletextpad=1.0,
        labelspacing=1.5,
        ncol=1,
        handlelength=1.0
    )

    # 8. Save figure
    fig.subplots_adjust(left=0.30, right=0.88, top=0.95, bottom=0.12)
    out_dir = os.path.dirname(outfile)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.6
    )
    plt.close()


if __name__ == "__main__":
    file_configs = [
        (
            "kegg_enrichment_results/protein_Control_vs_Pre_kegg.csv",
            "KEGG: Control vs Pre",
            "protein_kegg_bubble/KEGG_Control_vs_Pre.png"
        ),
        (
            "kegg_enrichment_results/protein_Pre_vs_Post_kegg.csv",
            "KEGG: Pre vs Post",
            "protein_kegg_bubble/KEGG_Pre_vs_Post.png"
        )
    ]

    for infile, title, outfile in file_configs:
        print(f"\nProcessing: {title}")
        try:
            if not os.path.exists(infile):
                raise FileNotFoundError(f"File not found: {infile}")
            df = pd.read_csv(infile)
            kegg_bubble_final(df, title=title, outfile=outfile)
        except Exception as e:
            print(f"Error: {str(e)}")
            continue