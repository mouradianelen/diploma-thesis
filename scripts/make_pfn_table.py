"""Render experiment_c_pfn_results.csv into a PFN-vs-KMeans ARI comparison table."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_FILE = "results/experiment_c_pfn_results.csv"
OUT_PNG = "experiment_c_plots/pfn_vs_kmeans_table.png"

ENCODERS = ["STAGATE", "GraphST", "SpaGCN"]
HEADS = ["PFN", "KMeans", "Louvain"]


def render_mpl_table(data, col_width=2.2, row_height=0.6, font_size=11,
                     header_color="#40466e", row_colors=("#f1f1f2", "w"),
                     edge_color="w", bold_mask=None):
    size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
    fig, ax = plt.subplots(figsize=size)
    ax.axis("off")
    table = ax.table(cellText=data.values, colLabels=data.columns,
                     rowLabels=data.index, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[row % len(row_colors)])
            if bold_mask is not None and 0 <= col < bold_mask.shape[1] and bold_mask.iloc[row - 1, col]:
                cell.set_text_props(weight="bold")
    return fig


def main():
    df = pd.read_csv(RESULTS_FILE)

    ari = pd.DataFrame(index=df["section"].astype(str))
    for enc in ENCODERS:
        for h in HEADS:
            ari[f"{enc}\n{h}"] = df[f"{enc}_{h}_ARI"].values

    ari.loc["mean"] = ari.mean(axis=0)

    # Bold the best head within each encoder group (blocks of len(HEADS)) per row.
    bold = pd.DataFrame(False, index=ari.index, columns=ari.columns)
    step = len(HEADS)
    for i in range(0, ari.shape[1], step):
        block = ari.iloc[:, i:i + step]
        winners = block.idxmax(axis=1)
        for r, col in winners.items():
            bold.loc[r, col] = True

    display = ari.round(3).astype(str)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig = render_mpl_table(display, bold_mask=bold)
    fig.suptitle("ARI by encoder and clustering head (DLPFC)", fontsize=13, y=1.02)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(display.to_string())
    print(f"\nSaved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
