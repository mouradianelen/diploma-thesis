"""Render the neighborhood Jaccard overlap results into a per-section table."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_FILE = "graph_results/neighborhood_overlap_all_sections.csv"
OUT_PNG = "graph_results/neighborhood_jaccard_table.png"


def render_mpl_table(data, col_width=2.4, row_height=0.6, font_size=11,
                     header_color="#40466e", row_colors=("#f1f1f2", "w"), edge_color="w"):
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
    return fig


def main():
    df = pd.read_csv(RESULTS_FILE)

    tbl = df.set_index("section")[["mean_jaccard", "median_jaccard", "std_jaccard"]]
    tbl.columns = ["mean\nJaccard", "median\nJaccard", "std\nJaccard"]
    tbl.loc["mean"] = tbl.mean(axis=0)
    tbl.index = tbl.index.astype(str)

    display = tbl.round(3).astype(str)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig = render_mpl_table(display)
    comparison = df["comparison"].iloc[0]
    fig.suptitle(f"Neighborhood Jaccard overlap \u2014 {comparison}", fontsize=12, y=1.02)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(display.to_string())
    print(f"\nSaved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
