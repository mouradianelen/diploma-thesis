"""Render PFN predicted-cluster-count diagnostics from experiment_c_pfn_results.csv."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_FILE = "results/experiment_c_pfn_results.csv"
OUT_PNG = "experiment_c_plots/pfn_pred_k_table.png"

ENCODERS = ["STAGATE", "GraphST", "SpaGCN"]


def render_mpl_table(data, col_width=2.0, row_height=0.6, font_size=11,
                     header_color="#40466e", row_colors=("#f1f1f2", "w"),
                     edge_color="w", collapse_mask=None):
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
            if collapse_mask is not None and 0 <= col < collapse_mask.shape[1] and collapse_mask.iloc[row - 1, col]:
                cell.set_facecolor("#f4b6b6")  # highlight collapsed predictions
    return fig


def main():
    df = pd.read_csv(RESULTS_FILE)

    tbl = pd.DataFrame(index=df["section"].astype(str))
    tbl["true k"] = df["n_clusters"].astype(int).values
    for enc in ENCODERS:
        tbl[f"{enc}\npred k"] = df[f"{enc}_PFN_pred_k"].astype(int).values

    # Flag encoder columns where PFN predicted far fewer clusters than truth.
    collapse = pd.DataFrame(False, index=tbl.index, columns=tbl.columns)
    for enc in ENCODERS:
        col = f"{enc}\npred k"
        collapse[col] = tbl[col] <= tbl["true k"] - 2

    display = tbl.astype(str)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig = render_mpl_table(display, collapse_mask=collapse)
    fig.suptitle("PFN predicted cluster count vs true k (red = collapse)", fontsize=12, y=1.02)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(display.to_string())
    print(f"\nSaved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
