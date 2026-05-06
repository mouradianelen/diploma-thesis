
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import transformer
import torch
import matplotlib.pyplot as plt
import textwrap
from matplotlib.lines import Line2D

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def pipeline_hvg_pca(adata):
    

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    # sc.pp.log1p(adata_hvg)
    sc.pp.scale(adata_hvg)

    sc.pp.pca(adata_hvg, n_comps=50)

    X = adata_hvg.obsm["X_pca"][:, :5]
    X = StandardScaler().fit_transform(X)

    return X

def pipeline_spatial_smoothing(adata):

    coords = adata.obsm["spatial"]

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    X_expr = adata.X
    if not isinstance(X_expr, np.ndarray):
        X_expr = X_expr.toarray()

    # smoothing
    X_smooth = np.zeros_like(X_expr)

    for i in range(X_expr.shape[0]):
        X_smooth[i] = X_expr[indices[i]].mean(axis=0)

    # PCA
    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=5).fit_transform(X_smooth)

    return X_pca

# def run_pfn(model, X):

#     X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
#     X_tensor = X_tensor.unsqueeze(1)

#     num_clusters = torch.zeros((X_tensor.shape[0] + 1, 1), dtype=torch.long).to(device)

#     with torch.no_grad():
#         output_points, _ = model(X_tensor, num_clusters)

#     pred = output_points.squeeze(1).argmax(dim=-1).cpu().numpy()

#     return pred

def run_pfn(model, X, cluster_count=7):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    X_tensor = X_tensor.unsqueeze(1)  # (S, 1, F)

    max_bucket = model.embedding.num_embeddings - 1
    if not (0 <= cluster_count <= max_bucket):
        raise ValueError(f"cluster_count must be in [0, {max_bucket}]")

    num_clusters = torch.full(
        (X_tensor.shape[0] + 1, 1),
        fill_value=cluster_count,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        output_points, _ = model(X_tensor, num_clusters)

    pred = output_points.squeeze(1).argmax(dim=-1).cpu().numpy()
    return pred

def save_comparison_plot(adata, gt_labels, pred, title, filename):

    coords = adata.obsm["spatial"]
    common_title = textwrap.fill(title, width=40)
    gt_cat = gt_labels.astype("category")
    gt_codes = gt_cat.cat.codes.to_numpy()
    gt_names = [str(name) for name in gt_cat.cat.categories]

    pred = np.asarray(pred)
    pred_clusters = np.unique(pred)

    # Name each predicted cluster by the most frequent GT label within that cluster.
    pred_name_map = {}
    for cluster_id in pred_clusters:
        mask = pred == cluster_id
        if mask.any():
            majority_gt_code = np.bincount(gt_codes[mask]).argmax()
            pred_name_map[int(cluster_id)] = gt_names[int(majority_gt_code)]
        else:
            pred_name_map[int(cluster_id)] = f"Cluster {int(cluster_id)}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    cmap = plt.get_cmap("tab20")

    gt_color_lookup = {
        idx: cmap(idx % cmap.N) for idx in range(len(gt_names))
    }
    pred_color_lookup = {
        int(cluster_id): cmap(i % cmap.N) for i, cluster_id in enumerate(pred_clusters)
    }

    # Ground truth
    gt_colors = [gt_color_lookup[int(code)] for code in gt_codes]
    axes[0].scatter(coords[:, 0], coords[:, 1], c=gt_colors, s=5)
    axes[0].set_title("Ground Truth")
    axes[0].invert_yaxis()
    axes[0].axis("off")

    gt_handles = [
        Line2D([0], [0], marker='o', color='w', label=gt_names[idx],
               markerfacecolor=gt_color_lookup[idx], markersize=5)
        for idx in range(len(gt_names))
    ]
    axes[0].legend(handles=gt_handles, loc="lower left", fontsize=6,
                   frameon=True, title="Clusters", title_fontsize=7)

    # Prediction
    pred_colors = [pred_color_lookup[int(cluster_id)] for cluster_id in pred]
    axes[1].scatter(coords[:, 0], coords[:, 1], c=pred_colors, s=5)
    axes[1].set_title("Prediction")
    axes[1].invert_yaxis()
    axes[1].axis("off")

    pred_handles = [
        Line2D([0], [0], marker='o', color='w',
               label=f"C{int(cluster_id)}: {pred_name_map[int(cluster_id)]}",
               markerfacecolor=pred_color_lookup[int(cluster_id)], markersize=5)
        for cluster_id in pred_clusters
    ]
    axes[1].legend(handles=pred_handles, loc="lower left", fontsize=6,
                   frameon=True, title="Pred Clusters", title_fontsize=7)

    fig.suptitle(common_title, fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    
def evaluate(adata, pred):
    
    gt = adata.obs["Ground Truth"]

    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)

    return ari, nmi

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax


sections = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

results = []


model = transformer.Transformer(
    d_model=256,
    nhead=4,
    nhid=512,
    nlayers=4,
    in_features=5,
    buckets_size=10
).to(device)

checkpoint = torch.load("pfn_hard_5D.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

for sec in sections:
    print(f"Processing section {sec}")

    adata = sc.read_h5ad(f"data/DLPFC_{sec}_ST_final.h5ad")  # adjust path
    gt_labels = adata.obs["Ground Truth"]
    # Pipeline 1
    X1 = pipeline_hvg_pca(adata.copy())
    pred1 = run_pfn(model, X1)
    ari1, nmi1 = evaluate(adata, pred1)

    # Pipeline 2
    X2 = pipeline_spatial_smoothing(adata.copy())
    pred2 = run_pfn(model, X2)
    ari2, nmi2 = evaluate(adata, pred2)

    results.append({
        "section": sec,
        "hvg_ari": ari1,
        "hvg_nmi": nmi1,
        "smooth_ari": ari2,
        "smooth_nmi": nmi2
    })
    save_comparison_plot(
            adata,
            gt_labels,
            pred1,
            f"PFN (HVG + PCA) {sec}",
            f"plots_with_given_k/{sec}_hvg_vs_truth.png"
        )

    save_comparison_plot(
        adata,
        gt_labels,
        pred2,
        f"PFN (Spatial Smoothing) {sec}",
        f"plots_with_given_k/{sec}_smoothing_vs_truth.png"
    )


df = pd.DataFrame(results)
fig,ax = render_mpl_table(df, header_columns=0, col_width=4.0)
fig.savefig("table_mpl1.png")
print(df)