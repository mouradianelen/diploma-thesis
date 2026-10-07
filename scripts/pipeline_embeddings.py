"""Run ClusterPFN inference on graph/spatial embeddings (STAGATE, GraphST, SpaGCN).

Instead of feeding raw HVG/PCA features to the PFN, we feed the embeddings learned
by the spatial encoders and let the PFN act as the clustering head (replacing KMeans).
Embeddings are cached to disk so the encoders only run once per section.
"""

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TensorFlow info/warning noise

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import clusterpfn.transformer as transformer


def log(msg):
    print(msg, flush=True)

# STAGATE/SpaGCN/GraphST still use the removed scipy `.A` attribute.
if not hasattr(sp.csr_matrix, "A"):
    sp.csr_matrix.A = property(lambda self: self.toarray())
if not hasattr(sp.csc_matrix, "A"):
    sp.csc_matrix.A = property(lambda self: self.toarray())

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

SECTIONS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

MODEL_PATH = "models/pfn_hard_5D.pt"
IN_FEATURES = 5
BUCKETS_SIZE = 10
EMB_DIR = "experiment_c_embeddings"
RESULTS_FILE = "results/experiment_c_pfn_results.csv"
STAGATE_EPOCHS = int(os.environ.get("STAGATE_EPOCHS", 1000))  # lower (e.g. 50) for a fast smoke test


# -----------------------------
# Encoders (mirrors embedding.ipynb)
# -----------------------------

def prepare_expression(adata, n_hvg=3000):
    adata = adata.copy()
    # Data is already normalized/log-transformed; do not normalize_total/log1p.
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def run_stagate(adata, k=6):
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()  # STAGATE relies on TF1 graph mode
    import STAGATE

    adata = prepare_expression(adata)
    log("      STAGATE: building spatial net")
    STAGATE.Cal_Spatial_Net(adata, k_cutoff=k, model="KNN", verbose=False)
    log(f"      STAGATE: training ({STAGATE_EPOCHS} epochs)")
    STAGATE.train_STAGATE(adata, hidden_dims=[512, 30], n_epochs=STAGATE_EPOCHS,
                          alpha=0, random_seed=42, verbose=True)
    return np.asarray(adata.obsm["STAGATE"])


def run_graphst(adata):
    from GraphST.GraphST import GraphST as GraphSTModel

    import GraphST

    adata = prepare_expression(adata)
    GraphST.construct_interaction(adata, n_neighbors=3)
    model = GraphSTModel(
        adata, device=torch.device("cpu"), learning_rate=0.001, weight_decay=0.0,
        epochs=10, dim_input=3000, dim_output=64, random_seed=42,
        alpha=10, beta=1, theta=0.1, lamda1=10, lamda2=1,
        deconvolution=False, datatype="10X",
    )
    model.train()
    return model.hiden_feat.detach().cpu().numpy()


def run_spagcn(adata):
    import SpaGCN

    adata = adata.copy()
    coords = np.asarray(adata.obsm["spatial"])
    x = coords[:, 0].tolist()
    y = coords[:, 1].tolist()

    adj = SpaGCN.calculate_adj_matrix(x=x, y=y, histology=False)
    l = SpaGCN.search_l(0.5, adj)
    if l is None:
        raise RuntimeError("SpaGCN search_l failed to find l.")

    clf = SpaGCN.SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, num_pcs=50, init="louvain", res=0.4, tol=1e-3)

    z, _ = clf.model.predict(clf.embed, clf.adj_exp)
    return z.detach().cpu().numpy()


ENCODERS = {"STAGATE": run_stagate, "GraphST": run_graphst, "SpaGCN": run_spagcn}


def get_embedding(method, sec, adata):
    """Return the method's embedding for a section, computing and caching it if needed."""
    os.makedirs(EMB_DIR, exist_ok=True)
    cache = os.path.join(EMB_DIR, f"{sec}_{method}_embedding.npy")
    if os.path.exists(cache):
        log(f"    {method}: loaded cached embedding")
        return np.load(cache)
    log(f"    {method}: computing embedding (cache miss)")
    t0 = time.time()
    emb = ENCODERS[method](adata)
    np.save(cache, emb)
    log(f"    {method}: done in {time.time() - t0:.1f}s, shape {emb.shape}")
    return emb


# -----------------------------
# Clustering heads
# -----------------------------

def prepare_for_pfn(embedding, in_features=IN_FEATURES):
    """Reduce an embedding to the PFN input width and scale to the prior's [0, 1] range."""
    X = StandardScaler().fit_transform(np.asarray(embedding))
    if X.shape[1] > in_features:
        X = PCA(n_components=in_features, random_state=42).fit_transform(X)
    X = MinMaxScaler().fit_transform(X)
    return X.astype(np.float32)


def run_pfn(model, X, cluster_count=0):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)  # (S, 1, F)

    max_bucket = model.embedding.num_embeddings - 1
    if not (0 <= cluster_count <= max_bucket):
        raise ValueError(f"cluster_count must be in [0, {max_bucket}]")

    num_clusters = torch.full((X_tensor.shape[0] + 1, 1), cluster_count, dtype=torch.long, device=device)
    with torch.no_grad():
        output_points, _ = model(X_tensor, num_clusters)
    return output_points.squeeze(1).argmax(dim=-1).cpu().numpy()


def cluster_kmeans(embedding, n_clusters, seed=42):
    X = StandardScaler().fit_transform(np.asarray(embedding))
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(X)


def cluster_louvain(embedding, n_clusters, seed=42, n_neighbors=15, max_iter=25):
    """Louvain on a kNN graph; binary-search the resolution to hit n_clusters communities."""
    ad = sc.AnnData(StandardScaler().fit_transform(np.asarray(embedding)).astype(np.float32))
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X", random_state=seed)

    lo, hi, best = 0.1, 3.0, None
    for _ in range(max_iter):
        res = (lo + hi) / 2
        sc.tl.louvain(ad, resolution=res, random_state=seed)
        labels = ad.obs["louvain"].astype(int).to_numpy()
        found = len(np.unique(labels))
        if found == n_clusters:
            return labels
        best = labels  # keep closest attempt
        if found < n_clusters:
            lo = res
        else:
            hi = res
    return best


def evaluate(adata, pred):
    gt = adata.obs["Ground Truth"].astype(str)
    return adjusted_rand_score(gt, pred), normalized_mutual_info_score(gt, pred)


def load_model():
    model = transformer.Transformer(
        d_model=256, nhead=4, nhid=512, nlayers=4,
        in_features=IN_FEATURES, buckets_size=BUCKETS_SIZE,
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    model = load_model()
    results = []

    for sec in SECTIONS:
        log(f"\n=== Section {sec} ===")
        adata = sc.read_h5ad(f"data/DLPFC_{sec}_ST_final.h5ad")
        n_clusters = adata.obs["Ground Truth"].nunique()

        row = {"section": sec, "n_spots": adata.n_obs, "n_clusters": n_clusters}

        for method in ENCODERS:
            try:
                emb = get_embedding(method, sec, adata)
                X = prepare_for_pfn(emb)

                # PFN conditioned on the known k (fair vs KMeans given k).
                pfn_pred = run_pfn(model, X, cluster_count=int(n_clusters))
                pfn_ari, pfn_nmi = evaluate(adata, pfn_pred)
                pfn_k = int(np.unique(pfn_pred).size)  # collapse diagnostic

                km_pred = cluster_kmeans(emb, n_clusters)
                km_ari, km_nmi = evaluate(adata, km_pred)

                lou_pred = cluster_louvain(emb, n_clusters)
                lou_ari, lou_nmi = evaluate(adata, lou_pred)
                lou_k = int(np.unique(lou_pred).size)

                row[f"{method}_PFN_ARI"] = pfn_ari
                row[f"{method}_PFN_NMI"] = pfn_nmi
                row[f"{method}_PFN_pred_k"] = pfn_k
                row[f"{method}_KMeans_ARI"] = km_ari
                row[f"{method}_KMeans_NMI"] = km_nmi
                row[f"{method}_Louvain_ARI"] = lou_ari
                row[f"{method}_Louvain_NMI"] = lou_nmi
                row[f"{method}_Louvain_pred_k"] = lou_k
                log(f"    {method:8s} | PFN ARI {pfn_ari:.3f} NMI {pfn_nmi:.3f} "
                    f"pred_k {pfn_k}/{n_clusters} | KMeans ARI {km_ari:.3f} NMI {km_nmi:.3f} "
                    f"| Louvain ARI {lou_ari:.3f} NMI {lou_nmi:.3f} pred_k {lou_k}/{n_clusters}")
            except Exception as e:  # keep going if one encoder fails
                log(f"    {method} ERROR: {e!r}")
                for col in (f"{method}_PFN_ARI", f"{method}_PFN_NMI", f"{method}_PFN_pred_k",
                            f"{method}_KMeans_ARI", f"{method}_KMeans_NMI",
                            f"{method}_Louvain_ARI", f"{method}_Louvain_NMI", f"{method}_Louvain_pred_k"):
                    row[col] = np.nan

        results.append(row)
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    df = pd.DataFrame(results)
    print("\n", df.round(3).to_string(index=False))
    print(f"\nSaved -> {RESULTS_FILE}")
    return df


if __name__ == "__main__":
    main()
