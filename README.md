# Spatial-domain clustering of DLPFC spatial transcriptomics

Thesis project that investigates different spatially resolved transcriptomics (SRT) methods
and clustering approaches, including ClusterPFN, on 10x Visium DLPFC sections.

## Repository layout

```
clusterpfn/        Core model: transformer, synthetic prior, training entry point, utils
scripts/           Batch pipelines and table/figure generators (run as modules from the root)
notebooks/         Analysis notebooks (data analysis, embedding generation, fusion, graph stats, ...)
models/            Trained ClusterPFN checkpoints (pfn_hard_5D.pt is the working model)
data/              DLPFC .h5ad sections (downloaded separately, see below)
figures/           Exported figures
results/           CSV / pickle / JSON / LaTeX result artifacts
```

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `requirements.txt` pins STAGATE to a specific GitHub commit and includes TensorFlow
(needed only for STAGATE). The remaining encoders (SpaGCN, GraphST) are PyTorch based.

## Data

The processed DLPFC sections are **not stored in Git** because they total roughly 4 GB.
They are published on figshare and must be downloaded into `data/`:

- Dataset: **DLPFC data**, Lihua Zhang (2025), figshare, CC BY 4.0
- DOI: https://doi.org/10.6084/m9.figshare.29146307.v1
- Page: https://figshare.com/articles/dataset/DLPFC_data/29146307

The dataset contains 12 sections, one `.h5ad` file each
(`DLPFC_1515{07..10}` and `DLPFC_1516{69..76}`), about 280 to 460 MB per file.

### Option A: download helper script (recommended)

The helper queries the figshare API, downloads every file into `data/`, verifies each
MD5 checksum, and skips files that are already present:

```bash
python scripts/download_data.py
```

### Option B: download the whole archive as a single zip

```bash
mkdir -p data
curl -L "https://ndownloader.figshare.com/articles/29146307/versions/1" -o data/dlpfc.zip
unzip data/dlpfc.zip -d data/
rm data/dlpfc.zip
```

### Option C: download individual files

Each file has a direct `ndownloader` URL, for example section 151507:

```bash
mkdir -p data
curl -L "https://ndownloader.figshare.com/files/54819434" -o data/DLPFC_151507_ST_final.h5ad
```

After downloading, `data/` should contain the twelve `DLPFC_<section>_ST_final.h5ad` files.
The notebooks and scripts load them with paths like `data/DLPFC_151507_ST_final.h5ad`.

## Usage

Run scripts and notebooks from the repository root so the relative `data/`, `models/`,
`figures/`, and `results/` paths resolve correctly.

Scripts are modules:

```bash
python -m scripts.pipeline_embeddings      # ClusterPFN vs KMeans vs Louvain on encoder embeddings
python -m scripts.make_pfn_table           # render result tables
```

Train a ClusterPFN model (synthetic data is generated on the fly):

```bash
python -m clusterpfn.main --in_features 5 --num_epochs 300000 --check_point pfn_hard_5D
```

Notebooks assume the working directory is the repository root
(`jupyter.notebookFileRoot` is set to the workspace folder in `.vscode/settings.json`);
restart the kernel after cloning so this takes effect.

## Citation

If you use the DLPFC data, cite the figshare dataset:

> Zhang, Lihua (2025). DLPFC data. figshare. Dataset. https://doi.org/10.6084/m9.figshare.29146307.v1
