# Lymphoma-GMM

Graph-based pipeline for **lymphoma subtype classification** (CLL / FL / MCL)
from H&E pathology images. H&E patches are converted into graphs and fed to a
graph neural network (GNN).

End-to-end pipeline:

1. **Download** the datasets (`scripts/download_data.py`)
2. **Construct graphs** from the images (`util/` package + `scripts/build_graphs.py`)
3. **Train** a GAT graph classifier (`models/` package + `scripts/train_gnn.py`)

---

## Setup

The project uses [uv](https://docs.astral.sh/uv/) with Python ≥ 3.12.

```bash
uv sync                 # core dependencies into .venv
uv sync --extra stardist  # + StarDist/TensorFlow (optional; see "StarDist nuclei")
```

Run any command with `uv run python ...` (no manual venv activation needed).

---

## 1. Download datasets

```bash
uv run python scripts/download_data.py                 # both datasets → ./data
uv run python scripts/download_data.py --dataset lymphoma
uv run python scripts/download_data.py --dataset nct --nct-val-only
```

| Flag | Meaning |
|------|---------|
| `--dataset {lymphoma,nct,all}` | which dataset(s) to fetch (default `all`) |
| `--data-dir PATH` | destination (default `./data`) |
| `--nct-val-only` | for NCT, fetch only the smaller 800 MB validation set |
| `--keep-zips` | keep the downloaded `.zip` archives after extraction |

### Datasets

| Name | Content | Source | Notes |
|------|---------|--------|-------|
| **lymphoma** | Malignant Lymphoma, 374 images, CLL/FL/MCL | Kaggle `andrewmvd/malignant-lymphoma-classification` | primary task; needs Kaggle credentials |
| **nct** | NCT-CRC-HE-100K + CRC-VAL-HE-7K, 224×224 H&E | Zenodo record `1214456` | public benchmark; train zip ≈ 11.7 GB |

**Kaggle credentials** (lymphoma only): create a token at
<https://www.kaggle.com/settings> → *Create New Token*, then save the file to
`~/.kaggle/kaggle.json` (`chmod 600`). Alternatively set `KAGGLE_USERNAME` /
`KAGGLE_KEY` env vars.

Resulting layout:

```
data/
├── lymphoma/
│   ├── CLL/   *.tif
│   ├── FL/    *.tif
│   └── MCL/   *.tif
└── nct-crc-he-100k/
    ├── NCT-CRC-HE-100K/
    └── CRC-VAL-HE-7K/
```

---

## 2. Build graphs

Convert a folder of images into `torch_geometric` graphs saved as a `.pt` file.

```bash
uv run python scripts/build_graphs.py                       # cell graphs, full lymphoma set
uv run python scripts/build_graphs.py --strategy patch
uv run python scripts/build_graphs.py --strategy both --limit 30   # quick test on 30 images
uv run python scripts/build_graphs.py \
    --image-dir data/lymphoma --out data/graphs/lymphoma_cell.pt
```

| Flag | Meaning |
|------|---------|
| `--image-dir PATH` | image folder, searched recursively (default `data/lymphoma`) |
| `--nuclei-cache PATH` | build cell graphs from a `segment_nuclei.py` cache (StarDist; see below) |
| `--out PATH` | output `.pt` (default `data/graphs/<dir>_<strategy>.pt`) |
| `--strategy {cell,patch,both}` | which graph(s) to build (default `cell`) |
| `--limit N` | only process the first N images (quick smoke test) |

The output `.pt` is a dict of lists of `Data` objects, e.g.
`{"cell_graphs": [Data, Data, ...]}`. Load it with `torch.load(path)`.

By default `build_graphs.py` detects nuclei with **watershed** (in-process,
no extra deps). For much better nuclei (StarDist), use the two-stage flow below.

### StarDist nuclei (recommended, two-stage)

[StarDist](https://github.com/stardist/stardist) gives far better nuclei than
watershed (~1700 vs ~180 nuclei per image on this dataset). It needs TensorFlow,
which **deadlocks if imported in the same process as PyTorch** (a known macOS
OpenMP issue). So segmentation runs in its own torch-free process first, caching
nuclei to disk; graph building then reads that cache in the torch process.

```bash
# one-time: install the optional StarDist extra (TensorFlow etc.)
uv sync --extra stardist

# stage 1 — StarDist segmentation (torch-free) → nuclei cache  (~5 min, 374 imgs)
uv run python scripts/segment_nuclei.py --out data/graphs/nuclei_stardist.pkl

# stage 2 — build cell graphs from the cache (torch)
uv run python scripts/build_graphs.py --nuclei-cache data/graphs/nuclei_stardist.pkl \
    --out data/graphs/lymphoma_cell_stardist.pt
```

`scripts/segment_nuclei.py` imports only the torch-free `util.nuclei` module
(it asserts torch is never loaded). `--method watershed` is also available there
for an apples-to-apples cache.

---

## 3. Train the GNN

Train the GAT graph classifier on the `.pt` graphs (stratified 70/15/15
train/val/test split, class-weighted loss, best-on-val model selection).

```bash
uv run python scripts/train_gnn.py                                  # defaults
uv run python scripts/train_gnn.py --graphs data/graphs/lymphoma_cell.pt
uv run python scripts/train_gnn.py --layers 6 --hidden 128 --epochs 100
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--graphs PATH` | `data/graphs/lymphoma_cell.pt` | input graphs |
| `--epochs N` | 80 | training epochs |
| `--layers N` | 4 | number of GAT layers (depth) |
| `--hidden N` | 64 | GAT hidden width |
| `--heads N` | 4 | attention heads per layer |
| `--pool` | `mean` | graph readout: `mean` / `max` / `mean+max` |
| `--lr` / `--weight-decay` | 5e-3 / 5e-4 | optimiser settings |
| `--dropout` | 0.5 | dropout probability |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` (prefer `cpu` here) |
| `--out PATH` | `data/graphs/gat_best.pt` | where to save the best model weights |

The model (`models/gat.py`) is a **configurable-depth GAT**: each block keeps a
constant width (`hidden × heads`) with a **residual** connection and
**BatchNorm**, so extra layers don't immediately over-smooth. Training
auto-selects the device (CUDA → MPS → CPU), prints per-class
precision/recall/F1 + a confusion matrix, and saves the best-on-val
`state_dict`.

> **Status — depth does *not* help here.** Sweeping 2/4/6 layers (same seed)
> leaves test accuracy bouncing 0.51–0.56 with no trend, and macro-F1 slightly
> *decreasing* with depth; deeper runs just move which class collapses rather
> than separating all three. With only ~57 test graphs the split is very noisy,
> so these differences aren't meaningful. The bottleneck is data/features/
> pooling, not depth — see *Next steps*.

| Depth | Params | Test acc | Macro-F1 |
|-------|--------|----------|----------|
| 2 | 71k  | 0.509 | 0.484 |
| 4 | 205k | 0.509 | 0.439 |
| 6 | 339k | 0.561 | 0.433 |

The model lives in `models/gat.py` and is importable:

```python
from models import GATGraphClassifier
model = GATGraphClassifier(in_channels=10, num_classes=3, heads=4, num_layers=4)
```

> **Tip:** on Apple Silicon, prefer `--device cpu`. These graphs are tiny
> (~180 nodes), so MPS per-op overhead makes it *much* slower than CPU here
> (a 5-fold run is ~4 min on CPU vs ~25 min on MPS).

### Cross-validation (recommended for real estimates)

A single split is too noisy to trust. `scripts/cross_validate.py` runs
stratified k-fold CV: each fold is held out as the test set once, a small
stratified slice of the rest is the validation set, and every graph gets exactly
one out-of-fold (OOF) prediction.

```bash
uv run python scripts/cross_validate.py --folds 5 --layers 2 --device cpu
```

It reports per-fold test accuracy + macro-F1, the mean ± std across folds, and
an OOF classification report + confusion matrix pooled over all 374 graphs.

**5-fold result** (`--layers 2`, 80 epochs): **accuracy 0.543 ± 0.013**,
macro-F1 0.479 ± 0.029. The tight ±0.013 std confirms the single-split swings
were just noise — true performance is ~54%. The OOF confusion matrix shows a
**systematic FL bias** (FL recall 0.91) with CLL and MCL frequently confused
(MCL recall only 0.26) — a consistent, real weakness, not a one-split artefact.

**Pooling sweep** (same config, CV): `mean` / `max` / `mean+max` give accuracy
0.543 / 0.532 / 0.537 and macro-F1 0.479 / 0.462 / 0.487 — all within one
std of each other. The readout is **not** the bottleneck; richer pooling only
trades CLL recall against MCL recall without a net gain. Default stays `mean`.
The remaining bottleneck is upstream — the **node features / graph construction**
don't carry enough signal to separate CLL from MCL.

**StarDist nuclei — the change that worked** (same CV config, graphs from
`lymphoma_cell_stardist.pt`): replacing watershed with StarDist nuclei (~1700 vs
~180 nuclei/image) lifts accuracy **0.543 → 0.612** and macro-F1 **0.479 →
0.583**. The gain is exactly where CV said the problem was — CLL↔MCL:

| Nuclei | Accuracy | Macro-F1 | CLL-F1 | FL-F1 | MCL-F1 |
|--------|----------|----------|--------|-------|--------|
| watershed | 0.543 ± 0.013 | 0.479 | 0.436 | 0.720 | 0.332 |
| **StarDist** | **0.612 ± 0.063** | **0.583** | **0.518** | 0.706 | **0.562** |

MCL F1 nearly doubles (0.33 → 0.56) and the FL bias eases. This confirms the
CV-driven diagnosis: **better nuclei, not a bigger/deeper model, is what moved
the needle.** (StarDist's per-fold variance is higher, ±0.063, so the exact
number is noisier — but the central improvement is real and consistent.)

```bash
uv run python scripts/cross_validate.py \
    --graphs data/graphs/lymphoma_cell_stardist.pt --folds 5 --layers 2 --device cpu
```

---

## 4. The `util` package

The graph-construction logic lives in `util/` and can be imported directly when
you build the GNN.

```python
from util import load_image, build_cell_graph, build_patch_graph

rgb = load_image("data/lymphoma/CLL/sj-03-2810_001.tif")
data, stats = build_cell_graph(rgb, label_y=0)

data.x          # [N, 10] node features
data.edge_index # [2, 2E] bidirectional edges
data.y          # [1]     graph-level label
data.pos        # [N, 2]  node (row, col) coordinates
```

Batch a whole directory in code (equivalent to `build_graphs.py`):

```python
from util import process_dataset

process_dataset(
    "data/lymphoma",
    out_pt_path="data/graphs/lymphoma_cell.pt",
    label_map={"CLL": 0, "FL": 1, "MCL": 2},   # matched by path substring
    strategies=("cell",),                      # or ("patch",) / ("cell", "patch")
)
```

### Two graph-construction strategies

| | **Cell-Graph** | **Patch-Graph** |
|--|----------------|-----------------|
| Nodes | individual nuclei (StarDist two-stage, or in-process watershed) | SLIC superpixels (~150) |
| Edges | Delaunay triangulation, pruned at 150 px | k-NN (k=6) on centroids |
| Features (10-dim) | morphology + mean RGB | colour mean/std + LBP texture + luminance |

> **Note:** both strategies emit 10-dim node features, so a GNN uses
> `in_channels=10` — but the two feature spaces are **not** interchangeable
> (cell = morphology, patch = colour/texture). Don't mix graphs from the two
> strategies in one model.

### Visual inspection / demo

`util/visualization.py` provides plotting helpers and a synthetic-patch
generator. Run the self-contained demo (no dataset needed):

```bash
uv run python -m util.demo      # writes figures to ./util_demo_output/
```

### Package layout

```
util/
├── __init__.py            # lazy public API (importing the package won't load torch)
├── nuclei.py              # TORCH-FREE nuclei detection + features (StarDist/watershed)
├── graph_construction.py  # core: build_cell_graph / build_patch_graph / process_dataset
├── visualization.py       # plotting + synthetic H&E patch generator (optional)
└── demo.py                # smoke test on synthetic patches
```

---

## Project structure

```
.
├── data/                          # downloaded datasets (git-ignored)
├── scripts/
│   ├── download_data.py           # fetch datasets → ./data
│   ├── segment_nuclei.py          # StarDist nuclei → cache (torch-free, stage 1)
│   ├── build_graphs.py            # images/cache → graphs (.pt)
│   ├── train_gnn.py               # train GAT classifier (single split)
│   └── cross_validate.py          # k-fold cross-validation
├── util/                          # graph-construction package
├── models/                        # GNN models (GAT classifier)
├── GNN/                           # original deliverable (pipeline script + report)
├── pyproject.toml                 # uv project + dependencies
└── README.md
```

---

## Next steps

CV ruled out the model side (depth, pooling) and pointed upstream — and
switching watershed → **StarDist nuclei confirmed it** (accuracy 0.54 → 0.61,
MCL-F1 0.33 → 0.56). Build on that:

- **Richer per-nucleus features** now that nuclei are accurate: chromatin/texture
  descriptors, neighbourhood statistics — rebuild the StarDist cache and re-run CV.
- **Reduce StarDist fold variance** (±0.063): more folds / repeated CV, or light
  regularisation tuning, to firm up the estimate.
- Add **edge features/weights** (e.g. inverse centroid distance) to the GAT.
- Compare the **patch-graph** strategy against the StarDist cell-graph (same CV).
- Always evaluate against the **CV** estimate, never a single split.
