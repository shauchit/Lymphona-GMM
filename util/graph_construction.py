"""
graph_construction.py
=====================
Core graph-construction utilities for H&E pathology images.

Two strategies turn an H&E patch into a ``torch_geometric.data.Data`` object,
ready to feed into a downstream GNN:

Strategy 1 — CELL-GRAPH
    Nuclei detection (StarDist or watershed fallback)
    -> Delaunay triangulation with distance-threshold pruning
    -> 10-dim morphological + colour node features

Strategy 2 — PATCH-GRAPH
    SLIC superpixel segmentation
    -> k-NN connectivity on centroid coordinates
    -> 10-dim colour + texture node features

Public API
----------
    load_image(path)                  -> RGB uint8 array
    build_cell_graph(rgb, ...)        -> (Data, GraphStats)
    build_patch_graph(rgb, ...)       -> (Data, GraphStats)
    process_dataset(image_dir, ...)   -> dict of graphs + stats
    summarise_stats(stats)            -> dict

Adapted from the ISEF lymphoma graph-construction pipeline
(original author: Shaunak Chitnis).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import Delaunay, KDTree
from skimage import color, segmentation
from skimage.feature import local_binary_pattern
from torch_geometric.data import Data

from .nuclei import FEATURE_DIM, detect_nuclei, load_image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Default hyper-parameters (override per-call via function arguments)
# ─────────────────────────────────────────────────────────────────────────────
DELAUNAY_DIST_THRESHOLD_PX = 150   # prune Delaunay edges longer than this
SLIC_N_SEGMENTS            = 150   # target superpixel count
SLIC_COMPACTNESS           = 20    # shape vs colour trade-off
KNN_K                      = 6     # neighbours in patch-graph
LBP_RADIUS                 = 3
LBP_N_POINTS               = 8 * LBP_RADIUS
# FEATURE_DIM is the cell-graph node-feature width, imported from .nuclei
# (currently 18). Patch-graph features are a separate 10-dim set.


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GraphStats:
    strategy:          str
    num_nodes:         int
    num_edges:         int
    avg_degree:        float
    node_feature_dim:  int
    construction_time: float          # seconds
    image_shape:       Tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 1 — CELL-GRAPH
#
#  Nuclei detection + feature extraction live in the torch-free ``util.nuclei``
#  module (StarDist/TF and torch deadlock in one process). build_cell_graph runs
#  detection in-process with method="watershed" by default; for StarDist, segment
#  offline (scripts/segment_nuclei.py) and use build_cell_graph_from_nuclei.
# ─────────────────────────────────────────────────────────────────────────────
def _cell_graph_from_arrays(
    centroids: np.ndarray,
    feats: np.ndarray,
    label_y: int,
    image_shape: Tuple[int, int],
    dist_threshold: float,
    t0: float,
) -> Tuple[Data, GraphStats]:
    """Build a cell-graph PyG Data + stats from centroids/features."""
    if len(centroids) < 3:
        # Degenerate — too few nuclei; return a trivial single-node graph
        data = Data(
            x          = torch.zeros((1, FEATURE_DIM), dtype=torch.float),
            edge_index = torch.zeros((2, 0), dtype=torch.long),
            y          = torch.tensor([label_y], dtype=torch.long),
            pos        = torch.zeros((1, 2), dtype=torch.float),
        )
        stats = GraphStats("cell_graph", 1, 0, 0.0, FEATURE_DIM,
                           time.perf_counter() - t0, image_shape)
        return data, stats

    # Delaunay triangulation → candidate edges (pruned by distance)
    tri = Delaunay(centroids)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                d    = np.linalg.norm(centroids[a] - centroids[b])
                if d <= dist_threshold:
                    edges.add((min(a, b), max(a, b)))

    edge_index = _edges_to_index(edges)
    data = Data(
        x          = torch.tensor(feats,     dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor([label_y], dtype=torch.long),
        pos        = torch.tensor(centroids, dtype=torch.float),
    )
    stats = GraphStats(
        strategy          = "cell_graph",
        num_nodes         = len(centroids),
        num_edges         = edge_index.shape[1] // 2,
        avg_degree        = edge_index.shape[1] / len(centroids),
        node_feature_dim  = FEATURE_DIM,
        construction_time = time.perf_counter() - t0,
        image_shape       = image_shape,
    )
    return data, stats


def build_cell_graph_from_nuclei(
    centroids: np.ndarray,
    feats: np.ndarray,
    label_y: int = -1,
    image_shape: Tuple[int, int] = (0, 0),
    dist_threshold: float = DELAUNAY_DIST_THRESHOLD_PX,
) -> Tuple[Data, GraphStats]:
    """
    Build a cell-graph from precomputed nuclei (centroids + 10-dim features).

    Use this with nuclei produced offline by scripts/segment_nuclei.py (StarDist),
    so the torch process never imports TensorFlow.
    """
    centroids = np.asarray(centroids, dtype=np.float32).reshape(-1, 2)
    feats     = np.asarray(feats,     dtype=np.float32).reshape(-1, FEATURE_DIM)
    return _cell_graph_from_arrays(
        centroids, feats, label_y, image_shape, dist_threshold, time.perf_counter())


def build_cell_graph(
    rgb: np.ndarray,
    dist_threshold: float = DELAUNAY_DIST_THRESHOLD_PX,
    label_y: int = -1,
    method: str = "watershed",
) -> Tuple[Data, GraphStats]:
    """
    Strategy 1: cell-graph via nuclei detection + Delaunay triangulation.

    Parameters
    ----------
    rgb            : H×W×3 uint8 image
    dist_threshold : prune Delaunay edges longer than this many pixels
    label_y        : graph-level class label (-1 = unknown)
    method         : "watershed" (default, safe in a torch process) or
                     "stardist". StarDist deadlocks alongside torch — only pass
                     "stardist" from a torch-free process; otherwise segment
                     offline and use build_cell_graph_from_nuclei.

    Returns
    -------
    data  : torch_geometric.data.Data  (x, edge_index, y, pos)
    stats : GraphStats
    """
    t0 = time.perf_counter()
    centroids, feats = detect_nuclei(rgb, method=method)
    return _cell_graph_from_arrays(
        centroids, feats, label_y, rgb.shape[:2], dist_threshold, t0)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 2 — PATCH-GRAPH
# ─────────────────────────────────────────────────────────────────────────────
def _extract_patch_features(
    rgb: np.ndarray, seg_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each SLIC superpixel compute a 10-dim feature vector.

    Features:
      0-2 mean R/G/B   3-5 std R/G/B   6 LBP mean   7 LBP var
      8 area fraction  9 mean luminance (CIELab L*)

    Returns
    -------
    centroids : (N, 2) float  [row, col]
    features  : (N, 10) float32
    """
    gray      = color.rgb2gray(rgb)
    lbp_map   = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    lab_image = color.rgb2lab(rgb / 255.0)
    total_px  = rgb.shape[0] * rgb.shape[1]

    centroids, feats = [], []
    for region_id in np.unique(seg_map):
        mask       = seg_map == region_id
        pix_rgb    = rgb[mask].astype(np.float32) / 255.0
        pix_lbp    = lbp_map[mask]
        pix_lab    = lab_image[mask]
        rows, cols = np.where(mask)
        cy, cx     = rows.mean(), cols.mean()

        feat = np.array([
            pix_rgb[:, 0].mean(),
            pix_rgb[:, 1].mean(),
            pix_rgb[:, 2].mean(),
            pix_rgb[:, 0].std(),
            pix_rgb[:, 1].std(),
            pix_rgb[:, 2].std(),
            pix_lbp.mean() / (LBP_N_POINTS + 2),   # LBP mean (normalised)
            min(pix_lbp.var() / 100.0, 1.0),       # LBP var  (clamped)
            mask.sum() / total_px,                 # area fraction
            pix_lab[:, 0].mean() / 100.0,          # L* luminance [0,1]
        ], dtype=np.float32)

        centroids.append([cy, cx])
        feats.append(feat)

    return np.array(centroids, dtype=np.float32), np.array(feats, dtype=np.float32)


def build_patch_graph(
    rgb: np.ndarray,
    n_segments: int    = SLIC_N_SEGMENTS,
    compactness: float = SLIC_COMPACTNESS,
    k: int             = KNN_K,
    label_y: int       = -1,
) -> Tuple[Data, GraphStats]:
    """
    Strategy 2: patch-graph via SLIC superpixels + spatial k-NN.

    Parameters
    ----------
    rgb         : H×W×3 uint8 image
    n_segments  : target number of SLIC superpixels
    compactness : SLIC compactness parameter
    k           : k-NN neighbourhood size
    label_y     : graph-level class label

    Returns
    -------
    data  : torch_geometric.data.Data  (x, edge_index, y, pos)
    stats : GraphStats
    """
    t0 = time.perf_counter()

    # 1. SLIC segmentation
    rgb_f   = rgb.astype(np.float64) / 255.0
    seg_map = segmentation.slic(
        rgb_f, n_segments=n_segments, compactness=compactness,
        channel_axis=2, start_label=0,
    )

    # 2. Per-superpixel features and centroids
    centroids, feats = _extract_patch_features(rgb, seg_map)
    N = len(centroids)

    # 3. k-NN on centroid Euclidean distance (query k+1 to skip self)
    tree    = KDTree(centroids)
    _, idx  = tree.query(centroids, k=min(k + 1, N))
    idx     = np.atleast_2d(idx)
    edges   = set()
    for i, neighbours in enumerate(idx):
        for j in neighbours[1:]:           # skip self (index 0)
            edges.add((min(int(i), int(j)), max(int(i), int(j))))

    edge_index = _edges_to_index(edges)

    data = Data(
        x          = torch.tensor(feats,     dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor([label_y], dtype=torch.long),
        pos        = torch.tensor(centroids, dtype=torch.float),
    )

    stats = GraphStats(
        strategy          = "patch_graph",
        num_nodes         = N,
        num_edges         = edge_index.shape[1] // 2,
        avg_degree        = (edge_index.shape[1] / N) if N > 0 else 0.0,
        node_feature_dim  = feats.shape[1] if feats.ndim == 2 else 0,
        construction_time = time.perf_counter() - t0,
        image_shape       = rgb.shape[:2],
    )
    return data, stats


# ─────────────────────────────────────────────────────────────────────────────
#  Edge helper
# ─────────────────────────────────────────────────────────────────────────────
def _edges_to_index(edges: set) -> torch.Tensor:
    """Undirected edge set -> (2, 2E) bidirectional edge_index tensor."""
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    ei  = np.array(list(edges), dtype=np.int64).T       # (2, E)
    src = np.concatenate([ei[0], ei[1]])
    dst = np.concatenate([ei[1], ei[0]])
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch processing — run a strategy over a directory of patches
# ─────────────────────────────────────────────────────────────────────────────
def process_dataset(
    image_dir: str | Path,
    out_pt_path: Optional[str | Path] = None,
    label_map: Optional[Dict[str, int]] = None,
    strategies: Tuple[str, ...] = ("cell", "patch"),
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    verbose: bool = True,
) -> Dict[str, List]:
    """
    Run the chosen strategy/strategies on every image in ``image_dir``.

    Parameters
    ----------
    image_dir   : folder with image files (searched recursively)
    out_pt_path : optional path to save the resulting graphs as a .pt file
    label_map   : dict mapping a path substring -> int label
    strategies  : any of {"cell", "patch"}
    extensions  : accepted image file extensions
    verbose     : print progress

    Returns
    -------
    dict with keys 'cell_graphs'/'patch_graphs' and matching '*_stats'
    (only for the requested strategies)
    """
    image_dir = Path(image_dir)
    paths     = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in extensions)
    if verbose:
        print(f"Found {len(paths)} images in {image_dir}")

    do_cell  = "cell" in strategies
    do_patch = "patch" in strategies
    cell_graphs, patch_graphs = [], []
    cell_stats,  patch_stats  = [], []

    for i, path in enumerate(paths):
        # Infer label from any matching substring in the path
        label_y = -1
        if label_map:
            for key, val in label_map.items():
                if key.lower() in str(path).lower():
                    label_y = val
                    break

        try:
            rgb = load_image(path)
        except Exception as e:           # noqa: BLE001 - skip unreadable files
            if verbose:
                print(f"  [skip] {path.name}: {e}")
            continue

        if do_cell:
            cg, cs = build_cell_graph(rgb, label_y=label_y)
            cell_graphs.append(cg); cell_stats.append(cs)
        if do_patch:
            pg, ps = build_patch_graph(rgb, label_y=label_y)
            patch_graphs.append(pg); patch_stats.append(ps)

        if verbose and (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(paths)}")

    out: Dict[str, List] = {}
    if do_cell:
        out["cell_graphs"]  = cell_graphs
        out["cell_stats"]   = cell_stats
    if do_patch:
        out["patch_graphs"] = patch_graphs
        out["patch_stats"]  = patch_stats

    if out_pt_path is not None:
        payload = {k: out[k] for k in ("cell_graphs", "patch_graphs") if k in out}
        torch.save(payload, str(out_pt_path))
        if verbose:
            print(f"Saved graphs → {out_pt_path}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics summary
# ─────────────────────────────────────────────────────────────────────────────
def summarise_stats(stats: List[GraphStats]) -> Dict:
    """Return mean / std of key metrics over a list of GraphStats."""
    nodes   = [s.num_nodes  for s in stats]
    edges   = [s.num_edges  for s in stats]
    degrees = [s.avg_degree for s in stats]
    times   = [s.construction_time for s in stats]
    return {
        "strategy":         stats[0].strategy,
        "n_graphs":         len(stats),
        "nodes_mean±std":   f"{np.mean(nodes):.1f} ± {np.std(nodes):.1f}",
        "edges_mean±std":   f"{np.mean(edges):.1f} ± {np.std(edges):.1f}",
        "avg_deg_mean±std": f"{np.mean(degrees):.2f} ± {np.std(degrees):.2f}",
        "time_mean_s":      f"{np.mean(times):.3f}",
        "feature_dim":      stats[0].node_feature_dim,
    }
