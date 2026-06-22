"""
graph_construction_pipeline.py
=================================
Two graph construction strategies for H&E pathology images.

Strategy 1 — CELL-GRAPH
  Nuclei detection (StarDist or watershed fallback)
  → Delaunay triangulation with distance-threshold pruning
  → 10-dim morphological + colour node features

Strategy 2 — PATCH-GRAPH
  SLIC superpixel segmentation
  → k-NN connectivity (k=6) on centroid coordinates
  → 10-dim colour + texture node features

Both produce torch_geometric.data.Data objects.
Both are benchmarked for graph-size statistics and wall-clock cost.

Dataset tested against: NCT-CRC-HE-100K patches (224×224 px, 0.5 µm/px)
Pipeline is generalised to any H&E patch of compatible resolution.

Author: Shaunak Chitnis (adapted for ISEF lymphoma pipeline)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import torch
from scipy.spatial import Delaunay
from skimage import color, exposure, filters, measure, morphology, segmentation
from skimage.feature import local_binary_pattern
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
DELAUNAY_DIST_THRESHOLD_PX = 150   # prune Delaunay edges longer than this
SLIC_N_SEGMENTS             = 150   # target superpixel count
SLIC_COMPACTNESS            = 20    # shape vs colour trade-off
KNN_K                       = 6     # neighbours in patch-graph
LBP_RADIUS                  = 3
LBP_N_POINTS                = 8 * LBP_RADIUS
FEATURE_DIM                 = 10    # shared for both strategies


# ─────────────────────────────────────────────────────────────────────────────
#  Data class for statistics
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
#  Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """Read an H&E image as uint8 RGB (H×W×3)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _safe_norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; handle constant columns."""
    lo, hi = x.min(0), x.max(0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (x - lo) / rng


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 1 — CELL-GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def _detect_nuclei_watershed(rgb: np.ndarray) -> np.ndarray:
    """
    Lightweight nucleus detection when StarDist is unavailable.
    Returns a label image (int32, 0 = background).
    """
    gray   = color.rgb2gray(rgb)
    # Enhance nuclei with a Gaussian difference
    smooth = filters.gaussian(gray, sigma=1.5)
    thresh = filters.threshold_otsu(smooth)
    binary = smooth < thresh                   # nuclei are darker in H&E
    binary = morphology.remove_small_objects(binary, min_size=30)
    binary = morphology.remove_small_holes(binary, area_threshold=100)
    dist   = morphology.binary_erosion(binary, morphology.disk(2)).astype(float)
    # Use distance transform to find seeds
    from scipy.ndimage import distance_transform_edt, label as nd_label
    dist_map = distance_transform_edt(binary)
    seeds, _  = nd_label(dist_map > dist_map.max() * 0.4)
    labels    = segmentation.watershed(-dist_map, seeds, mask=binary)
    return labels.astype(np.int32)


def _try_stardist(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Attempt StarDist 2D_versatile_he; return None on failure."""
    try:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
        model   = StarDist2D.from_pretrained("2D_versatile_he")
        norm    = normalize(rgb, 1, 99.8, axis=(0, 1))
        labels, _ = model.predict_instances(norm)
        return labels.astype(np.int32)
    except Exception:
        return None


def _extract_cell_features(rgb: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each labelled nucleus compute a 10-dim feature vector.

    Features (matching existing ISEF pipeline):
      0  area (normalised)
      1  eccentricity
      2  solidity
      3  perimeter (normalised)
      4  compactness  = 4π·area / perimeter²
      5  orientation  (mapped to [0,1])
      6  extent
      7  mean_R
      8  mean_G
      9  mean_B

    Returns
    -------
    centroids : (N, 2) float  [row, col]
    features  : (N, 10) float32
    """
    props = measure.regionprops(labels, intensity_image=rgb)
    centroids, feats = [], []
    for p in props:
        if p.area < 20:          # skip tiny artefacts
            continue
        cy, cx   = p.centroid
        area     = p.area
        perim    = max(p.perimeter, 1e-6)
        compact  = (4 * np.pi * area) / (perim ** 2)
        # mean colour in the nucleus bounding box
        minr, minc, maxr, maxc = p.bbox
        patch_rgb = rgb[minr:maxr, minc:maxc]
        mask_roi  = labels[minr:maxr, minc:maxc] == p.label
        r_vals    = patch_rgb[:, :, 0][mask_roi]
        g_vals    = patch_rgb[:, :, 1][mask_roi]
        b_vals    = patch_rgb[:, :, 2][mask_roi]
        feat = np.array([
            area,
            p.eccentricity,
            p.solidity,
            perim,
            compact,
            (p.orientation + np.pi / 2) / np.pi,   # → [0,1]
            p.extent,
            r_vals.mean() / 255.0,
            g_vals.mean() / 255.0,
            b_vals.mean() / 255.0,
        ], dtype=np.float32)
        centroids.append([cy, cx])
        feats.append(feat)

    centroids = np.array(centroids, dtype=np.float32)
    feats     = np.array(feats,     dtype=np.float32)
    # Normalise geometric features (cols 0-6) to [0,1]
    feats[:, :7] = _safe_norm(feats[:, :7])
    return centroids, feats


def build_cell_graph(
    rgb: np.ndarray,
    dist_threshold: float = DELAUNAY_DIST_THRESHOLD_PX,
    label_y: int = -1,
) -> Tuple[Data, GraphStats]:
    """
    Strategy 1: Cell-graph via nuclei detection + Delaunay triangulation.

    Parameters
    ----------
    rgb            : H×W×3 uint8 image
    dist_threshold : prune Delaunay edges longer than this many pixels
    label_y        : graph-level class label (-1 = unknown)

    Returns
    -------
    data  : torch_geometric.data.Data
    stats : GraphStats
    """
    t0 = time.perf_counter()

    # 1. Nuclei detection
    labels = _try_stardist(rgb)
    if labels is None:
        labels = _detect_nuclei_watershed(rgb)

    # 2. Per-nucleus features and centroids
    centroids, feats = _extract_cell_features(rgb, labels)
    if len(centroids) < 3:
        # Degenerate — too few nuclei detected; return a trivial graph
        data = Data(
            x       = torch.zeros((1, FEATURE_DIM), dtype=torch.float),
            edge_index = torch.zeros((2, 0), dtype=torch.long),
            y       = torch.tensor([label_y], dtype=torch.long),
        )
        stats = GraphStats("cell_graph", 1, 0, 0.0, FEATURE_DIM, time.perf_counter() - t0, rgb.shape[:2])
        return data, stats

    # 3. Delaunay triangulation → candidate edges
    tri = Delaunay(centroids)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                d    = np.linalg.norm(centroids[a] - centroids[b])
                if d <= dist_threshold:
                    edges.add((min(a, b), max(a, b)))

    # 4. Build edge_index (undirected → two directed edges)
    if edges:
        ei  = np.array(list(edges), dtype=np.int64).T       # (2, E)
        src = np.concatenate([ei[0], ei[1]])
        dst = np.concatenate([ei[1], ei[0]])
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # 5. Assemble PyG Data
    x    = torch.tensor(feats,       dtype=torch.float)
    y    = torch.tensor([label_y],   dtype=torch.long)
    pos  = torch.tensor(centroids,   dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)

    n_edges = edge_index.shape[1] // 2
    avg_deg = (edge_index.shape[1] / len(centroids)) if len(centroids) > 0 else 0.0
    stats   = GraphStats(
        strategy          = "cell_graph",
        num_nodes         = len(centroids),
        num_edges         = n_edges,
        avg_degree        = avg_deg,
        node_feature_dim  = FEATURE_DIM,
        construction_time = time.perf_counter() - t0,
        image_shape       = rgb.shape[:2],
    )
    return data, stats


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 2 — PATCH-GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def _extract_patch_features(rgb: np.ndarray, seg_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each SLIC superpixel compute a 10-dim feature vector.

    Features:
      0  mean_R            (normalised)
      1  mean_G
      2  mean_B
      3  std_R
      4  std_G
      5  std_B
      6  LBP mean          (uniform LBP, captures texture)
      7  LBP variance
      8  area fraction     (superpixel pixels / total pixels)
      9  mean luminance    (CIELab L channel)

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
        mask     = seg_map == region_id
        pix_rgb  = rgb[mask].astype(np.float32) / 255.0
        pix_lbp  = lbp_map[mask]
        pix_lab  = lab_image[mask]
        rows, cols = np.where(mask)
        cy, cx   = rows.mean(), cols.mean()

        feat = np.array([
            pix_rgb[:, 0].mean(),           # mean R
            pix_rgb[:, 1].mean(),           # mean G
            pix_rgb[:, 2].mean(),           # mean B
            pix_rgb[:, 0].std(),            # std  R
            pix_rgb[:, 1].std(),            # std  G
            pix_rgb[:, 2].std(),            # std  B
            pix_lbp.mean() / (LBP_N_POINTS + 2),  # LBP mean (normalised)
            min(pix_lbp.var() / 100.0, 1.0),       # LBP var  (clamped)
            mask.sum() / total_px,                  # area fraction
            pix_lab[:, 0].mean() / 100.0,           # L* luminance [0,1]
        ], dtype=np.float32)

        centroids.append([cy, cx])
        feats.append(feat)

    return np.array(centroids, dtype=np.float32), np.array(feats, dtype=np.float32)


def build_patch_graph(
    rgb: np.ndarray,
    n_segments: int   = SLIC_N_SEGMENTS,
    compactness: float = SLIC_COMPACTNESS,
    k: int            = KNN_K,
    label_y: int      = -1,
) -> Tuple[Data, GraphStats]:
    """
    Strategy 2: Patch-graph via SLIC superpixels + spatial k-NN.

    Parameters
    ----------
    rgb         : H×W×3 uint8 image
    n_segments  : target number of SLIC superpixels
    compactness : SLIC compactness parameter
    k           : k-NN neighbourhood size
    label_y     : graph-level class label

    Returns
    -------
    data  : torch_geometric.data.Data
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

    # 3. k-NN on centroid Euclidean distance
    from scipy.spatial import KDTree
    tree   = KDTree(centroids)
    # query k+1 because the point itself is its own nearest neighbour
    _, idx = tree.query(centroids, k=min(k + 1, N))
    edges  = set()
    for i, neighbours in enumerate(idx):
        for j in neighbours[1:]:           # skip self (index 0)
            edges.add((min(int(i), int(j)), max(int(i), int(j))))

    # 4. Build edge_index
    if edges:
        ei  = np.array(list(edges), dtype=np.int64).T
        src = np.concatenate([ei[0], ei[1]])
        dst = np.concatenate([ei[1], ei[0]])
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    x    = torch.tensor(feats,     dtype=torch.float)
    y    = torch.tensor([label_y], dtype=torch.long)
    pos  = torch.tensor(centroids, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)

    n_edges = edge_index.shape[1] // 2
    avg_deg = (edge_index.shape[1] / N) if N > 0 else 0.0
    stats   = GraphStats(
        strategy          = "patch_graph",
        num_nodes         = N,
        num_edges         = n_edges,
        avg_degree        = avg_deg,
        node_feature_dim  = FEATURE_DIM,
        construction_time = time.perf_counter() - t0,
        image_shape       = rgb.shape[:2],
    )
    return data, stats


# ─────────────────────────────────────────────────────────────────────────────
#  Batch processing — run both strategies over a directory of patches
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    image_dir: str | Path,
    out_pt_path: str | Path,
    label_map: Optional[Dict[str, int]] = None,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".tif", ".tiff"),
) -> Dict[str, List]:
    """
    Run both strategies on every image in `image_dir`.

    Parameters
    ----------
    image_dir   : folder with image files
    out_pt_path : path to save {cell_graphs, patch_graphs} .pt
    label_map   : dict mapping folder/filename pattern to int label
    extensions  : accepted image file extensions

    Returns
    -------
    dict with keys 'cell_graphs', 'patch_graphs',
    'cell_stats', 'patch_stats'
    """
    image_dir   = Path(image_dir)
    paths       = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in extensions])
    print(f"Found {len(paths)} images in {image_dir}")

    cell_graphs, patch_graphs = [], []
    cell_stats,  patch_stats  = [], []

    for i, path in enumerate(paths):
        # Infer label from parent folder name
        label_y = -1
        if label_map:
            for key, val in label_map.items():
                if key.lower() in str(path).lower():
                    label_y = val
                    break

        try:
            rgb = load_image(path)
        except Exception as e:
            print(f"  [skip] {path.name}: {e}")
            continue

        cg, cs = build_cell_graph(rgb,  label_y=label_y)
        pg, ps = build_patch_graph(rgb, label_y=label_y)
        cell_graphs.append(cg);  cell_stats.append(cs)
        patch_graphs.append(pg); patch_stats.append(ps)

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(paths)}")

    torch.save({"cell_graphs": cell_graphs, "patch_graphs": patch_graphs},
               str(out_pt_path))
    print(f"Saved graphs → {out_pt_path}")

    return {
        "cell_graphs":  cell_graphs,
        "patch_graphs": patch_graphs,
        "cell_stats":   cell_stats,
        "patch_stats":  patch_stats,
    }


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


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _make_nx_graph(data: Data) -> nx.Graph:
    """Convert torch_geometric Data → networkx Graph with position metadata."""
    G = nx.Graph()
    pos_arr = data.pos.numpy() if data.pos is not None else np.zeros((data.num_nodes, 2))
    for i in range(data.num_nodes):
        G.add_node(i, pos=(pos_arr[i, 1], pos_arr[i, 0]))   # (x=col, y=row)
    ei = data.edge_index.numpy()
    for k in range(ei.shape[1]):
        G.add_edge(int(ei[0, k]), int(ei[1, k]))
    return G


def visualise_comparison(
    rgb: np.ndarray,
    cell_data: Data,
    patch_data: Data,
    seg_map: Optional[np.ndarray] = None,
    labels_mask: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
    title: str = "",
) -> plt.Figure:
    """
    4-panel figure: image | cell-graph overlay | superpixel map | patch-graph overlay.
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=130)

    # Panel 0 — raw image
    axes[0].imshow(rgb)
    axes[0].set_title("H&E Patch", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 1 — cell-graph overlay
    ax = axes[1]
    ax.imshow(rgb, alpha=0.6)
    if cell_data.num_nodes > 0 and cell_data.pos is not None:
        G_cell = _make_nx_graph(cell_data)
        pos    = {n: G_cell.nodes[n]["pos"] for n in G_cell.nodes}
        nx.draw_networkx_edges(G_cell, pos, ax=ax, edge_color="#FF6B6B",
                               alpha=0.5, width=0.8)
        nx.draw_networkx_nodes(G_cell, pos, ax=ax, node_size=15,
                               node_color="#FF6B6B", alpha=0.9)
    ax.set_title(
        f"Strategy 1 — Cell-Graph\n"
        f"{cell_data.num_nodes} nodes · "
        f"{cell_data.edge_index.shape[1]//2} edges",
        fontsize=10, fontweight="bold",
    )
    ax.axis("off")

    # Panel 2 — superpixel map
    ax = axes[2]
    if seg_map is not None:
        boundaries = segmentation.mark_boundaries(rgb / 255.0, seg_map, color=(1, 0.6, 0))
        ax.imshow(boundaries)
    else:
        # recompute seg_map for display
        rgb_f   = rgb.astype(np.float64) / 255.0
        seg_tmp = segmentation.slic(rgb_f, n_segments=SLIC_N_SEGMENTS,
                                    compactness=SLIC_COMPACTNESS, channel_axis=2, start_label=0)
        boundaries = segmentation.mark_boundaries(rgb_f, seg_tmp, color=(1, 0.6, 0))
        ax.imshow(boundaries)
    ax.set_title("Strategy 2 — SLIC Superpixels", fontsize=10, fontweight="bold")
    ax.axis("off")

    # Panel 3 — patch-graph overlay
    ax = axes[3]
    ax.imshow(rgb, alpha=0.6)
    if patch_data.num_nodes > 0 and patch_data.pos is not None:
        G_patch = _make_nx_graph(patch_data)
        pos     = {n: G_patch.nodes[n]["pos"] for n in G_patch.nodes}
        nx.draw_networkx_edges(G_patch, pos, ax=ax, edge_color="#4ECDC4",
                               alpha=0.5, width=0.8)
        nx.draw_networkx_nodes(G_patch, pos, ax=ax, node_size=30,
                               node_color="#4ECDC4", alpha=0.9)
    ax.set_title(
        f"Strategy 2 — Patch-Graph (k={KNN_K})\n"
        f"{patch_data.num_nodes} nodes · "
        f"{patch_data.edge_index.shape[1]//2} edges",
        fontsize=10, fontweight="bold",
    )
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


def visualise_degree_distribution(
    cell_stats:  List[GraphStats],
    patch_stats: List[GraphStats],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side degree distribution histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    for ax, stats, colour, label in [
        (ax1, cell_stats,  "#FF6B6B", "Cell-Graph (Delaunay)"),
        (ax2, patch_stats, "#4ECDC4", f"Patch-Graph (k-NN, k={KNN_K})"),
    ]:
        degs = [s.avg_degree for s in stats]
        ax.hist(degs, bins=20, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(degs), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean = {np.mean(degs):.2f}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Average Node Degree")
        ax.set_ylabel("Count (graphs)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Quick demo — runs when executed directly (uses synthetic H&E-like images)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_he_patch(size: int = 256, n_nuclei: int = 60,
                                  seed: int = 0) -> np.ndarray:
    """
    Synthesise a plausible H&E patch for testing purposes.
    Background is pink (eosin); nuclei are dark purple (haematoxylin).
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), fill_value=[240, 200, 220], dtype=np.uint8)
    for _ in range(n_nuclei):
        cx, cy = rng.integers(15, size - 15, size=2)
        rx     = rng.integers(6, 15)
        ry     = rng.integers(6, 15)
        # draw an ellipse
        Y, X   = np.ogrid[:size, :size]
        mask   = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
        # nuclear colour: dark blue-purple
        img[mask, 0] = rng.integers(40,  90)
        img[mask, 1] = rng.integers(20,  60)
        img[mask, 2] = rng.integers(100, 160)
    # Add some texture
    noise        = rng.integers(-12, 12, size=(size, size, 3), dtype=np.int16)
    img          = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    from pathlib import Path

    OUT = Path("graph_construction_output")
    OUT.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Graph Construction Pipeline — Demo")
    print("  Using synthetic H&E-like patches (no data download needed)")
    print("=" * 60)

    all_cell_stats, all_patch_stats = [], []

    for img_idx in range(6):
        rgb = _generate_synthetic_he_patch(size=256, n_nuclei=55 + img_idx * 5, seed=img_idx)

        cg, cs = build_cell_graph(rgb,  label_y=img_idx % 3)
        pg, ps = build_patch_graph(rgb, label_y=img_idx % 3)
        all_cell_stats.append(cs)
        all_patch_stats.append(ps)

        print(f"\n  Image {img_idx + 1}")
        print(f"    Cell-Graph  : {cs.num_nodes} nodes, {cs.num_edges} edges, "
              f"deg={cs.avg_degree:.2f}, t={cs.construction_time:.3f}s")
        print(f"    Patch-Graph : {ps.num_nodes} nodes, {ps.num_edges} edges, "
              f"deg={ps.avg_degree:.2f}, t={ps.construction_time:.3f}s")

        if img_idx < 3:
            seg_map = segmentation.slic(
                rgb.astype(np.float64) / 255.0,
                n_segments=SLIC_N_SEGMENTS, compactness=SLIC_COMPACTNESS,
                channel_axis=2, start_label=0,
            )
            fig = visualise_comparison(
                rgb, cg, pg, seg_map=seg_map,
                save_path=OUT / f"comparison_img{img_idx+1}.png",
                title=f"Graph Construction Comparison — Sample {img_idx + 1}",
            )
            plt.close(fig)
            print(f"    → saved comparison_img{img_idx+1}.png")

    # Degree distribution plot
    fig2 = visualise_degree_distribution(
        all_cell_stats, all_patch_stats,
        save_path=OUT / "degree_distributions.png",
    )
    plt.close(fig2)

    # Print aggregate statistics table
    print("\n" + "=" * 60)
    print("  Aggregate Statistics")
    print("=" * 60)
    for stats_list in [all_cell_stats, all_patch_stats]:
        s = summarise_stats(stats_list)
        print(f"\n  {s['strategy'].upper().replace('_', '-')}")
        for k, v in s.items():
            if k != "strategy":
                print(f"    {k:25s}: {v}")

    print(f"\nAll outputs saved to {OUT.resolve()}")
