"""
visualization.py
================
Plotting helpers for the graph-construction utilities, plus a synthetic
H&E patch generator for quick local testing (no dataset download required).

These are optional and only needed for inspection / debugging — the GNN
training path only needs ``graph_construction.build_*`` functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage import segmentation
from torch_geometric.data import Data

from .graph_construction import (
    KNN_K,
    SLIC_COMPACTNESS,
    SLIC_N_SEGMENTS,
    GraphStats,
)


def _make_nx_graph(data: Data) -> nx.Graph:
    """Convert a torch_geometric Data → networkx Graph with node positions."""
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
    save_path: Optional[str | Path] = None,
    title: str = "",
) -> plt.Figure:
    """4-panel figure: image | cell-graph | superpixel map | patch-graph."""
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
        f"{cell_data.num_nodes} nodes · {cell_data.edge_index.shape[1] // 2} edges",
        fontsize=10, fontweight="bold",
    )
    ax.axis("off")

    # Panel 2 — superpixel map
    ax = axes[2]
    if seg_map is None:
        seg_map = segmentation.slic(
            rgb.astype(np.float64) / 255.0, n_segments=SLIC_N_SEGMENTS,
            compactness=SLIC_COMPACTNESS, channel_axis=2, start_label=0,
        )
    boundaries = segmentation.mark_boundaries(rgb / 255.0, seg_map, color=(1, 0.6, 0))
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
        f"{patch_data.num_nodes} nodes · {patch_data.edge_index.shape[1] // 2} edges",
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
    cell_stats: List[GraphStats],
    patch_stats: List[GraphStats],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side average-degree histograms for the two strategies."""
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


def generate_synthetic_he_patch(
    size: int = 256, n_nuclei: int = 60, seed: int = 0
) -> np.ndarray:
    """
    Synthesise a plausible H&E patch for testing.
    Background is pink (eosin); nuclei are dark purple (haematoxylin).
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), fill_value=[240, 200, 220], dtype=np.uint8)
    for _ in range(n_nuclei):
        cx, cy = rng.integers(15, size - 15, size=2)
        rx, ry = rng.integers(6, 15), rng.integers(6, 15)
        Y, X   = np.ogrid[:size, :size]
        mask   = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
        img[mask, 0] = rng.integers(40,  90)
        img[mask, 1] = rng.integers(20,  60)
        img[mask, 2] = rng.integers(100, 160)
    noise = rng.integers(-12, 12, size=(size, size, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
