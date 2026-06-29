"""
scripts/add_edge_features.py
==============================
Enrich existing cell-graph .pt files with geometric edge attributes.

For every directed edge (i → j) we compute:
  feat[0]  =  exp(−d / σ)          normalised inverse distance (Gaussian kernel)
  feat[1]  =  sin(θ)               vertical component of edge direction
  feat[2]  =  cos(θ)               horizontal component of edge direction

where d = ||pos_i − pos_j|| (pixels) and σ is the median edge length across the
dataset (auto-computed).  The Gaussian kernel maps short edges → 1 and long edges
→ 0, giving a continuous proximity weight that GATv2Conv can condition on.

sin/cos encoding avoids the discontinuity at ±π that a raw angle would have.

Usage
-----
    uv run python scripts/add_edge_features.py \\
        --graphs data/graphs/lymphoma_cell_stardist.pt \\
        --out    data/graphs/lymphoma_cell_stardist_edgefeat.pt

The output .pt has the same structure but every Data object gains a
data.edge_attr tensor of shape [2E, 3].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
#  Core feature computation
# ─────────────────────────────────────────────────────────────────────────────

def _edge_lengths(data: Data) -> np.ndarray:
    """Return the Euclidean length of every directed edge in the graph."""
    if data.pos is None or data.edge_index.shape[1] == 0:
        return np.array([])
    pos = data.pos.numpy()                          # (N, 2)  [row, col]
    src, dst = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    delta = pos[dst] - pos[src]                     # (E, 2)
    return np.linalg.norm(delta, axis=1)            # (E,)


def compute_edge_attr(data: Data, sigma: float) -> torch.Tensor:
    """
    Compute 3-dim edge attributes for a single graph.

    Parameters
    ----------
    data  : PyG Data object with data.pos and data.edge_index
    sigma : Gaussian bandwidth (median edge length across dataset)

    Returns
    -------
    edge_attr : (2E, 3) float32 tensor
    """
    if data.edge_index.shape[1] == 0:
        return torch.zeros((0, 3), dtype=torch.float)

    pos = data.pos.numpy()
    src = data.edge_index[0].numpy()
    dst = data.edge_index[1].numpy()
    delta = pos[dst] - pos[src]                     # (E, 2)  [Δrow, Δcol]
    d     = np.linalg.norm(delta, axis=1)           # (E,)

    # Gaussian proximity weight
    proximity = np.exp(-d / (sigma + 1e-8))

    # Direction (sin / cos of angle in image coordinates)
    angle = np.arctan2(delta[:, 0], delta[:, 1])   # row-axis = vertical
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    feats = np.stack([proximity, sin_a, cos_a], axis=1).astype(np.float32)
    return torch.tensor(feats)


def add_edge_features(graphs: list[Data], sigma: float | None = None) -> list[Data]:
    """
    Add edge_attr to a list of graphs.

    Parameters
    ----------
    graphs : list of PyG Data objects (must have data.pos)
    sigma  : Gaussian bandwidth; if None, computed as the median edge length
             across the full dataset (recommended)

    Returns
    -------
    enriched graphs (in-place modification + return)
    """
    if sigma is None:
        # Compute sigma from all edges in the dataset
        all_lengths = np.concatenate(
            [_edge_lengths(g) for g in graphs if g.edge_index.shape[1] > 0]
        )
        sigma = float(np.median(all_lengths)) if len(all_lengths) > 0 else 50.0
        print(f"  σ (median edge length) = {sigma:.1f} px")

    for g in graphs:
        g.edge_attr = compute_edge_attr(g, sigma)

    return graphs


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Add geometric edge features to cell-graphs.")
    parser.add_argument("--graphs", required=True,
                        help="Input .pt file (output of build_graphs.py)")
    parser.add_argument("--out", default=None,
                        help="Output .pt path (default: <input>_edgefeat.pt)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Gaussian σ in pixels (default: auto = median edge length)")
    args = parser.parse_args()

    in_path  = Path(args.graphs)
    out_path = Path(args.out) if args.out else in_path.with_stem(in_path.stem + "_edgefeat")

    print(f"Loading {in_path} …")
    store = torch.load(str(in_path), weights_only=False)

    # Support both raw list and dict-of-lists formats
    if isinstance(store, dict):
        keys = [k for k in store if isinstance(store[k], list) and len(store[k]) > 0
                and isinstance(store[k][0], Data)]
    else:
        store = {"graphs": store}
        keys  = ["graphs"]

    for key in keys:
        graphs = store[key]
        print(f"  Processing '{key}': {len(graphs)} graphs …")
        store[key] = add_edge_features(graphs, sigma=args.sigma)
        # Quick sanity check
        sample = store[key][0]
        print(f"    edge_attr shape: {sample.edge_attr.shape}  "
              f"(expected [{sample.edge_index.shape[1]}, 3])")

    torch.save(store, str(out_path))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
