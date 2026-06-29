"""
models/gat_hier.py
====================
Two-level hierarchical GAT with SAGPool coarsening.

Motivation (from CV analysis):
  - Depth/pooling changes to the flat GATGraphClassifier had no effect.
  - MCL recall is ~0.47 — the weakest class.
  - MCL is defined by its MANTLE-ZONE architecture (a ring of small lymphocytes
    around reactive follicles), which is a tissue-level pattern not visible from
    single-nucleus features averaged by global mean-pool.
  - A hierarchical readout lets the model first reason at cell level, then
    coarsen to a tissue-level graph and reason about spatial clusters of cells.

Architecture:
    Input cell-graph (N nodes, 18 dim)
         │
    [Cell-level GATConv blocks × cell_layers]   ← learns per-nucleus context
         │
    [SAGPool (ratio)]                            ← selects ~ratio×N 'landmark' cells
         │                                         and rewires a coarser graph
    [Tissue-level GATConv blocks × tissue_layers] ← learns cluster interactions
         │
    [global mean-pool + global max-pool]          ← richer readout (2×hidden)
         │
    [MLP classifier]

Optionally accepts edge_attr (e.g. inverse distance + angle) on the input graph.
If the graphs have no edge_attr the model falls back to plain GATConv.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    BatchNorm,
    GATConv,
    GATv2Conv,
    SAGPooling,
    global_max_pool,
    global_mean_pool,
)


class _GATBlock(nn.Module):
    """Single GATConv (or GATv2Conv) + BatchNorm + residual."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        heads: int,
        dropout: float = 0.3,
        edge_dim: int | None = None,
    ):
        super().__init__()
        out = hidden * heads
        Conv = GATv2Conv if edge_dim else GATConv
        kw = dict(edge_dim=edge_dim) if edge_dim else {}
        self.conv = Conv(in_channels, hidden, heads=heads, dropout=dropout,
                         concat=True, **kw)
        self.bn   = BatchNorm(out)
        self.proj = nn.Linear(in_channels, out) if in_channels != out else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv(x, edge_index, edge_attr) if edge_attr is not None else self.conv(x, edge_index)
        h = self.bn(h)
        h = F.elu(h + self.proj(x))
        return h


class HierarchicalGAT(nn.Module):
    """
    Hierarchical GAT: cell-level → SAGPool coarsening → tissue-level → MLP.

    Parameters
    ----------
    in_channels   : node feature dimension (18 for StarDist cell-graphs)
    num_classes   : number of output classes (3: CLL / FL / MCL)
    hidden        : width per attention head
    heads         : attention heads (applied at both levels)
    cell_layers   : number of GATConv blocks at the cell level (before pooling)
    tissue_layers : number of GATConv blocks at the tissue level (after pooling)
    pool_ratio    : fraction of nodes SAGPool keeps (0 < ratio ≤ 1)
    dropout       : dropout probability
    edge_dim      : edge-attribute dimension (None → ignore edge_attr)
    """

    def __init__(
        self,
        in_channels:   int   = 18,
        num_classes:   int   = 3,
        hidden:        int   = 64,
        heads:         int   = 4,
        cell_layers:   int   = 2,
        tissue_layers: int   = 2,
        pool_ratio:    float = 0.5,
        dropout:       float = 0.3,
        edge_dim:      int | None = None,
    ):
        super().__init__()
        self.dropout    = dropout
        self.pool_ratio = pool_ratio
        cell_out = hidden * heads

        # ── Cell-level blocks ──────────────────────────────────────────────
        self.cell_blocks = nn.ModuleList()
        ch = in_channels
        for _ in range(cell_layers):
            self.cell_blocks.append(_GATBlock(ch, hidden, heads, dropout,
                                              edge_dim=edge_dim))
            ch = cell_out
        self.cell_out_dim = ch

        # ── SAGPool ───────────────────────────────────────────────────────
        # SAGPool uses a learnable scoring function on node features.
        self.sagpool = SAGPooling(ch, ratio=pool_ratio)

        # ── Tissue-level blocks ───────────────────────────────────────────
        # After SAGPool the graph still has cell_out_dim features.
        self.tissue_blocks = nn.ModuleList()
        for _ in range(tissue_layers):
            self.tissue_blocks.append(_GATBlock(ch, hidden, heads, dropout))
            ch = cell_out

        # ── Classifier ────────────────────────────────────────────────────
        # Concatenate global mean-pool and global max-pool → 2×cell_out_dim
        cls_in = 2 * ch
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, cls_in // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_in // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        # ── Cell-level processing ──────────────────────────────────────────
        for blk in self.cell_blocks:
            x = blk(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ── SAGPool coarsening ────────────────────────────────────────────
        # Returns coarsened (x, edge_index, edge_attr, batch, perm, score)
        x, edge_index, _, batch, _, _ = self.sagpool(x, edge_index, batch=batch)

        # ── Tissue-level processing ────────────────────────────────────────
        # edge_attr is discarded after pooling (SAGPool doesn't propagate it)
        for blk in self.tissue_blocks:
            x = blk(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Readout: mean + max concatenation ────────────────────────────
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=-1)

        return self.classifier(x)
