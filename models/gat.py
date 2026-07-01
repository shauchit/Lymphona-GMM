"""
gat.py
======
A configurable-depth GAT graph classifier for H&E cell-/patch-graphs.

Architecture:
    GATConv(in → hidden, heads=H)                        → BN → ELU
    (num_layers-1) × GATConv(hidden*H → hidden, heads=H) → BN → ELU  (+ residual)
    global mean pool over nodes
    Linear head → num_classes

Deepening a plain GAT tends to over-smooth (node representations collapse).
To make extra depth actually help we keep every block at the same width
(hidden*heads), add a **residual** connection across each hidden block, and
**BatchNorm** between layers to stabilise training.

Input feature width is read from the data (``in_channels``); the graph-level
label lives in ``data.y``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import BatchNorm, GATConv, global_max_pool, global_mean_pool

POOLINGS = ("mean", "max", "mean+max")


class GATGraphClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 64,
        num_classes: int = 3,
        heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.5,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if pool not in POOLINGS:
            raise ValueError(f"pool must be one of {POOLINGS}")
        self.dropout = dropout
        self.pool = pool
        width = hidden_channels * heads          # common block width

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input block: in_channels -> width
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                                  dropout=dropout))
        self.norms.append(BatchNorm(width))

        # Hidden blocks: width -> width (residual-friendly, constant width)
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(width, hidden_channels, heads=heads,
                                      dropout=dropout))
            self.norms.append(BatchNorm(width))

        # mean+max concatenation doubles the pooled dimension.
        pooled_dim = width * (2 if pool == "mean+max" else 1)
        self.lin = nn.Linear(pooled_dim, num_classes)

    def _readout(self, x, batch) -> torch.Tensor:
        if self.pool == "mean":
            return global_mean_pool(x, batch)
        if self.pool == "max":
            return global_max_pool(x, batch)
        return torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Input block (no residual — dimensions change here).
        x = F.elu(self.norms[0](self.convs[0](x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hidden blocks with residual connections (constant width).
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            h = F.elu(norm(conv(x, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h                                  # residual

        x = self._readout(x, batch)                    # [num_graphs, pooled_dim]
        return self.lin(x)                             # raw logits
