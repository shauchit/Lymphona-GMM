"""
edge_gat.py
===========
Edge-feature-aware graph classifier (GATv2Conv with `edge_dim`).

Mirrors the flat baseline GAT (residual + BatchNorm, configurable depth) but
uses `GATv2Conv` consuming per-edge features so message passing is geometry-aware
(not just topological). Edge features are computed from node positions by
`attach_edge_features` in the training harness: [inverse distance, cos, sin] of
the edge vector — 3-dim by default.

forward(data) reads data.x, data.edge_index, data.edge_attr, data.batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import BatchNorm, GATv2Conv, global_mean_pool


class EdgeGATGraphClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 18,
        hidden_channels: int = 64,
        num_classes: int = 3,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.5,
        edge_dim: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        width = hidden_channels * heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads,
                                    edge_dim=edge_dim, dropout=dropout))
        self.norms.append(BatchNorm(width))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(width, hidden_channels, heads=heads,
                                        edge_dim=edge_dim, dropout=dropout))
            self.norms.append(BatchNorm(width))
        self.lin = nn.Linear(width, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.elu(self.norms[0](self.convs[0](x, edge_index, ea)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            h = F.elu(norm(conv(x, edge_index, ea)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        x = global_mean_pool(x, batch)
        return self.lin(x)
