"""
sagpool.py
==========
Hierarchical-pooling graph classifier (SAGPool) for cell-/patch-graphs.

Three GAT blocks, each followed by self-attention pooling (`SAGPooling`) that
keeps a learned fraction (`ratio`) of nodes. A jumping-knowledge readout sums a
mean+max pool after every block, so the classifier sees coarsened, tissue-level
structure rather than a single global average — the motivation for trying it on
the MCL architecture problem.

forward(data) signature (data carries x, edge_index, batch).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, SAGPooling, global_max_pool, global_mean_pool


class SAGPoolGraphClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 18,
        hidden_channels: int = 64,
        num_classes: int = 3,
        heads: int = 4,
        ratio: float = 0.5,
        dropout: float = 0.5,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(num_blocks):
            cin = in_channels if i == 0 else hidden_channels
            # heads collapsed to 1 so each block outputs `hidden_channels`.
            self.convs.append(GATConv(cin, hidden_channels, heads=heads, concat=False,
                                      dropout=dropout))
            self.pools.append(SAGPooling(hidden_channels, ratio=ratio))
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        readout = 0
        for conv, pool in zip(self.convs, self.pools):
            x = F.elu(conv(x, edge_index))
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
            readout = readout + torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        h = F.relu(self.lin1(readout))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)
