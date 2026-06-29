"""
training.py
===========
Shared training utilities for the GAT graph classifier — reused by both the
single-split trainer (scripts/train_gnn.py) and the k-fold cross-validator
(scripts/cross_validate.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .gat import GATGraphClassifier

CLASS_NAMES = ["CLL", "FL", "MCL"]


def get_device() -> torch.device:
    """Pick the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_graphs(path: str | Path, verbose: bool = True) -> List:
    """Load the first available *_graphs list from a build_graphs .pt file."""
    blob = torch.load(path, weights_only=False)
    for key in ("cell_graphs", "patch_graphs"):
        if key in blob and blob[key]:
            graphs = blob[key]
            # `pos` is unused by the model and is missing on degenerate graphs
            # (<3 nuclei) — drop it so PyG can batch a consistent attribute set.
            for g in graphs:
                if "pos" in g:
                    del g.pos
            if verbose:
                print(f"Loaded {len(graphs)} graphs from '{key}' in {path}")
            return graphs
    raise SystemExit(f"No graphs found in {path}")


def labels_of(graphs: List) -> np.ndarray:
    """Graph-level class labels as an int array."""
    return np.array([int(g.y.item()) for g in graphs])


def class_weights(train_g: List, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights from the training split."""
    counts = np.bincount(labels_of(train_g), minlength=num_classes)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float, device=device)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (accuracy, preds, trues) over a loader."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds.append(logits.argmax(1).cpu())
        trues.append(batch.y.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return float((preds == trues).mean()), preds, trues


def train_model(
    train_g: List,
    val_g: List,
    *,
    in_channels: int,
    num_classes: int,
    hidden: int = 64,
    heads: int = 4,
    num_layers: int = 4,
    dropout: float = 0.5,
    pool: str = "mean",
    lr: float = 5e-3,
    weight_decay: float = 5e-4,
    epochs: int = 80,
    batch_size: int = 16,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Tuple[GATGraphClassifier, float]:
    """
    Train a GAT classifier with best-on-validation model selection.

    Returns the model (restored to its best-on-val weights) and the best
    validation accuracy.
    """
    device = device or get_device()

    train_loader = DataLoader(train_g, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_g,   batch_size=batch_size)
    weights = class_weights(train_g, num_classes, device)

    model = GATGraphClassifier(
        in_channels=in_channels, hidden_channels=hidden,
        num_classes=num_classes, heads=heads,
        num_layers=num_layers, dropout=dropout, pool=pool,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, best_epoch = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y, weight=weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        val_acc, _, _ = evaluate(model, val_loader, device)
        if val_acc >= best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            train_acc, _, _ = evaluate(model, train_loader, device)
            print(f"  epoch {epoch:3d} | loss {total_loss/len(train_g):.4f} "
                  f"| train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  best val acc {best_val:.3f} @ epoch {best_epoch}")
    return model, best_val
