"""
train_gnn.py
============
Train the GAT graph classifier on the graphs built by scripts/build_graphs.py.

Usage
-----
    uv run python scripts/train_gnn.py
    uv run python scripts/train_gnn.py --graphs data/graphs/lymphoma_cell.pt
    uv run python scripts/train_gnn.py --layers 6 --hidden 128 --epochs 100

Does a stratified train/val/test split (70/15/15), trains with best-on-val
model selection, and reports test accuracy + a per-class report.

For a less noisy estimate of generalisation, use scripts/cross_validate.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from models.training import (
    CLASS_NAMES,
    evaluate,
    get_device,
    labels_of,
    load_graphs,
    train_model,
)
from torch_geometric.loader import DataLoader


def stratified_split(graphs: list, seed: int):
    """70/15/15 stratified split on graph-level labels."""
    labels = labels_of(graphs)
    idx = np.arange(len(graphs))
    train_idx, tmp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=labels[tmp_idx], random_state=seed)
    pick = lambda ids: [graphs[i] for i in ids]
    return pick(train_idx), pick(val_idx), pick(test_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", type=Path, default=Path("data/graphs/lymphoma_cell.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4, help="number of GAT layers")
    parser.add_argument("--pool", choices=["mean", "max", "mean+max"],
                        default="mean", help="graph readout pooling")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto", help="compute device (default: auto)")
    parser.add_argument("--out", type=Path, default=Path("data/graphs/gat_best.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    graphs = load_graphs(args.graphs)
    train_g, val_g, test_g = stratified_split(graphs, args.seed)
    print(f"Split  : train={len(train_g)}  val={len(val_g)}  test={len(test_g)}")

    in_channels = graphs[0].x.shape[1]
    num_classes = int(labels_of(graphs).max()) + 1

    model, best_val = train_model(
        train_g, val_g,
        in_channels=in_channels, num_classes=num_classes,
        hidden=args.hidden, heads=args.heads, num_layers=args.layers,
        dropout=args.dropout, pool=args.pool,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size,
        device=device, verbose=True,
    )
    print(f"\nBest val acc {best_val:.3f}")

    test_loader = DataLoader(test_g, batch_size=args.batch_size)
    test_acc, preds, trues = evaluate(model, test_loader, device)
    print(f"Test acc     {test_acc:.3f}\n")
    names = CLASS_NAMES[:num_classes]
    print(classification_report(trues, preds, target_names=names, digits=3,
                                zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(trues, preds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nSaved best model → {args.out}")


if __name__ == "__main__":
    main()
