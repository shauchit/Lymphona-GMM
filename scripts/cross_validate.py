"""
cross_validate.py
=================
Stratified k-fold cross-validation for the GAT graph classifier.

Why: a single 70/15/15 split on N=374 leaves only ~57 test graphs, so one
accuracy number is very noisy. k-fold gives every graph exactly one held-out
(out-of-fold) prediction and reports mean ± std across folds.

Each fold: the held-out fold is the test set; a small stratified slice of the
remaining data is the validation set (for best-on-val model selection); the
rest is training data.

Usage
-----
    uv run python scripts/cross_validate.py
    uv run python scripts/cross_validate.py --folds 5 --layers 2 --epochs 80
    uv run python scripts/cross_validate.py --graphs data/graphs/lymphoma_cell.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader

from models import EdgeGATGraphClassifier, GATGraphClassifier, SAGPoolGraphClassifier
from models.training import (
    CLASS_NAMES,
    attach_edge_features,
    evaluate,
    get_device,
    labels_of,
    load_graphs,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", type=Path, default=Path("data/graphs/lymphoma_cell.pt"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="fraction of each fold's train split held out for val")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4, help="number of GAT layers")
    parser.add_argument("--model", choices=["flat", "sagpool", "edge"], default="flat",
                        help="architecture: flat GAT, SAGPool, or edge-feature GATv2")
    parser.add_argument("--pool-ratio", type=float, default=0.5,
                        help="SAGPool keep-ratio (only for --model sagpool)")
    parser.add_argument("--pool", choices=["mean", "max", "mean+max"],
                        default="mean", help="graph readout pooling (flat model)")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto", help="compute device (default: auto)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    # Edge-feature model needs node positions to derive edge_attr; keep `pos`.
    graphs = load_graphs(args.graphs, strip_pos=(args.model != "edge"))
    labels = labels_of(graphs)
    in_channels = graphs[0].x.shape[1]
    num_classes = int(labels.max()) + 1
    names = CLASS_NAMES[:num_classes]

    if args.model == "edge":
        attach_edge_features(graphs)

    # Build a fresh-model factory for the chosen architecture.
    if args.model == "sagpool":
        def model_factory():
            return SAGPoolGraphClassifier(
                in_channels=in_channels, hidden_channels=args.hidden,
                num_classes=num_classes, heads=args.heads,
                ratio=args.pool_ratio, dropout=args.dropout)
        cfg = f"SAGPool ratio={args.pool_ratio}"
    elif args.model == "edge":
        def model_factory():
            return EdgeGATGraphClassifier(
                in_channels=in_channels, hidden_channels=args.hidden,
                num_classes=num_classes, heads=args.heads,
                num_layers=args.layers, dropout=args.dropout)
        cfg = "edge-GATv2 (inv-dist + angle)"
    else:
        model_factory = None                       # default flat GAT in train_model
        cfg = f"flat GAT {args.layers}L pool={args.pool}"

    print(f"Config: {args.folds}-fold CV | model={args.model} ({cfg}) | "
          f"hidden={args.hidden} | heads={args.heads} | epochs={args.epochs}\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    idx = np.arange(len(graphs))

    fold_acc, fold_f1 = [], []
    oof_pred = np.full(len(graphs), -1, dtype=int)   # out-of-fold predictions

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(idx, labels), start=1):
        # Carve a stratified validation slice out of the train+val portion.
        tr_idx, val_idx = train_test_split(
            trainval_idx, test_size=args.val_frac,
            stratify=labels[trainval_idx], random_state=args.seed)

        train_g = [graphs[i] for i in tr_idx]
        val_g   = [graphs[i] for i in val_idx]
        test_g  = [graphs[i] for i in test_idx]

        model, best_val = train_model(
            train_g, val_g,
            in_channels=in_channels, num_classes=num_classes,
            hidden=args.hidden, heads=args.heads, num_layers=args.layers,
            dropout=args.dropout, pool=args.pool,
            lr=args.lr, weight_decay=args.weight_decay,
            epochs=args.epochs, batch_size=args.batch_size,
            device=device, model_factory=model_factory, verbose=False,
        )

        test_loader = DataLoader(test_g, batch_size=args.batch_size)
        acc, preds, trues = evaluate(model, test_loader, device)
        macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        oof_pred[test_idx] = preds

        fold_acc.append(acc)
        fold_f1.append(macro_f1)
        print(f"  fold {fold}/{args.folds} | n_test={len(test_g):3d} "
              f"| val_acc={best_val:.3f} | test_acc={acc:.3f} | macro_F1={macro_f1:.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    acc_arr, f1_arr = np.array(fold_acc), np.array(fold_f1)
    print("\n" + "=" * 60)
    print("  Cross-validation summary")
    print("=" * 60)
    print(f"  test accuracy : {acc_arr.mean():.3f} ± {acc_arr.std():.3f}")
    print(f"  macro F1      : {f1_arr.mean():.3f} ± {f1_arr.std():.3f}")

    # Out-of-fold report: every graph predicted exactly once, held out.
    print("\n  Out-of-fold report (all graphs pooled):")
    print(classification_report(labels, oof_pred, target_names=names,
                                digits=3, zero_division=0))
    print("  Out-of-fold confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(labels, oof_pred))


if __name__ == "__main__":
    main()
