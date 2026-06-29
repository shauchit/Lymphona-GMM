"""
scripts/cross_validate_improved.py
=====================================
Extended k-fold cross-validation supporting all four promising directions
from the README, compared against the baseline (flat GATGraphClassifier).

Models available via --model:
  baseline   Flat GAT + global mean-pool  (README best: acc 0.679 ± 0.043)
  hier       Hierarchical GAT + SAGPool   (direction 1 — highest expected payoff)
  hier_edge  Hierarchical GAT + SAGPool + edge features  (directions 1 + 3)

Typical usage
-------------
# Direction 1: hierarchical pooling
uv run python scripts/cross_validate_improved.py \\
    --graphs data/graphs/lymphoma_cell_stardist.pt \\
    --model hier --folds 5 --device cpu

# Directions 1 + 3: hierarchical + edge features
uv run python scripts/cross_validate_improved.py \\
    --graphs data/graphs/lymphoma_cell_stardist_edgefeat.pt \\
    --model hier_edge --folds 5 --device cpu

# Direction 2: patch-graph baseline (flat model, but different input)
uv run python scripts/cross_validate_improved.py \\
    --graphs data/graphs/lymphoma_patch.pt \\
    --model baseline --folds 5 --device cpu

All results are printed as per-fold metrics + mean ± std + OOF confusion matrix.
The best model state from each fold is saved to --out-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    BatchNorm, GATConv, global_mean_pool, global_max_pool
)
import torch.nn as nn

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.gat_hier import HierarchicalGAT

# Try to import the baseline from the project; fall back to a self-contained copy.
try:
    from models import GATGraphClassifier
    _HAVE_BASELINE = True
except ImportError:
    _HAVE_BASELINE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Fallback baseline model (flat GAT, matches README spec)
# ─────────────────────────────────────────────────────────────────────────────

class _GATBlock(nn.Module):
    def __init__(self, in_ch, hidden, heads, dropout=0.3):
        super().__init__()
        out = hidden * heads
        self.conv = GATConv(in_ch, hidden, heads=heads, dropout=dropout, concat=True)
        self.bn   = BatchNorm(out)
        self.proj = nn.Linear(in_ch, out) if in_ch != out else nn.Identity()

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return F.elu(self.bn(h) + self.proj(x))


class _BaselineGAT(nn.Module):
    """Flat GAT + global mean-pool — replicates README GATGraphClassifier."""
    def __init__(self, in_channels=18, num_classes=3, hidden=64, heads=4,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.blocks  = nn.ModuleList()
        ch = in_channels
        for _ in range(num_layers):
            self.blocks.append(_GATBlock(ch, hidden, heads, dropout))
            ch = hidden * heads
        self.clf = nn.Sequential(
            nn.Linear(ch, ch // 2), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(ch // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        for blk in self.blocks:
            x = F.dropout(blk(x, edge_index), p=self.dropout, training=self.training)
        return self.clf(global_mean_pool(x, batch))


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, in_channels: int, num_classes: int, args) -> nn.Module:
    if name == "baseline":
        if _HAVE_BASELINE:
            return GATGraphClassifier(in_channels=in_channels,
                                      num_classes=num_classes,
                                      heads=args.heads,
                                      num_layers=args.layers)
        return _BaselineGAT(in_channels=in_channels, num_classes=num_classes,
                            hidden=args.hidden, heads=args.heads,
                            num_layers=args.layers, dropout=args.dropout)
    elif name in ("hier", "hier_edge"):
        edge_dim = 3 if name == "hier_edge" else None
        return HierarchicalGAT(
            in_channels   = in_channels,
            num_classes   = num_classes,
            hidden        = args.hidden,
            heads         = args.heads,
            cell_layers   = args.cell_layers,
            tissue_layers = args.tissue_layers,
            pool_ratio    = args.pool_ratio,
            dropout       = args.dropout,
            edge_dim      = edge_dim,
        )
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose baseline / hier / hier_edge")


# ─────────────────────────────────────────────────────────────────────────────
#  Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def _batch_graphs(graphs: list[Data], indices: list[int], device) -> Batch:
    return Batch.from_data_list([graphs[i] for i in indices]).to(device)


def _forward(model, batch, use_edge_attr: bool):
    edge_attr = batch.edge_attr if (use_edge_attr and hasattr(batch, "edge_attr")
                                    and batch.edge_attr is not None) else None
    return model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)


def train_epoch(model, graphs, train_idx, optimizer, class_weights, device,
                use_edge_attr: bool, batch_size: int = 32):
    model.train()
    np.random.shuffle(train_idx)
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, len(train_idx), batch_size):
        chunk = train_idx[start:start + batch_size]
        batch = _batch_graphs(graphs, chunk, device)
        optimizer.zero_grad()
        out  = _forward(model, batch, use_edge_attr)
        loss = F.cross_entropy(out, batch.y.view(-1),
                               weight=class_weights.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, graphs, indices, device, use_edge_attr: bool):
    model.eval()
    all_pred, all_true = [], []
    for start in range(0, len(indices), 64):
        chunk = indices[start:start + 64]
        batch = _batch_graphs(graphs, chunk, device)
        out   = _forward(model, batch, use_edge_attr)
        pred  = out.argmax(dim=1).cpu().numpy()
        true  = batch.y.view(-1).cpu().numpy()
        all_pred.extend(pred); all_true.extend(true)
    acc = np.mean(np.array(all_pred) == np.array(all_true))
    return acc, np.array(all_pred), np.array(all_true)


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-validation loop
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["CLL", "FL", "MCL"]


def run_cv(graphs: list[Data], args, device) -> None:
    labels = np.array([g.y.item() for g in graphs])
    use_edge_attr = args.model == "hier_edge"
    in_channels   = graphs[0].x.shape[1]
    num_classes   = len(np.unique(labels))

    # Class weights (inverse frequency)
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_accs, fold_f1s = [], []
    oof_pred = np.full(len(graphs), -1, dtype=int)
    oof_true = labels.copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(graphs)), labels)):
        print(f"\n{'='*60}")
        print(f"  Fold {fold+1}/{args.folds}  |  model={args.model}")
        print(f"{'='*60}")

        # Split trainval → train / val (85/15)
        val_size  = max(1, int(len(trainval_idx) * 0.15))
        rng       = np.random.default_rng(fold)
        val_idx   = rng.choice(trainval_idx, size=val_size, replace=False)
        train_idx = np.array([i for i in trainval_idx if i not in set(val_idx)])

        model     = build_model(args.model, in_channels, num_classes, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_acc, best_state, patience_ctr = 0.0, None, 0

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, graphs, train_idx.tolist(), optimizer,
                               weights, device, use_edge_attr)
            val_acc, _, _ = evaluate(model, graphs, val_idx.tolist(), device, use_edge_attr)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d}  loss={loss:.4f}  val_acc={val_acc:.3f}  "
                      f"best={best_val_acc:.3f}")

            if patience_ctr >= args.patience:
                print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                break

        # Evaluate on test set with best model
        model.load_state_dict(best_state)
        test_acc, pred, true = evaluate(model, graphs, test_idx.tolist(), device, use_edge_attr)

        # Per-class F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
        fold_accs.append(test_acc)
        fold_f1s.append(macro_f1)
        oof_pred[test_idx] = pred

        print(f"\n  Fold {fold+1} test:  acc={test_acc:.3f}  macro-F1={macro_f1:.3f}")
        print(classification_report(true, pred, target_names=CLASS_NAMES,
                                    zero_division=0))
        # Save best model
        torch.save(best_state, str(out_dir / f"fold{fold+1}_{args.model}.pt"))

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {args.folds}-Fold Summary  |  model={args.model}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
    print(f"  Macro-F1  : {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
    print(f"\n  OOF Classification Report:")
    print(classification_report(oof_true, oof_pred, target_names=CLASS_NAMES,
                                zero_division=0))
    print("  OOF Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(oof_true, oof_pred)
    header = f"{'':>6}" + "".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>4}" + "".join(f"{v:>8}" for v in row))

    # ── Comparison table vs README baseline ───────────────────────────────────
    print(f"\n  README baseline (StarDist 18-dim flat GAT):")
    print(f"    accuracy 0.679 ± 0.043   macro-F1 0.659")
    delta_acc = np.mean(fold_accs) - 0.679
    delta_f1  = np.mean(fold_f1s)  - 0.659
    sign_acc  = "+" if delta_acc >= 0 else ""
    sign_f1   = "+" if delta_f1  >= 0 else ""
    print(f"  This run ({args.model}):")
    print(f"    accuracy {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}  "
          f"(Δ {sign_acc}{delta_acc:.3f})")
    print(f"    macro-F1 {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}  "
          f"(Δ {sign_f1}{delta_f1:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Cross-validate improved GNN models on lymphoma cell-graphs.")
    # Data
    p.add_argument("--graphs", default="data/graphs/lymphoma_cell_stardist.pt",
                   help=".pt file produced by build_graphs.py")
    p.add_argument("--graph-key", default=None,
                   help="Key in the .pt dict (auto-detected if omitted)")
    # Model
    p.add_argument("--model", default="hier",
                   choices=["baseline", "hier", "hier_edge"],
                   help="Model architecture to evaluate")
    p.add_argument("--hidden",        type=int,   default=64)
    p.add_argument("--heads",         type=int,   default=4)
    p.add_argument("--layers",        type=int,   default=2,
                   help="Layers for baseline model")
    p.add_argument("--cell-layers",   type=int,   default=2,
                   help="Cell-level GAT layers (hier models)")
    p.add_argument("--tissue-layers", type=int,   default=2,
                   help="Tissue-level GAT layers after SAGPool (hier models)")
    p.add_argument("--pool-ratio",    type=float, default=0.5,
                   help="SAGPool keep fraction (hier models)")
    p.add_argument("--dropout",       type=float, default=0.3)
    # Training
    p.add_argument("--folds",         type=int,   default=5)
    p.add_argument("--epochs",        type=int,   default=80)
    p.add_argument("--patience",      type=int,   default=20,
                   help="Early-stop patience (epochs without val improvement)")
    p.add_argument("--lr",            type=float, default=5e-3)
    p.add_argument("--weight-decay",  type=float, default=5e-4)
    p.add_argument("--device",        default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    # Output
    p.add_argument("--out-dir", default="data/checkpoints",
                   help="Directory to save per-fold best model weights")
    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load graphs
    print(f"Loading {args.graphs} …")
    store = torch.load(args.graphs, weights_only=False)

    if isinstance(store, list):
        graphs = store
    elif isinstance(store, dict):
        if args.graph_key:
            graphs = store[args.graph_key]
        else:
            # Auto-detect: pick first key that is a non-empty list of Data
            for key, val in store.items():
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], Data):
                    graphs = val
                    print(f"  Using key '{key}' ({len(val)} graphs)")
                    break
            else:
                raise ValueError(f"No graph list found in {args.graphs}. "
                                 f"Available keys: {list(store.keys())}")
    else:
        raise ValueError(f"Unexpected .pt format: {type(store)}")

    # Validate edge_attr presence for hier_edge
    if args.model == "hier_edge":
        sample = graphs[0]
        if not hasattr(sample, "edge_attr") or sample.edge_attr is None:
            raise ValueError(
                "Model 'hier_edge' requires edge_attr. "
                "Run scripts/add_edge_features.py first."
            )
        print(f"  Edge features: {sample.edge_attr.shape[1]}-dim")

    print(f"  {len(graphs)} graphs  |  "
          f"node features: {graphs[0].x.shape[1]}  |  "
          f"classes: {len(torch.unique(torch.cat([g.y for g in graphs])))}")

    run_cv(graphs, args, device)


if __name__ == "__main__":
    main()
