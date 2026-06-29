"""
build_graphs.py
===============
Convert a directory of H&E images into torch_geometric graphs (.pt) using the
``util`` graph-construction pipeline, ready for GNN training.

Defaults target the Malignant Lymphoma dataset laid out as
``data/lymphoma/{CLL,FL,MCL}/*.tif`` (as produced by scripts/download_data.py).

Two ways to build cell graphs:
  * in-process detection (watershed) straight from images, or
  * from a StarDist nuclei cache produced by scripts/segment_nuclei.py
    (``--nuclei-cache``) — recommended, since StarDist can't run in a torch
    process. Patch graphs are always built from images.

Usage
-----
    uv run python scripts/build_graphs.py                          # watershed cells
    uv run python scripts/build_graphs.py --strategy patch
    uv run python scripts/build_graphs.py --strategy both --limit 30
    # StarDist (two-stage):
    uv run python scripts/segment_nuclei.py --out data/graphs/nuclei_stardist.pkl
    uv run python scripts/build_graphs.py --nuclei-cache data/graphs/nuclei_stardist.pkl \
        --out data/graphs/lymphoma_cell_stardist.pt

Output
------
A .pt file holding a dict of lists of torch_geometric.data.Data objects, e.g.
``{"cell_graphs": [...]}`` — load with torch.load(path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so ``import util`` works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from util import (
    build_cell_graph_from_nuclei,
    process_dataset,
    summarise_stats,
)

# Default label mapping for the lymphoma subtypes (folder name -> class id).
LYMPHOMA_LABEL_MAP = {"CLL": 0, "FL": 1, "MCL": 2}


def build_from_nuclei_cache(cache_path: Path, out_path: Path) -> None:
    """Stage 2: build cell graphs from a segment_nuclei.py cache (torch process)."""
    import pickle

    with open(cache_path, "rb") as f:
        blob = pickle.load(f)
    records = blob["records"]
    print(f"Loaded {len(records)} nuclei records (method={blob.get('method')}) "
          f"from {cache_path}")

    graphs, stats = [], []
    for r in records:
        data, st = build_cell_graph_from_nuclei(
            r["centroids"], r["feats"], label_y=r["label_y"], image_shape=r["shape"])
        graphs.append(data)
        stats.append(st)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cell_graphs": graphs}, str(out_path))

    s = summarise_stats(stats)
    print("\n  CELL-GRAPH (from StarDist cache)")
    for k, v in s.items():
        if k != "strategy":
            print(f"    {k:18s}: {v}")
    labels = [int(g.y.item()) for g in graphs]
    print("\n  Label distribution:")
    for name, cid in LYMPHOMA_LABEL_MAP.items():
        print(f"    {name:4s} (={cid}): {labels.count(cid)}")
    if labels.count(-1):
        print(f"    UNKNOWN (-1): {labels.count(-1)}  ← check label_map")
    print(f"\nSaved {len(graphs)} graphs → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, default=Path("data/lymphoma"),
                        help="folder of images, searched recursively (default: data/lymphoma)")
    parser.add_argument("--nuclei-cache", type=Path, default=None,
                        help="build cell graphs from a segment_nuclei.py .pkl cache")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .pt path (default: data/graphs/<dir>_<strategy>.pt)")
    parser.add_argument("--strategy", choices=["cell", "patch", "both"], default="cell",
                        help="graph construction strategy (default: cell)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N images (quick test)")
    args = parser.parse_args()

    # Stage-2 path: build from a precomputed StarDist nuclei cache.
    if args.nuclei_cache is not None:
        if not args.nuclei_cache.exists():
            sys.exit(f"Nuclei cache not found: {args.nuclei_cache}\n"
                     f"Run: uv run python scripts/segment_nuclei.py")
        out_path = args.out or Path("data/graphs") / f"{args.nuclei_cache.stem}.pt"
        build_from_nuclei_cache(args.nuclei_cache, out_path)
        return

    if not args.image_dir.exists():
        sys.exit(f"Image dir not found: {args.image_dir}\n"
                 f"Run: uv run python scripts/download_data.py --dataset lymphoma")

    strategies = ("cell", "patch") if args.strategy == "both" else (args.strategy,)

    out_path = args.out
    if out_path is None:
        out_path = Path("data/graphs") / f"{args.image_dir.name}_{args.strategy}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # `--limit` keeps the full pipeline but caps the image list via a temp view.
    image_dir = args.image_dir
    if args.limit is not None:
        image_dir = _limited_view(args.image_dir, args.limit)

    print(f"Image dir : {args.image_dir}")
    print(f"Strategy  : {args.strategy}")
    print(f"Output    : {out_path}\n")

    result = process_dataset(
        image_dir,
        out_pt_path=out_path,
        label_map=LYMPHOMA_LABEL_MAP,
        strategies=strategies,
    )

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for strat in strategies:
        stats = result[f"{strat}_stats"]
        if not stats:
            continue
        s = summarise_stats(stats)
        print(f"\n  {s['strategy'].upper().replace('_', '-')}")
        for k, v in s.items():
            if k != "strategy":
                print(f"    {k:18s}: {v}")

    # Report label distribution as a sanity check on the label_map.
    any_graphs = next(v for k, v in result.items() if k.endswith("_graphs"))
    labels = [int(g.y.item()) for g in any_graphs]
    print("\n  Label distribution:")
    for name, cid in LYMPHOMA_LABEL_MAP.items():
        print(f"    {name:4s} (={cid}): {labels.count(cid)}")
    unknown = labels.count(-1)
    if unknown:
        print(f"    UNKNOWN (-1): {unknown}  ← check folder names vs label_map")

    print(f"\nSaved {len(any_graphs)} graphs → {out_path}")


def _limited_view(image_dir: Path, limit: int) -> Path:
    """Symlink the first `limit` images into a temp dir for a quick run."""
    import tempfile

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)[:limit]
    tmp = Path(tempfile.mkdtemp(prefix="build_graphs_"))
    for p in paths:
        # Preserve parent folder name so the label_map still matches.
        link_dir = tmp / p.parent.name
        link_dir.mkdir(exist_ok=True)
        (link_dir / p.name).symlink_to(p.resolve())
    print(f"[limit] processing first {len(paths)} images via {tmp}")
    return tmp


if __name__ == "__main__":
    main()
