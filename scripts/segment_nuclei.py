"""
segment_nuclei.py  —  STAGE 1 of the cell-graph pipeline (torch-free)
====================================================================
Run nucleus detection (StarDist by default) over a directory of H&E images and
cache the per-image nuclei (centroids + 10-dim features) to disk.

**Why a separate script:** StarDist (TensorFlow) and PyTorch deadlock when
imported in the same process on macOS. This script imports ONLY the torch-free
``util.nuclei`` module, so it never loads torch. Stage 2
(scripts/build_graphs.py --nuclei-cache ...) reads this cache and builds the PyG
graphs in a torch process that never imports TensorFlow.

Usage
-----
    uv run python scripts/segment_nuclei.py                       # StarDist → cache
    uv run python scripts/segment_nuclei.py --method watershed
    uv run python scripts/segment_nuclei.py --limit 20            # quick test
    uv run python scripts/segment_nuclei.py \
        --image-dir data/lymphoma --out data/graphs/nuclei_stardist.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# IMPORTANT: import only the torch-free module. Do NOT import torch / models /
# util.graph_construction here, or StarDist will deadlock.
from util.nuclei import detect_nuclei, load_image

assert "torch" not in sys.modules, "torch must not be imported in the StarDist stage"

LYMPHOMA_LABEL_MAP = {"CLL": 0, "FL": 1, "MCL": 2}
EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, default=Path("data/lymphoma"))
    parser.add_argument("--out", type=Path, default=None,
                        help="output .pkl (default: data/graphs/nuclei_<method>.pkl)")
    parser.add_argument("--method", choices=["stardist", "watershed"], default="stardist")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N images (quick test)")
    args = parser.parse_args()

    if not args.image_dir.exists():
        sys.exit(f"Image dir not found: {args.image_dir}\n"
                 f"Run: uv run python scripts/download_data.py --dataset lymphoma")

    out_path = args.out or Path("data/graphs") / f"nuclei_{args.method}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.image_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if args.limit is not None:
        paths = paths[:args.limit]
    print(f"Found {len(paths)} images | method={args.method} | out={out_path}")

    records = []
    t0 = time.perf_counter()
    for i, path in enumerate(paths):
        label_y = next((v for k, v in LYMPHOMA_LABEL_MAP.items()
                        if k.lower() in str(path).lower()), -1)
        try:
            rgb = load_image(path)
        except Exception as e:           # noqa: BLE001 - skip unreadable files
            print(f"  [skip] {path.name}: {e}")
            continue

        centroids, feats = detect_nuclei(rgb, method=args.method)
        records.append({
            "path":      str(path),
            "label_y":   label_y,
            "centroids": centroids,
            "feats":     feats,
            "shape":     tuple(rgb.shape[:2]),
        })
        if (i + 1) % 20 == 0 or (i + 1) == len(paths):
            rate = (time.perf_counter() - t0) / (i + 1)
            print(f"  {i+1}/{len(paths)} | last={len(centroids)} nuclei "
                  f"| {rate:.2f}s/img", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump({"method": args.method, "records": records}, f)

    n_nuclei = [len(r["centroids"]) for r in records]
    mean_n = sum(n_nuclei) / max(len(n_nuclei), 1)
    print(f"\nSaved {len(records)} records → {out_path}")
    print(f"Nuclei per image: mean {mean_n:.0f}, min {min(n_nuclei, default=0)}, "
          f"max {max(n_nuclei, default=0)}")


if __name__ == "__main__":
    main()
