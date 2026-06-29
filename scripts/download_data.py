"""
download_data.py
================
Download the datasets used by the lymphoma graph pipeline into ``./data``.

Datasets
--------
lymphoma : Malignant Lymphoma classification (CLL / FL / MCL, 374 images)
           Source: Kaggle `andrewmvd/malignant-lymphoma-classification`
           Downloaded via kagglehub (needs Kaggle credentials, see below).
nct      : NCT-CRC-HE-100K colorectal H&E patches (224x224) + CRC-VAL-HE-7K
           Source: Zenodo record 1214456 (public, no auth, several GB).

Usage
-----
    uv run python scripts/download_data.py --dataset all
    uv run python scripts/download_data.py --dataset lymphoma
    uv run python scripts/download_data.py --dataset nct --nct-val-only
    uv run python scripts/download_data.py --dataset nct --data-dir ./data

Kaggle credentials (for the lymphoma dataset)
---------------------------------------------
kagglehub reads ~/.kaggle/kaggle.json (or KAGGLE_USERNAME / KAGGLE_KEY env
vars). Create a token at https://www.kaggle.com/settings -> "Create New Token",
then place the downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset definitions
# ─────────────────────────────────────────────────────────────────────────────
LYMPHOMA_KAGGLE_SLUG = "andrewmvd/malignant-lymphoma-classification"

ZENODO_BASE = "https://zenodo.org/records/1214456/files"
NCT_FILES = {
    "train": f"{ZENODO_BASE}/NCT-CRC-HE-100K.zip?download=1",   # ~7.4 GB
    "val":   f"{ZENODO_BASE}/CRC-VAL-HE-7K.zip?download=1",     # ~0.8 GB
}


# ─────────────────────────────────────────────────────────────────────────────
#  Generic helpers
# ─────────────────────────────────────────────────────────────────────────────
class _DownloadProgress(tqdm):
    """tqdm hook compatible with urllib.request.urlretrieve."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total: int = -1) -> None:
        if total not in (-1, None):
            self.total = total
        self.update(blocks * block_size - self.n)


def download_file(url: str, dest: Path) -> Path:
    """Stream a URL to ``dest`` with a progress bar; resume-safe via .part."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already downloaded: {dest.name}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {dest.name}")
    with _DownloadProgress(unit="B", unit_scale=True, unit_divisor=1024,
                           miniters=1, desc=f"  {dest.name}") as bar:
        urllib.request.urlretrieve(url, tmp, reporthook=bar.update_to)
    tmp.rename(dest)
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip archive, skipping if it looks already extracted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        top = out_dir / members[0].split("/")[0] if members else None
        if top is not None and top.exists():
            print(f"  [skip] already extracted: {zip_path.name}")
            return
        print(f"  extracting {zip_path.name} → {out_dir}")
        for m in tqdm(members, desc=f"  {zip_path.name}", unit="file"):
            zf.extract(m, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset downloaders
# ─────────────────────────────────────────────────────────────────────────────
def download_lymphoma(data_dir: Path) -> None:
    """Download the Malignant Lymphoma dataset via kagglehub into data_dir."""
    print("\n[lymphoma] Malignant Lymphoma (CLL/FL/MCL, 374 images)")
    try:
        import kagglehub
    except ImportError:
        sys.exit("  kagglehub not installed. Run: uv add kagglehub")

    try:
        cache_path = Path(kagglehub.dataset_download(LYMPHOMA_KAGGLE_SLUG))
    except Exception as e:  # noqa: BLE001 - surface auth/network errors clearly
        sys.exit(
            f"  Kaggle download failed: {e}\n"
            f"  Make sure ~/.kaggle/kaggle.json exists (or KAGGLE_USERNAME / "
            f"KAGGLE_KEY are set). Token: https://www.kaggle.com/settings"
        )

    target = data_dir / "lymphoma"
    if target.exists():
        print(f"  [skip] already present: {target}")
    else:
        print(f"  copying {cache_path} → {target}")
        shutil.copytree(cache_path, target)
    print(f"  done → {target}")


def download_nct(data_dir: Path, val_only: bool = False) -> None:
    """Download NCT-CRC-HE-100K (+ CRC-VAL-HE-7K) from Zenodo into data_dir."""
    print("\n[nct] NCT-CRC-HE-100K colorectal H&E patches")
    target = data_dir / "nct-crc-he-100k"
    keys = ["val"] if val_only else ["train", "val"]
    for key in keys:
        url      = NCT_FILES[key]
        zip_name = url.split("/")[-1].split("?")[0]
        zip_path = target / zip_name
        download_file(url, zip_path)
        extract_zip(zip_path, target)
    print(f"  done → {target}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["lymphoma", "nct", "all"], default="all",
                        help="which dataset(s) to download (default: all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="destination directory (default: ./data)")
    parser.add_argument("--nct-val-only", action="store_true",
                        help="for NCT, fetch only the smaller CRC-VAL-HE-7K set")
    parser.add_argument("--keep-zips", action="store_true",
                        help="keep downloaded .zip archives after extraction")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into {data_dir}")

    if args.dataset in ("lymphoma", "all"):
        download_lymphoma(data_dir)
    if args.dataset in ("nct", "all"):
        download_nct(data_dir, val_only=args.nct_val_only)

    if not args.keep_zips:
        for z in (data_dir / "nct-crc-he-100k").glob("*.zip"):
            z.unlink()
            print(f"  removed archive {z.name}")

    print("\nAll requested downloads complete.")


if __name__ == "__main__":
    main()
