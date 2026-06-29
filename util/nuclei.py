"""
nuclei.py
=========
Nucleus detection and per-nucleus feature extraction for H&E images.

**This module is deliberately torch-free.** StarDist (TensorFlow) and PyTorch
deadlock when imported in the same process on macOS, so nuclei segmentation must
run in a process that never imports torch. Keep this module's imports limited to
numpy / OpenCV / scikit-image / scipy / StarDist.

Pipeline:
    detect_nuclei(rgb, method="stardist"|"watershed")
        -> (centroids[N,2], features[N,10])
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label
from skimage import color, filters, measure, morphology, segmentation

warnings.filterwarnings("ignore")

FEATURE_DIM = 18
MIN_NUCLEUS_AREA = 20   # skip regions smaller than this (px²)

# Columns that are scale-variant (raw pixel/length units) and get per-image
# min-max normalised so they sit in [0,1] like the rest. The remaining columns
# are already intrinsically bounded (ratios, [0,1] colour, orientation).
_NORM_COLS = [0, 3, 10, 11, 16, 17]   # area, perim, major/minor axis, hema mean/std


def load_image(path: str | Path) -> np.ndarray:
    """Read an H&E image as uint8 RGB (H×W×3)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _safe_norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; handle constant columns."""
    lo, hi = x.min(0), x.max(0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    return (x - lo) / rng


# ─────────────────────────────────────────────────────────────────────────────
#  Detectors → integer label image (0 = background)
# ─────────────────────────────────────────────────────────────────────────────
def detect_nuclei_watershed(rgb: np.ndarray) -> np.ndarray:
    """Lightweight watershed nucleus detection (no deep learning)."""
    gray   = color.rgb2gray(rgb)
    smooth = filters.gaussian(gray, sigma=1.5)
    thresh = filters.threshold_otsu(smooth)
    binary = smooth < thresh                   # nuclei are darker in H&E
    binary = morphology.remove_small_objects(binary, min_size=30)
    binary = morphology.remove_small_holes(binary, area_threshold=100)
    dist_map = distance_transform_edt(binary)
    seeds, _ = nd_label(dist_map > dist_map.max() * 0.4)
    labels   = segmentation.watershed(-dist_map, seeds, mask=binary)
    return labels.astype(np.int32)


# Lazy module-level cache so the pretrained model is loaded at most once.
_STARDIST_MODEL = None
_STARDIST_FAILED = False


def get_stardist_model():
    """Load (and cache) the StarDist 2D_versatile_he model; None if unavailable."""
    global _STARDIST_MODEL, _STARDIST_FAILED
    if _STARDIST_MODEL is not None or _STARDIST_FAILED:
        return _STARDIST_MODEL
    try:
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from stardist.models import StarDist2D
        _STARDIST_MODEL = StarDist2D.from_pretrained("2D_versatile_he")
    except Exception:
        _STARDIST_FAILED = True
    return _STARDIST_MODEL


def detect_nuclei_stardist(rgb: np.ndarray) -> Optional[np.ndarray]:
    """StarDist 2D_versatile_he instance segmentation; None if unavailable.

    Must run in a torch-free process (see module docstring)."""
    model = get_stardist_model()
    if model is None:
        return None
    try:
        from csbdeep.utils import normalize
        norm      = normalize(rgb, 1, 99.8, axis=(0, 1))
        labels, _ = model.predict_instances(norm)
        return labels.astype(np.int32)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-nucleus features
# ─────────────────────────────────────────────────────────────────────────────
def extract_cell_features(
    rgb: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each labelled nucleus compute an 18-dim feature vector.

    Shape (0-6, 10-12):
      0 area            1 eccentricity   2 solidity      3 perimeter
      4 compactness     5 orientation    6 extent
      10 major axis     11 minor axis    12 axis ratio (minor/major)
    Colour (7-9, 13-15):
      7-9   mean R/G/B          13-15 std R/G/B (colour heterogeneity)
    Chromatin / texture (16-17):
      16 mean haematoxylin (rgb2hed H — DNA/chromatin density)
      17 std  haematoxylin (chromatin texture / heterogeneity)

    Scale-variant columns (_NORM_COLS) are min-max normalised per image; the
    rest are already bounded.

    Returns
    -------
    centroids : (N, 2) float  [row, col]
    features  : (N, 18) float32
    """
    # Precompute whole-image stain channels once (cheap vs per-nucleus).
    hema = color.rgb2hed(rgb)[:, :, 0]          # haematoxylin optical density

    props = measure.regionprops(labels, intensity_image=rgb)
    centroids, feats = [], []
    for p in props:
        if p.area < MIN_NUCLEUS_AREA:
            continue
        cy, cx  = p.centroid
        area    = p.area
        perim   = max(p.perimeter, 1e-6)
        compact = (4 * np.pi * area) / (perim ** 2)
        major   = p.major_axis_length
        minor   = p.minor_axis_length
        axis_ratio = minor / (major + 1e-6)

        minr, minc, maxr, maxc = p.bbox
        patch_rgb = rgb[minr:maxr, minc:maxc]
        mask_roi  = labels[minr:maxr, minc:maxc] == p.label
        r_vals    = patch_rgb[:, :, 0][mask_roi].astype(np.float32)
        g_vals    = patch_rgb[:, :, 1][mask_roi].astype(np.float32)
        b_vals    = patch_rgb[:, :, 2][mask_roi].astype(np.float32)
        hema_vals = hema[minr:maxr, minc:maxc][mask_roi]

        feat = np.array([
            area,
            p.eccentricity,
            p.solidity,
            perim,
            compact,
            (p.orientation + np.pi / 2) / np.pi,   # → [0,1]
            p.extent,
            r_vals.mean() / 255.0,
            g_vals.mean() / 255.0,
            b_vals.mean() / 255.0,
            major,
            minor,
            axis_ratio,
            r_vals.std() / 255.0,
            g_vals.std() / 255.0,
            b_vals.std() / 255.0,
            hema_vals.mean(),
            hema_vals.std(),
        ], dtype=np.float32)
        centroids.append([cy, cx])
        feats.append(feat)

    centroids = np.array(centroids, dtype=np.float32).reshape(-1, 2)
    feats     = np.array(feats,     dtype=np.float32).reshape(-1, FEATURE_DIM)
    if len(feats):
        feats[:, _NORM_COLS] = _safe_norm(feats[:, _NORM_COLS])
    return centroids, feats


def detect_nuclei(
    rgb: np.ndarray, method: str = "stardist"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect nuclei and return (centroids[N,2], features[N,18]).

    method:
      "stardist"  — StarDist 2D_versatile_he (torch-free process only); falls
                    back to watershed if StarDist is unavailable.
      "watershed" — classical watershed (safe in any process).
    """
    labels = None
    if method == "stardist":
        labels = detect_nuclei_stardist(rgb)
    if labels is None:
        labels = detect_nuclei_watershed(rgb)
    return extract_cell_features(rgb, labels)
