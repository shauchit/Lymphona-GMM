"""
util — graph-construction tools for H&E pathology images.

Turn an H&E patch into a ``torch_geometric.data.Data`` graph, ready for a GNN.

Note on imports
---------------
Names are exposed **lazily** (PEP 562). Importing ``util`` or ``util.nuclei``
does NOT import torch/torch_geometric — this matters because the torch-free
``util.nuclei`` module is used by the StarDist segmentation stage, and StarDist
(TensorFlow) deadlocks if torch is loaded in the same process. The torch-backed
names below are only imported the first time you access them.

Quick start
-----------
    from util import load_image, build_cell_graph, build_patch_graph

    rgb = load_image("patch.png")
    cell_data,  cell_stats  = build_cell_graph(rgb,  label_y=0)
    patch_data, patch_stats = build_patch_graph(rgb, label_y=0)
"""

import importlib

# name -> submodule that defines it (imported on first access)
_LAZY = {
    "load_image":                 "util.nuclei",
    "FEATURE_DIM":                "util.nuclei",
    "detect_nuclei":              "util.nuclei",
    "build_cell_graph":           "util.graph_construction",
    "build_cell_graph_from_nuclei": "util.graph_construction",
    "build_patch_graph":          "util.graph_construction",
    "process_dataset":            "util.graph_construction",
    "summarise_stats":            "util.graph_construction",
    "GraphStats":                 "util.graph_construction",
    "DELAUNAY_DIST_THRESHOLD_PX": "util.graph_construction",
    "SLIC_N_SEGMENTS":            "util.graph_construction",
    "SLIC_COMPACTNESS":           "util.graph_construction",
    "KNN_K":                      "util.graph_construction",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module 'util' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)
