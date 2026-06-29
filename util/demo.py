"""
Demo / smoke-test for the graph-construction utilities.

Run with:
    uv run python -m util.demo

Uses synthetic H&E-like patches (no dataset download required) and writes
comparison figures + a degree-distribution plot to ``util_demo_output/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from skimage import segmentation

from .graph_construction import (
    SLIC_COMPACTNESS,
    SLIC_N_SEGMENTS,
    build_cell_graph,
    build_patch_graph,
    summarise_stats,
)
from .visualization import (
    generate_synthetic_he_patch,
    visualise_comparison,
    visualise_degree_distribution,
)


def main() -> None:
    out = Path("util_demo_output")
    out.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Graph Construction Utils — Demo (synthetic H&E patches)")
    print("=" * 60)

    all_cell_stats, all_patch_stats = [], []

    for img_idx in range(6):
        rgb = generate_synthetic_he_patch(size=256, n_nuclei=55 + img_idx * 5, seed=img_idx)

        cg, cs = build_cell_graph(rgb,  label_y=img_idx % 3)
        pg, ps = build_patch_graph(rgb, label_y=img_idx % 3)
        all_cell_stats.append(cs)
        all_patch_stats.append(ps)

        print(f"\n  Image {img_idx + 1}")
        print(f"    Cell-Graph  : {cs.num_nodes} nodes, {cs.num_edges} edges, "
              f"deg={cs.avg_degree:.2f}, t={cs.construction_time:.3f}s")
        print(f"    Patch-Graph : {ps.num_nodes} nodes, {ps.num_edges} edges, "
              f"deg={ps.avg_degree:.2f}, t={ps.construction_time:.3f}s")

        if img_idx < 3:
            seg_map = segmentation.slic(
                rgb.astype(float) / 255.0, n_segments=SLIC_N_SEGMENTS,
                compactness=SLIC_COMPACTNESS, channel_axis=2, start_label=0,
            )
            fig = visualise_comparison(
                rgb, cg, pg, seg_map=seg_map,
                save_path=out / f"comparison_img{img_idx + 1}.png",
                title=f"Graph Construction Comparison — Sample {img_idx + 1}",
            )
            plt.close(fig)
            print(f"    → saved comparison_img{img_idx + 1}.png")

    fig2 = visualise_degree_distribution(
        all_cell_stats, all_patch_stats, save_path=out / "degree_distributions.png"
    )
    plt.close(fig2)

    print("\n" + "=" * 60)
    print("  Aggregate Statistics")
    print("=" * 60)
    for stats_list in (all_cell_stats, all_patch_stats):
        s = summarise_stats(stats_list)
        print(f"\n  {s['strategy'].upper().replace('_', '-')}")
        for k, v in s.items():
            if k != "strategy":
                print(f"    {k:25s}: {v}")

    print(f"\nAll outputs saved to {out.resolve()}")


if __name__ == "__main__":
    main()
