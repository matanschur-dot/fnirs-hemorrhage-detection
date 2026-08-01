"""
visualize_geometry.py

Two required plots, kept visually distinct on purpose:

1. PCB-coordinate view: raw pcb_x_mm/pcb_y_mm, matplotlib y-axis inverted
   ONLY for display so it visually matches how KiCad itself renders the
   board (KiCad also treats +y as downward on screen). This axis flip is
   a plotting convenience, not a coordinate transform - the numbers
   plotted are the untouched pcb_x_mm/pcb_y_mm values.

2. Head-coordinate view: transformed head_x_mm/head_y_mm/head_z_mm
   (translation-only, see pcb_geometry.py), with the simulation volume
   boundary and an explicit note that the "physical right / posterior /
   inward" axis labels are the placeholder convention from
   configs/geometry_config.py, not a confirmed physical mapping.

Plus a distance histogram with the analysis bin edges from distance_stats.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from configs.hardware_mapping import LED_740_ORDER, LED_850_ORDER, DETECTOR_ORDER
from geometry.channel_map import ChannelRow
from geometry.distance_stats import BIN_EDGES_MM, DistanceStats


def _legend_handles():
    return [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="crimson", markersize=10, label="LED 740nm"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="darkorange", markersize=10, label="LED 850nm"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue", markersize=10, label="Detector"),
    ]


def plot_pcb_view(components, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    def plot_group(refs, marker, color):
        for ref in refs:
            c = components[ref]
            ax.scatter(c.pcb_x_mm, c.pcb_y_mm, marker=marker, s=140, c=color,
                       edgecolor="black", zorder=3)
            ax.annotate(ref, (c.pcb_x_mm, c.pcb_y_mm), textcoords="offset points",
                       xytext=(4, 4), fontsize=8)

    plot_group(LED_740_ORDER, "^", "crimson")
    plot_group(LED_850_ORDER, "^", "darkorange")
    plot_group(DETECTOR_ORDER, "o", "royalblue")

    all_refs = LED_740_ORDER + LED_850_ORDER + DETECTOR_ORDER
    xs = [components[r].pcb_x_mm for r in all_refs]
    ys = [components[r].pcb_y_mm for r in all_refs]
    pad = 5
    ax.add_patch(plt.Rectangle((min(xs) - pad, min(ys) - pad),
                                (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad,
                                fill=False, edgecolor="gray", linestyle=":", label="PCB bounding box"))

    ax.set_xlabel("PCB x (mm)  [KiCad board coordinates, unmodified]")
    ax.set_ylabel("PCB y (mm)  [KiCad board coordinates, unmodified]")
    ax.set_title("PCB-coordinate view\n(matches KiCad on-screen orientation; y-axis inverted for DISPLAY ONLY)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(handles=_legend_handles(), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_head_view(rows: list[ChannelRow], volume_cfg, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    # collect unique source/detector head positions from the channel map
    seen = {}
    for r in rows:
        seen[("src", r.source_reference)] = (r.source_head_x_mm, r.source_head_y_mm, r.wavelength_nm)
        seen[("det", r.detector_reference)] = (r.detector_head_x_mm, r.detector_head_y_mm, None)

    for (kind, ref), (x, y, wl) in seen.items():
        if kind == "det":
            ax.scatter(x, y, marker="o", s=110, c="royalblue", edgecolor="black", zorder=3)
        else:
            color = "crimson" if wl == 740 else "darkorange"
            ax.scatter(x, y, marker="^", s=140, c=color, edgecolor="black", zorder=3)
        ax.annotate(ref, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    dim_x_mm = volume_cfg.dim_x * volume_cfg.voxel_size_mm
    dim_y_mm = volume_cfg.dim_y * volume_cfg.voxel_size_mm
    ax.add_patch(plt.Rectangle((0, 0), dim_x_mm, dim_y_mm, fill=False,
                                edgecolor="black", linewidth=1.5, linestyle="--",
                                label="simulation volume boundary"))

    ax.annotate("+x_head (\"physical right\" - PLACEHOLDER, unconfirmed)",
                (dim_x_mm * 0.5, -4), ha="center", fontsize=8, color="dimgray")
    ax.annotate("+y_head (\"posterior\" - PLACEHOLDER, unconfirmed)",
                (-6, dim_y_mm * 0.5), rotation=90, va="center", fontsize=8, color="dimgray")
    ax.set_title("Head-coordinate view (translation-only transform)\n"
                 "Axis directions are a documented PLACEHOLDER pending physical confirmation "
                 "- see README", fontsize=10)
    ax.set_xlabel("head x (mm)")
    ax.set_ylabel("head y (mm)")
    ax.set_aspect("equal")
    ax.set_xlim(-10, dim_x_mm + 10)
    ax.set_ylim(dim_y_mm + 10, -10)  # keep same on-screen sense as PCB view
    ax.legend(handles=_legend_handles() + [
        Line2D([0], [0], color="black", linestyle="--", label="simulation volume boundary")
    ], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_distance_histogram(rows: list[ChannelRow], stats: DistanceStats, out_path):
    dists = [r.source_detector_distance_mm for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = [e for e in BIN_EDGES_MM if e != float("inf")] + [max(dists) + 1]
    ax.hist(dists, bins=edges, color="steelblue", edgecolor="black")
    ax.axvline(stats.mean_mm, color="crimson", linestyle="--", label=f"mean={stats.mean_mm:.1f}mm")
    ax.axvline(stats.median_mm, color="darkorange", linestyle="--", label=f"median={stats.median_mm:.1f}mm")
    ax.set_xlabel("source-detector distance (mm)")
    ax.set_ylabel("channel count")
    ax.set_title("Source-detector distance distribution (256 channels)\n"
                 "bins are engineering analysis categories, not quality labels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
