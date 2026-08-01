"""
visualize_anatomy.py

Required visualizations for a paired subject:
  - x-y, x-z, y-z slices through the hemorrhage center (healthy vs hemorrhage)
  - healthy-vs-hemorrhage volume comparison (part of the slice figure)
  - multi-slice hemorrhage-mask visualization
  - PCB sources/detectors relative to the volume
  - layer-thickness visualization

All axes are labeled in millimeters where the axis represents a physical
quantity.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

from anatomy.labels import AIR, SCALP, SKULL, CSF, BRAIN, HEMORRHAGE, LABEL_NAMES
from anatomy.subject_generator import SubjectSample

_LABEL_COLORS = {
    AIR: "#f0f0f0", SCALP: "#e8b98a", SKULL: "#d9d9d9",
    CSF: "#8ecae6", BRAIN: "#a8dadc", HEMORRHAGE: "#e63946",
}
_CMAP = ListedColormap([_LABEL_COLORS[i] for i in range(6)])
_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1), _CMAP.N)


def _label_legend():
    return [Line2D([0], [0], marker="s", color="w", markerfacecolor=_LABEL_COLORS[i],
                    markersize=12, label=LABEL_NAMES[i]) for i in range(6)]


def plot_pair_slices(healthy: SubjectSample, hemorrhage: SubjectSample, out_path: str) -> None:
    """2x3 grid: healthy (top) vs hemorrhage (bottom), xy/xz/yz slices
    through the hemorrhage center, with mm axes."""
    vox = hemorrhage.voxel_size_mm
    cx, cy, cz = [int(v) for v in hemorrhage.hemorrhage_center_vox]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    def show(ax, arr2d, extent, title):
        ax.imshow(arr2d.T, origin="lower", cmap=_CMAP, norm=_NORM, extent=extent, aspect="auto")
        ax.set_title(title, fontsize=10)

    dim_x, dim_y, dim_z = healthy.volume_shape
    extent_xy = [0, dim_x * vox, 0, dim_y * vox]
    extent_xz = [0, dim_x * vox, 0, dim_z * vox]
    extent_yz = [0, dim_y * vox, 0, dim_z * vox]

    show(axes[0, 0], healthy.tissue_volume[:, :, cz], extent_xy, f"healthy x-y @ z={cz*vox:.1f}mm")
    show(axes[0, 1], healthy.tissue_volume[:, cy, :], extent_xz, f"healthy x-z @ y={cy*vox:.1f}mm")
    show(axes[0, 2], healthy.tissue_volume[cx, :, :], extent_yz, f"healthy y-z @ x={cx*vox:.1f}mm")

    show(axes[1, 0], hemorrhage.tissue_volume[:, :, cz], extent_xy, f"hemorrhage x-y @ z={cz*vox:.1f}mm")
    show(axes[1, 1], hemorrhage.tissue_volume[:, cy, :], extent_xz, f"hemorrhage x-z @ y={cy*vox:.1f}mm")
    show(axes[1, 2], hemorrhage.tissue_volume[cx, :, :], extent_yz, f"hemorrhage y-z @ x={cx*vox:.1f}mm")

    for ax in axes.flat:
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")

    fig.suptitle(
        f"{hemorrhage.subject_id}: healthy vs hemorrhage, slices through hemorrhage center "
        f"({cx*vox:.1f}, {cy*vox:.1f}, {cz*vox:.1f}) mm",
        fontsize=12,
    )
    fig.legend(handles=_label_legend(), loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_hemorrhage_multislice(hemorrhage: SubjectSample, out_path: str, n_slices: int = 6) -> None:
    vox = hemorrhage.voxel_size_mm
    cz = int(hemorrhage.hemorrhage_center_vox[2])
    rz = int(hemorrhage.hemorrhage_radii_vox[2])
    dim_z = hemorrhage.volume_shape[2]

    z_indices = np.linspace(max(0, cz - rz - 1), min(dim_z - 1, cz + rz + 1), n_slices).astype(int)
    z_indices = sorted(set(z_indices.tolist()))

    fig, axes = plt.subplots(1, len(z_indices), figsize=(3 * len(z_indices), 3.2))
    if len(z_indices) == 1:
        axes = [axes]
    for ax, z in zip(axes, z_indices):
        ax.imshow(hemorrhage.hemorrhage_mask[:, :, z].T, origin="lower", cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"z={z*vox:.1f}mm", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{hemorrhage.subject_id}: hemorrhage mask multi-slice (x-y, around center)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_layer_thickness(sample: SubjectSample, out_path: str) -> None:
    """Step chart of tissue label vs. depth (z, mm) along the volume
    center column, i.e. a 1-D profile of the layered phantom."""
    vox = sample.voxel_size_mm
    dim_x, dim_y, dim_z = sample.volume_shape
    profile = sample.tissue_volume[dim_x // 2, dim_y // 2, :]
    z_mm = np.arange(dim_z) * vox

    fig, ax = plt.subplots(figsize=(4, 6))
    for z, lbl in zip(z_mm, profile):
        ax.barh(z, 1, height=vox, color=_LABEL_COLORS[int(lbl)], edgecolor="none")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("depth z (mm)")
    ax.invert_yaxis()
    ax.set_title(f"{sample.sample_id}\nlayer profile (center column)\n"
                 f"scalp={sample.scalp_thickness_mm:.1f}mm skull={sample.skull_thickness_mm:.1f}mm "
                 f"csf={sample.csf_thickness_mm:.1f}mm", fontsize=9)
    ax.legend(handles=_label_legend(), loc="upper right", fontsize=7, bbox_to_anchor=(1.6, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sensors_on_volume(sample: SubjectSample, out_path: str) -> None:
    """Top-down (x-y) view of the volume boundary with PCB-derived source
    and detector positions, at this stage's flat-phantom placement."""
    vox = sample.voxel_size_mm
    dim_x, dim_y, dim_z = sample.volume_shape

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Rectangle((0, 0), dim_x * vox, dim_y * vox, fill=False,
                                edgecolor="black", linestyle="--", label="volume boundary"))

    for x, y, _ in sample.source_positions_740_mm:
        ax.scatter(x, y, marker="^", s=120, c="crimson", edgecolor="black", zorder=3)
    for x, y, _ in sample.source_positions_850_mm:
        ax.scatter(x, y, marker="^", s=120, c="darkorange", edgecolor="black", zorder=3)
    for x, y, _ in sample.detector_positions_mm:
        ax.scatter(x, y, marker="o", s=90, c="royalblue", edgecolor="black", zorder=3)

    ax.set_xlabel("head x (mm)")
    ax.set_ylabel("head y (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"{sample.subject_id}: PCB sources/detectors over simulation volume\n"
                 f"(flat-phantom placement, surface_z - see README)", fontsize=10)
    ax.legend(handles=[
        Line2D([0], [0], marker="^", color="w", markerfacecolor="crimson", markersize=10, label="LED 740nm"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="darkorange", markersize=10, label="LED 850nm"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue", markersize=10, label="Detector"),
        Line2D([0], [0], color="black", linestyle="--", label="volume boundary"),
    ], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
