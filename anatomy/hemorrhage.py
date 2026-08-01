"""
hemorrhage.py

Injects a 3D ellipsoidal hemorrhage into a copy of a healthy layered
volume, entirely within brain tissue. Retries with fresh random
center/radii up to `max_placement_attempts` and raises
HemorrhageGenerationError (rather than silently placing an invalid
sample) if it can't find a valid placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anatomy.labels import BRAIN, HEMORRHAGE
from anatomy.head_model import LayerMeta
from configs.anatomy_config import HemorrhageConfig
from configs.geometry_config import VolumeConfig


class HemorrhageGenerationError(Exception):
    pass


@dataclass
class HemorrhageMeta:
    center_vox: tuple[int, int, int]
    center_mm: tuple[float, float, float]
    radii_vox: tuple[int, int, int]
    radii_mm: tuple[float, float, float]
    volume_voxels: int
    volume_mm3: float
    depth_from_surface_mm: float
    depth_from_brain_start_mm: float
    n_attempts_used: int


def _ellipsoid_voxel_indices(cx, cy, cz, rx, ry, rz, dim_x, dim_y, dim_z):
    x_lo, x_hi = max(0, cx - rx), min(dim_x - 1, cx + rx)
    y_lo, y_hi = max(0, cy - ry), min(dim_y - 1, cy + ry)
    z_lo, z_hi = max(0, cz - rz), min(dim_z - 1, cz + rz)

    xs, ys, zs = np.meshgrid(
        np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1), np.arange(z_lo, z_hi + 1),
        indexing="ij",
    )
    ell = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 + ((zs - cz) / rz) ** 2
    inside = ell <= 1.0
    return xs[inside], ys[inside], zs[inside]


def inject_hemorrhage(
    healthy_volume: np.ndarray,
    brain_mask: np.ndarray,
    layer_meta: LayerMeta,
    volume_cfg: VolumeConfig,
    hem_cfg: HemorrhageConfig,
    rng: np.random.Generator,
    sensor_xy_mm: np.ndarray | None = None,
) -> tuple[np.ndarray, HemorrhageMeta]:

    dim_x, dim_y, dim_z = volume_cfg.dim_x, volume_cfg.dim_y, volume_cfg.dim_z
    vox = volume_cfg.voxel_size_mm
    margin = hem_cfg.xy_margin_vox

    if not brain_mask.any():
        raise HemorrhageGenerationError("brain_mask is empty; cannot place a hemorrhage.")

    brain_z_indices = np.where(brain_mask.any(axis=(0, 1)))[0]
    z_brain_lo, z_brain_hi = int(brain_z_indices.min()), int(brain_z_indices.max())

    last_reason = "unknown"
    for attempt in range(1, hem_cfg.max_placement_attempts + 1):
        rx = int(rng.integers(hem_cfg.rx_range_vox[0], hem_cfg.rx_range_vox[1] + 1))
        ry = int(rng.integers(hem_cfg.ry_range_vox[0], hem_cfg.ry_range_vox[1] + 1))
        rz = int(rng.integers(hem_cfg.rz_range_vox[0], hem_cfg.rz_range_vox[1] + 1))

        x_lo, x_hi = margin + rx, dim_x - 1 - margin - rx
        y_lo, y_hi = margin + ry, dim_y - 1 - margin - ry
        if sensor_xy_mm is not None and len(sensor_xy_mm) and np.ptp(sensor_xy_mm[:,0]) > 5*vox and np.ptp(sensor_xy_mm[:,1]) > 5*vox:
            sx0 = int(np.floor(sensor_xy_mm[:, 0].min() / vox + hem_cfg.sensor_coverage_margin_mm / vox))
            sx1 = int(np.ceil(sensor_xy_mm[:, 0].max() / vox - hem_cfg.sensor_coverage_margin_mm / vox))
            sy0 = int(np.floor(sensor_xy_mm[:, 1].min() / vox + hem_cfg.sensor_coverage_margin_mm / vox))
            sy1 = int(np.ceil(sensor_xy_mm[:, 1].max() / vox - hem_cfg.sensor_coverage_margin_mm / vox))
            x_lo, x_hi = max(x_lo, sx0 + rx), min(x_hi, sx1 - rx)
            y_lo, y_hi = max(y_lo, sy0 + ry), min(y_hi, sy1 - ry)
        requested_z_lo = int(np.ceil((layer_meta.tissue_surface_z_mm + hem_cfg.depth_range_mm[0]) / vox))
        requested_z_hi = int(np.floor((layer_meta.tissue_surface_z_mm + hem_cfg.depth_range_mm[1]) / vox))
        z_lo = max(z_brain_lo + rz, requested_z_lo)
        z_hi = min(z_brain_hi - rz, requested_z_hi)

        if x_lo > x_hi or y_lo > y_hi or z_lo > z_hi:
            last_reason = (
                f"radii (rx={rx},ry={ry},rz={rz}) too large for the available "
                f"brain volume with margin={margin} at attempt {attempt}."
            )
            continue

        cx = int(rng.integers(x_lo, x_hi + 1))
        cy = int(rng.integers(y_lo, y_hi + 1))
        cz = int(rng.integers(z_lo, z_hi + 1))

        xs, ys, zs = _ellipsoid_voxel_indices(cx, cy, cz, rx, ry, rz, dim_x, dim_y, dim_z)
        if xs.size == 0:
            last_reason = f"ellipsoid at attempt {attempt} produced zero voxels."
            continue

        if hem_cfg.require_fully_inside_brain:
            all_inside_brain = bool(brain_mask[xs, ys, zs].all())
            if not all_inside_brain:
                last_reason = (
                    f"attempt {attempt}: ellipsoid at center=({cx},{cy},{cz}) "
                    f"radii=({rx},{ry},{rz}) extends outside brain tissue."
                )
                continue

        # success
        hemorrhage_volume = healthy_volume.copy()
        hemorrhage_volume[xs, ys, zs] = HEMORRHAGE
        n_voxels = int(xs.size)
        voxel_volume_mm3 = vox ** 3

        center_mm = (cx * vox, cy * vox, cz * vox)
        depth_from_surface_mm = cz * vox - layer_meta.tissue_surface_z_mm
        depth_from_brain_start_mm = (cz - layer_meta.brain_start_z_vox) * vox

        meta = HemorrhageMeta(
            center_vox=(cx, cy, cz), center_mm=center_mm,
            radii_vox=(rx, ry, rz), radii_mm=(rx * vox, ry * vox, rz * vox),
            volume_voxels=n_voxels, volume_mm3=n_voxels * voxel_volume_mm3,
            depth_from_surface_mm=depth_from_surface_mm,
            depth_from_brain_start_mm=depth_from_brain_start_mm,
            n_attempts_used=attempt,
        )
        return hemorrhage_volume, meta

    raise HemorrhageGenerationError(
        f"Failed to place a valid hemorrhage entirely inside brain tissue after "
        f"{hem_cfg.max_placement_attempts} attempts. Last reason: {last_reason}. "
        f"Consider widening the brain region (larger volume / thinner scalp+skull+CSF) "
        f"or narrowing HemorrhageConfig radius ranges."
    )


def healthy_hemorrhage_targets(volume_cfg: VolumeConfig) -> dict:
    """The fixed invalid-value convention for healthy (non-hemorrhage) samples."""
    return dict(
        label=0,
        hemorrhage_mask=np.zeros((volume_cfg.dim_x, volume_cfg.dim_y, volume_cfg.dim_z), dtype=np.uint8),
        hemorrhage_center_vox=np.array([-1, -1, -1], dtype=np.int32),
        hemorrhage_center_mm=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        hemorrhage_radii_vox=np.array([0, 0, 0], dtype=np.int32),
        hemorrhage_radii_mm=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        hemorrhage_volume_voxels=0,
        hemorrhage_volume_mm3=0.0,
        hemorrhage_depth_from_surface_mm=-1.0,
        hemorrhage_depth_from_brain_start_mm=-1.0,
        regression_target_valid=0,
    )


def hemorrhage_targets(meta: HemorrhageMeta, hemorrhage_volume: np.ndarray) -> dict:
    mask = (hemorrhage_volume == HEMORRHAGE).astype(np.uint8)
    return dict(
        label=1,
        hemorrhage_mask=mask,
        hemorrhage_center_vox=np.array(meta.center_vox, dtype=np.int32),
        hemorrhage_center_mm=np.array(meta.center_mm, dtype=np.float32),
        hemorrhage_radii_vox=np.array(meta.radii_vox, dtype=np.int32),
        hemorrhage_radii_mm=np.array(meta.radii_mm, dtype=np.float32),
        hemorrhage_volume_voxels=meta.volume_voxels,
        hemorrhage_volume_mm3=meta.volume_mm3,
        hemorrhage_depth_from_surface_mm=meta.depth_from_surface_mm,
        hemorrhage_depth_from_brain_start_mm=meta.depth_from_brain_start_mm,
        regression_target_valid=1,
    )
