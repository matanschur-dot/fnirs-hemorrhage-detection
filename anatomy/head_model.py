"""Flat layered phantom: 0 air, 1 scalp, 2 skull, 3 CSF, 4 brain."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from anatomy.labels import AIR, SCALP, SKULL, CSF, BRAIN
from configs.anatomy_config import LayerThicknessConfig
from configs.geometry_config import VolumeConfig

@dataclass
class LayerMeta:
    air_thickness_mm: float
    scalp_thickness_mm: float
    skull_thickness_mm: float
    csf_thickness_mm: float
    tissue_surface_z_mm: float
    scalp_start_z_vox: int
    brain_start_z_mm: float
    brain_start_z_vox: int
    scalp_end_z_vox: int
    skull_end_z_vox: int
    csf_end_z_vox: int

def build_layered_volume(volume_cfg: VolumeConfig, thickness_cfg: LayerThicknessConfig,
                         rng: np.random.Generator) -> tuple[np.ndarray, LayerMeta]:
    vox = volume_cfg.voxel_size_mm
    dx, dy, dz = volume_cfg.dim_x, volume_cfg.dim_y, volume_cfg.dim_z
    air_vox = max(1, int(round(thickness_cfg.air_thickness_mm / vox)))
    scalp_vox = max(1, int(round(float(rng.uniform(*thickness_cfg.scalp_thickness_mm_range)) / vox)))
    skull_vox = max(1, int(round(float(rng.uniform(*thickness_cfg.skull_thickness_mm_range)) / vox)))
    csf_vox = max(1, int(round(float(rng.uniform(*thickness_cfg.csf_thickness_mm_range)) / vox)))
    s0 = air_vox
    s1 = s0 + scalp_vox
    s2 = s1 + skull_vox
    s3 = s2 + csf_vox
    assert 0 < s0 < s1 < s2 < s3 < dz, (
        f"Layers do not fit dim_z={dz}: air={s0}, scalp_end={s1}, skull_end={s2}, csf_end={s3}")
    volume = np.full((dx, dy, dz), AIR, dtype=np.uint8)
    volume[:, :, s0:s1] = SCALP
    volume[:, :, s1:s2] = SKULL
    volume[:, :, s2:s3] = CSF
    volume[:, :, s3:] = BRAIN
    meta = LayerMeta(
        air_thickness_mm=s0 * vox,
        scalp_thickness_mm=scalp_vox * vox,
        skull_thickness_mm=skull_vox * vox,
        csf_thickness_mm=csf_vox * vox,
        tissue_surface_z_mm=s0 * vox,
        scalp_start_z_vox=s0,
        brain_start_z_mm=s3 * vox,
        brain_start_z_vox=s3,
        scalp_end_z_vox=s1,
        skull_end_z_vox=s2,
        csf_end_z_vox=s3,
    )
    return volume, meta

def build_brain_mask(volume: np.ndarray) -> np.ndarray:
    return volume == BRAIN
