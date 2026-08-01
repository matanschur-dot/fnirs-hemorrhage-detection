"""
subject_generator.py

The single, unified generator: builds ONE synthetic subject (one anatomy,
one set of layer thicknesses, one PCB-derived sensor geometry), then
derives a matched healthy sample and hemorrhage sample from it. There is
no separate "healthy generator" / "hemorrhage generator" code path - both
samples are produced from the same in-memory volume by this one function,
which is what pairing.validate_pair() then checks.

Seed handling: `subject_seed = master_seed + subject_id`. Anatomy (layer
thicknesses) is drawn from a Generator seeded with `subject_seed`.
Hemorrhage geometry is drawn from a SEPARATE Generator seeded with
`subject_seed + HEMORRHAGE_SEED_OFFSET`, so hemorrhage randomness can
never perturb the anatomy draw order (and vice versa), while both remain
fully deterministic for a given subject_id.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from anatomy.head_model import build_layered_volume, build_brain_mask, LayerMeta
from anatomy.hemorrhage import (
    inject_hemorrhage, healthy_hemorrhage_targets, hemorrhage_targets,
    HemorrhageGenerationError,
)
from anatomy.pairing import validate_pair, PairValidationResult
from configs.anatomy_config import AnatomyRunConfig, ANATOMY_VERSION
from configs.geometry_config import VolumeConfig
from configs.hardware_mapping import MAPPING_VERSION
from configs.geometry_config import GEOMETRY_VERSION

HEMORRHAGE_SEED_OFFSET = 1_000_000


@dataclass
class SensorGeometry:
    """PCB-derived sensor positions carried through from the geometry
    stage, unchanged, so anatomy/hemorrhage samples stay consistent with
    the verified hardware geometry. No PMCX / optical simulation happens
    here."""
    source_positions_740_mm: np.ndarray   # (8, 3)
    source_positions_850_mm: np.ndarray   # (8, 3)
    detector_positions_mm: np.ndarray     # (16, 3)


@dataclass
class SubjectSample:
    subject_id: str
    sample_id: str
    label: int
    subject_seed: int
    anatomy_seed: int
    hemorrhage_seed: int

    tissue_volume: np.ndarray
    brain_mask: np.ndarray

    scalp_thickness_mm: float
    skull_thickness_mm: float
    csf_thickness_mm: float
    brain_start_z_mm: float

    hemorrhage_mask: np.ndarray
    hemorrhage_center_vox: np.ndarray
    hemorrhage_center_mm: np.ndarray
    hemorrhage_radii_vox: np.ndarray
    hemorrhage_radii_mm: np.ndarray
    hemorrhage_volume_voxels: int
    hemorrhage_volume_mm3: float
    hemorrhage_depth_from_surface_mm: float
    hemorrhage_depth_from_brain_start_mm: float
    regression_target_valid: int

    source_positions_740_mm: np.ndarray
    source_positions_850_mm: np.ndarray
    detector_positions_mm: np.ndarray

    voxel_size_mm: float
    volume_shape: tuple[int, int, int]

    anatomy_version: str
    geometry_version: str
    mapping_version: str


def build_paired_subject(
    subject_id: str,
    master_seed: int,
    volume_cfg: VolumeConfig,
    anatomy_cfg: AnatomyRunConfig,
    sensors: SensorGeometry,
) -> tuple[SubjectSample, SubjectSample, PairValidationResult, LayerMeta]:
    """Build one subject's anatomy and derive matched healthy + hemorrhage
    samples. Raises HemorrhageGenerationError or pairing.PairValidationError
    rather than returning an invalid pair."""

    numeric_id = _stable_int_hash(subject_id)
    subject_seed = master_seed + numeric_id
    anatomy_seed = subject_seed
    hemorrhage_seed = subject_seed + HEMORRHAGE_SEED_OFFSET

    anatomy_rng = np.random.default_rng(anatomy_seed)
    hemorrhage_rng = np.random.default_rng(hemorrhage_seed)

    healthy_volume, layer_meta = build_layered_volume(volume_cfg, anatomy_cfg.layers, anatomy_rng)
    brain_mask = build_brain_mask(healthy_volume)

    hemorrhage_volume, hem_meta = inject_hemorrhage(
        healthy_volume, brain_mask, layer_meta, volume_cfg, anatomy_cfg.hemorrhage, hemorrhage_rng,
        sensor_xy_mm=np.vstack([sensors.source_positions_740_mm[:, :2], sensors.source_positions_850_mm[:, :2], sensors.detector_positions_mm[:, :2]]),
    )

    healthy_targets = healthy_hemorrhage_targets(volume_cfg)
    hem_targets = hemorrhage_targets(hem_meta, hemorrhage_volume)

    pair_result = validate_pair(healthy_volume, hemorrhage_volume, hem_targets["hemorrhage_mask"])

    src740 = sensors.source_positions_740_mm.copy()
    src850 = sensors.source_positions_850_mm.copy()
    dets = sensors.detector_positions_mm.copy()
    src740[:, 2] = layer_meta.tissue_surface_z_mm + anatomy_cfg.placement.source_offset_into_scalp_mm
    src850[:, 2] = layer_meta.tissue_surface_z_mm + anatomy_cfg.placement.source_offset_into_scalp_mm
    dets[:, 2] = layer_meta.tissue_surface_z_mm + anatomy_cfg.placement.detector_offset_into_scalp_mm

    shared = dict(
        subject_id=subject_id,
        subject_seed=subject_seed, anatomy_seed=anatomy_seed, hemorrhage_seed=hemorrhage_seed,
        brain_mask=brain_mask,
        scalp_thickness_mm=layer_meta.scalp_thickness_mm,
        skull_thickness_mm=layer_meta.skull_thickness_mm,
        csf_thickness_mm=layer_meta.csf_thickness_mm,
        brain_start_z_mm=layer_meta.brain_start_z_mm,
        source_positions_740_mm=src740,
        source_positions_850_mm=src850,
        detector_positions_mm=dets,
        voxel_size_mm=volume_cfg.voxel_size_mm,
        volume_shape=(volume_cfg.dim_x, volume_cfg.dim_y, volume_cfg.dim_z),
        anatomy_version=ANATOMY_VERSION, geometry_version=GEOMETRY_VERSION, mapping_version=MAPPING_VERSION,
    )

    healthy_sample = SubjectSample(
        sample_id=f"{subject_id}_healthy", tissue_volume=healthy_volume, **healthy_targets, **shared,
    )
    hemorrhage_sample = SubjectSample(
        sample_id=f"{subject_id}_hemorrhage", tissue_volume=hemorrhage_volume, **hem_targets, **shared,
    )

    return healthy_sample, hemorrhage_sample, pair_result, layer_meta


def _stable_int_hash(subject_id: str) -> int:
    """Deterministic (across processes/runs - unlike Python's built-in
    hash() for strings) small non-negative int derived from subject_id,
    used to offset master_seed. Not cryptographic; just needs to be stable
    and well-distributed enough that different subject_ids get different
    seeds."""
    import hashlib
    h = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16)
