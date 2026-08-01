"""
io_npz.py

Saves a SubjectSample as a .npz containing anatomy and targets ONLY - no
PMCX measurement matrices (none exist yet at this stage; this revision
explicitly does not launch PMCX).
"""

from __future__ import annotations

import os

import numpy as np

from anatomy.subject_generator import SubjectSample


def save_subject_npz(sample: SubjectSample, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sample.sample_id}.npz")
    np.savez_compressed(
        path,
        subject_id=sample.subject_id,
        sample_id=sample.sample_id,
        label=np.int8(sample.label),
        subject_seed=sample.subject_seed,
        anatomy_seed=sample.anatomy_seed,
        hemorrhage_seed=sample.hemorrhage_seed,

        tissue_volume=sample.tissue_volume.astype(np.uint8),
        brain_mask=sample.brain_mask.astype(np.uint8),

        scalp_thickness_mm=np.float32(sample.scalp_thickness_mm),
        skull_thickness_mm=np.float32(sample.skull_thickness_mm),
        csf_thickness_mm=np.float32(sample.csf_thickness_mm),
        brain_start_z_mm=np.float32(sample.brain_start_z_mm),

        hemorrhage_mask=sample.hemorrhage_mask.astype(np.uint8),
        hemorrhage_center_vox=sample.hemorrhage_center_vox,
        hemorrhage_center_mm=sample.hemorrhage_center_mm,
        hemorrhage_radii_vox=sample.hemorrhage_radii_vox,
        hemorrhage_radii_mm=sample.hemorrhage_radii_mm,
        hemorrhage_volume_voxels=np.int32(sample.hemorrhage_volume_voxels),
        hemorrhage_volume_mm3=np.float32(sample.hemorrhage_volume_mm3),
        hemorrhage_depth_from_surface_mm=np.float32(sample.hemorrhage_depth_from_surface_mm),
        hemorrhage_depth_from_brain_start_mm=np.float32(sample.hemorrhage_depth_from_brain_start_mm),
        regression_target_valid=np.int8(sample.regression_target_valid),

        source_positions_740_mm=sample.source_positions_740_mm,
        source_positions_850_mm=sample.source_positions_850_mm,
        detector_positions_mm=sample.detector_positions_mm,

        voxel_size_mm=np.float32(sample.voxel_size_mm),
        volume_shape=np.array(sample.volume_shape, dtype=np.int32),

        anatomy_version=sample.anatomy_version,
        geometry_version=sample.geometry_version,
        mapping_version=sample.mapping_version,
    )
    return path


def sample_to_metadata_row(sample: SubjectSample) -> dict:
    return dict(
        sample_id=sample.sample_id,
        subject_id=sample.subject_id,
        label=sample.label,
        subject_seed=sample.subject_seed,
        anatomy_seed=sample.anatomy_seed,
        hemorrhage_seed=sample.hemorrhage_seed,
        voxel_size_mm=sample.voxel_size_mm,
        volume_shape=sample.volume_shape,
        scalp_thickness_mm=sample.scalp_thickness_mm,
        skull_thickness_mm=sample.skull_thickness_mm,
        csf_thickness_mm=sample.csf_thickness_mm,
        brain_start_z_mm=sample.brain_start_z_mm,
        hemorrhage_center_x_mm=sample.hemorrhage_center_mm[0],
        hemorrhage_center_y_mm=sample.hemorrhage_center_mm[1],
        hemorrhage_center_z_mm=sample.hemorrhage_center_mm[2],
        hemorrhage_radii_x_mm=sample.hemorrhage_radii_mm[0],
        hemorrhage_radii_y_mm=sample.hemorrhage_radii_mm[1],
        hemorrhage_radii_z_mm=sample.hemorrhage_radii_mm[2],
        hemorrhage_volume_voxels=sample.hemorrhage_volume_voxels,
        hemorrhage_volume_mm3=sample.hemorrhage_volume_mm3,
        hemorrhage_depth_from_surface_mm=sample.hemorrhage_depth_from_surface_mm,
        hemorrhage_depth_from_brain_start_mm=sample.hemorrhage_depth_from_brain_start_mm,
        regression_target_valid=sample.regression_target_valid,
        anatomy_version=sample.anatomy_version,
        geometry_version=sample.geometry_version,
        mapping_version=sample.mapping_version,
    )
