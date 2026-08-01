"""
validate_anatomy.py

Per-subject and pilot-level structural checks for the anatomy/hemorrhage
stage, plus a human-readable report writer.
"""

from __future__ import annotations

import numpy as np

from anatomy.labels import AIR, SCALP, SKULL, CSF, BRAIN, HEMORRHAGE
from anatomy.subject_generator import SubjectSample
from anatomy.pairing import PairValidationResult


def validate_layer_order(sample: SubjectSample) -> None:
    dim_x, dim_y, dim_z = sample.volume_shape
    profile = sample.tissue_volume[dim_x // 2, dim_y // 2, :]
    z1 = int(round(sample.scalp_thickness_mm / sample.voxel_size_mm))
    z2 = z1 + int(round(sample.skull_thickness_mm / sample.voxel_size_mm))
    z3 = z2 + int(round(sample.csf_thickness_mm / sample.voxel_size_mm))
    assert z1 < z2 < z3 < dim_z, f"Invalid layer order/thicknesses for {sample.sample_id}: z1={z1} z2={z2} z3={z3} dim_z={dim_z}"
    assert set(np.unique(profile[:z1])) <= {SCALP}, f"Non-scalp voxel found in scalp region of {sample.sample_id}"
    assert set(np.unique(profile[z1:z2])) <= {SKULL}, f"Non-skull voxel found in skull region of {sample.sample_id}"
    assert set(np.unique(profile[z2:z3])) <= {CSF}, f"Non-CSF voxel found in CSF region of {sample.sample_id}"


def validate_brain_mask_nonempty(sample: SubjectSample) -> None:
    assert sample.brain_mask.any(), f"{sample.sample_id}: brain_mask is empty"


def validate_healthy_sample(sample: SubjectSample) -> None:
    assert sample.label == 0, f"{sample.sample_id}: expected label=0"
    assert not sample.hemorrhage_mask.any(), f"{sample.sample_id}: healthy hemorrhage_mask is not all-zero"
    assert sample.regression_target_valid == 0, f"{sample.sample_id}: expected regression_target_valid=0"
    assert tuple(sample.hemorrhage_center_vox) == (-1, -1, -1)
    assert sample.hemorrhage_depth_from_surface_mm == -1.0
    assert HEMORRHAGE not in np.unique(sample.tissue_volume), f"{sample.sample_id}: healthy volume contains HEMORRHAGE label"


def validate_hemorrhage_sample(sample: SubjectSample) -> None:
    assert sample.label == 1, f"{sample.sample_id}: expected label=1"
    assert sample.hemorrhage_mask.any(), f"{sample.sample_id}: hemorrhage_mask is empty"
    assert sample.regression_target_valid == 1, f"{sample.sample_id}: expected regression_target_valid=1"
    mask_bool = sample.hemorrhage_mask.astype(bool)
    assert bool(sample.brain_mask[mask_bool].all()), (
        f"{sample.sample_id}: hemorrhage mask extends outside the original brain region"
    )
    expected_vol_mm3 = sample.hemorrhage_volume_voxels * (sample.voxel_size_mm ** 3)
    assert abs(sample.hemorrhage_volume_mm3 - expected_vol_mm3) < 1e-3, (
        f"{sample.sample_id}: hemorrhage_volume_mm3 does not match voxel_count * voxel_volume"
    )


def validate_pair_result(pair_result: PairValidationResult, subject_id: str) -> None:
    assert pair_result.ok, f"{subject_id}: pair validation failed: {pair_result}"


def render_pilot_report(rows: list[dict], pair_results: dict[str, PairValidationResult]) -> str:
    lines = ["fNIRS anatomy/hemorrhage pilot validation report", "=" * 50]
    lines.append(f"Subjects generated: {len(pair_results)}")
    lines.append(f"Samples generated: {len(rows)}")
    lines.append("")
    for sid, pr in pair_results.items():
        lines.append(f"{sid}: pair_ok={pr.ok} n_changed_voxels={pr.n_changed_voxels}")
    lines.append("")
    lines.append("OVERALL STATUS: " + ("PASSED" if all(pr.ok for pr in pair_results.values()) else "FAILED"))
    return "\n".join(lines)
