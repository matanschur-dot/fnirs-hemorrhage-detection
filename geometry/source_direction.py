"""
source_direction.py

Actual implementation of the source-direction validation utility that the
previous submission only referenced without writing. Not used for a real
MCX run yet (no head model exists at this geometry-only stage) - it is
implemented and tested here against a simple placeholder layered volume,
per the revision request.
"""

from __future__ import annotations

import numpy as np


def validate_source_direction(
    source_position: tuple[float, float, float],
    source_direction: tuple[float, float, float],
    tissue_volume: np.ndarray,
    air_label: int = 0,
    step_mm: float = 1.0,
    voxel_size_mm: float = 1.0,
) -> dict:
    """Validate a candidate source position+direction against a tissue
    volume. Returns a dict of individual pass/fail checks plus an overall
    'ok' flag; never silently assumes correctness.

    Checks performed:
      1. source_position is inside the volume bounds.
      2. source_direction is (approximately) a unit vector.
      3. stepping ONE step_mm along source_direction lands in a non-air
         voxel (i.e. direction points INTO tissue).
      4. stepping ONE step_mm along the OPPOSITE direction does NOT also
         land in a deeper non-air voxel in a way that would indicate the
         direction sign is ambiguous (guards against e.g. a volume with
         tissue on both sides of the source).
    """
    result = {
        "position_in_bounds": False,
        "direction_is_unit_vector": False,
        "forward_step_enters_tissue": False,
        "reverse_step_is_not_also_tissue": False,
        "ok": False,
        "detail": [],
    }

    dim_x, dim_y, dim_z = tissue_volume.shape
    sx, sy, sz = source_position
    dx, dy, dz = source_direction

    # 1. bounds
    in_bounds = (0 <= sx < dim_x) and (0 <= sy < dim_y) and (0 <= sz < dim_z)
    result["position_in_bounds"] = in_bounds
    if not in_bounds:
        result["detail"].append(
            f"source_position {source_position} is outside volume bounds "
            f"{tissue_volume.shape}."
        )
        return result

    # 2. unit vector
    norm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    is_unit = abs(norm - 1.0) < 1e-3
    result["direction_is_unit_vector"] = is_unit
    if not is_unit:
        result["detail"].append(
            f"source_direction {source_direction} has norm {norm:.6f}, expected ~1.0. "
            f"Normalize it before use."
        )
        return result

    # 3. forward step enters tissue
    step_vox = step_mm / voxel_size_mm
    fx, fy, fz = sx + dx * step_vox, sy + dy * step_vox, sz + dz * step_vox
    fxi, fyi, fzi = int(round(fx)), int(round(fy)), int(round(fz))
    forward_in_bounds = (0 <= fxi < dim_x) and (0 <= fyi < dim_y) and (0 <= fzi < dim_z)
    if not forward_in_bounds:
        result["detail"].append(
            f"Forward step from {source_position} along {source_direction} "
            f"({step_mm}mm) leaves the volume at voxel ({fxi},{fyi},{fzi})."
        )
        return result
    forward_label = int(tissue_volume[fxi, fyi, fzi])
    forward_ok = forward_label != air_label
    result["forward_step_enters_tissue"] = forward_ok
    if not forward_ok:
        result["detail"].append(
            f"Forward step lands on label {forward_label} (air_label={air_label}) "
            f"at voxel ({fxi},{fyi},{fzi}) - direction does NOT point into tissue."
        )
        return result

    # 4. reverse step sanity check
    rx, ry, rz = sx - dx * step_vox, sy - dy * step_vox, sz - dz * step_vox
    rxi, ryi, rzi = int(round(rx)), int(round(ry)), int(round(rz))
    reverse_in_bounds = (0 <= rxi < dim_x) and (0 <= ryi < dim_y) and (0 <= rzi < dim_z)
    if reverse_in_bounds:
        reverse_label = int(tissue_volume[rxi, ryi, rzi])
        reverse_is_tissue = reverse_label != air_label
        result["reverse_step_is_not_also_tissue"] = not reverse_is_tissue
        if reverse_is_tissue:
            result["detail"].append(
                f"WARNING: reverse step also lands on non-air label {reverse_label} at "
                f"({rxi},{ryi},{rzi}). Direction sign may be ambiguous in this volume "
                f"(e.g. tissue on both sides of the source) - inspect manually."
            )
    else:
        # reverse step leaving the volume (e.g. into open air above the head)
        # is the expected/healthy case, not a failure.
        result["reverse_step_is_not_also_tissue"] = True

    result["ok"] = (
        result["position_in_bounds"]
        and result["direction_is_unit_vector"]
        and result["forward_step_enters_tissue"]
        and result["reverse_step_is_not_also_tissue"]
    )
    return result


def make_placeholder_layered_volume(dim_x=20, dim_y=20, dim_z=20, air_thickness=5) -> np.ndarray:
    """A trivial layered volume for testing validate_source_direction only:
    label 0 = air for z < air_thickness, label 1 = tissue for z >= air_thickness.
    This is NOT the project's real head model (that doesn't exist yet at
    this geometry-only stage)."""
    vol = np.zeros((dim_x, dim_y, dim_z), dtype=np.uint8)
    vol[:, :, air_thickness:] = 1
    return vol
