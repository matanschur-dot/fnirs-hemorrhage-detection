from __future__ import annotations
import numpy as np

def sphere_mean(volume: np.ndarray, position_mm: np.ndarray, voxel_size_mm: float, radius_mm: float) -> float:
    pos = np.asarray(position_mm, dtype=float) / voxel_size_mm
    center = np.rint(pos).astype(int)
    r = max(1, int(np.ceil(radius_mm / voxel_size_mm)))
    values = []
    for x in range(center[0]-r, center[0]+r+1):
        for y in range(center[1]-r, center[1]+r+1):
            for z in range(center[2]-r, center[2]+r+1):
                if 0 <= x < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= z < volume.shape[2]:
                    if np.sum((np.array([x,y,z])-pos)**2) <= (radius_mm/voxel_size_mm)**2:
                        values.append(float(volume[x,y,z]))
    if not values:
        raise ValueError(f"Detector sampling region is empty at {position_mm.tolist()} mm")
    return float(np.mean(values))

def extract_detector_values(field: np.ndarray, detector_positions_mm: np.ndarray,
                            voxel_size_mm: float, mode: str, radius_mm: float) -> np.ndarray:
    if mode != "sphere_mean":
        raise NotImplementedError(f"Detector mode {mode!r} is not implemented; use sphere_mean")
    vals = np.array([sphere_mean(field, p, voxel_size_mm, radius_mm) for p in detector_positions_mm], dtype=np.float64)
    if vals.shape != (16,):
        raise AssertionError(f"Expected 16 detector values, got {vals.shape}")
    return vals
