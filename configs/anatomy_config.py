"""Configuration for the flat layered phantom and hemorrhage targets."""
from __future__ import annotations
from dataclasses import dataclass, field

ANATOMY_VERSION = "2.0.0"

@dataclass(frozen=True)
class LayerThicknessConfig:
    air_thickness_mm: float = 1.0
    scalp_thickness_mm_range: tuple[float, float] = (2.0, 3.0)
    skull_thickness_mm_range: tuple[float, float] = (5.0, 7.0)
    csf_thickness_mm_range: tuple[float, float] = (1.0, 2.0)

@dataclass(frozen=True)
class HemorrhageConfig:
    rx_range_vox: tuple[int, int] = (5, 7)
    ry_range_vox: tuple[int, int] = (5, 7)
    rz_range_vox: tuple[int, int] = (4, 5)

    max_placement_attempts: int = 1000
    require_fully_inside_brain: bool = True
    xy_margin_vox: int = 2

    # Center depth measured from the outer tissue surface.
    depth_range_mm: tuple[float, float] = (10.0, 20.0)
    # Keep centers within the optode x-y bounding box, shrunk by this margin.
    sensor_coverage_margin_mm: float = 2.0

@dataclass(frozen=True)
class SourceDetectorPlacementConfig:
    """For the flat phantom, optical points are placed in the first scalp voxel.
    This is an engineering approximation, not a validated coupling model."""
    source_offset_into_scalp_mm: float = 0.0
    detector_offset_into_scalp_mm: float = 0.0
    policy_name: str = "first_scalp_voxel_flat_phantom"

@dataclass(frozen=True)
class AnatomyRunConfig:
    layers: LayerThicknessConfig = field(default_factory=LayerThicknessConfig)
    hemorrhage: HemorrhageConfig = field(default_factory=HemorrhageConfig)
    placement: SourceDetectorPlacementConfig = field(default_factory=SourceDetectorPlacementConfig)
