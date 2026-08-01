"""
geometry_config.py

Configuration for the PCB->head coordinate transform and the (future) MCX
volume. Kept separate from hardware_mapping.py because this is
engineering/simulation configuration, not verified hardware identity.

TRANSFORM POLICY (explicit, not implied by code structure)
-----------------------------------------------------------------------
head = translation(pcb) -- and NOTHING else, by default.

- translation_mm: subtracts the min x/y over all source+detector
  footprints, so head coordinates are >= 0. This is the ONLY operation
  applied.
- rotation_deg: 0.0 (identity). A nonzero value would need to be
  justified by an actual known mounting rotation of the PCB relative to
  the head - we don't have that, so it stays at 0 and is documented as
  such.
- reflect_x / reflect_y: False. A mirror reflection preserves all
  pairwise distances (which is NOT sufficient justification - see
  module docstring in pcb_geometry.py) so it is never applied silently.
  If the physical PCB is later confirmed to be mounted mirrored
  relative to this convention, flip the relevant flag HERE, explicitly,
  with a comment citing the source of that confirmation.
- scale: 1.0, always. The PCB is not stretched to fit the volume; if it
  doesn't fit, we raise an error and ask for a larger volume instead.

PHYSICAL HEAD-COORDINATE CONVENTION (placeholder, unconfirmed)
-----------------------------------------------------------------------
+x_head = same sense as PCB +x ("physical right" -- ASSUMED, not verified
          against any physical mounting drawing)
+y_head = same sense as PCB +y ("posterior direction" -- ASSUMED)
+z_head = inward into the tissue (perpendicular to the board), with
          z=0 at the physical component-mounting plane.

This mapping of "PCB +x/+y" onto "anatomical right/posterior" is a
PLACEHOLDER pending physical confirmation of how the board is actually
oriented on the subject's head. It is used consistently throughout this
codebase, but it is explicitly flagged here as an assumption rather than
a fact - see README.md "Known assumptions requiring physical
confirmation".

SURFACE PLACEMENT POLICY (placeholder, unconfirmed)
-----------------------------------------------------------------------
`surface_z_mm` below places every source/detector at a single fixed
depth. This is a placeholder ("component center projected straight onto
an assumed flat tissue surface"). It does NOT yet distinguish between:
  - the LED/detector package's physical center,
  - the actual optical emission or active-sensing point,
  - the true air-tissue interface (which may not be flat),
  - detector active area / acceptance cone.
Do not treat source/detector placement as physically validated for MCX
until those are addressed (see README "Known limitations").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformConfig:
    rotation_deg: float = 0.0
    reflect_x: bool = False
    reflect_y: bool = False
    scale: float = 1.0


@dataclass(frozen=True)
class VolumeConfig:
    dim_x: int = 80
    dim_y: int = 80
    dim_z: int = 50
    voxel_size_mm: float = 1.0


@dataclass(frozen=True)
class SurfacePlacementConfig:
    policy_name: str = "component_center_projected_to_flat_surface_PLACEHOLDER"
    surface_z_mm: float = 1.0  # placeholder depth; see module docstring


@dataclass(frozen=True)
class GeometryRunConfig:
    transform: TransformConfig = TransformConfig()
    volume: VolumeConfig = VolumeConfig()
    surface: SurfacePlacementConfig = SurfacePlacementConfig()


GEOMETRY_VERSION = "2.0.0"
