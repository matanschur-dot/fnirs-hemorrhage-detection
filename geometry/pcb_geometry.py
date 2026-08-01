"""
pcb_geometry.py

Two jobs:
  1. Validate that the PCB's ACTUAL contents (by Value+Footprint, not just
     "does this designator exist") match the expected hardware mapping
     exactly - 16 LEDs, 16 detectors, no extras, no missing, no overlaps.
  2. Transform PCB millimeter coordinates into head coordinates using an
     explicit, documented, minimal transform (translation only by
     default - see configs/geometry_config.py for why).

COORDINATE SYSTEMS (three, all explicit)
-----------------------------------------------------------------------
A. PCB coordinates (pcb_x_mm, pcb_y_mm): raw KiCad board coordinates.
   Origin is the KiCad page/board origin. By KiCad/EDA convention x
   increases rightward and y increases DOWNWARD (screen convention).
   We do not alter this.

B. Head coordinates (head_x_mm, head_y_mm, head_z_mm): PCB coordinates
   after ONE operation - translation so all head coordinates are >= 0
   (shifted by the min x/y across all source+detector footprints, plus
   a centering margin so the array sits in the middle of the configured
   volume). No rotation, no reflection, no scale (see
   configs/geometry_config.TransformConfig - all off/identity by
   default). head_z_mm is a placeholder surface depth (see
   configs/geometry_config.SurfacePlacementConfig) - the same value for
   every source/detector.

   IMPORTANT: this translation-only transform preserves every pairwise
   distance and every relative angle EXACTLY. But distance preservation
   alone does not prove the transform is physically correct - a mirror
   reflection would preserve distances too. We do not apply a
   reflection, so this is a true rigid-motion (in fact just a
   translation) of the PCB layout, not merely a distance-preserving one.
   The mapping of "PCB +x/+y" onto real anatomical directions is a
   separate, still-open, explicitly-flagged assumption (see
   geometry_config.py and README.md).

C. MCX volume coordinates (voxel indices): head coordinates divided by
   voxel_size_mm and rounded to the nearest integer. `volume[x, y, z]`
   convention - x,y as above, z=0 at the mounting plane, +z into the
   tissue. Continuous head_mm coordinates are ALWAYS preserved
   alongside the rounded voxel indices; voxel indices never replace them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from geometry.kicad_parser import components_by_reference, PCBComponent
from configs.hardware_mapping import (
    LED_740_ORDER, LED_850_ORDER, DETECTOR_ORDER,
    LED_VALUE, LED_FOOTPRINT, DETECTOR_VALUE, DETECTOR_FOOTPRINT,
    MAPPING_VERSION,
)
from configs.geometry_config import GeometryRunConfig, GEOMETRY_VERSION


class HardwareValidationError(Exception):
    pass


# ----------------------------------------------------------------------
# 1. Validate actual PCB contents against the expected hardware mapping
# ----------------------------------------------------------------------

@dataclass
class HardwareMismatchReport:
    missing_leds: list[str]
    extra_leds: list[str]
    missing_detectors: list[str]
    extra_detectors: list[str]
    unexpected_values: list[str]      # "<ref>: expected X got Y"
    unexpected_footprints: list[str]  # "<ref>: expected X got Y"
    duplicate_designators: list[str]
    wavelength_overlap: list[str]     # designators in both LED_740 and LED_850
    led_detector_overlap: list[str]   # designators in both an LED list and DETECTOR list

    @property
    def ok(self) -> bool:
        return not any([
            self.missing_leds, self.extra_leds,
            self.missing_detectors, self.extra_detectors,
            self.unexpected_values, self.unexpected_footprints,
            self.duplicate_designators, self.wavelength_overlap,
            self.led_detector_overlap,
        ])

    def render(self) -> str:
        lines = ["Hardware content validation report", "=" * 40]
        lines.append(f"Status: {'PASSED' if self.ok else 'FAILED'}")
        for field in ["missing_leds", "extra_leds", "missing_detectors", "extra_detectors",
                      "unexpected_values", "unexpected_footprints", "duplicate_designators",
                      "wavelength_overlap", "led_detector_overlap"]:
            vals = getattr(self, field)
            lines.append(f"{field}: {vals if vals else 'none'}")
        return "\n".join(lines)


def validate_hardware(components: dict[str, PCBComponent]) -> HardwareMismatchReport:
    """Build pcb_led_refs / pcb_detector_refs from ACTUAL PCB content
    (matching on Value+Footprint), and compare exactly against the
    expected designator sets from hardware_mapping.py."""

    expected_leds = set(LED_740_ORDER + LED_850_ORDER)
    expected_detectors = set(DETECTOR_ORDER)

    if len(expected_leds) != 16:
        raise HardwareValidationError(
            f"configs/hardware_mapping.py is internally broken: "
            f"expected 16 unique LED designators, got {len(expected_leds)}. Fix the config."
        )
    if len(expected_detectors) != 16:
        raise HardwareValidationError(
            f"configs/hardware_mapping.py is internally broken: "
            f"expected 16 unique detector designators, got {len(expected_detectors)}. Fix the config."
        )

    # What does the PCB actually contain, by content (not by our lists)?
    pcb_led_refs = {
        ref for ref, c in components.items()
        if c.value == LED_VALUE and c.footprint == LED_FOOTPRINT
    }
    pcb_detector_refs = {
        ref for ref, c in components.items()
        if c.value == DETECTOR_VALUE and c.footprint == DETECTOR_FOOTPRINT
    }

    missing_leds = sorted(expected_leds - pcb_led_refs)
    extra_leds = sorted(pcb_led_refs - expected_leds)
    missing_detectors = sorted(expected_detectors - pcb_detector_refs)
    extra_detectors = sorted(pcb_detector_refs - expected_detectors)

    # Per-designator value/footprint sanity (catches "the ref exists but
    # isn't really the component type we think it is" even when it's not
    # missing/extra, e.g. someone repurposed a designator).
    unexpected_values = []
    unexpected_footprints = []
    for ref in expected_leds:
        c = components.get(ref)
        if c is None:
            continue
        if c.value != LED_VALUE:
            unexpected_values.append(f"{ref}: expected value='{LED_VALUE}' got '{c.value}'")
        if c.footprint != LED_FOOTPRINT:
            unexpected_footprints.append(f"{ref}: expected footprint='{LED_FOOTPRINT}' got '{c.footprint}'")
    for ref in expected_detectors:
        c = components.get(ref)
        if c is None:
            continue
        if c.value != DETECTOR_VALUE:
            unexpected_values.append(f"{ref}: expected value='{DETECTOR_VALUE}' got '{c.value}'")
        if c.footprint != DETECTOR_FOOTPRINT:
            unexpected_footprints.append(f"{ref}: expected footprint='{DETECTOR_FOOTPRINT}' got '{c.footprint}'")

    duplicate_designators = []  # components_by_reference() already raises on true PCB duplicates
    dup_check = set()
    for ref in LED_740_ORDER + LED_850_ORDER + DETECTOR_ORDER:
        if ref in dup_check:
            duplicate_designators.append(ref)
        dup_check.add(ref)

    wavelength_overlap = sorted(set(LED_740_ORDER) & set(LED_850_ORDER))
    led_detector_overlap = sorted((set(LED_740_ORDER) | set(LED_850_ORDER)) & set(DETECTOR_ORDER))

    report = HardwareMismatchReport(
        missing_leds=missing_leds, extra_leds=extra_leds,
        missing_detectors=missing_detectors, extra_detectors=extra_detectors,
        unexpected_values=unexpected_values, unexpected_footprints=unexpected_footprints,
        duplicate_designators=duplicate_designators,
        wavelength_overlap=wavelength_overlap, led_detector_overlap=led_detector_overlap,
    )

    assert len(expected_leds) == 16
    assert len(expected_detectors) == 16

    if not report.ok:
        raise HardwareValidationError("\n" + report.render())

    if pcb_led_refs != expected_leds:
        raise HardwareValidationError(
            f"Internal inconsistency: pcb_led_refs != expected_leds even though "
            f"report.ok was True. pcb={sorted(pcb_led_refs)} expected={sorted(expected_leds)}"
        )
    if pcb_detector_refs != expected_detectors:
        raise HardwareValidationError(
            f"Internal inconsistency: pcb_detector_refs != expected_detectors even though "
            f"report.ok was True. pcb={sorted(pcb_detector_refs)} expected={sorted(expected_detectors)}"
        )

    return report


# ----------------------------------------------------------------------
# 2. PCB -> head coordinate transform (translation only, documented)
# ----------------------------------------------------------------------

@dataclass
class TransformResult:
    translation_x_mm: float
    translation_y_mm: float
    rotation_deg: float
    reflect_x: bool
    reflect_y: bool
    scale: float


def compute_transform(components: dict[str, PCBComponent], cfg: GeometryRunConfig) -> TransformResult:
    all_refs = LED_740_ORDER + LED_850_ORDER + DETECTOR_ORDER
    xs = [components[r].pcb_x_mm for r in all_refs]
    ys = [components[r].pcb_y_mm for r in all_refs]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    vol_x_mm = cfg.volume.dim_x * cfg.volume.voxel_size_mm
    vol_y_mm = cfg.volume.dim_y * cfg.volume.voxel_size_mm

    margin_x = (vol_x_mm - span_x) / 2.0
    margin_y = (vol_y_mm - span_y) / 2.0
    if margin_x < 0 or margin_y < 0:
        raise HardwareValidationError(
            f"PCB source/detector footprint ({span_x:.1f}mm x {span_y:.1f}mm) does not fit "
            f"inside the configured volume ({vol_x_mm:.1f}mm x {vol_y_mm:.1f}mm). "
            f"Increase VolumeConfig.dim_x/dim_y rather than scaling the PCB layout."
        )

    if cfg.transform.rotation_deg != 0.0:
        raise NotImplementedError(
            "Nonzero rotation_deg is not yet implemented (no confirmed physical "
            "mounting rotation exists to apply). Leave TransformConfig.rotation_deg=0.0."
        )
    if cfg.transform.scale != 1.0:
        raise HardwareValidationError(
            "TransformConfig.scale must be exactly 1.0. The PCB layout must never be "
            "stretched to fit the volume; enlarge the volume instead."
        )
    if cfg.transform.reflect_x or cfg.transform.reflect_y:
        # Centering/translation (margin_x, margin_y, above) is computed from the
        # UN-reflected bounding box. Applying a reflection on top of that would
        # place the reflected geometry off-center (or outside the volume) rather
        # than recentering correctly. Until the transform pipeline recomputes the
        # bounding box AFTER reflection and re-centers on that, reflection is not
        # a safe, verified feature - so we refuse rather than silently producing
        # incorrectly-positioned geometry.
        raise NotImplementedError(
            "Geometry reflection is not safely implemented yet."
        )

    translation_x = -min(xs) + margin_x
    translation_y = -min(ys) + margin_y

    return TransformResult(
        translation_x_mm=translation_x, translation_y_mm=translation_y,
        rotation_deg=cfg.transform.rotation_deg,
        reflect_x=cfg.transform.reflect_x, reflect_y=cfg.transform.reflect_y,
        scale=cfg.transform.scale,
    )


def pcb_to_head(x_mm: float, y_mm: float, transform: TransformResult) -> tuple[float, float]:
    x = x_mm
    y = y_mm
    if transform.reflect_x:
        x = -x
    if transform.reflect_y:
        y = -y
    # rotation_deg is enforced to be 0.0 in compute_transform for now;
    # kept here so a future confirmed rotation has exactly one place to add it.
    x = x * transform.scale + transform.translation_x_mm
    y = y * transform.scale + transform.translation_y_mm
    return x, y


# ----------------------------------------------------------------------
# 3. Misc helpers
# ----------------------------------------------------------------------

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def distance_mm(ax, ay, az, bx, by, bz) -> float:
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))
