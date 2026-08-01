"""
channel_map.py

Builds the 256-row channel map (8x16 @740nm + 8x16 @850nm) with every
column required by the spec: PCB coordinates, head coordinates, distances,
component identity (value/footprint/rotation/board side), and placeholder
hardware-acquisition-index fields that are explicitly -1 because that
information isn't available from the supplied files (see
configs/hardware_mapping.py docstring).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict

from geometry.kicad_parser import PCBComponent
from geometry.pcb_geometry import compute_transform, pcb_to_head, distance_mm, TransformResult
from configs.hardware_mapping import (
    LED_740_ORDER, LED_850_ORDER, DETECTOR_ORDER, MAPPING_VERSION,
    UNKNOWN_INDEX,
)
from configs.geometry_config import GeometryRunConfig, GEOMETRY_VERSION


@dataclass
class ChannelRow:
    wavelength_nm: int
    source_reference: str
    detector_reference: str
    source_index: int
    detector_index: int
    matrix_row: int
    matrix_column: int

    source_pcb_x_mm: float
    source_pcb_y_mm: float
    detector_pcb_x_mm: float
    detector_pcb_y_mm: float

    source_head_x_mm: float
    source_head_y_mm: float
    source_head_z_mm: float
    detector_head_x_mm: float
    detector_head_y_mm: float
    detector_head_z_mm: float

    source_detector_distance_mm: float

    source_value: str
    source_footprint: str
    detector_value: str
    detector_footprint: str

    source_rotation_deg: float
    detector_rotation_deg: float

    source_board_side: str
    detector_board_side: str

    # Voxel indices, kept ALONGSIDE the continuous mm fields above, never
    # replacing them.
    source_voxel_x: int
    source_voxel_y: int
    source_voxel_z: int
    detector_voxel_x: int
    detector_voxel_y: int
    detector_voxel_z: int

    led_drive_index: int
    detector_adc_index: int
    hardware_channel_index: int

    geometry_version: str
    mapping_version: str


def build_channel_map(
    components: dict[str, PCBComponent],
    cfg: GeometryRunConfig,
) -> tuple[list[ChannelRow], TransformResult]:

    transform = compute_transform(components, cfg)
    surface_z_mm = cfg.surface.surface_z_mm
    vox = cfg.volume.voxel_size_mm

    def head_pos(ref: str) -> tuple[float, float, float]:
        c = components[ref]
        hx, hy = pcb_to_head(c.pcb_x_mm, c.pcb_y_mm, transform)
        return hx, hy, surface_z_mm

    def voxel_pos(hx: float, hy: float, hz: float) -> tuple[int, int, int]:
        return int(round(hx / vox)), int(round(hy / vox)), int(round(hz / vox))

    rows: list[ChannelRow] = []
    for wl, src_list in [(740, LED_740_ORDER), (850, LED_850_ORDER)]:
        for src_idx, src_ref in enumerate(src_list):
            src = components[src_ref]
            shx, shy, shz = head_pos(src_ref)
            svx, svy, svz = voxel_pos(shx, shy, shz)

            for det_idx, det_ref in enumerate(DETECTOR_ORDER):
                det = components[det_ref]
                dhx, dhy, dhz = head_pos(det_ref)
                dvx, dvy, dvz = voxel_pos(dhx, dhy, dhz)

                dist = distance_mm(shx, shy, shz, dhx, dhy, dhz)

                rows.append(ChannelRow(
                    wavelength_nm=wl,
                    source_reference=src_ref, detector_reference=det_ref,
                    source_index=src_idx, detector_index=det_idx,
                    matrix_row=src_idx, matrix_column=det_idx,

                    source_pcb_x_mm=src.pcb_x_mm, source_pcb_y_mm=src.pcb_y_mm,
                    detector_pcb_x_mm=det.pcb_x_mm, detector_pcb_y_mm=det.pcb_y_mm,

                    source_head_x_mm=shx, source_head_y_mm=shy, source_head_z_mm=shz,
                    detector_head_x_mm=dhx, detector_head_y_mm=dhy, detector_head_z_mm=dhz,

                    source_detector_distance_mm=dist,

                    source_value=src.value, source_footprint=src.footprint,
                    detector_value=det.value, detector_footprint=det.footprint,

                    source_rotation_deg=src.rotation_deg, detector_rotation_deg=det.rotation_deg,
                    source_board_side=src.board_side, detector_board_side=det.board_side,

                    source_voxel_x=svx, source_voxel_y=svy, source_voxel_z=svz,
                    detector_voxel_x=dvx, detector_voxel_y=dvy, detector_voxel_z=dvz,

                    led_drive_index=UNKNOWN_INDEX,
                    detector_adc_index=UNKNOWN_INDEX,
                    hardware_channel_index=UNKNOWN_INDEX,

                    geometry_version=GEOMETRY_VERSION,
                    mapping_version=MAPPING_VERSION,
                ))

    return rows, transform


def validate_channel_map(rows: list[ChannelRow]) -> None:
    """Structural checks on the built channel map. Raises AssertionError
    with a specific message on any failure - this is meant to be called
    right after construction, before anything is written to disk."""

    assert len(rows) == 256, f"Expected 256 channels, got {len(rows)}"

    n_740 = sum(1 for r in rows if r.wavelength_nm == 740)
    n_850 = sum(1 for r in rows if r.wavelength_nm == 850)
    assert n_740 == 128, f"Expected 128 channels at 740nm, got {n_740}"
    assert n_850 == 128, f"Expected 128 channels at 850nm, got {n_850}"

    keys = [(r.wavelength_nm, r.source_reference, r.detector_reference) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate (wavelength, source, detector) channel keys found."

    unique_740_sources = {r.source_reference for r in rows if r.wavelength_nm == 740}
    unique_850_sources = {r.source_reference for r in rows if r.wavelength_nm == 850}
    unique_detectors = {r.detector_reference for r in rows}
    assert len(unique_740_sources) == 8, f"Expected 8 unique 740nm sources, got {len(unique_740_sources)}"
    assert len(unique_850_sources) == 8, f"Expected 8 unique 850nm sources, got {len(unique_850_sources)}"
    assert len(unique_detectors) == 16, f"Expected 16 unique detectors, got {len(unique_detectors)}"

    for r in rows:
        for field_name in ["source_pcb_x_mm", "source_pcb_y_mm", "detector_pcb_x_mm", "detector_pcb_y_mm",
                            "source_head_x_mm", "source_head_y_mm", "source_head_z_mm",
                            "detector_head_x_mm", "detector_head_y_mm", "detector_head_z_mm",
                            "source_detector_distance_mm"]:
            val = getattr(r, field_name)
            assert val == val, f"NaN found in {field_name} for channel {r.wavelength_nm}/{r.source_reference}/{r.detector_reference}"  # NaN != NaN
        assert r.source_detector_distance_mm > 0, (
            f"Non-positive source-detector distance for {r.source_reference}->{r.detector_reference}"
        )

    # Reconstructed PCB distance must match the direct coordinate calculation
    # (i.e. the transform must be distance-preserving, checked numerically).
    for r in rows:
        pcb_dist = distance_mm(
            r.source_pcb_x_mm, r.source_pcb_y_mm, 0.0,
            r.detector_pcb_x_mm, r.detector_pcb_y_mm, 0.0,
        )
        head_dist_xy_only = distance_mm(
            r.source_head_x_mm, r.source_head_y_mm, 0.0,
            r.detector_head_x_mm, r.detector_head_y_mm, 0.0,
        )
        assert abs(pcb_dist - head_dist_xy_only) < 1e-6, (
            f"Transform did not preserve xy-distance for {r.source_reference}->{r.detector_reference}: "
            f"pcb={pcb_dist:.6f}mm head={head_dist_xy_only:.6f}mm"
        )


def save_channel_map(rows: list[ChannelRow], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "channel_map.csv")
    fieldnames = list(asdict(rows[0]).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return csv_path
