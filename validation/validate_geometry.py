"""
validate_geometry.py

Assembles the human-readable validation_report.txt and the machine-readable
geometry_metadata.json. Both are written regardless of pass/fail so a
failed run still leaves a clear record of what went wrong.
"""

from __future__ import annotations

import datetime
import json
import os

from configs.hardware_mapping import (
    LED_740_ORDER, LED_850_ORDER, DETECTOR_ORDER, MAPPING_VERSION,
    LED_DRIVE_INDEX_KNOWN, DETECTOR_ADC_INDEX_KNOWN, HARDWARE_CHANNEL_INDEX_KNOWN,
)
from configs.geometry_config import GeometryRunConfig, GEOMETRY_VERSION
from geometry.pcb_geometry import sha256_of_file, HardwareMismatchReport, TransformResult
from geometry.channel_map import ChannelRow
from geometry.distance_stats import DistanceStats


def write_validation_report(
    out_dir: str,
    pcb_path: str,
    hw_report: HardwareMismatchReport,
    n_channels: int,
    structural_checks_passed: bool,
    structural_check_error: str | None,
) -> str:
    lines = []
    lines.append("fNIRS geometry validation report")
    lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
    lines.append(f"PCB file: {pcb_path}")
    lines.append("")
    lines.append(hw_report.render())
    lines.append("")
    lines.append(f"Channel map rows: {n_channels} (expected 256)")
    lines.append(f"Structural checks: {'PASSED' if structural_checks_passed else 'FAILED'}")
    if structural_check_error:
        lines.append(f"  error: {structural_check_error}")
    lines.append("")
    lines.append("Hardware acquisition order (LED drive index / detector ADC index / "
                 "hardware channel index):")
    lines.append("  The acquisition order could not be verified from the supplied files. "
                 "All acquisition-order fields in channel_map.csv are set to -1.")
    lines.append("")
    overall_ok = hw_report.ok and structural_checks_passed
    lines.append(f"OVERALL STATUS: {'PASSED' if overall_ok else 'FAILED'}")

    text = "\n".join(lines)
    path = os.path.join(out_dir, "validation_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def write_geometry_metadata(
    out_dir: str,
    pcb_path: str,
    cfg: GeometryRunConfig,
    transform: TransformResult,
    dist_stats: DistanceStats,
    validation_status: str,
) -> str:
    meta = {
        "pcb_file_name": os.path.basename(pcb_path),
        "pcb_file_hash_sha256": sha256_of_file(pcb_path),
        "generated_at": datetime.datetime.now().isoformat(),
        "geometry_version": GEOMETRY_VERSION,
        "mapping_version": MAPPING_VERSION,
        "num_leds": len(LED_740_ORDER) + len(LED_850_ORDER),
        "num_detectors": len(DETECTOR_ORDER),
        "led_740_order": LED_740_ORDER,
        "led_850_order": LED_850_ORDER,
        "detector_order": DETECTOR_ORDER,
        "hardware_acquisition_order_known": {
            "led_drive_index": LED_DRIVE_INDEX_KNOWN,
            "detector_adc_index": DETECTOR_ADC_INDEX_KNOWN,
            "hardware_channel_index": HARDWARE_CHANNEL_INDEX_KNOWN,
        },
        "coordinate_convention": {
            "A_pcb": "Raw KiCad board coordinates (pcb_x_mm, pcb_y_mm). +x right, +y down (KiCad screen convention). Unmodified.",
            "B_head": (
                "head_x_mm/head_y_mm/head_z_mm = pcb coordinates after a TRANSLATION-ONLY "
                "transform (see transform_matrix/translation_vector below). "
                "+x_head/+y_head are labeled 'physical right'/'posterior' as a PLACEHOLDER "
                "convention, NOT confirmed against a physical mounting drawing. "
                "head_z_mm is a fixed placeholder surface depth, identical for every "
                "source/detector (see surface_placement_policy below)."
            ),
            "C_volume": (
                "volume[x,y,z] voxel array. voxel = round(head_mm / voxel_size_mm). "
                "z=0 at the assumed mounting/tissue-surface plane, +z increases into tissue "
                "(not yet empirically validated - no head model exists at this geometry-only "
                "stage; see geometry/source_direction.py for the validation utility that will "
                "be used once a head model exists)."
            ),
        },
        "surface_placement_policy": {
            "name": cfg.surface.policy_name,
            "surface_z_mm": cfg.surface.surface_z_mm,
            "note": (
                "Placeholder: component center projected onto a single fixed depth. Does not "
                "yet distinguish package center / optical emission point / true air-tissue "
                "interface / detector active area. Not yet claimed physically correct for MCX."
            ),
        },
        "transform_matrix": [[1.0 if not transform.reflect_x else -1.0, 0.0],
                              [0.0, 1.0 if not transform.reflect_y else -1.0]],
        "translation_vector_mm": [transform.translation_x_mm, transform.translation_y_mm],
        "rotation_deg": transform.rotation_deg,
        "scale": transform.scale,
        "reflection_flags": {"reflect_x": transform.reflect_x, "reflect_y": transform.reflect_y},
        "voxel_size_mm": cfg.volume.voxel_size_mm,
        "volume_dims": {"dim_x": cfg.volume.dim_x, "dim_y": cfg.volume.dim_y, "dim_z": cfg.volume.dim_z},
        "distance_statistics_mm": {
            "minimum": dist_stats.minimum_mm, "maximum": dist_stats.maximum_mm,
            "mean": dist_stats.mean_mm, "median": dist_stats.median_mm, "std": dist_stats.std_mm,
            "n_very_short_lt10": dist_stats.n_very_short_lt10,
            "n_short_10_20": dist_stats.n_short_10_20,
            "n_medium_20_30": dist_stats.n_medium_20_30,
            "n_long_30_40": dist_stats.n_long_30_40,
            "n_very_long_40_50": dist_stats.n_very_long_40_50,
            "n_extreme_gt50": dist_stats.n_extreme_gt50,
            "n_total": dist_stats.n_total,
        },
        "validation_status": validation_status,
    }
    path = os.path.join(out_dir, "geometry_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path
