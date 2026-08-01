#!/usr/bin/env python3
"""
Small end-to-end smoke test for the synthetic fNIRS dataset generator.

This script:
1. Runs generate_optical_dataset.py for 3 paired subjects.
2. Validates all 6 NPZ files and metadata.csv.
3. Produces a short PASS/FAIL report.

No learning model is created or trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REQUIRED_KEYS = {
    "I740",
    "I850",
    "label",
    "subject_id",
    "pair_id",
    "sample_id",
    "tissue_volume",
    "brain_mask",
    "hemorrhage_mask",
    "hemorrhage_center_vox",
    "hemorrhage_center_mm",
    "hemorrhage_radii_vox",
    "hemorrhage_radii_mm",
    "hemorrhage_depth_from_surface_mm",
    "hemorrhage_depth_from_brain_start_mm",
    "hemorrhage_volume_voxels",
    "hemorrhage_volume_mm3",
    "regression_target_valid",
    "source_positions_740_mm",
    "source_positions_850_mm",
    "detector_positions_mm",
    "optical_properties_740",
    "optical_properties_850",
    "photon_count",
    "noise_mode",
    "simulation_backend",
    "voxel_size_mm",
    "volume_shape",
    "subject_seed",
    "anatomy_seed",
    "hemorrhage_seed",
}


class ValidationFailure(RuntimeError):
    pass


def scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    return value.item() if np.asarray(value).shape == () else value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def validate_numeric_array(name: str, array: np.ndarray) -> None:
    require(np.issubdtype(array.dtype, np.number), f"{name} is not numeric")
    require(np.all(np.isfinite(array)), f"{name} contains NaN or Inf")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = REQUIRED_KEYS.difference(data.files)
            require(not missing, f"{path.name}: missing keys {sorted(missing)}")
            return {key: np.array(data[key]) for key in data.files}
    except ValidationFailure:
        raise
    except Exception as exc:
        raise ValidationFailure(f"Could not reopen {path.name}: {exc}") from exc


def validate_single(path: Path, expected_label: int, photons: int) -> dict[str, np.ndarray]:
    data = load_npz(path)

    require(int(data["label"].item()) == expected_label,
            f"{path.name}: wrong label")
    require(data["I740"].shape == (8, 16),
            f"{path.name}: I740 shape is {data['I740'].shape}, expected (8,16)")
    require(data["I850"].shape == (8, 16),
            f"{path.name}: I850 shape is {data['I850'].shape}, expected (8,16)")
    require(data["source_positions_740_mm"].shape == (8, 3),
            f"{path.name}: wrong 740 source positions shape")
    require(data["source_positions_850_mm"].shape == (8, 3),
            f"{path.name}: wrong 850 source positions shape")
    require(data["detector_positions_mm"].shape == (16, 3),
            f"{path.name}: wrong detector positions shape")
    require(data["tissue_volume"].shape == data["hemorrhage_mask"].shape,
            f"{path.name}: tissue/mask shape mismatch")
    require(tuple(data["volume_shape"].tolist()) == data["tissue_volume"].shape,
            f"{path.name}: volume_shape metadata mismatch")
    require(int(data["photon_count"].item()) == photons,
            f"{path.name}: photon_count mismatch")

    for key in (
        "I740", "I850", "source_positions_740_mm",
        "source_positions_850_mm", "detector_positions_mm",
        "optical_properties_740", "optical_properties_850",
    ):
        validate_numeric_array(f"{path.name}:{key}", data[key])

    require(np.any(data["I740"] > 0), f"{path.name}: I740 is all zero/nonpositive")
    require(np.any(data["I850"] > 0), f"{path.name}: I850 is all zero/nonpositive")

    mask = data["hemorrhage_mask"]
    if expected_label == 0:
        require(np.count_nonzero(mask) == 0,
                f"{path.name}: healthy hemorrhage mask is not empty")
        require(int(data["regression_target_valid"].item()) == 0,
                f"{path.name}: healthy regression_target_valid must be 0")
    else:
        require(np.count_nonzero(mask) > 0,
                f"{path.name}: hemorrhage mask is empty")
        require(int(data["regression_target_valid"].item()) == 1,
                f"{path.name}: hemorrhage regression_target_valid must be 1")
        require(int(data["hemorrhage_volume_voxels"].item()) == int(np.count_nonzero(mask)),
                f"{path.name}: hemorrhage voxel count mismatch")
        require(float(data["hemorrhage_volume_mm3"].item()) > 0,
                f"{path.name}: hemorrhage volume must be positive")

    return data


def validate_pair(
    healthy_path: Path,
    hemorrhage_path: Path,
    photons: int,
) -> dict[str, float]:
    healthy = validate_single(healthy_path, 0, photons)
    hemorrhage = validate_single(hemorrhage_path, 1, photons)

    for key in ("subject_id", "pair_id", "subject_seed", "anatomy_seed"):
        require(
            scalar_from_dict(healthy, key) == scalar_from_dict(hemorrhage, key),
            f"{healthy_path.stem}: pair mismatch in {key}",
        )

    for key in (
        "source_positions_740_mm",
        "source_positions_850_mm",
        "detector_positions_mm",
        "optical_properties_740",
        "optical_properties_850",
        "brain_mask",
    ):
        require(
            np.array_equal(healthy[key], hemorrhage[key]),
            f"{healthy_path.stem}: pair mismatch in {key}",
        )

    changed_voxels = healthy["tissue_volume"] != hemorrhage["tissue_volume"]
    hemorrhage_mask = hemorrhage["hemorrhage_mask"].astype(bool)

    require(np.any(changed_voxels), f"{healthy_path.stem}: tissue volumes are identical")
    require(
        np.array_equal(changed_voxels, hemorrhage_mask),
        f"{healthy_path.stem}: tissue changes do not exactly match hemorrhage mask",
    )

    delta740 = hemorrhage["I740"] - healthy["I740"]
    delta850 = hemorrhage["I850"] - healthy["I850"]

    require(np.any(delta740 != 0), f"{healthy_path.stem}: no 740nm paired difference")
    require(np.any(delta850 != 0), f"{healthy_path.stem}: no 850nm paired difference")

    return {
        "mean_abs_delta_740": float(np.mean(np.abs(delta740))),
        "mean_abs_delta_850": float(np.mean(np.abs(delta850))),
        "max_abs_delta_740": float(np.max(np.abs(delta740))),
        "max_abs_delta_850": float(np.max(np.abs(delta850))),
        "healthy_mean_740": float(np.mean(healthy["I740"])),
        "healthy_mean_850": float(np.mean(healthy["I850"])),
    }


def scalar_from_dict(data: dict[str, np.ndarray], key: str):
    value = data[key]
    return value.item() if value.shape == () else value.tolist()


def validate_metadata(metadata_path: Path, num_subjects: int) -> None:
    require(metadata_path.is_file(), "metadata.csv was not created")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    require(len(rows) == 2 * num_subjects,
            f"metadata.csv has {len(rows)} rows, expected {2 * num_subjects}")

    required_columns = {
        "sample_id", "subject_id", "pair_id", "file_name",
        "label", "status", "photon_count", "noise_mode",
        "backend", "pair_mean_abs_delta_740",
        "pair_mean_abs_delta_850",
    }
    require(required_columns.issubset(rows[0].keys()),
            "metadata.csv is missing required columns")

    for row in rows:
        require(row["status"] == "complete",
                f"metadata row {row['sample_id']} is not complete")
        require(row["label"] in {"0", "1"},
                f"metadata row {row['sample_id']} has invalid label")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate and validate a 3-pair fNIRS dataset smoke test."
    )
    parser.add_argument("--pcb", required=True)
    parser.add_argument(
        "--generator",
        default="generate_optical_dataset.py",
    )
    parser.add_argument(
        "--output",
        default="output/dataset_smoke_test",
    )
    parser.add_argument("--num-subjects", type=int, default=3)
    parser.add_argument("--photons", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--backend",
        choices=["pmcx", "analytic_debug"],
        default="pmcx",
    )
    args = parser.parse_args()

    generator = Path(args.generator)
    pcb = Path(args.pcb)
    output = Path(args.output)

    require(generator.is_file(), f"Generator not found: {generator.resolve()}")
    require(pcb.is_file(), f"PCB not found: {pcb.resolve()}")
    require(args.num_subjects > 0, "--num-subjects must be positive")
    require(args.photons > 0, "--photons must be positive")

    command = [
        sys.executable,
        str(generator),
        "--pcb", str(pcb),
        "--output", str(output),
        "--num-subjects", str(args.num_subjects),
        "--seed", str(args.seed),
        "--photons", str(args.photons),
        "--backend", args.backend,
        "--noise-mode", "none",
        "--overwrite",
        "--visualize-first", str(min(args.num_subjects, 3)),
    ]

    print("Running dataset generator:")
    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    require(completed.returncode == 0,
            f"Dataset generator failed with exit code {completed.returncode}")

    samples_dir = output / "samples"
    require(samples_dir.is_dir(), "samples directory was not created")

    results = []
    for index in range(args.num_subjects):
        subject = f"subject_{index:05d}"
        healthy_path = samples_dir / f"{subject}_healthy.npz"
        hemorrhage_path = samples_dir / f"{subject}_hemorrhage.npz"

        require(healthy_path.is_file(), f"Missing {healthy_path.name}")
        require(hemorrhage_path.is_file(), f"Missing {hemorrhage_path.name}")

        pair_result = validate_pair(healthy_path, hemorrhage_path, args.photons)
        pair_result["subject_id"] = subject
        results.append(pair_result)

    validate_metadata(output / "metadata.csv", args.num_subjects)

    config_path = output / "config_snapshot.json"
    require(config_path.is_file(), "config_snapshot.json was not created")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    require(
        config["simulation"]["pmcx"]["detector_mode"]
        == "explicit_detected_photons",
        "Smoke test did not use explicit_detected_photons",
    )
    require(
        config["simulation"]["pmcx"]["paired_seed_mode"] == "identical",
        "Smoke test did not use identical paired seeds",
    )

    report_lines = [
        "DATASET SMOKE TEST: PASS",
        f"Subjects: {args.num_subjects}",
        f"Samples: {2 * args.num_subjects}",
        f"Photons per source: {args.photons:,}",
        f"Backend: {args.backend}",
        "Noise mode: none",
        "Detector mode: explicit_detected_photons",
        "",
        "Validated:",
        "- all expected NPZ files exist and reopen",
        "- I740 and I850 shapes are (8,16)",
        "- no NaN/Inf in measurements or geometry",
        "- measurements are not all zero",
        "- healthy masks are empty",
        "- hemorrhage masks are non-empty",
        "- paired anatomy, geometry and optical properties match",
        "- tissue differences exactly equal the hemorrhage mask",
        "- paired measurements differ at both wavelengths",
        "- metadata.csv and config_snapshot.json are consistent",
        "",
        "Pair deltas:",
    ]

    for row in results:
        report_lines.append(
            f"{row['subject_id']}: "
            f"mean|delta740|={row['mean_abs_delta_740']:.3e}, "
            f"mean|delta850|={row['mean_abs_delta_850']:.3e}, "
            f"max|delta740|={row['max_abs_delta_740']:.3e}, "
            f"max|delta850|={row['max_abs_delta_850']:.3e}"
        )

    report = "\n".join(report_lines)
    report_path = output / "smoke_test_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"\nReport saved to: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationFailure as exc:
        print(f"\nDATASET SMOKE TEST: FAIL\n{exc}", file=sys.stderr)
        raise SystemExit(2)
