from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fnirs_dataset import (
    DualInputNormalizer,
    MeasurementAugmentation,
    paired_healthy_path,
)
from medt_multitask import DualBranchMedT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", default="../output/final_prediction_v2")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    normalizer = DualInputNormalizer.from_state_dict(
        checkpoint["normalizer"]
    )
    augmentation = MeasurementAugmentation(
        **checkpoint["augmentation"]
    )
    threshold = float(checkpoint.get("classification_threshold", 0.5))

    model = DualBranchMedT(
        volume_shape=tuple(checkpoint["volume_shape"]),
        voxel_size_mm=float(checkpoint["voxel_size_mm"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sample_path = Path(args.sample)
    with np.load(sample_path, allow_pickle=False) as data:
        subject_id = str(np.asarray(data["subject_id"]).item())
        current_740_clean = np.asarray(data["I740"], dtype=np.float32)
        current_850_clean = np.asarray(data["I850"], dtype=np.float32)

    healthy_path = paired_healthy_path(sample_path, subject_id)
    with np.load(healthy_path, allow_pickle=False) as healthy:
        baseline_740_clean = np.asarray(healthy["I740"], dtype=np.float32)
        baseline_850_clean = np.asarray(healthy["I850"], dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    baseline_740_noisy = augmentation.apply(baseline_740_clean, rng)
    baseline_850_noisy = augmentation.apply(baseline_850_clean, rng)
    current_740_noisy = augmentation.apply(current_740_clean, rng)
    current_850_noisy = augmentation.apply(current_850_clean, rng)

    x_absolute = torch.from_numpy(
        np.stack(
            [
                baseline_740_noisy,
                baseline_850_noisy,
                current_740_noisy,
                current_850_noisy,
            ],
            axis=0,
        )
    ).float()
    x_absolute = normalizer.transform_absolute(x_absolute).unsqueeze(0).to(device)

    x_residual = torch.from_numpy(
        np.stack(
            [
                current_740_clean - baseline_740_clean,
                current_850_clean - baseline_850_clean,
            ],
            axis=0,
        ).astype(np.float32)
    )
    x_residual = normalizer.transform_residual(x_residual).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x_absolute, x_residual)

    probability = float(torch.sigmoid(outputs["class_logit"])[0].cpu())
    predicted_label = int(probability >= threshold)

    mask_probability = outputs["soft_mask"][0].cpu().numpy()
    binary_mask = (
        (mask_probability >= 0.5).astype(np.uint8)
        if predicted_label == 1
        else np.zeros_like(mask_probability, dtype=np.uint8)
    )

    result = {
        "sample": str(sample_path),
        "hemorrhage_probability": probability,
        "classification_threshold": threshold,
        "predicted_label": predicted_label,
        "predicted_center_vox_native_order": (
            outputs["center_vox"][0].cpu().tolist()
            if predicted_label == 1 else None
        ),
        "predicted_depth_mm": (
            float(outputs["depth_mm"][0].cpu())
            if predicted_label == 1 else None
        ),
        "predicted_radii_vox_native_order": (
            outputs["radii_vox"][0].cpu().tolist()
            if predicted_label == 1 else None
        ),
        "predicted_volume_mm3": (
            float(outputs["volume_mm3"][0].cpu())
            if predicted_label == 1 else 0.0
        ),
        "predicted_mask_voxels": int(binary_mask.sum()),
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "prediction.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "prediction.npz",
        hemorrhage_probability=np.float32(probability),
        classification_threshold=np.float32(threshold),
        predicted_label=np.int64(predicted_label),
        predicted_center_vox=outputs["center_vox"][0].cpu().numpy().astype(np.float32),
        predicted_depth_mm=np.float32(outputs["depth_mm"][0].cpu()),
        predicted_radii_vox=outputs["radii_vox"][0].cpu().numpy().astype(np.float32),
        predicted_volume_mm3=np.float32(outputs["volume_mm3"][0].cpu()),
        predicted_mask_probability=mask_probability.astype(np.float32),
        predicted_mask=binary_mask,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
