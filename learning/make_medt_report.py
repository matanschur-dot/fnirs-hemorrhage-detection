from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from fnirs_dataset import (
    DualInputNormalizer,
    FNIRSDualBranchDataset,
    MeasurementAugmentation,
    make_subject_split,
)
from medt_multitask import DualBranchMedT


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def projection_xy(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=2)


def projection_xz(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=1)


def projection_yz(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=0).T


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    return float(2.0 * intersection / max(denominator, 1))


def overlay_rgb(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    rgb = np.zeros((*target.shape, 3), dtype=np.float32)
    rgb[..., 0] = target.astype(np.float32)
    rgb[..., 1] = prediction.astype(np.float32)
    return rgb


def add_image(ax, image, title, cmap=None, vmin=None, vmax=None):
    shown = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return shown


def save_example(example: dict, output_path: Path, heading: str) -> None:
    gt = example["gt"]
    probability = example["probability_mask"]
    prediction = example["binary_mask"]

    gt_views = [projection_xy(gt), projection_xz(gt), projection_yz(gt)]
    probability_views = [
        projection_xy(probability),
        projection_xz(probability),
        projection_yz(probability),
    ]
    prediction_views = [
        projection_xy(prediction),
        projection_xz(prediction),
        projection_yz(prediction),
    ]
    names = ["XY", "XZ", "YZ"]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, name in enumerate(names):
        add_image(axes[row, 0], gt_views[row], f"Ground truth — {name}", cmap="gray")
        im = add_image(
            axes[row, 1],
            probability_views[row],
            f"Predicted probability — {name}",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
        add_image(
            axes[row, 2],
            prediction_views[row],
            f"Binary prediction — {name}",
            cmap="gray",
        )
        add_image(
            axes[row, 3],
            overlay_rgb(gt_views[row], prediction_views[row]),
            f"Overlay — {name}\nred=GT, green=prediction, yellow=overlap",
        )

    fig.suptitle(
        f"{heading}: {example['sample_id']}\n"
        f"Dice={example['dice']:.3f} | "
        f"hemorrhage probability={example['class_probability']:.3f} | "
        f"center error={example['center_error']:.2f} vox",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(best: dict, median: dict, worst: dict, output_path: Path) -> None:
    rows = [("Best", best), ("Median", median), ("Worst", worst)]
    fig, axes = plt.subplots(3, 5, figsize=(19, 11))

    for row, (name, example) in enumerate(rows):
        gt = projection_xy(example["gt"])
        probability = projection_xy(example["probability_mask"])
        prediction = projection_xy(example["binary_mask"])

        add_image(
            axes[row, 0],
            example["residual_740"],
            f"{name}: residual 740\n(normalized signed-log)",
        )
        add_image(
            axes[row, 1],
            example["residual_850"],
            f"{name}: residual 850\n(normalized signed-log)",
        )
        add_image(
            axes[row, 2],
            gt,
            f"Ground truth — XY\n{example['sample_id']}",
            cmap="gray",
        )
        add_image(
            axes[row, 3],
            probability,
            "Probability — XY",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        add_image(
            axes[row, 4],
            overlay_rgb(gt, prediction),
            f"Overlay — XY\nDice={example['dice']:.3f}, "
            f"center={example['center_error']:.2f} vox",
        )

    fig.suptitle("Representative test-set segmentation results", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def regenerate_training_plots(run_dir: Path, script_dir: Path) -> None:
    plotting_script = script_dir / "plot_training_curves.py"
    if not plotting_script.is_file():
        raise FileNotFoundError(f"Missing plotting script: {plotting_script}")

    subprocess.run(
        [
            sys.executable,
            str(plotting_script),
            "--history",
            str(run_dir / "history.json"),
            "--metrics",
            str(run_dir / "test_metrics.json"),
            "--output",
            str(run_dir / "plots"),
        ],
        check=True,
    )


def build_test_loader(
    dataset_root: Path,
    checkpoint: dict,
    batch_size: int,
    num_workers: int,
):
    training_args = checkpoint.get("args", {})
    split_seed = int(training_args.get("seed", 42))

    split = make_subject_split(
        metadata_csv=dataset_root / "metadata.csv",
        samples_dir=dataset_root / "samples",
        seed=split_seed,
    )

    normalizer = DualInputNormalizer.from_state_dict(checkpoint["normalizer"])
    augmentation = MeasurementAugmentation(**checkpoint["augmentation"])

    test_dataset = FNIRSDualBranchDataset(
        split.test,
        normalizer,
        training=False,
        augmentation=augmentation,
        seed=split_seed + 2,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(checkpoint: dict, device: torch.device) -> DualBranchMedT:
    model = DualBranchMedT(
        volume_shape=tuple(checkpoint["volume_shape"]),
        voxel_size_mm=float(checkpoint["voxel_size_mm"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def collect_positive_examples(model, loader, device) -> list[dict]:
    examples: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            x_absolute = batch["x_absolute"].to(device, non_blocking=True)
            x_residual = batch["x_residual"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            true_masks = batch["mask"].to(device, non_blocking=True)
            true_centers = batch["center_vox"].to(device, non_blocking=True)

            outputs = model(x_absolute, x_residual)
            class_probabilities = torch.sigmoid(outputs["class_logit"])
            probability_masks = outputs["soft_mask"]
            binary_masks = probability_masks >= 0.5

            positive_indices = torch.where(labels > 0.5)[0]

            for index in positive_indices.tolist():
                gt = to_numpy(true_masks[index])
                probability_mask = to_numpy(probability_masks[index])
                binary_mask = to_numpy(binary_masks[index]).astype(np.uint8)
                predicted_center = outputs["center_vox"][index]

                center_error = float(
                    torch.linalg.vector_norm(
                        predicted_center - true_centers[index]
                    ).cpu()
                )

                examples.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "subject_id": str(batch["subject_id"][index]),
                        "gt": gt,
                        "probability_mask": probability_mask,
                        "binary_mask": binary_mask,
                        "residual_740": to_numpy(x_residual[index, 0]),
                        "residual_850": to_numpy(x_residual[index, 1]),
                        "class_probability": float(
                            class_probabilities[index].cpu()
                        ),
                        "dice": binary_dice(binary_mask, gt >= 0.5),
                        "center_error": center_error,
                        "predicted_center": to_numpy(
                            predicted_center
                        ).tolist(),
                        "true_center": to_numpy(
                            true_centers[index]
                        ).tolist(),
                        "predicted_radii": to_numpy(
                            outputs["radii_vox"][index]
                        ).tolist(),
                        "predicted_depth_mm": float(
                            outputs["depth_mm"][index].cpu()
                        ),
                        "predicted_volume_mm3": float(
                            outputs["volume_mm3"][index].cpu()
                        ),
                    }
                )

    return examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate final training graphs and create segmentation "
            "figures for the report."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset_root).resolve()
    run_dir = Path(args.run_dir).resolve()

    required_paths = [
        dataset_root / "metadata.csv",
        dataset_root / "samples",
        run_dir / "best_model.pt",
        run_dir / "history.json",
        run_dir / "test_metrics.json",
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    print("[1/4] Regenerating corrected training graphs...")
    regenerate_training_plots(run_dir, script_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[2/4] Loading checkpoint and test data on {device}...")

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
    )
    loader = build_test_loader(
        dataset_root,
        checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = load_model(checkpoint, device)

    print("[3/4] Running segmentation inference on positive test samples...")
    examples = collect_positive_examples(model, loader, device)

    if not examples:
        raise RuntimeError(
            "No positive hemorrhage samples were found in the test split."
        )

    examples.sort(key=lambda item: item["dice"])
    worst = examples[0]
    median = examples[len(examples) // 2]
    best = examples[-1]

    segmentation_dir = run_dir / "segmentation_visuals"
    report_dir = run_dir / "report_figures"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    save_example(
        best,
        segmentation_dir / "best_segmentation.png",
        "Best segmentation",
    )
    save_example(
        median,
        segmentation_dir / "median_segmentation.png",
        "Median segmentation",
    )
    save_example(
        worst,
        segmentation_dir / "worst_segmentation.png",
        "Worst segmentation",
    )
    save_summary(
        best,
        median,
        worst,
        report_dir / "13_segmentation_examples_summary.png",
    )

    excluded = {
        "gt",
        "probability_mask",
        "binary_mask",
        "residual_740",
        "residual_850",
    }
    summary = {
        "device": str(device),
        "number_of_positive_test_examples": len(examples),
        "mask_threshold": 0.5,
        "classification_threshold": float(
            checkpoint.get("classification_threshold", 0.5)
        ),
        "best": {
            key: value
            for key, value in best.items()
            if key not in excluded
        },
        "median": {
            key: value
            for key, value in median.items()
            if key not in excluded
        },
        "worst": {
            key: value
            for key, value in worst.items()
            if key not in excluded
        },
    }
    (
        report_dir / "segmentation_visualization_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("[4/4] Finished successfully.")
    print(f"Training graphs: {run_dir / 'plots'}")
    print(f"Segmentation figures: {segmentation_dir}")
    print(
        "Report summary figure: "
        f"{report_dir / '13_segmentation_examples_summary.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
