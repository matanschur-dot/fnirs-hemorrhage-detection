from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fnirs_dataset import (
    DualInputNormalizer,
    FNIRSDualBranchDataset,
    MeasurementAugmentation,
    filter_positive_files,
    make_subject_split,
)
from medt_multitask import DualBranchMedT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive_dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def localization_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Stage-2 localization loss.

    Important change:
    center supervision is expressed directly in voxels, with both coordinate
    Smooth-L1 and Euclidean distance terms.
    """
    center_delta = outputs["center_vox"] - batch["center_vox"]

    center_coordinate = torch.nn.functional.smooth_l1_loss(
        center_delta / 5.0,
        torch.zeros_like(center_delta),
    )
    center_distance = torch.linalg.vector_norm(
        center_delta,
        dim=1,
    ).mean() / 5.0

    radii = torch.nn.functional.smooth_l1_loss(
        outputs["radii_vox"] / 8.0,
        batch["radii_vox"] / 8.0,
    )
    depth = torch.nn.functional.smooth_l1_loss(
        outputs["depth_mm"] / 20.0,
        batch["depth_mm"] / 20.0,
    )
    volume = torch.nn.functional.smooth_l1_loss(
        torch.log1p(outputs["volume_mm3"]) / 8.0,
        torch.log1p(batch["volume_mm3"]) / 8.0,
    )
    dice = positive_dice_loss(
        outputs["soft_mask"],
        batch["mask"],
    )
    mask_bce = torch.nn.functional.binary_cross_entropy(
        outputs["soft_mask"].clamp(1e-6, 1.0 - 1e-6),
        batch["mask"],
    )

    total = (
        2.0 * center_coordinate
        + 2.5 * center_distance
        + 1.25 * radii
        + 0.75 * depth
        + 0.50 * volume
        + 3.0 * dice
        + 0.15 * mask_bce
    )

    return total, {
        "center_coordinate": float(center_coordinate.detach()),
        "center_distance": float(center_distance.detach()),
        "radii": float(radii.detach()),
        "depth": float(depth.detach()),
        "volume": float(volume.detach()),
        "dice": float(dice.detach()),
        "mask_bce": float(mask_bce.detach()),
    }


def joint_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    label = batch["label"]
    positive = label > 0.5

    classification = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["class_logit"], label
    )

    if positive.any():
        positive_outputs = {
            key: value[positive]
            for key, value in outputs.items()
            if key != "class_logit"
        }
        positive_batch = {
            key: value[positive]
            for key, value in batch.items()
            if torch.is_tensor(value) and key not in {"x_absolute", "x_residual", "label"}
        }
        loc_total, loc_parts = localization_loss(
            positive_outputs,
            positive_batch,
        )
    else:
        loc_total = outputs["center_vox"].sum() * 0.0
        loc_parts = {}

    total = 1.5 * classification + loc_total
    return total, {
        "classification": float(classification.detach()),
        **loc_parts,
    }


def roc_auc_binary(labels: list[float], scores: list[float]) -> float:
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(scores, dtype=np.float64)
    positives = int(labels_np.sum())
    negatives = int(len(labels_np) - positives)

    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores_np)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores_np) + 1)

    _, inverse, counts = np.unique(
        scores_np,
        return_inverse=True,
        return_counts=True,
    )
    for group_index, count in enumerate(counts):
        if count > 1:
            members = np.where(inverse == group_index)[0]
            ranks[members] = ranks[members].mean()

    positive_rank_sum = ranks[labels_np == 1].sum()
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def select_threshold(
    labels: list[float],
    probabilities: list[float],
) -> float:
    """
    Choose threshold on validation only by maximizing Youden's J.
    Ties prefer the threshold nearest 0.5.
    """
    labels_np = np.asarray(labels, dtype=np.int64)
    probabilities_np = np.asarray(probabilities, dtype=np.float64)

    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0]),
                probabilities_np,
            ]
        )
    )

    best_threshold = 0.5
    best_j = -float("inf")
    best_distance = float("inf")

    for threshold in candidates:
        predictions = (probabilities_np >= threshold).astype(np.int64)

        tp = int(((predictions == 1) & (labels_np == 1)).sum())
        tn = int(((predictions == 0) & (labels_np == 0)).sum())
        fp = int(((predictions == 1) & (labels_np == 0)).sum())
        fn = int(((predictions == 0) & (labels_np == 1)).sum())

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        j = sensitivity + specificity - 1.0
        distance = abs(float(threshold) - 0.5)

        if j > best_j or (math.isclose(j, best_j) and distance < best_distance):
            best_j = j
            best_distance = distance
            best_threshold = float(threshold)

    return best_threshold


@torch.no_grad()
def evaluate(
    model: DualBranchMedT,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    return_raw_classification: bool = False,
) -> tuple[dict[str, float], list[float] | None, list[float] | None]:
    model.eval()

    labels: list[float] = []
    probabilities: list[float] = []
    center_errors: list[float] = []
    depth_errors: list[float] = []
    radii_errors: list[float] = []
    volume_errors: list[float] = []
    dice_scores: list[float] = []

    for batch in loader:
        tensor_batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        outputs = model(
            tensor_batch["x_absolute"],
            tensor_batch["x_residual"],
        )

        probability = torch.sigmoid(outputs["class_logit"])
        labels.extend(tensor_batch["label"].cpu().tolist())
        probabilities.extend(probability.cpu().tolist())

        positive = tensor_batch["label"] > 0.5
        if positive.any():
            center_errors.extend(
                torch.linalg.vector_norm(
                    outputs["center_vox"][positive]
                    - tensor_batch["center_vox"][positive],
                    dim=1,
                ).cpu().tolist()
            )
            depth_errors.extend(
                torch.abs(
                    outputs["depth_mm"][positive]
                    - tensor_batch["depth_mm"][positive]
                ).cpu().tolist()
            )
            radii_errors.extend(
                torch.abs(
                    outputs["radii_vox"][positive]
                    - tensor_batch["radii_vox"][positive]
                ).mean(dim=1).cpu().tolist()
            )
            volume_errors.extend(
                torch.abs(
                    outputs["volume_mm3"][positive]
                    - tensor_batch["volume_mm3"][positive]
                ).cpu().tolist()
            )

            predicted_mask = outputs["soft_mask"][positive] >= 0.5
            true_mask = tensor_batch["mask"][positive] >= 0.5
            intersection = (predicted_mask & true_mask).flatten(1).sum(dim=1).float()
            denominator = (
                predicted_mask.flatten(1).sum(dim=1)
                + true_mask.flatten(1).sum(dim=1)
            )
            dice_scores.extend(
                (2.0 * intersection / denominator.clamp_min(1)).cpu().tolist()
            )

    labels_np = np.asarray(labels, dtype=np.int64)
    probabilities_np = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities_np >= threshold).astype(np.int64)

    tp = int(((predictions == 1) & (labels_np == 1)).sum())
    tn = int(((predictions == 0) & (labels_np == 0)).sum())
    fp = int(((predictions == 1) & (labels_np == 0)).sum())
    fn = int(((predictions == 0) & (labels_np == 1)).sum())

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    metrics = {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(len(labels_np), 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "roc_auc": roc_auc_binary(labels, probabilities),
        "center_error_vox": mean(center_errors),
        "depth_mae_mm": mean(depth_errors),
        "radii_mae_vox": mean(radii_errors),
        "volume_mae_mm3": mean(volume_errors),
        "positive_dice": mean(dice_scores),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    if return_raw_classification:
        return metrics, labels, probabilities
    return metrics, None, None


def localization_score(metrics: dict[str, float]) -> float:
    dice = metrics["positive_dice"]
    center = metrics["center_error_vox"]
    radii = metrics["radii_mae_vox"]

    if not math.isfinite(dice):
        dice = 0.0
    if not math.isfinite(center):
        center = 100.0
    if not math.isfinite(radii):
        radii = 10.0

    center_quality = math.exp(-center / 6.0)
    radii_quality = math.exp(-radii / 1.0)
    return 0.60 * dice + 0.30 * center_quality + 0.10 * radii_quality


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default="../output/medt_training_final_v2")
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage2-patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = make_subject_split(
        metadata_csv=dataset_root / "metadata.csv",
        samples_dir=dataset_root / "samples",
        seed=args.seed,
    )

    normalizer = DualInputNormalizer.fit(split.train)
    augmentation = MeasurementAugmentation()

    train_dataset = FNIRSDualBranchDataset(
        split.train,
        normalizer,
        training=True,
        augmentation=augmentation,
        seed=args.seed,
    )
    positive_train_dataset = FNIRSDualBranchDataset(
        filter_positive_files(split.train),
        normalizer,
        training=True,
        augmentation=augmentation,
        seed=args.seed,
    )
    validation_dataset = FNIRSDualBranchDataset(
        split.val,
        normalizer,
        training=False,
        augmentation=augmentation,
        seed=args.seed + 1,
    )
    test_dataset = FNIRSDualBranchDataset(
        split.test,
        normalizer,
        training=False,
        augmentation=augmentation,
        seed=args.seed + 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    positive_train_loader = DataLoader(
        positive_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchMedT(
        volume_shape=train_dataset.volume_shape,
        voxel_size_mm=train_dataset.voxel_size_mm,
    ).to(device)

    print(f"Device: {device}")
    print(
        "Train/positive-train/val/test samples: "
        f"{len(train_dataset)}/{len(positive_train_dataset)}/"
        f"{len(validation_dataset)}/{len(test_dataset)}"
    )
    print(f"Volume shape: {train_dataset.volume_shape}")
    print("Stage 1: joint detection + localization")
    print("Stage 2: frozen detector, positive-only localization fine-tuning")
    print("Checkpoint selection: Dice + center + radii")
    print("Classification threshold: selected on validation only")

    history: list[dict] = []

    # ---------------------------
    # Stage 1: joint optimization
    # ---------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.stage1_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        losses: list[float] = []

        for batch in train_loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                tensor_batch["x_absolute"],
                tensor_batch["x_residual"],
            )
            loss, _ = joint_loss(outputs, tensor_batch)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in stage 1, epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        val_metrics, _, _ = evaluate(
            model, validation_loader, device, threshold=0.5
        )
        record = {
            "stage": 1,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print(
            f"Stage1 {epoch:03d} | "
            f"loss={record['train_loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"auc={val_metrics['roc_auc']:.4f} | "
            f"dice={val_metrics['positive_dice']:.4f} | "
            f"center={val_metrics['center_error_vox']:.3f} vox | "
            f"depth={val_metrics['depth_mae_mm']:.3f} mm | "
            f"radii={val_metrics['radii_mae_vox']:.3f} vox"
        )

    # Calibrate threshold after detection training, using validation only.
    _, validation_labels, validation_probabilities = evaluate(
        model,
        validation_loader,
        device,
        threshold=0.5,
        return_raw_classification=True,
    )
    calibrated_threshold = select_threshold(
        validation_labels or [],
        validation_probabilities or [],
    )
    calibrated_val_metrics, _, _ = evaluate(
        model,
        validation_loader,
        device,
        threshold=calibrated_threshold,
    )
    print(
        f"Validation-selected threshold after stage 1: "
        f"{calibrated_threshold:.6f}"
    )
    print(
        "Calibrated validation classification | "
        f"acc={calibrated_val_metrics['accuracy']:.4f} | "
        f"sens={calibrated_val_metrics['sensitivity']:.4f} | "
        f"spec={calibrated_val_metrics['specificity']:.4f}"
    )

    # Preserve the final stage-1 detector state.
    detection_state = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if key.startswith("detection_encoder.")
        or key.startswith("detection_head.")
    }

    # -------------------------------------------
    # Stage 2: positive-only localization tuning
    # -------------------------------------------
    model.freeze_detection_branch()

    localization_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        localization_parameters,
        lr=args.stage2_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )

    best_score = -float("inf")
    bad_epochs = 0

    for stage2_epoch in range(1, args.stage2_epochs + 1):
        model.train()
        model.detection_encoder.eval()
        model.detection_head.eval()

        losses: list[float] = []

        for batch in positive_train_loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                tensor_batch["x_absolute"],
                tensor_batch["x_residual"],
            )
            loss, _ = localization_loss(outputs, tensor_batch)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in stage 2, epoch {stage2_epoch}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                localization_parameters,
                max_norm=1.0,
            )
            optimizer.step()
            losses.append(float(loss.detach()))

        val_metrics, _, _ = evaluate(
            model,
            validation_loader,
            device,
            threshold=calibrated_threshold,
        )
        score = localization_score(val_metrics)
        scheduler.step(score)

        record = {
            "stage": 2,
            "epoch": stage2_epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        print(
            f"Stage2 {stage2_epoch:03d} | "
            f"loss={record['train_loss']:.4f} | "
            f"dice={val_metrics['positive_dice']:.4f} | "
            f"center={val_metrics['center_error_vox']:.3f} vox | "
            f"depth={val_metrics['depth_mae_mm']:.3f} mm | "
            f"radii={val_metrics['radii_mae_vox']:.3f} vox | "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0

            full_state = model.state_dict()
            for key, value in detection_state.items():
                full_state[key] = value.to(full_state[key].device)

            torch.save(
                {
                    "model_state": full_state,
                    "normalizer": normalizer.state_dict(),
                    "volume_shape": train_dataset.volume_shape,
                    "voxel_size_mm": train_dataset.voxel_size_mm,
                    "augmentation": augmentation.__dict__,
                    "classification_threshold": calibrated_threshold,
                    "args": vars(args),
                    "best_validation_metrics": val_metrics,
                    "best_localization_score": score,
                },
                output_dir / "best_model.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.stage2_patience:
                print("Stage 2 early stopping.")
                break

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["classification_threshold"])

    test_metrics, _, _ = evaluate(
        model,
        test_loader,
        device,
        threshold=threshold,
    )

    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    print("\nFinal test metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved model: {output_dir / 'best_model.pt'}")
    print(f"Saved metrics: {output_dir / 'test_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
