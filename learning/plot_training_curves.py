from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_by_stage(history: list[dict]) -> tuple[list[dict], list[dict]]:
    stage1 = [row for row in history if int(row.get("stage", 1)) == 1]
    stage2 = [row for row in history if int(row.get("stage", 1)) == 2]
    return stage1, stage2


def get_value(row: dict, key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def make_axis(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def add_stage_boundary(ax, last_stage1_epoch: int) -> None:
    boundary = last_stage1_epoch + 0.5
    ax.axvline(boundary, linestyle="--", linewidth=1.5)
    ymin, ymax = ax.get_ylim()
    ax.text(
        boundary + 0.5,
        ymax - 0.07 * (ymax - ymin),
        "Stage 2 begins\nDetection branch frozen",
        rotation=90,
        va="top",
        ha="left",
    )


def finish(fig, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    history_path = Path(args.history)
    metrics_path = Path(args.metrics)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove every old PNG first, so stale graphs from the previous script
    # cannot remain in the folder under obsolete filenames.
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()

    history = load_json(history_path)
    metrics = load_json(metrics_path)
    stage1, stage2 = split_by_stage(history)

    if not stage1:
        raise ValueError("No Stage 1 rows found in history.json")
    if not stage2:
        raise ValueError("No Stage 2 rows found in history.json")

    stage1_epochs = np.array([int(row["epoch"]) for row in stage1], dtype=int)
    last_stage1_epoch = int(stage1_epochs.max())

    # IMPORTANT: Stage 2 begins after Stage 1.
    stage2_epochs_continuous = np.array(
        [last_stage1_epoch + int(row["epoch"]) for row in stage2],
        dtype=int,
    )

    continuous_last_epoch = int(stage2_epochs_continuous.max())

    # 01 — training objectives
    fig, ax = make_axis(
        "Training objectives across both stages",
        "Continuous epoch",
        "Loss",
    )
    ax.plot(
        stage1_epochs,
        [get_value(row, "train_loss") for row in stage1],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Stage 1 joint loss",
    )
    ax.plot(
        stage2_epochs_continuous,
        [get_value(row, "train_loss") for row in stage2],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Stage 2 localization loss",
    )
    add_stage_boundary(ax, last_stage1_epoch)
    ax.set_xlim(1, continuous_last_epoch)
    ax.legend()
    finish(fig, output_dir / "01_training_objectives.png")

    # 02 — accuracy, Stage 1 only because detector is frozen in Stage 2
    fig, ax = make_axis(
        "Validation accuracy during Stage 1 detection training",
        "Stage 1 epoch",
        "Accuracy",
    )
    ax.plot(
        stage1_epochs,
        [get_value(row, "val_accuracy") for row in stage1],
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax.set_ylim(0.45, 1.02)
    finish(fig, output_dir / "02_validation_accuracy.png")

    # 03 — AUC, Stage 1 only
    fig, ax = make_axis(
        "Validation ROC-AUC during Stage 1 detection training",
        "Stage 1 epoch",
        "ROC-AUC",
    )
    ax.plot(
        stage1_epochs,
        [get_value(row, "val_roc_auc") for row in stage1],
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax.set_ylim(0.85, 1.005)
    finish(fig, output_dir / "03_validation_auc.png")

    # 04 — sensitivity and specificity, Stage 1 only
    fig, ax = make_axis(
        "Validation sensitivity and specificity during Stage 1",
        "Stage 1 epoch",
        "Metric value",
    )
    ax.plot(
        stage1_epochs,
        [get_value(row, "val_sensitivity") for row in stage1],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Sensitivity",
    )
    ax.plot(
        stage1_epochs,
        [get_value(row, "val_specificity") for row in stage1],
        marker="o",
        linewidth=2,
        markersize=4,
        label="Specificity",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    finish(fig, output_dir / "04_sensitivity_specificity.png")

    def two_stage_metric(
        filename: str,
        title: str,
        ylabel: str,
        history_key: str,
        test_key: str,
        test_label: str,
    ) -> None:
        fig, ax = make_axis(title, "Continuous epoch", ylabel)
        ax.plot(
            stage1_epochs,
            [get_value(row, history_key) for row in stage1],
            marker="o",
            linewidth=2,
            markersize=4,
            label="Stage 1",
        )
        ax.plot(
            stage2_epochs_continuous,
            [get_value(row, history_key) for row in stage2],
            marker="o",
            linewidth=2,
            markersize=4,
            label="Stage 2",
        )
        add_stage_boundary(ax, last_stage1_epoch)
        ax.axhline(
            float(metrics[test_key]),
            linestyle=":",
            linewidth=1.5,
            label=test_label.format(float(metrics[test_key])),
        )
        ax.set_xlim(1, continuous_last_epoch)
        ax.legend()
        finish(fig, output_dir / filename)

    # 05–08 — all continue through epochs 31–49
    two_stage_metric(
        "05_validation_dice.png",
        "Segmentation Dice across both training stages",
        "Positive-case Dice",
        "val_positive_dice",
        "positive_dice",
        "Test Dice = {:.3f}",
    )
    two_stage_metric(
        "06_center_error.png",
        "Center localization error across both training stages",
        "Error [voxels]",
        "val_center_error_vox",
        "center_error_vox",
        "Test error = {:.2f} vox",
    )
    two_stage_metric(
        "07_depth_error.png",
        "Depth estimation error across both training stages",
        "MAE [mm]",
        "val_depth_mae_mm",
        "depth_mae_mm",
        "Test MAE = {:.2f} mm",
    )
    two_stage_metric(
        "08_radii_error.png",
        "Radii estimation error across both training stages",
        "MAE [voxels]",
        "val_radii_mae_vox",
        "radii_mae_vox",
        "Test MAE = {:.3f} vox",
    )

    # 09 — Stage 2 checkpoint score
    fig, ax = make_axis(
        "Stage 2 checkpoint selection score",
        "Stage 2 epoch",
        "Selection score",
    )
    ax.plot(
        [int(row["epoch"]) for row in stage2],
        [get_value(row, "selection_score") for row in stage2],
        marker="o",
        linewidth=2,
        markersize=4,
    )
    finish(fig, output_dir / "09_selection_score.png")

    # 10 — final classification metrics
    names = ["Accuracy", "Sensitivity", "Specificity", "ROC-AUC"]
    values = [
        float(metrics["accuracy"]),
        float(metrics["sensitivity"]),
        float(metrics["specificity"]),
        float(metrics["roc_auc"]),
    ]
    fig, ax = make_axis(
        "Final test classification metrics",
        "Metric",
        "Value",
    )
    positions = np.arange(len(names))
    bars = ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylim(0.0, 1.05)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    finish(fig, output_dir / "10_final_classification_metrics.png")

    # 11 — final localization table
    rows = [
        ["Positive Dice", f"{float(metrics['positive_dice']):.3f}", "unitless"],
        ["Center error", f"{float(metrics['center_error_vox']):.2f}", "voxels"],
        ["Depth MAE", f"{float(metrics['depth_mae_mm']):.2f}", "mm"],
        ["Radii MAE", f"{float(metrics['radii_mae_vox']):.3f}", "voxels"],
        ["Volume MAE", f"{float(metrics['volume_mae_mm3']):.2f}", "mm³"],
    ]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axis("off")
    ax.set_title("Final test localization and segmentation metrics", pad=18)
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value", "Unit"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)
    finish(fig, output_dir / "11_final_localization_metrics.png")

    # 12 — confusion matrix
    confusion = np.array(
        [
            [int(metrics["tn"]), int(metrics["fp"])],
            [int(metrics["fn"]), int(metrics["tp"])],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(confusion)
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted healthy", "Predicted hemorrhage"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True healthy", "True hemorrhage"])
    ax.set_title("Final test confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=14)
    finish(fig, output_dir / "12_confusion_matrix.png")

    summary = {
        "stage1_epochs": len(stage1),
        "stage2_epochs": len(stage2),
        "stage1_epoch_range": [1, last_stage1_epoch],
        "stage2_continuous_epoch_range": [
            int(stage2_epochs_continuous.min()),
            int(stage2_epochs_continuous.max()),
        ],
        "deleted_old_pngs_before_generation": True,
        "generated_files": sorted(path.name for path in output_dir.glob("*.png")),
    }
    (output_dir / "plots_summary_corrected_v2.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Corrected plots saved to: {output_dir}")
    print(
        f"Stage 1 shown as epochs 1-{last_stage1_epoch}; "
        f"Stage 2 shown as epochs "
        f"{int(stage2_epochs_continuous.min())}-"
        f"{int(stage2_epochs_continuous.max())}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
