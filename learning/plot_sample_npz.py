from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def safe_log10(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a non-negative intensity matrix to log10 scale.

    Zeros are replaced by a small positive floor derived from the smallest
    positive value in the same matrix, so no -inf values are produced.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    positive = matrix[matrix > 0]

    if positive.size == 0:
        return np.zeros_like(matrix, dtype=np.float64)

    floor = max(float(positive.min()) * 0.1, np.finfo(np.float64).tiny)
    return np.log10(np.clip(matrix, floor, None))


def signed_log10(matrix: np.ndarray) -> np.ndarray:
    """
    Signed logarithmic transform for residual matrices.

    Positive and negative changes are preserved:
        sign(x) * log10(1 + |x| / scale)
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    nonzero = np.abs(matrix[matrix != 0])

    if nonzero.size == 0:
        return np.zeros_like(matrix, dtype=np.float64)

    scale = max(float(np.median(nonzero)), np.finfo(np.float64).tiny)
    return np.sign(matrix) * np.log10(1.0 + np.abs(matrix) / scale)


def save_heatmap(
    matrix: np.ndarray,
    title: str,
    colorbar_label: str,
    output_path: Path,
    symmetric: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    if symmetric:
        limit = float(np.max(np.abs(matrix)))
        if limit == 0.0:
            limit = 1.0
        image = ax.imshow(
            matrix,
            aspect="auto",
            vmin=-limit,
            vmax=limit,
        )
    else:
        image = ax.imshow(matrix, aspect="auto")

    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_xlabel("Detector index")
    ax.set_ylabel("Source index")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_mask_projection(
    mask: np.ndarray,
    axis: int,
    title: str,
    output_path: Path,
) -> None:
    projection = mask.sum(axis=axis)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(projection)
    fig.colorbar(image, ax=ax, label="Projected hemorrhage mask")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def paired_healthy_path(sample_path: Path, subject_id: str) -> Path:
    path = sample_path.parent / f"{subject_id}_healthy.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find paired healthy sample: {path}"
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        required=True,
        help="Path to a healthy or hemorrhage NPZ sample",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory in which the PNG files will be saved",
    )
    parser.add_argument(
        "--include-linear",
        action="store_true",
        help="Also save the original linear-scale matrices",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(sample_path, allow_pickle=False) as data:
        i740 = np.asarray(data["I740"], dtype=np.float64)
        i850 = np.asarray(data["I850"], dtype=np.float64)
        mask = np.asarray(data["hemorrhage_mask"], dtype=np.float32)
        label = int(np.asarray(data["label"]).item())
        sample_id = str(np.asarray(data["sample_id"]).item())
        subject_id = str(np.asarray(data["subject_id"]).item())

    # Main figures: logarithmic intensity scale.
    save_heatmap(
        safe_log10(i740),
        f"{sample_id} | 740 nm measurement (log scale)",
        "log10 intensity",
        output_dir / "I740_log10.png",
    )
    save_heatmap(
        safe_log10(i850),
        f"{sample_id} | 850 nm measurement (log scale)",
        "log10 intensity",
        output_dir / "I850_log10.png",
    )

    # Optional linear versions for comparison.
    if args.include_linear:
        save_heatmap(
            i740,
            f"{sample_id} | 740 nm measurement (linear scale)",
            "Intensity",
            output_dir / "I740_linear.png",
        )
        save_heatmap(
            i850,
            f"{sample_id} | 850 nm measurement (linear scale)",
            "Intensity",
            output_dir / "I850_linear.png",
        )

    # For hemorrhage samples, compare against the paired healthy baseline and
    # plot signed-log residuals.
    if label == 1:
        healthy_path = paired_healthy_path(sample_path, subject_id)

        with np.load(healthy_path, allow_pickle=False) as healthy:
            healthy_740 = np.asarray(healthy["I740"], dtype=np.float64)
            healthy_850 = np.asarray(healthy["I850"], dtype=np.float64)

        residual_740 = i740 - healthy_740
        residual_850 = i850 - healthy_850

        save_heatmap(
            signed_log10(residual_740),
            f"{sample_id} | 740 nm hemorrhage-minus-healthy residual",
            "Signed logarithmic residual",
            output_dir / "residual_740_signed_log.png",
            symmetric=True,
        )
        save_heatmap(
            signed_log10(residual_850),
            f"{sample_id} | 850 nm hemorrhage-minus-healthy residual",
            "Signed logarithmic residual",
            output_dir / "residual_850_signed_log.png",
            symmetric=True,
        )

        if mask.sum() > 0:
            save_mask_projection(
                mask,
                axis=0,
                title=f"{sample_id} | mask projection over axis 0",
                output_path=output_dir / "mask_projection_axis0.png",
            )
            save_mask_projection(
                mask,
                axis=1,
                title=f"{sample_id} | mask projection over axis 1",
                output_path=output_dir / "mask_projection_axis1.png",
            )
            save_mask_projection(
                mask,
                axis=2,
                title=f"{sample_id} | mask projection over axis 2",
                output_path=output_dir / "mask_projection_axis2.png",
            )

    print(f"Saved logarithmic matrix plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
