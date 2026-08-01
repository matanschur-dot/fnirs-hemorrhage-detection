from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SplitFiles:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def make_subject_split(
    metadata_csv: str | Path,
    samples_dir: str | Path,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> SplitFiles:
    """Split strictly by subject_id so paired samples never cross splits."""
    metadata_csv = Path(metadata_csv)
    samples_dir = Path(samples_dir)

    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {metadata_csv}")

    subject_ids = sorted({row["subject_id"] for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    if n_subjects < 3:
        raise ValueError("At least three subjects are required")

    n_train = int(round(train_fraction * n_subjects))
    n_val = int(round(val_fraction * n_subjects))
    n_train = min(max(n_train, 1), n_subjects - 2)
    n_val = min(max(n_val, 1), n_subjects - n_train - 1)

    train_ids = set(subject_ids[:n_train])
    val_ids = set(subject_ids[n_train:n_train + n_val])
    test_ids = set(subject_ids[n_train + n_val:])

    def collect(ids: set[str]) -> list[Path]:
        result: list[Path] = []
        for row in rows:
            if row["subject_id"] not in ids:
                continue
            if row.get("status", "complete") != "complete":
                continue
            path = samples_dir / row["file_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
            result.append(path)
        return sorted(result)

    return SplitFiles(
        train=collect(train_ids),
        val=collect(val_ids),
        test=collect(test_ids),
    )


def filter_positive_files(files: list[Path]) -> list[Path]:
    """Keep only hemorrhage samples for stage-2 localization fine-tuning."""
    positives: list[Path] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            label = int(np.asarray(data["label"]).item())
        if label == 1:
            positives.append(path)
    if not positives:
        raise ValueError("No positive samples were found")
    return positives


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)


def paired_healthy_path(sample_path: Path, subject_id: str) -> Path:
    path = sample_path.parent / f"{subject_id}_healthy.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing paired healthy sample: {path}")
    return path


@dataclass(frozen=True)
class MeasurementAugmentation:
    led_gain_std: float = 0.010
    detector_gain_std: float = 0.008
    multiplicative_std: float = 0.010
    additive_fraction_std: float = 0.0015

    def apply(self, matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)

        led_gain = rng.normal(1.0, self.led_gain_std, (matrix.shape[0], 1)).astype(np.float32)
        detector_gain = rng.normal(1.0, self.detector_gain_std, (1, matrix.shape[1])).astype(np.float32)
        multiplicative = rng.normal(1.0, self.multiplicative_std, matrix.shape).astype(np.float32)

        positive = matrix[matrix > 0]
        reference = float(np.median(positive)) if positive.size else 1e-8
        additive = rng.normal(
            0.0,
            self.additive_fraction_std * reference,
            matrix.shape,
        ).astype(np.float32)

        result = matrix * led_gain * detector_gain * multiplicative + additive
        return np.clip(result, 0.0, None).astype(np.float32)


@dataclass
class DualInputNormalizer:
    absolute_mean: torch.Tensor
    absolute_std: torch.Tensor
    residual_mean: torch.Tensor
    residual_std: torch.Tensor
    scale: float = 1e6

    @staticmethod
    def _absolute_transform_np(x: np.ndarray, scale: float) -> np.ndarray:
        return np.log1p(np.maximum(x, 0.0) * scale)

    @staticmethod
    def _residual_transform_np(x: np.ndarray, scale: float) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x) * scale)

    @classmethod
    def fit(cls, files: list[Path], scale: float = 1e6) -> "DualInputNormalizer":
        absolute_values: list[np.ndarray] = []
        residual_values: list[np.ndarray] = []

        for path in files:
            with np.load(path, allow_pickle=False) as data:
                subject_id = str(np.asarray(data["subject_id"]).item())
                label = int(np.asarray(data["label"]).item())
                current_740 = np.asarray(data["I740"], dtype=np.float32)
                current_850 = np.asarray(data["I850"], dtype=np.float32)

            healthy_path = paired_healthy_path(path, subject_id)
            with np.load(healthy_path, allow_pickle=False) as healthy:
                baseline_740 = np.asarray(healthy["I740"], dtype=np.float32)
                baseline_850 = np.asarray(healthy["I850"], dtype=np.float32)

            absolute = np.stack(
                [baseline_740, baseline_850, current_740, current_850],
                axis=0,
            )
            absolute_values.append(cls._absolute_transform_np(absolute, scale))

            if label == 1:
                residual = np.stack(
                    [current_740 - baseline_740, current_850 - baseline_850],
                    axis=0,
                )
                residual_values.append(cls._residual_transform_np(residual, scale))

        if not residual_values:
            raise ValueError("No positive samples were found for residual normalization")

        abs_stack = np.stack(absolute_values, axis=0)
        res_stack = np.stack(residual_values, axis=0)

        absolute_mean = abs_stack.mean(axis=(0, 2, 3), keepdims=True)
        absolute_std = np.maximum(abs_stack.std(axis=(0, 2, 3), keepdims=True), 1e-6)
        residual_mean = res_stack.mean(axis=(0, 2, 3), keepdims=True)
        residual_std = np.maximum(res_stack.std(axis=(0, 2, 3), keepdims=True), 1e-6)

        return cls(
            absolute_mean=torch.from_numpy(absolute_mean.astype(np.float32)),
            absolute_std=torch.from_numpy(absolute_std.astype(np.float32)),
            residual_mean=torch.from_numpy(residual_mean.astype(np.float32)),
            residual_std=torch.from_numpy(residual_std.astype(np.float32)),
            scale=float(scale),
        )

    def transform_absolute(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(torch.clamp(x, min=0.0) * self.scale)
        mean = self.absolute_mean.to(x.device).squeeze(0)
        std = self.absolute_std.to(x.device).squeeze(0)
        return (x - mean) / std

    def transform_residual(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sign(x) * torch.log1p(torch.abs(x) * self.scale)
        mean = self.residual_mean.to(x.device).squeeze(0)
        std = self.residual_std.to(x.device).squeeze(0)
        return (x - mean) / std

    def state_dict(self) -> dict:
        return {
            "absolute_mean": self.absolute_mean,
            "absolute_std": self.absolute_std,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "scale": self.scale,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DualInputNormalizer":
        return cls(
            absolute_mean=state["absolute_mean"],
            absolute_std=state["absolute_std"],
            residual_mean=state["residual_mean"],
            residual_std=state["residual_std"],
            scale=float(state["scale"]),
        )


def mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argwhere(mask > 0)
    if indices.size == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    low = indices.min(axis=0).astype(np.float32)
    high = indices.max(axis=0).astype(np.float32)
    center = 0.5 * (low + high)
    radii = np.maximum(0.5 * (high - low), 1.0)
    return center, radii


class FNIRSDualBranchDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        normalizer: DualInputNormalizer,
        training: bool,
        augmentation: MeasurementAugmentation | None = None,
        seed: int = 42,
    ) -> None:
        self.files = list(files)
        self.normalizer = normalizer
        self.training = bool(training)
        self.augmentation = augmentation or MeasurementAugmentation()
        self.seed = int(seed)

        if not self.files:
            raise ValueError("Dataset received no files")

        with np.load(self.files[0], allow_pickle=False) as data:
            self.volume_shape = tuple(int(v) for v in data["volume_shape"])
            self.voxel_size_mm = float(np.asarray(data["voxel_size_mm"]).item())

    def __len__(self) -> int:
        return len(self.files)

    def _rng(self, sample_id: str, stream: str) -> np.random.Generator:
        if self.training:
            return np.random.default_rng()
        return np.random.default_rng(stable_seed(f"{sample_id}:{stream}", self.seed))

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]

        with np.load(path, allow_pickle=False) as data:
            subject_id = str(np.asarray(data["subject_id"]).item())
            sample_id = str(np.asarray(data["sample_id"]).item())
            label = float(np.asarray(data["label"]).item())
            current_740_clean = np.asarray(data["I740"], dtype=np.float32)
            current_850_clean = np.asarray(data["I850"], dtype=np.float32)
            mask = np.asarray(data["hemorrhage_mask"], dtype=np.float32)
            depth_mm = float(np.asarray(data["hemorrhage_depth_from_surface_mm"]).item())
            volume_mm3 = float(np.asarray(data["hemorrhage_volume_mm3"]).item())

        healthy_path = paired_healthy_path(path, subject_id)
        with np.load(healthy_path, allow_pickle=False) as healthy:
            baseline_740_clean = np.asarray(healthy["I740"], dtype=np.float32)
            baseline_850_clean = np.asarray(healthy["I850"], dtype=np.float32)

        baseline_740_noisy = self.augmentation.apply(
            baseline_740_clean, self._rng(sample_id, "baseline740")
        )
        baseline_850_noisy = self.augmentation.apply(
            baseline_850_clean, self._rng(sample_id, "baseline850")
        )
        current_740_noisy = self.augmentation.apply(
            current_740_clean, self._rng(sample_id, "current740")
        )
        current_850_noisy = self.augmentation.apply(
            current_850_clean, self._rng(sample_id, "current850")
        )

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
        x_absolute = self.normalizer.transform_absolute(x_absolute)

        x_residual = torch.from_numpy(
            np.stack(
                [
                    current_740_clean - baseline_740_clean,
                    current_850_clean - baseline_850_clean,
                ],
                axis=0,
            ).astype(np.float32)
        )
        x_residual = self.normalizer.transform_residual(x_residual)

        center_vox, radii_vox = mask_geometry(mask)
        if label == 0.0:
            depth_mm = 0.0
            volume_mm3 = 0.0

        return {
            "x_absolute": x_absolute,
            "x_residual": x_residual,
            "label": torch.tensor(label, dtype=torch.float32),
            "center_vox": torch.from_numpy(center_vox),
            "radii_vox": torch.from_numpy(radii_vox),
            "depth_mm": torch.tensor(depth_mm, dtype=torch.float32),
            "volume_mm3": torch.tensor(volume_mm3, dtype=torch.float32),
            "mask": torch.from_numpy(mask),
            "sample_id": sample_id,
            "subject_id": subject_id,
        }
