"""
pairing.py

Verifies that a healthy/hemorrhage volume pair differs ONLY in the
hemorrhage voxels - the single most important correctness property of the
whole synthetic-data pipeline (see project brief section on avoiding
class-specific artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anatomy.labels import BRAIN, HEMORRHAGE


class PairValidationError(Exception):
    pass


@dataclass
class PairValidationResult:
    difference_mask_matches_hemorrhage_mask: bool
    all_changed_healthy_voxels_were_brain: bool
    all_changed_hemorrhage_voxels_are_hemorrhage_label: bool
    all_unchanged_voxels_identical: bool
    n_changed_voxels: int

    @property
    def ok(self) -> bool:
        return all([
            self.difference_mask_matches_hemorrhage_mask,
            self.all_changed_healthy_voxels_were_brain,
            self.all_changed_hemorrhage_voxels_are_hemorrhage_label,
            self.all_unchanged_voxels_identical,
        ])


def validate_pair(
    healthy_volume: np.ndarray,
    hemorrhage_volume: np.ndarray,
    hemorrhage_mask: np.ndarray,
) -> PairValidationResult:
    if healthy_volume.shape != hemorrhage_volume.shape:
        raise PairValidationError(
            f"Shape mismatch: healthy={healthy_volume.shape} hemorrhage={hemorrhage_volume.shape}"
        )

    difference_mask = healthy_volume != hemorrhage_volume
    hem_mask_bool = hemorrhage_mask.astype(bool)

    diff_matches_mask = bool(np.array_equal(difference_mask, hem_mask_bool))
    changed_healthy_were_brain = bool(np.all(healthy_volume[difference_mask] == BRAIN)) if difference_mask.any() else True
    changed_hemorrhage_are_hem = bool(np.all(hemorrhage_volume[difference_mask] == HEMORRHAGE)) if difference_mask.any() else True
    unchanged_identical = bool(np.array_equal(
        healthy_volume[~difference_mask], hemorrhage_volume[~difference_mask]
    ))

    result = PairValidationResult(
        difference_mask_matches_hemorrhage_mask=diff_matches_mask,
        all_changed_healthy_voxels_were_brain=changed_healthy_were_brain,
        all_changed_hemorrhage_voxels_are_hemorrhage_label=changed_hemorrhage_are_hem,
        all_unchanged_voxels_identical=unchanged_identical,
        n_changed_voxels=int(difference_mask.sum()),
    )

    if not result.ok:
        raise PairValidationError(
            f"Pair validation FAILED: {result}\n"
            f"This means the healthy/hemorrhage pair differs by more than just the "
            f"hemorrhage region - a model trained on this pair could learn a "
            f"non-hemorrhage artifact instead of hemorrhage physics."
        )

    return result
