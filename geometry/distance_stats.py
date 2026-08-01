"""
distance_stats.py

Source-detector distance analysis. The bin labels (very_short..very_long)
are ENGINEERING ANALYSIS CATEGORIES for eyeballing the distance
distribution - not automatic data-quality labels, and nothing here drops
or flags channels as bad based on them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from geometry.channel_map import ChannelRow

BIN_EDGES_MM = [0, 10, 20, 30, 40, 50, float("inf")]
BIN_LABELS = ["very_short (<10mm)", "short (10-20mm)", "medium (20-30mm)",
              "long (30-40mm)", "very_long (40-50mm)", "extreme (>50mm)"]


@dataclass
class DistanceStats:
    minimum_mm: float
    maximum_mm: float
    mean_mm: float
    median_mm: float
    std_mm: float
    n_very_short_lt10: int
    n_short_10_20: int
    n_medium_20_30: int
    n_long_30_40: int
    n_very_long_40_50: int
    n_extreme_gt50: int
    n_total: int


def compute_distance_stats(rows: list[ChannelRow]) -> DistanceStats:
    dists = [r.source_detector_distance_mm for r in rows]

    bins = [0, 0, 0, 0, 0, 0]
    for d in dists:
        if d < 10:
            bins[0] += 1
        elif d < 20:
            bins[1] += 1
        elif d < 30:
            bins[2] += 1
        elif d < 40:
            bins[3] += 1
        elif d < 50:
            bins[4] += 1
        else:
            bins[5] += 1

    return DistanceStats(
        minimum_mm=min(dists), maximum_mm=max(dists),
        mean_mm=statistics.mean(dists), median_mm=statistics.median(dists),
        std_mm=statistics.pstdev(dists),
        n_very_short_lt10=bins[0], n_short_10_20=bins[1], n_medium_20_30=bins[2],
        n_long_30_40=bins[3], n_very_long_40_50=bins[4], n_extreme_gt50=bins[5],
        n_total=len(dists),
    )
