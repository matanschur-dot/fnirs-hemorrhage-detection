"""Optical simulation configuration. Coefficients are preliminary assumptions in 1/mm."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SIMULATION_VERSION = "1.0.0"

BASE_PROPERTIES_740 = np.array([
    [0.0,   0.0,   1.0,  1.0],
    [0.025, 8.5,   0.90, 1.40],
    [0.018, 10.5,  0.90, 1.40],
    [0.004, 0.05,  0.99, 1.33],
    [0.035, 12.5,  0.90, 1.40],
    [0.150, 15.0,  0.90, 1.40],
], dtype=np.float32)
BASE_PROPERTIES_850 = np.array([
    [0.0,   0.0,   1.0,  1.0],
    [0.015, 8.0,   0.90, 1.40],
    [0.010, 10.0,  0.90, 1.40],
    [0.003, 0.03,  0.99, 1.33],
    [0.025, 12.0,  0.90, 1.40],
    [0.120, 14.0,  0.90, 1.40],
], dtype=np.float32)

@dataclass(frozen=True)
class OpticalVariationConfig:
    enabled: bool = True
    mua_scale_range: tuple[float, float] = (0.95, 1.05)
    mus_scale_range: tuple[float, float] = (0.95, 1.05)
    hemorrhage_mua_scale_range: tuple[float, float] = (0.95, 1.05)
    hemorrhage_mus_scale_range: tuple[float, float] = (0.95, 1.05)

@dataclass(frozen=True)
class PMCXConfig:
    photon_count: int = 2000000
    tstart: float = 0.0
    tstep: float = 5e-9
    tend: float = 5e-8
    srcdir: tuple[float, float, float] = (0.0, 0.0, 1.0)
    issrcfrom0: int = 1
    backend: str = "pmcx"  # pmcx or analytic_debug
    detector_mode: str = "explicit_detected_photons"
    detector_radius_mm: float = 1.54
    paired_seed_mode: str = "identical"

@dataclass(frozen=True)
class NoiseConfig:
    mode: str = "none"  # none, basic
    led_gain_std: float = 0.01
    detector_gain_std: float = 0.008
    multiplicative_white_std: float = 0.01
    additive_fraction_std: float = 0.0015

@dataclass(frozen=True)
class SimulationRunConfig:
    pmcx: PMCXConfig = field(default_factory=PMCXConfig)
    variation: OpticalVariationConfig = field(default_factory=OpticalVariationConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
