from __future__ import annotations
import numpy as np
from configs.simulation_config import BASE_PROPERTIES_740, BASE_PROPERTIES_850, OpticalVariationConfig

def sample_optical_properties(seed: int, cfg: OpticalVariationConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    p740 = BASE_PROPERTIES_740.copy()
    p850 = BASE_PROPERTIES_850.copy()
    meta = {}
    if not cfg.enabled:
        return p740, p850, meta
    for wavelength, props in ((740, p740), (850, p850)):
        for label in range(1, 6):
            if label == 5:
                ar, sr = cfg.hemorrhage_mua_scale_range, cfg.hemorrhage_mus_scale_range
            else:
                ar, sr = cfg.mua_scale_range, cfg.mus_scale_range
            a = float(rng.uniform(*ar)); s = float(rng.uniform(*sr))
            props[label, 0] *= a; props[label, 1] *= s
            meta[f"{wavelength}_label{label}_mua_scale"] = a
            meta[f"{wavelength}_label{label}_mus_scale"] = s
    return p740, p850, meta
