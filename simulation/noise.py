from __future__ import annotations
import numpy as np
from configs.simulation_config import NoiseConfig

def apply_noise(i740: np.ndarray, i850: np.ndarray, seed: int, cfg: NoiseConfig) -> tuple[np.ndarray,np.ndarray,dict]:
    if cfg.mode == "none":
        return i740.copy(), i850.copy(), {"noise_mode":"none"}
    if cfg.mode != "basic":
        raise ValueError(f"Unsupported noise mode: {cfg.mode}")
    rng = np.random.default_rng(seed)
    det_gain = 1.0 + rng.normal(0, cfg.detector_gain_std, size=(1,16))
    out=[]
    for arr in (i740,i850):
        led_gain = 1.0 + rng.normal(0, cfg.led_gain_std, size=(8,1))
        mult = 1.0 + rng.normal(0, cfg.multiplicative_white_std, size=arr.shape)
        scale=max(float(arr.mean()), np.finfo(float).tiny)
        add = rng.normal(0, scale*cfg.additive_fraction_std, size=arr.shape)
        out.append(np.clip(arr*led_gain*det_gain*mult+add,0,None))
    return out[0],out[1],{"noise_mode":"basic"}
