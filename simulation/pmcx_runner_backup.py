"""PMCX optical runner plus an explicitly non-physical analytic_debug backend for CI only."""
from __future__ import annotations
import time
import numpy as np
from configs.simulation_config import PMCXConfig
from simulation.detector_model import extract_detector_values

class PMCXUnavailableError(RuntimeError): pass
class OpticalSimulationError(RuntimeError): pass

def _validate_source(volume: np.ndarray, source_mm: np.ndarray, voxel_size_mm: float, srcdir) -> np.ndarray:
    p=np.asarray(source_mm,dtype=float)/voxel_size_mm
    d=np.asarray(srcdir,dtype=float); n=np.linalg.norm(d)
    if not np.isfinite(p).all() or not np.isfinite(d).all() or n == 0:
        raise OpticalSimulationError("Invalid source position or direction")
    d=d/n
    idx=np.rint(p).astype(int)
    if np.any(idx<0) or np.any(idx>=np.array(volume.shape)):
        raise OpticalSimulationError(f"Source outside volume: {source_mm.tolist()} mm")
    if volume[tuple(idx)] == 0:
        raise OpticalSimulationError(f"Source is in air at voxel {idx.tolist()}; expected first scalp voxel")
    step=np.rint(p+d).astype(int)
    if np.any(step<0) or np.any(step>=np.array(volume.shape)) or volume[tuple(step)] == 0:
        raise OpticalSimulationError("Source direction does not point into tissue")
    return d

def _run_pmcx(volume, source_mm, detectors_mm, props, voxel_size_mm, cfg, seed):
    try:
        import pmcx
    except ImportError as exc:
        raise PMCXUnavailableError("pmcx is not installed. Install PMCX/CUDA and rerun with --backend pmcx.") from exc
    d=_validate_source(volume,source_mm,voxel_size_mm,cfg.srcdir)
    source_vox=np.asarray(source_mm,dtype=np.float32)/voxel_size_mm
    mcx_cfg={
        "nphoton": int(cfg.photon_count), "vol": np.asarray(volume,dtype=np.uint8),
        "srcpos": source_vox, "srcdir": d.astype(np.float32),
        "prop": np.asarray(props,dtype=np.float32), "tstart":cfg.tstart,
        "tstep":cfg.tstep, "tend":cfg.tend, "issrcfrom0":cfg.issrcfrom0,
        "seed": int(seed % 2_147_483_647),
    }
    t=time.perf_counter()
    try: result=pmcx.run(mcx_cfg)
    except Exception as exc: raise OpticalSimulationError(f"PMCX failed: {exc}") from exc
    field=result.get("flux") if isinstance(result,dict) else None
    if field is None and isinstance(result,dict): field=result.get("data")
    if field is None: raise OpticalSimulationError(f"PMCX returned no flux/data field; keys={list(result) if isinstance(result,dict) else type(result)}")
    field=np.asarray(field)
    if field.ndim==4: field=field.sum(axis=3)
    if field.shape != volume.shape:
        raise OpticalSimulationError(f"Unexpected PMCX field shape {field.shape}, expected {volume.shape}")
    vals=extract_detector_values(field,detectors_mm,voxel_size_mm,cfg.detector_mode,cfg.detector_radius_mm)
    return vals,time.perf_counter()-t

def _run_analytic_debug(volume, source_mm, detectors_mm, props, voxel_size_mm, cfg, seed):
    """Fast deterministic smoke-test backend. NOT MCX and NOT valid training data."""
    _validate_source(volume,source_mm,voxel_size_mm,cfg.srcdir)
    s=np.asarray(source_mm,float); ds=np.asarray(detectors_mm,float)
    dist=np.linalg.norm(ds-s[None,:],axis=1)
    # Path-weighted label absorption, sampled along straight paths only.
    vals=[]
    for det,d in zip(ds,dist):
        n=max(10,int(np.ceil(d/voxel_size_mm))*2)
        pts=np.linspace(s,det,n)
        idx=np.rint(pts/voxel_size_mm).astype(int)
        idx=np.clip(idx,[0,0,0],np.array(volume.shape)-1)
        labels=volume[idx[:,0],idx[:,1],idx[:,2]]
        mua=float(np.mean(props[labels,0])); mus=float(np.mean(props[labels,1]))
        vals.append(np.exp(-(mua+0.01*mus)*max(d,1.0))/(max(d,1.0)**2))
    return np.asarray(vals,float),0.0

def run_source(volume, source_mm, detectors_mm, props, voxel_size_mm, cfg:PMCXConfig, seed:int):
    if cfg.backend=="pmcx": return _run_pmcx(volume,source_mm,detectors_mm,props,voxel_size_mm,cfg,seed)
    if cfg.backend=="analytic_debug": return _run_analytic_debug(volume,source_mm,detectors_mm,props,voxel_size_mm,cfg,seed)
    raise ValueError(f"Unknown backend {cfg.backend!r}")

def run_wavelength(volume, sources_mm, detectors_mm, props, voxel_size_mm, cfg, base_seed,
                   progress_callback=None):
    if len(sources_mm)!=8 or len(detectors_mm)!=16: raise AssertionError("Hardware requires 8 sources and 16 detectors")
    matrix=np.zeros((8,16),dtype=np.float64); times=[]
    for i,src in enumerate(sources_mm):
        vals,dt=run_source(volume,src,detectors_mm,props,voxel_size_mm,cfg,base_seed+i)
        matrix[i]=vals; times.append(dt)
        if progress_callback: progress_callback(i,dt)
    validate_measurement_matrix(matrix)
    return matrix,np.asarray(times)

def validate_measurement_matrix(m):
    if m.shape!=(8,16): raise OpticalSimulationError(f"Expected (8,16), got {m.shape}")
    if not np.isfinite(m).all(): raise OpticalSimulationError("Measurement contains NaN/Inf")
    if (m<0).any(): raise OpticalSimulationError("Measurement contains negative values")
    if np.any(np.all(m==0,axis=1)): raise OpticalSimulationError("At least one complete source row is zero")

def matrix_stats(m):
    pos=m[m>0]
    return {"min":float(m.min()),"max":float(m.max()),"mean":float(m.mean()),"median":float(np.median(m)),
            "std":float(m.std()),"zero_count":int((m==0).sum()),"negative_count":int((m<0).sum()),
            "nan_count":int(np.isnan(m).sum()),"inf_count":int(np.isinf(m).sum()),
            "min_positive":float(pos.min()) if pos.size else float("nan")}
