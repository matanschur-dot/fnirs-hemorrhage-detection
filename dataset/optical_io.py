from __future__ import annotations
import os, tempfile
import numpy as np
from anatomy.subject_generator import SubjectSample
from configs.simulation_config import SIMULATION_VERSION

def save_optical_sample(sample: SubjectSample, i740: np.ndarray, i850: np.ndarray,
                        props740: np.ndarray, props850: np.ndarray, photon_count:int,
                        noise_mode:str, backend:str, out_dir:str, extra:dict|None=None) -> str:
    os.makedirs(out_dir,exist_ok=True)
    final=os.path.join(out_dir,f"{sample.sample_id}.npz")
    payload=dict(
        I740=np.asarray(i740,dtype=np.float32), I850=np.asarray(i850,dtype=np.float32),
        label=np.int8(sample.label), subject_id=sample.subject_id, pair_id=sample.subject_id,
        sample_id=sample.sample_id, tissue_volume=sample.tissue_volume.astype(np.uint8),
        brain_mask=sample.brain_mask.astype(np.uint8), hemorrhage_mask=sample.hemorrhage_mask.astype(np.uint8),
        hemorrhage_center_vox=sample.hemorrhage_center_vox, hemorrhage_center_mm=sample.hemorrhage_center_mm,
        hemorrhage_radii_vox=sample.hemorrhage_radii_vox, hemorrhage_radii_mm=sample.hemorrhage_radii_mm,
        hemorrhage_depth_from_surface_mm=np.float32(sample.hemorrhage_depth_from_surface_mm),
        hemorrhage_depth_from_brain_start_mm=np.float32(sample.hemorrhage_depth_from_brain_start_mm),
        hemorrhage_volume_voxels=np.int32(sample.hemorrhage_volume_voxels),
        hemorrhage_volume_mm3=np.float32(sample.hemorrhage_volume_mm3),
        regression_target_valid=np.int8(sample.regression_target_valid),
        source_positions_740_mm=sample.source_positions_740_mm.astype(np.float32),
        source_positions_850_mm=sample.source_positions_850_mm.astype(np.float32),
        detector_positions_mm=sample.detector_positions_mm.astype(np.float32),
        optical_properties_740=np.asarray(props740,dtype=np.float32), optical_properties_850=np.asarray(props850,dtype=np.float32),
        photon_count=np.int64(photon_count), noise_mode=noise_mode, simulation_backend=backend,
        voxel_size_mm=np.float32(sample.voxel_size_mm), volume_shape=np.asarray(sample.volume_shape,dtype=np.int32),
        subject_seed=np.int64(sample.subject_seed), anatomy_seed=np.int64(sample.anatomy_seed),
        hemorrhage_seed=np.int64(sample.hemorrhage_seed), anatomy_version=sample.anatomy_version,
        geometry_version=sample.geometry_version, mapping_version=sample.mapping_version,
        simulation_version=SIMULATION_VERSION,
    )
    if extra:
        for k,v in extra.items():
            if isinstance(v,(dict,list,tuple)) and not isinstance(v,np.ndarray): continue
            payload[k]=v
    fd,tmp=tempfile.mkstemp(prefix=".tmp_",suffix=".npz",dir=out_dir); os.close(fd)
    try:
        np.savez_compressed(tmp,**payload); os.replace(tmp,final)
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return final

def is_valid_completed(path:str)->bool:
    if not os.path.isfile(path): return False
    try:
        with np.load(path,allow_pickle=False) as d:
            return d["I740"].shape==(8,16) and d["I850"].shape==(8,16) and np.isfinite(d["I740"]).all() and np.isfinite(d["I850"]).all()
    except Exception: return False
