#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,logging,os,sys,time
from dataclasses import replace,asdict
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from geometry.kicad_parser import components_by_reference
from geometry.pcb_geometry import validate_hardware
from configs.geometry_config import GeometryRunConfig
from configs.anatomy_config import AnatomyRunConfig
from configs.simulation_config import SimulationRunConfig
from anatomy.sensor_geometry import build_sensor_geometry
from anatomy.subject_generator import build_paired_subject
from simulation.optical_properties import sample_optical_properties
from simulation.pmcx_runner import run_wavelength,matrix_stats,PMCXUnavailableError,OpticalSimulationError
from simulation.noise import apply_noise
from dataset.optical_io import save_optical_sample,is_valid_completed
from dataset.visualize_optical import plot_pair

OPTICAL_SEED_OFFSET=2_000_000; NOISE_SEED_OFFSET=3_000_000

def setup_log(out):
    os.makedirs(out,exist_ok=True); log=os.path.join(out,'generation.log')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[logging.FileHandler(log,encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)

def write_csv_atomic(path,rows):
    if not rows:return
    tmp=path+'.tmp'
    with open(tmp,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    os.replace(tmp,path)

def load_existing_rows(path):
    if not os.path.isfile(path):return []
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(description='Generate paired synthetic fNIRS optical data')
    ap.add_argument('--pcb',required=True); ap.add_argument('--output',default='output/optical_dataset')
    ap.add_argument('--num-subjects',type=int,default=5); ap.add_argument('--seed',type=int,default=12345)
    ap.add_argument('--photons',type=int,default=100000); ap.add_argument('--backend',choices=['pmcx','analytic_debug'],default='pmcx')
    ap.add_argument('--noise-mode',choices=['none','basic'],default='none'); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--visualize-first',type=int,default=5)
    args=ap.parse_args(); setup_log(args.output)
    if not os.path.isfile(args.pcb): raise FileNotFoundError(os.path.abspath(args.pcb))
    if args.backend=='analytic_debug': logging.warning('analytic_debug is NOT physical MCX data and MUST NOT be used for training.')
    comps=components_by_reference(args.pcb); validate_hardware(comps)
    gcfg=GeometryRunConfig(); acfg=AnatomyRunConfig(); scfg=SimulationRunConfig()
    scfg=replace(scfg,pmcx=replace(scfg.pmcx,photon_count=args.photons,backend=args.backend),noise=replace(scfg.noise,mode=args.noise_mode))
    sensors=build_sensor_geometry(comps,gcfg,acfg.placement)
    samples_dir=os.path.join(args.output,'samples'); viz_dir=os.path.join(args.output,'visualizations'); os.makedirs(samples_dir,exist_ok=True); os.makedirs(viz_dir,exist_ok=True)
    config_snapshot={'arguments':vars(args),'simulation':{'pmcx':asdict(scfg.pmcx),'variation':asdict(scfg.variation),'noise':asdict(scfg.noise)}}
    with open(os.path.join(args.output,'config_snapshot.json'),'w',encoding='utf-8') as f: json.dump(config_snapshot,f,indent=2)
    metadata_path=os.path.join(args.output,'metadata.csv'); rows=load_existing_rows(metadata_path) if args.resume else []
    existing_ids={r.get('sample_id') for r in rows}; total_runs=args.num_subjects*32; completed_runs=0; start=time.perf_counter()
    for idx in range(args.num_subjects):
        sid=f'subject_{idx:05d}'; hp=os.path.join(samples_dir,sid+'_healthy.npz'); xp=os.path.join(samples_dir,sid+'_hemorrhage.npz')
        if args.resume and not args.overwrite and is_valid_completed(hp) and is_valid_completed(xp):
            logging.info('Skipping completed pair %s',sid); completed_runs+=32; continue
        pair_start=time.perf_counter(); healthy,hem,pair_result,layers=build_paired_subject(sid,args.seed,gcfg.volume,acfg,sensors)
        props740,props850,prop_meta=sample_optical_properties(healthy.subject_seed+OPTICAL_SEED_OFFSET,scfg.variation)
        pair_data=[]
        for case_i,sample in enumerate((healthy,hem)):
            case='healthy' if sample.label==0 else 'hemorrhage'
            seed_base=healthy.subject_seed+OPTICAL_SEED_OFFSET
            if scfg.pmcx.paired_seed_mode=='independent': seed_base+=case_i*10000
            def cb_factory(wave):
                def cb(source_i,dt):
                    nonlocal completed_runs
                    completed_runs+=1; elapsed=time.perf_counter()-start; avg=elapsed/max(completed_runs,1); eta=avg*(total_runs-completed_runs)
                    logging.info('Subject %d/%d | %s | %dnm | source %d/8 | runs %d/%d | %.2fs | ETA %.1fmin',idx+1,args.num_subjects,case,wave,source_i+1,completed_runs,total_runs,dt,eta/60)
                return cb
            i740,t740=run_wavelength(sample.tissue_volume,sample.source_positions_740_mm,sample.detector_positions_mm,props740,sample.voxel_size_mm,scfg.pmcx,seed_base+740,cb_factory(740))
            i850,t850=run_wavelength(sample.tissue_volume,sample.source_positions_850_mm,sample.detector_positions_mm,props850,sample.voxel_size_mm,scfg.pmcx,seed_base+850,cb_factory(850))
            i740,i850,noise_meta=apply_noise(i740,i850,sample.subject_seed+NOISE_SEED_OFFSET+case_i,scfg.noise)
            pair_data.append((i740,i850,t740,t850,noise_meta))
        h740,h850,ht740,ht850,_=pair_data[0]; x740,x850,xt740,xt850,_=pair_data[1]
        delta740=x740-h740; delta850=x850-h850
        pair_stats={'pair_mean_abs_delta_740':float(np.mean(np.abs(delta740))),'pair_mean_abs_delta_850':float(np.mean(np.abs(delta850))),
                    'pair_max_abs_delta_740':float(np.max(np.abs(delta740))),'pair_max_abs_delta_850':float(np.max(np.abs(delta850)))}
        for sample,(i740,i850,t740,t850,noise_meta) in zip((healthy,hem),pair_data):
            out=save_optical_sample(sample,i740,i850,props740,props850,args.photons,args.noise_mode,args.backend,samples_dir)
            s740=matrix_stats(i740);s850=matrix_stats(i850)
            row={'sample_id':sample.sample_id,'subject_id':sid,'pair_id':sid,'file_name':os.path.basename(out),'label':sample.label,'status':'complete',
                 'scalp_thickness_mm':sample.scalp_thickness_mm,'skull_thickness_mm':sample.skull_thickness_mm,'csf_thickness_mm':sample.csf_thickness_mm,
                 'brain_start_z_mm':sample.brain_start_z_mm,'hemorrhage_center_x_mm':sample.hemorrhage_center_mm[0],'hemorrhage_center_y_mm':sample.hemorrhage_center_mm[1],
                 'hemorrhage_center_z_mm':sample.hemorrhage_center_mm[2],'hemorrhage_depth_from_surface_mm':sample.hemorrhage_depth_from_surface_mm,
                 'hemorrhage_rx_mm':sample.hemorrhage_radii_mm[0],'hemorrhage_ry_mm':sample.hemorrhage_radii_mm[1],'hemorrhage_rz_mm':sample.hemorrhage_radii_mm[2],
                 'hemorrhage_volume_mm3':sample.hemorrhage_volume_mm3,'photon_count':args.photons,'noise_mode':args.noise_mode,'backend':args.backend,
                 'subject_seed':sample.subject_seed,'I740_min':s740['min'],'I740_max':s740['max'],'I740_mean':s740['mean'],'I740_zero_count':s740['zero_count'],
                 'I850_min':s850['min'],'I850_max':s850['max'],'I850_mean':s850['mean'],'I850_zero_count':s850['zero_count'],
                 'runtime_seconds':float(t740.sum()+t850.sum()),**pair_stats,'error_message':''}
            if sample.sample_id not in existing_ids: rows.append(row); existing_ids.add(sample.sample_id)
        write_csv_atomic(metadata_path,rows)
        if idx<args.visualize_first: plot_pair(h740,h850,x740,x850,hem.hemorrhage_mask,os.path.join(viz_dir,sid+'_optical_pair.png'))
        logging.info('Completed pair %s in %.1fs; mean abs delta=(%.3g, %.3g)',sid,time.perf_counter()-pair_start,pair_stats['pair_mean_abs_delta_740'],pair_stats['pair_mean_abs_delta_850'])
    logging.info('Dataset generation complete: %s',args.output); return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except (PMCXUnavailableError,OpticalSimulationError) as e:
        logging.exception('Optical generation failed: %s',e); raise SystemExit(2)
