"""Recover actual target-camera extrinsics without changing frozen RGB or labels."""
import gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015');DATA=Path('/root/autodl-tmp/jepa_vln_data_20261014')

def extrinsic(cam):
    y,p,h=cam['yaw'],cam.get('pitch',0.),cam.get('height',1.25)
    cy,sy,cp,sp=math.cos(y),math.sin(y),math.cos(p),math.sin(p)
    ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]);rx=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    matrix=np.eye(4);matrix[:3,:3]=ry@rx;matrix[:3,3]=np.array(cam['position'])+matrix[:3,:3]@np.array([0.,h,0.])
    return matrix

if __name__=='__main__':
    ds=DATA/'datasets/hssd_image_nav_v1';out=ds/'camera_extrinsics_v1';out.mkdir(exist_ok=True)
    stats=[];examples=[]
    for file in sorted((ds/'instances').glob('*.json')):
        scene=file.stem;rows={}
        for target in json.loads(file.read_text()):
            for view in target['views']:rows[view['image']]=view['camera']
        # Place cameras have zero pitch but are included to make the convention explicit.
        for source in (ds/'episodes').glob('*/'+scene+'.place.jsonl.gz'):
            with gzip.open(source,'rt') as f:
                for line in f:
                    e=json.loads(line);rows[e['goal_image']]=e['goal_camera']
        records=[];deviations=[];heights=[]
        for image,cam in sorted(rows.items()):
            m=extrinsic(cam);nominal=np.array(cam['position'])+np.array([0,cam['height'],0])
            delta=float(np.linalg.norm(m[:3,3]-nominal));deviations.append(delta);heights.append(float(m[1,3]-cam['position'][1]))
            cv=m.copy();cv[:3,:3]=m[:3,:3]@np.diag([1.,-1.,-1.])
            records.append({'goal_image':image,'world_from_sensor_gl':m.tolist(),'world_from_optical_cv':cv.tolist()})
        payload={'scene_id':scene,'convention':'column vectors; Habitat camera x-right y-up -z-forward; optical CV x-right y-down z-forward',
                 'source_position_semantics':'agent origin, not optical center','height_semantics':'local sensor offset before agent yaw/pitch rotation',
                 'does_not_change_frozen_images_or_success_regions':True,'records':records}
        dest=out/(scene+'.json');dest.write_text(json.dumps(payload,separators=(',',':')))
        stats.append({'scene':scene,'images':len(rows),'nonzero_pitch':sum(abs(c.get('pitch',0.))>1e-8 for c in rows.values()),
                      'max_optical_center_shift_from_vertical_offset_m':max(deviations),'minimum_camera_height_above_agent_origin_m':min(heights),
                      'sidecar_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
        for image,cam in sorted(rows.items(),key=lambda kv:abs(kv[1].get('pitch',0.)),reverse=True)[:1]:examples.append((abs(cam.get('pitch',0.)),scene,cam))
    sys.path.insert(0,str(DATA/'tools'));from nav_build import opensim,setpose
    verified=[]
    for _,scene,cam in sorted(examples,reverse=True)[:3]:
        sim=opensim(scene,DATA/'raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json')
        try:
            for pitch in [0.,cam['pitch']]:
                cc=dict(cam,pitch=pitch);setpose(sim,cc['position'],cc['yaw'],pitch);st=sim.get_agent(0).get_state().sensor_states['rgb']
                import quaternion
                rotation=quaternion.as_rotation_matrix(st.rotation);pred=extrinsic(cc)
                poserr=float(np.max(np.abs(np.asarray(st.position)-pred[:3,3])));roterr=float(np.max(np.abs(rotation-pred[:3,:3])))
                assert poserr<1e-4 and roterr<1e-5
                verified.append({'scene':scene,'pitch_deg':pitch*180/math.pi,'max_position_component_error_m':poserr,'max_rotation_component_error':roterr})
        finally:sim.close()
    result={'status':'passed','scenes':len(stats),'target_images':sum(s['images'] for s in stats),'nonzero_pitch_target_images':sum(s['nonzero_pitch'] for s in stats),
            'maximum_optical_center_shift_from_nominal_vertical_offset_m':max(s['max_optical_center_shift_from_vertical_offset_m'] for s in stats),
            'minimum_camera_height_above_agent_origin_m':min(s['minimum_camera_height_above_agent_origin_m'] for s in stats),
            'live_sensor_transform_checks':verified,'per_scene_sidecars':stats,'sidecar_directory':str(out),
            'source_goal_camera_position_is_agent_origin':True,'source_height_is_local_sensor_offset':True,
            'benchmark_images_panels_and_success_regions_unchanged':True,
            'scope':'Sidecars recover the actual optical center and orientation of existing images; they do not rerender goals or change navigation scores. Observed yaw-only camera stays at 1.25m, but pitched goal cameras have varying optical centers. Target extrinsics remain supervision/evaluator metadata, never default policy inputs.'}
    (BASE/'results/camera_extrinsics_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['per_scene_sidecars','live_sensor_transform_checks']}),flush=True)
