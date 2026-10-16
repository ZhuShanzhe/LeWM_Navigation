"""Recompute JEPLO motion measurements from sensor logs and native extrinsics."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
NATIVE=Path('/root/autodl-tmp/jeplo_eval_20260928/official_alignment')
OUT=ROOT/'results/jeplo_native_motion_validated'
manifest=json.loads((NATIVE/'native_build_manifest.json').read_text())
for relative,expected in manifest['artifacts'].items():
    assert hashlib.file_digest((NATIVE/relative).open('rb'),'sha256').hexdigest()==expected,relative
profile=json.loads((NATIVE/'official_readme_profile.json').read_text())
for name,expected in profile['models_sha256'].items():
    assert hashlib.file_digest((NATIVE/'source'/profile['model_subdir']/name).open('rb'),'sha256').hexdigest()==expected,name
site=ET.parse(NATIVE/'source/deployment/unitree_mujoco/unitree_robots/go2/go2.xml').find('.//site[@name="lidar"]')
offset=np.fromstring(site.get('pos'),sep=' ')
assert np.array_equal(np.fromstring(site.get('quat'),sep=' '),[0,1,0,0])
records=[]
for case in ['stand','forward','rotate','stairs','boxes']:
    for repeat in range(1,4):
        folder=OUT/f'{case}_{repeat}'
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['complete'] and summary['case']==case
        raw=[json.loads(x) for x in (folder/'lidar_pose.jsonl').read_text().splitlines()]
        raw=[r for r in raw if r['wall_seconds']>summary['ready_seconds']+2]
        t=np.asarray([r['wall_seconds'] for r in raw])
        rotations=np.asarray([r['sensor_rotation'] for r in raw]).reshape(-1,3,3)@np.diag([1,-1,-1])
        assert np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3)))<1e-4
        position=np.asarray([r['sensor_xyz'] for r in raw])-np.einsum('nij,j->ni',rotations,offset)
        assert len(t)==summary['pose_metrics']['samples'] and np.all(np.diff(t)>0)
        delta=position[-1]-position[0]
        duration=t[-1]-t[0]
        yaw=np.unwrap(np.arctan2(rotations[:,1,0],rotations[:,0,0]))
        yaw_rate=np.diff(yaw)/np.diff(t)
        velocity=np.diff(position,axis=0)/np.diff(t)[:,None]
        body_velocity=np.einsum('nji,nj->ni',rotations[:-1],velocity)
        forward_mean=float(delta[:2]@rotations[0,:2,0]/duration)
        omega_mean=float((yaw[-1]-yaw[0])/duration)
        assert abs(omega_mean-summary['pose_metrics']['mean_yaw_rate_rad_s'])<1e-8
        assert abs(np.linalg.norm(delta[:2])-summary['pose_metrics']['xy_displacement_m'])<1e-8
        upright=rotations[:,2,2]
        longest=0.
        start=None
        for stamp,bad in zip(t,upright<.5):
            if bad and start is None:
                start=stamp
            if bad:
                longest=max(longest,stamp-start)
            else:
                start=None
        command=np.asarray(summary['command'])
        record={'case':case,'repeat':repeat,'runtime_complete':True,'samples':len(t),
            'measurement_seconds':float(duration),'command':command.tolist(),
            'body_forward_mean_from_net_displacement_m_s':forward_mean,
            'mean_yaw_rate_rad_s':omega_mean,
            'body_forward_velocity_rmse_m_s':float(np.sqrt(np.mean((body_velocity[:,0]-command[0])**2))),
            'yaw_rate_rmse_rad_s':float(np.sqrt(np.mean((yaw_rate-command[2])**2))),
            'xy_net_displacement_m':float(np.linalg.norm(delta[:2])),
            'max_base_z_m':float(position[:,2].max()),'minimum_upright_cos':float(upright.min()),
            'longest_tilt_over_60deg_seconds':float(longest),
            'sustained_tilt_event':bool(longest>=1),
            'ascent_self_defined_gate_held':bool(summary.get('ascent_goal_held',False)) if case in {'stairs','boxes'} else None,
            'safety_stop_detected':summary['safety_stop_detected'],
            'pose_log_sha256':hashlib.sha256((folder/'lidar_pose.jsonl').read_bytes()).hexdigest()}
        records.append(record)
groups={}
for case in ['stand','forward','rotate','stairs','boxes']:
    values=[r for r in records if r['case']==case]
    metrics={}
    for key in ['body_forward_mean_from_net_displacement_m_s','mean_yaw_rate_rad_s',
                'body_forward_velocity_rmse_m_s','yaw_rate_rmse_rad_s','xy_net_displacement_m','max_base_z_m']:
        array=np.asarray([r[key] for r in values])
        metrics[key]={'mean':float(array.mean()),'std_population':float(array.std()),'runs':array.tolist()}
    groups[case]={'runs':3,'runtime_completed':sum(r['runtime_complete'] for r in values),
        'sustained_tilt_events':sum(r['sustained_tilt_event'] for r in values),
        'self_defined_ascent_gates_held':sum(r['ascent_self_defined_gate_held'] is True for r in values),
        'measurements':metrics}
result={'acceptance':'passed','runs':15,'source_and_model_hashes_verified':True,
    'scope':'Native JEPLO deployment simulation; frozen commands and project-defined ascent/fall criteria, not paper success-rate reproduction',
    'fall_diagnostic':'Base upright cosine<0.5 continuously for at least 1 second, derived from native LiDAR-site pose',
    'repeats':'Uncontrolled upstream cage random_device retained; not controlled training seeds',
    'groups':groups,'records':records}
(OUT/'accepted_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in {'records','groups'}},indent=2))
for case,value in groups.items():
    print(json.dumps({'case':case,'completed':value['runtime_completed'],'tilt_events':value['sustained_tilt_events'],
        'ascent_gates_held':value['self_defined_ascent_gates_held'],
        'forward_mean':value['measurements']['body_forward_mean_from_net_displacement_m_s']['mean'],
        'yaw_rate_mean':value['measurements']['mean_yaw_rate_rad_s']['mean']}))
