"""Summarize recorded native probes without turning completion into success."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('runs',nargs='+');a=p.parse_args()
cases=[];trajectories=[]
for name in a.runs:
    folder=a.root/name;summary=json.loads((folder/'summary.json').read_text())
    row={'run':name,'driver':summary,'controller_cpp_fixture_pass':False}
    log=(folder/'controller.log').read_text(errors='replace')
    row['controller_cpp_fixture_pass']='passed ONNX test cases' in log
    row['ood_positive_log_lines']=len(re.findall(r'SAFETY ood=1\b',log))
    row['safety_fault_logged']='OOD safety fault:' in log
    row['sensor_periodic_statistics']=[line for line in (folder/'sensor.log').read_text(errors='replace').splitlines() if 'frames in ' in line]
    row['raw_files_sha256']={f:hashlib.file_digest((folder/f).open('rb'),'sha256').hexdigest() for f in ['summary.json','controller.log','simulator.log','sensor.log','lidar_pose.jsonl']}
    if summary.get('ready_seconds') is not None:
        rows=[json.loads(s) for s in (folder/'lidar_pose.jsonl').read_text().splitlines()]
        rows=[r for r in rows if r['wall_seconds']>summary['ready_seconds']+2]
        if len(rows)>10:
            t=np.array([r['wall_seconds'] for r in rows]);xyz=np.array([r['sensor_xyz'] for r in rows]);rot=np.array([r['sensor_rotation'] for r in rows]).reshape(-1,3,3)@np.diag([1,-1,-1])
            base=xyz-np.einsum('nij,j->ni',rot,np.array([.275,0,.149]));yaw=np.unwrap(np.arctan2(rot[:,1,0],rot[:,0,0]));duration=t[-1]-t[0]
            row['recomputed']={'elapsed_seconds':float(duration),'raw_lidar_hz':float((len(t)-1)/duration),'base_xy_displacement_m':float(np.linalg.norm(base[-1,:2]-base[0,:2])),'mean_displacement_speed_m_s':float(np.linalg.norm(base[-1,:2]-base[0,:2])/duration),'mean_yaw_rate_rad_s':float((yaw[-1]-yaw[0])/duration),'min_base_z_m':float(base[:,2].min()),'max_base_z_m':float(base[:,2].max()),'min_base_upright_cos':float(rot[:,2,2].min()),'final_base_xyz':base[-1].tolist()}
            trajectories.extend({'run':name,'wall_seconds':float(t[i]),'base_xyz':base[i].tolist(),'unwrapped_yaw_rad':float(yaw[i]),'upright_cos':float(rot[i,2,2])} for i in range(len(t)))
            if summary.get('pose_metrics'):
                assert abs(row['recomputed']['base_xy_displacement_m']-summary['pose_metrics']['xy_displacement_m'])<1e-9
    cases.append(row)
failures=[]
for path in sorted(a.root.glob('native*/summary.json')):
    failed=json.loads(path.read_text())
    if not failed.get('complete'):
        failures.append({'run':path.parent.name,'error':failed.get('error'),'ready_seconds':failed.get('ready_seconds'),'counted_as_locomotion_trial':failed.get('ready_seconds') is not None,'classification':'post-ready failure retained' if failed.get('ready_seconds') is not None else 'pre-ready setup failure'})
report={'scope':'native README-aligned deployment probes; self-defined constant commands; no paper success-rate claim','source_commit':'460e2272907af053d6e66b57fec71fdbe6ca1774','mujoco_candidate':'3.3.7','sdk_candidate_commit':'63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36','cage_mask':'enabled, upstream random_device realization per process','paper_protocol_confirmed':False,'paper_performance_reproduced':False,'cases':cases,'initial_infrastructure_failures':failures,'completion_is_not_task_success':True}
a.output.write_text(json.dumps(report,indent=2));a.output.with_name('official_native_trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trajectories));print(json.dumps({'cases':len(cases),'completed':sum(c['driver']['complete'] for c in cases),'paper_reproduced':False}))
