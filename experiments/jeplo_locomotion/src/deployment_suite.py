"""Pre-registered frozen-policy deployment-readiness simulation, not hardware validation."""
import argparse, collections, hashlib, json, math, platform, subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
import deployment_eval_v2 as ev

def audit(out):
    report={'scope':'CPU ONNX + synchronous MuJoCo; no DDS/ROS/hardware test',
            'commit':subprocess.check_output(['git','-C',str(ev.REPO),'rev-parse','HEAD'],text=True).strip(),
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'onnxruntime':ort.__version__,'mujoco':ev.mj.__version__},
            'models':{},'interface_checks':{}}
    for name in ['policy','sensor_estimator']:
        p=ev.MODEL_DIR/(name+'.onnx');opts=ort.SessionOptions();opts.intra_op_num_threads=1
        sess=ort.InferenceSession(str(p),opts,providers=['CPUExecutionProvider'])
        errors=[]
        for f in sorted((ev.MODEL_DIR/'test_cases'/name).glob('*.npz')):
            with np.load(f) as a:
                feeds={s.name:a['input__'+s.name] if 'input__'+s.name in a else a[s.name] for s in sess.get_inputs()}
                for spec,pred in zip(sess.get_outputs(),sess.run(None,feeds)):
                    ref=a['output__'+spec.name] if 'output__'+spec.name in a else a[spec.name];delta=np.abs(pred-ref)
                    errors.append({'sample':f.name,'output':spec.name,'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'strict_allclose':bool(np.allclose(pred,ref,rtol=1e-4,atol=1e-5))})
        assert errors,'Missing official fixtures'
        tol=.1 if name=='policy' else .5
        report['models'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'inputs':{x.name:x.shape for x in sess.get_inputs()},'outputs':{x.name:x.shape for x in sess.get_outputs()},'errors':errors,'official_deployment_tolerance_pass':all(e['max_abs']<=tol and e['mean_abs']<=.1 for e in errors),'strict_allclose_pass':all(e['strict_allclose'] for e in errors),'max_abs':max(e['max_abs'] for e in errors),'worst_mean_abs':max(e['mean_abs'] for e in errors)}
        twin=ev.REPO/'training/logs/rsl_rl/go2_loco/pretrained/exported'/p.name
        report['models'][name]['training_export_identical']=p.read_bytes()==twin.read_bytes()
    motor_names=[leg+'_'+joint+'_joint' for leg in ['FR','FL','RR','RL'] for joint in ['hip','thigh','calf']]
    mapping=[motor_names.index(n) for n in ev.NAMES]
    assert mapping==[3,0,9,6,4,1,10,7,5,2,11,8]
    sent=np.arange(12);assert np.array_equal(sent,np.array([sent[mapping.index(i)] for i in range(12)])[mapping])
    report['interface_checks']['motor_permutation_roundtrip']=True
    report['interface_checks']['joint2motor_idx']=mapping
    pol=ev.Policy();obs=np.arange(45,dtype=np.float32)/100
    class Stub:
        def __init__(self,kind):self.kind=kind;self.feeds=[]
        def run(self,names,feeds):
            self.feeds.append({k:v.copy() for k,v in feeds.items()})
            return [np.ones((1,12),np.float32)] if self.kind=='policy' else [np.zeros((1,512),np.float32)]*2
    pol.pol=Stub('policy');pol.est=Stub('est')
    target=pol.step(obs,np.ones((25,60),np.float32),0)
    assert np.allclose(target-ev.Q0,[.1]*4+[.2]*8)
    pol.step(obs+1,np.zeros((25,60),np.float32),1)
    assert np.array_equal(pol.hist[0,:45],obs+1) and np.array_equal(pol.hist[0,45:90],obs)
    for k in range(2,11):pol.step(obs,np.zeros((25,60),np.float32),k)
    assert len(pol.est.feeds)==3 and len(pol.pol.feeds)==11
    report['interface_checks'].update(history_newest_first=True,default_smoothing_and_hip_scale=True,policy_50Hz_estimator_10Hz=True,depth_shape=[1,2,25,60],depth_units='range/2 metres, missing=1',command_frame='body x forward,y left,yaw CCW',joint_position_units='radians',imu_quaternion_order='wxyz',physics_hz=500,lidar_hz=20,lidar_accumulation=5,depth_publish_hz=10,lidar_downsample=1)
    report['limitations']=['Official default action processing reimplemented from C++; no full C++/DDS transport execution.','Synchronous clocks and deterministic initial standing replace asynchronous hardware timing and stand-up sequence.','Official deployment tolerance passing is not bitwise parity; strict tolerance is reported independently.','Threshold safety stop terminates trial; subsequent hardware damping trajectory is not simulated.','CPU timings exclude network, sensor preprocessing, Jetson scheduling and motor transport.']
    (out/'interface_audit.json').write_text(json.dumps(report,indent=2))
    assert all(v['official_deployment_tolerance_pass'] for v in report['models'].values())
    return report

def suite(n):
    cases=[]
    for mode in ['deployment_default','raw_actions']:
        for name,cmd in [('stand',[0,0,0]),('forward025',[.25,0,0]),('forward075',[.75,0,0]),('forward125',[1.25,0,0]),('backward',[-.5,0,0]),('left',[0,.5,0]),('right',[0,-.5,0]),('rotate_left',[0,0,1]),('rotate_right',[0,0,-1]),('curve',[.5,0,.5]),('start_stop',[.75,0,0])]:
            for i in range(n):
                r=np.random.default_rng(9100+i)
                c=dict(terrain='flat',height=0.,condition='clean',seed=9100+i,x=float(r.uniform(-.1,.1)),y=float(r.uniform(-.1,.1)),yaw=float(r.uniform(-.15,.15)),speed=0.,scan_phase=int(r.integers(1000)),duration=10.,task='command_tracking',group=name,action_mode=mode,split='deployment_diagnostic')
                c['command_segments']=[dict(start=1.,end=6. if name=='start_stop' else 10.,cmd=cmd)]
                if name=='start_stop':c['stop_at']=6.
                cases.append(c)
        for terrain,height in [('box',.15),('box',.3),('box',.4),('box',.5),('stairs',.1),('stairs',.15),('stairs',.2)]:
            for i in range(n):
                r=np.random.default_rng(9200+i)
                cases.append(dict(terrain=terrain,height=height,condition='clean',seed=9200+i,x=float(r.uniform(-.1,.1)),y=float(r.uniform(-.1,.1)),yaw=float(r.uniform(-.15,.15)),speed=.75,scan_phase=int(r.integers(1000)),duration=18.,group=f'{terrain}_{height}',action_mode=mode,split='deployment_diagnostic'))
    for condition in ['clean','sparse1','mask0.4','noise5cm','delay100','delay300','blackout1s','push_lateral','friction_low','mass_plus20']:
        for i in range(n):
            r=np.random.default_rng(9300+i)
            cases.append(dict(terrain='stairs',height=.15,condition=condition,seed=9300+i,x=float(r.uniform(-.1,.1)),y=float(r.uniform(-.1,.1)),yaw=float(r.uniform(-.15,.15)),speed=.75,scan_phase=int(r.integers(1000)),duration=18.,group='robustness_'+condition,action_mode='deployment_default',split='deployment_diagnostic'))
    for i,c in enumerate(cases):c['episode_id']=f'v2_{i:04d}'
    return cases

def summarize(out,rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['action_mode'],r['group'])].append(r)
    summary=[]
    for (mode,group),rs in groups.items():
        k=sum(r['success'] for r in rs);n=len(rs);p=k/n;z=1.96;den=1+z*z/n
        centre=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        item=dict(mode=mode,group=group,n=n,completed=k,rate=p,wilson95=[centre-half,centre+half],reasons=dict(collections.Counter(r['reason'] for r in rs)))
        for key in ['tracking_xy_rmse','tracking_yaw_rmse','policy_ms_p95','estimator_ms_p95','max_tau_proxy_Nm','stop_final_speed_m_s','stop_displacement_m','scan_coverage']:
            vals=[r[key] for r in rs if r[key] is not None];item[key]=float(np.mean(vals)) if vals else None
        summary.append(item)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--trials',type=int,default=5);ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=audit(out)
    print(json.dumps({'audit':'official_tolerance_pass','strict_parity':{k:v['strict_allclose_pass'] for k,v in report['models'].items()}}),flush=True)
    if args.audit_only:return
    cases=suite(args.trials);(out/'manifest.json').write_text(json.dumps(cases,indent=2))
    for source in [Path(__file__),Path(ev.__file__)]:
        (out/source.name).write_bytes(source.read_bytes())
    policy=ev.Policy();rows=[]
    for i,c in enumerate(cases):
        r,trace=ev.run(c,policy,True)
        rows.append(r)
        with (out/'results.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        (out/(c['episode_id']+'.trace.json')).write_text(json.dumps(trace))
        summarize(out,rows)
        print(json.dumps({'i':i,'total':len(cases),'group':c['group'],'mode':c['action_mode'],'reason':r['reason'],'xy_rmse':r['tracking_xy_rmse'],'wall_s':r['wall_seconds']}),flush=True)
    (out/'complete').write_text('complete\n')

if __name__=='__main__':main()
