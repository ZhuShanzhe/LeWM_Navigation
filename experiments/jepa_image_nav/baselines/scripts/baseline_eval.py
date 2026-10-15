"""Closed-loop image-goal evaluation. Ground truth stays inside the evaluator.

RGB policy receives goal RGB, observed RGB and its own action history ONLY.
No pathfinder, target identity, success flag, pose or depth enters that process.
"""
import argparse, concurrent.futures as cf, gzip, hashlib, json, math, os, subprocess, sys, tempfile, time, traceback
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
DATA=Path('/root/autodl-tmp/jepa_vln_data_20261014')
sys.path.insert(0,str(DATA/'tools'))

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def readpanel(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def metrics(stopped,d,d0,length,collisions,moves,visible=None):
    success=bool(stopped and d<(1. if visible is None else .5) and (visible is None or visible))
    efficiency=d0/max(d0,length,1e-8)
    return {'success':int(success),'spl':int(success)*efficiency,'softspl':max(0.,1.-d/max(d0,1e-8))*efficiency,
            'collision_rate':collisions/max(moves,1),'final_distance_m':d,'executed_length_m':length}

def evaluate_scene(args):
    # A per-scene OS lock permits safe cooperative workers and cache rechecks.
    # Scoring and policy computation are unchanged.
    import fcntl
    folder=BASE/'results'/args.run/args.scene;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _evaluate_scene(args)

def _evaluate_scene(args):
    from nav_build import opensim,setpose,pathlength
    from habitat_sim.utils.common import quat_to_coeffs
    from PIL import Image
    eps=[e for e in readpanel(args.panel) if e['scene_id']==args.scene]
    if args.limit:eps=eps[:args.limit]
    ds=DATA/'datasets'/eps[0]['dataset'];hssd=eps[0]['dataset'].startswith('hssd')
    scene=args.scene if hssd else str(DATA/'raw/scene_datasets/gibson'/f'{args.scene}.glb')
    config=DATA/'raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json' if hssd else None
    output=BASE/'results'/args.run/args.scene
    output.mkdir(parents=True,exist_ok=True)
    complete=output/'summary.json'
    panel_sha=hashlib.sha256(Path(args.panel).read_bytes()).hexdigest()
    if complete.exists():
        old=json.loads(complete.read_text())
        assert old['panel_sha256']==panel_sha and old['max_steps']==args.max_steps and old.get('turn_deg',30)==args.turn_deg
        return old
    sim=opensim(scene,config);objects={};targetmeta={}
    for action_name in ['turn_left','turn_right']:
        sim.get_agent(0).agent_config.action_space[action_name].actuation.amount=args.turn_deg
    if hssd:
        objs=sim.get_rigid_object_manager().get_objects_by_handle_substring('')
        for i,(handle,obj) in enumerate(sorted(objs.items()),1):obj.semantic_id=i
        targetmeta={i['instance_id']:i for i in json.loads((ds/'instances'/f'{args.scene}.json').read_text())}
    policy_log=(output/'policy_stderr.log').open('w')
    policy_command=([str(BASE/'policy_env/bin/python'),str(BASE/'tools/baseline_modular.py')] if args.model=='lg_frontier'
                    else [str(BASE/'policy_env/bin/python'),str(BASE/'tools/baseline_policy.py'),'--model',args.model])
    proc=subprocess.Popen(policy_command,
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=policy_log,text=True,bufsize=1)
    def rpc(req):
        proc.stdin.write(json.dumps(req)+'\n');proc.stdin.flush()
        line=proc.stdout.readline()
        if not line:raise RuntimeError('Policy process exited; see policy_stderr.log')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    ready=json.loads(proc.stdout.readline());assert ready['ready']
    records=[];started=time.time()
    try:
        with tempfile.TemporaryDirectory(prefix='image_nav_',dir='/dev/shm') as temp:
            temp=Path(temp);gp=temp/'goal.npy';rp=temp/'rgb.npy';dp=temp/'depth.npy'
            with (output/'episodes.jsonl').open('w') as results, gzip.open(output/'steps.jsonl.gz','wt') as trace:
                for e in eps:
                    np.save(gp,np.asarray(Image.open(ds/e['goal_image']).convert('RGB')))
                    rpc({'op':'reset','goal':str(gp),'task':e['task']})
                    obs=setpose(sim,e['start_position'],e['start_yaw'])
                    instance=e['task']=='instance_image_nav';target=targetmeta.get(e.get('instance_id'))
                    goal=e.get('goal_position');views=e.get('valid_approach_views',[])
                    def distance(p):
                        vals=([pathlength(sim,p,v['position']) for v in views] if instance else [pathlength(sim,p,goal)])
                        return min((v for v in vals if v is not None and np.isfinite(v)),default=1e6)
                    d0=distance(e['start_position']);dmin=d0;length=0.;collisions=0;moves=0;stopped=False;lat=[]
                    near_ever=False;visible_ever=False;first_near=None;positions=[];actions=[];target_fraction=0.
                    for step in range(args.max_steps):
                        st=sim.get_agent(0).get_state();before=np.array(st.position)
                        d=distance(before);dmin=min(dmin,d)
                        target_fraction=float((obs['semantic']==target['semantic_id']).mean()) if instance else None
                        seen=bool(instance and target_fraction>=.025);visible_ever |= seen
                        near=(d<(.5 if instance else 1.))
                        near_ever |= near and (seen or not instance)
                        if near and (seen or not instance) and first_near is None:first_near=step
                        np.save(rp,obs['rgb'][:,:,:3]);request={'op':'act','rgb':str(rp)}
                        if args.model=='lg_frontier':
                            import quaternion
                            direction=quaternion.rotate_vectors(st.rotation,np.array([0.,0.,-1.]))
                            yaw=math.atan2(-direction[0],-direction[2]);dx,dz=(before-np.array(e['start_position']))[[0,2]]
                            sy=e['start_yaw'];relative=[math.cos(sy)*dx-math.sin(sy)*dz,math.sin(sy)*dx+math.cos(sy)*dz,
                                                      (yaw-sy+math.pi)%(2*math.pi)-math.pi]
                            np.save(dp,obs['depth']);request.update(depth=str(dp),odometry=relative)
                        decision=rpc(request)
                        action=int(decision['action']);assert 0<=action<4
                        lat.append(decision['latency_ms']);actions.append(action);positions.append(before.tolist())
                        trace.write(json.dumps({'episode_id':e['episode_id'],'t':step,'action':action,'position':before.tolist(),
                                                'rotation_xyzw':quat_to_coeffs(st.rotation).tolist(),'distance_m':d,
                                                'target_fraction':target_fraction,'logits':decision.get('logits'),
                                                'diagnostics':decision.get('diagnostics')})+'\n')
                        if action==0:stopped=True;break
                        cmd={1:'move_forward',2:'turn_left',3:'turn_right'}[action]
                        obs=sim.step(cmd);after=np.array(sim.get_agent(0).get_state().position)
                        length+=float(np.linalg.norm(after-before))
                        if action==1:moves+=1;collisions+=int(bool(obs['collided']))
                    final=np.array(sim.get_agent(0).get_state().position);d=distance(final);dmin=min(dmin,d)
                    target_fraction=float((obs['semantic']==target['semantic_id']).mean()) if instance else None
                    visible=bool(target_fraction>=.025) if instance else None
                    near_ever |= bool(d<(.5 if instance else 1.) and (visible or not instance))
                    wrong=False;wrong_id=None
                    if instance and stopped:
                        for other in targetmeta.values():
                            if other['category']!=target['category'] or other['instance_id']==target['instance_id']:continue
                            if float((obs['semantic']==other['semantic_id']).mean())<.025:continue
                            if min((pathlength(sim,final,v['position']) or 1e6 for v in other['valid_approach_views']),default=1e6)<.5:
                                wrong=True;wrong_id=other['instance_id'];break
                    m=metrics(stopped,d,d0,length,collisions,moves,visible)
                    if m['success']:reason='success'
                    elif wrong:reason='wrong_instance_stop'
                    elif near_ever:reason='reached_success_region_but_failed_to_stop'
                    elif stopped:reason='premature_stop'
                    elif collisions/max(moves,1)>.5:reason='collision_dominated_timeout'
                    else:reason='search_or_localization_timeout'
                    # These are observable outcome tags, not causal claims about internals.
                    r={'episode_id':e['episode_id'],'scene_id':args.scene,'dataset':e['dataset'],'task':e['task'],
                       'distance_bin':e['distance_bin'],'detour_bin':e['detour_bin'],'category':e.get('category'),
                       'start_goal_yaw_gap_deg':e.get('start_goal_yaw_gap_deg'),
                       'same_category_distractors':len(e.get('same_category_distractors',[])),
                       'steps':len(actions),'stopped':stopped,'timeout':not stopped,'initial_distance_m':d0,
                       'minimum_distance_m':dmin,'reached_success_region':near_ever,'first_near_step':first_near,
                       'target_visible_ever':visible_ever if instance else None,'target_visible_at_stop':visible,
                       'target_fraction_at_end':target_fraction,'wrong_instance_stop':wrong,'wrong_instance_id':wrong_id,
                       'outcome_tag':reason,'inference_ms_mean':float(np.mean(lat)),
                       'inference_ms_p95':float(np.percentile(lat,95)),'action_counts':dict(Counter(actions)),**m}
                    if not instance:
                        q=sim.get_agent(0).get_state().rotation
                        direction=__import__('quaternion').rotate_vectors(q,np.array([0.,0.,-1.]))
                        yaw=math.atan2(-direction[0],-direction[2])
                        gap=abs((yaw-e['goal_camera']['yaw']+math.pi)%(2*math.pi)-math.pi)*180/math.pi
                        r['final_yaw_error_deg']=gap;r['strict_pose_success']=int(stopped and d<.25 and gap<15.)
                    records.append(r);results.write(json.dumps(r)+'\n');results.flush()
                    print(json.dumps({'scene':args.scene,'completed':len(records),'sr':sum(x['success'] for x in records)/len(records)}),flush=True)
        summary={'model':args.model,'panel':Path(args.panel).name,'panel_sha256':panel_sha,'max_steps':args.max_steps,'turn_deg':args.turn_deg,
                 'scene':args.scene,'episodes':len(records),'successes':sum(x['success'] for x in records),
                 'SR':float(np.mean([r['success'] for r in records])),'SPL':float(np.mean([r['spl'] for r in records])),
                 'SoftSPL':float(np.mean([r['softspl'] for r in records])),'outcomes':dict(Counter(r['outcome_tag'] for r in records)),
                 'wall_seconds':time.time()-started,'policy':ready,'status':'complete'}
        dump(complete,summary);return summary
    finally:
        if proc.poll() is None:
            try:rpc({'op':'close'})
            except Exception:pass
            proc.terminate();proc.wait(timeout=15)
        policy_log.close();sim.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--model',default='fgprompt')
    p.add_argument('--run',required=True);p.add_argument('--scene');p.add_argument('--workers',type=int,default=3)
    p.add_argument('--max-steps',type=int,default=500);p.add_argument('--turn-deg',type=float,default=30.)
    p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    if args.scene:return evaluate_scene(args)
    scenes=sorted({e['scene_id'] for e in readpanel(args.panel)})
    def child(scene):
        log=BASE/'logs'/f'{args.run}_{scene}.log'
        env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,__file__,'--panel',args.panel,'--model',args.model,'--run',args.run,'--scene',scene,
             '--max-steps',str(args.max_steps),'--turn-deg',str(args.turn_deg),'--limit',str(args.limit)]
        with log.open('w') as f:
            for attempt in range(3):
                result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=3600)
                if result.returncode==0:break
                f.write('\nRETRY_SCENE_PROCESS '+str(attempt+1)+'\n');f.flush();time.sleep(2)
        return {'scene':scene,'code':result.returncode}
    started=time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        statuses=list(pool.map(child,scenes))
    dump(BASE/'results'/args.run/'execution.json',{'statuses':statuses,'wall_seconds':time.time()-started,
         'all_scene_processes_succeeded':all(s['code']==0 for s in statuses)})
    print(json.dumps(statuses),flush=True)
    if any(s['code']!=0 for s in statuses):raise SystemExit(1)

if __name__=='__main__':main()
