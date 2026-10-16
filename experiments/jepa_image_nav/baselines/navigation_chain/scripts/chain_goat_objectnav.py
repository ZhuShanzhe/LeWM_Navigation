"""Official multimodal GOAT category policy on all-instance HSSD labels.

The primary project score uses visible proximity to ANY recognized instance of
the requested class. Sparse verified-view SPL is a separate secondary metric;
it is not incorrectly presented as shortest-path SPL for the primary goal set.
"""
import argparse
from collections import Counter
import concurrent.futures as cf
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

import numpy as np
from instance_eval_goat import ROOT, DATA, PROFILES, opensim, sensor_values, write
sys.path.insert(0, str(DATA/'tools'))
from nav_build import setpose, pathlength

DATASET = ROOT/'datasets/hssd_objectnav_v3'
RUN = ROOT/'results/goat_category_hssd300_v3'

def scene_evaluate(args):
    scene_data = json.loads((DATASET/'scenes'/(args.scene+'.json')).read_text())
    episodes = scene_data['episodes'][:args.limit or None]
    assert all(e['status']=='complete' for e in episodes)
    out = RUN/args.scene
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'summary.json').exists(), 'Use a new run or skip a finished scene'
    sim = opensim(args.scene, PROFILES['native_camera_common_geometry'])
    log = (out/'policy_stderr.log').open('w')
    process = subprocess.Popen([str(ROOT/'goat_env/bin/python'),str(ROOT/'tools/instance_policy_goat.py')],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ)
    def rpc(request=None):
        if request is not None:
            process.stdin.write(json.dumps(request)+'\n')
            process.stdin.flush()
        if not selector.select(90):
            raise TimeoutError('GOAT policy response timeout')
        answer=json.loads(process.stdout.readline())
        assert 'error' not in answer,answer
        return answer
    records=[]
    try:
        policy=rpc()
        assert policy['ready'] and policy['checkpoint_keys']==349 and policy['strict_all_state_keys']
        write(out/'policy.json',policy)
        write(out/'configuration.json',{'task':'category_goal_nav',
            'profile':'native_camera_common_geometry','max_steps':500,
            'native_goal_input':'Original GOAT category-name CLIP cache, unnormalized 1024D',
            'primary_success':'action6 and any requested-category instance pixel fraction>=0.025 and horizontal AABB gap<=1m',
            'secondary_success':'action6 and geodesic to sparse verified category views<0.25m',
            'SPL':'Only secondary verified-view success is assigned verified-view SPL',
            'policy_inputs':['category','rgb','episodic_gps','episodic_compass'],
            'score_only':['semantic_id','all_category_instances','AABBs','navmesh','verified_views'],
            'scene_labels_sha256':hashlib.sha256((DATASET/'scenes'/(args.scene+'.json')).read_bytes()).hexdigest()})
        with tempfile.TemporaryDirectory(prefix='goat_cat_',dir='/dev/shm') as temporary:
            rgb_file=Path(temporary)/'rgb.npy'
            for e in episodes:
                dest=out/(e['episode_id']+'.json')
                if dest.exists():
                    records.append(json.loads(dest.read_text()))
                    continue
                goals=[g for g in scene_data['instances'] if g['category']==e['category']]
                views=[v['position'] for g in goals for v in g['valid_approach_views']]
                assert goals and views
                def view_distance(position):
                    values=[pathlength(sim,position,v) for v in views]
                    return min((d for d in values if d is not None and np.isfinite(d)),default=1e6)
                def visible_near(obs,position):
                    evidence=[]
                    p=np.asarray(position)[[0,2]]
                    for g in goals:
                        fraction=float((obs['semantic']==g['semantic_id']).mean())
                        if fraction<.025:
                            continue
                        lo,hi=np.asarray(g['aabb_min'])[[0,2]],np.asarray(g['aabb_max'])[[0,2]]
                        gap=float(np.linalg.norm(np.maximum(np.maximum(lo-p,p-hi),0)))
                        evidence.append({'semantic_id':g['semantic_id'],'fraction':fraction,'horizontal_aabb_gap_m':gap})
                    return any(x['horizontal_aabb_gap_m']<=1. for x in evidence),evidence
                rpc({'op':'reset','category':e['category']})
                obs=setpose(sim,e['start_position'],e['start_yaw'])
                initial=sim.get_agent(0).get_state()
                origin,rotation=np.array(initial.position),initial.rotation
                assert sim.pathfinder.is_navigable(origin,max_y_delta=.05)
                d0=view_distance(origin)
                assert 1.49<=d0<=30.01
                length=0.
                near_ever=False
                visible_ever=False
                actions=[]
                latencies=[]
                collisions=0
                with gzip.open(out/(e['episode_id']+'.steps.jsonl.gz'),'wt') as trace:
                    for t in range(500):
                        state=sim.get_agent(0).get_state()
                        before=np.asarray(state.position).copy()
                        gps,compass=sensor_values(before,state.rotation,origin,rotation)
                        near,evidence=visible_near(obs,before)
                        near_ever|=near
                        visible_ever|=bool(evidence)
                        np.save(rgb_file,obs['rgb'][:,:,:3])
                        decision=rpc({'op':'act','rgb':str(rgb_file),'gps':gps,'compass':compass})
                        action=int(decision['action'])
                        assert 0<=action<=6
                        actions.append(action)
                        latencies.append(decision['latency_ms'])
                        if action not in {0,6}:
                            obs=sim.step({1:'move_forward',2:'turn_left',3:'turn_right',4:'look_up',5:'look_down'}[action])
                        after=np.asarray(sim.get_agent(0).get_state().position).copy()
                        length+=float(np.linalg.norm(after-before))
                        collision=bool(obs.get('collided',False)) if action==1 else False
                        collisions+=int(collision)
                        trace.write(json.dumps({'t':t,'action':action,'before':before.tolist(),'after':after.tolist(),
                            'gps':gps,'compass':compass,'logits':decision['logits'],'collision':collision,
                            'visible_category_evidence':evidence})+'\n')
                        if action in {0,6}:
                            break
                near,evidence=visible_near(obs,sim.get_agent(0).get_state().position)
                final_d=view_distance(sim.get_agent(0).get_state().position)
                success=int(actions[-1]==6 and near)
                secondary=int(actions[-1]==6 and final_d<.25)
                record={'episode_id':e['episode_id'],'scene':args.scene,'category':e['category'],
                    'status':'complete','success':success,'steps':len(actions),'final_action':actions[-1],
                    'path_length_m':length,'initial_verified_view_geodesic_m':d0,'final_verified_view_geodesic_m':final_d,
                    'verified_view_success':secondary,'verified_view_spl':secondary*d0/max(d0,length,1e-8),
                    'category_visible_ever':bool(visible_ever or evidence),'near_correct_class_ever':bool(near_ever or near),
                    'final_category_evidence':evidence,'collisions':collisions,
                    'action_counts':dict(Counter(actions)),'inference_ms_mean':float(np.mean(latencies)),
                    'outcome':'success' if success else 'global_abort' if actions[-1]==0 else
                        'premature_subtask_stop' if actions[-1]==6 else 'timeout'}
                records.append(record)
                write(dest,record)
                print(json.dumps({'episode':e['episode_id'],'success':success,'steps':len(actions)}),flush=True)
        write(out/'summary.json',{'scene':args.scene,'episodes':len(records),'status':'complete',
            'successes':sum(r['success'] for r in records),'SR':float(np.mean([r['success'] for r in records])),
            'outcomes':dict(Counter(r['outcome'] for r in records))})
    finally:
        process.terminate()
        process.wait(timeout=20)
        selector.close()
        log.close()
        sim.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene')
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if args.scene:
        scene_evaluate(args)
    else:
        scenes=sorted(p.stem for p in (DATASET/'scenes').glob('*.json'))
        def child(scene):
            if (RUN/scene/'summary.json').exists():
                return {'scene':scene,'exit_code':0,'already_complete':True}
            with (ROOT/'logs'/('goat_category_'+scene+'.log')).open('w') as log:
                env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                p=subprocess.run([sys.executable,__file__,'--scene',scene,'--limit',str(args.limit)],
                    stdout=log,stderr=subprocess.STDOUT,env=env)
            return {'scene':scene,'exit_code':p.returncode}
        with (ROOT/'gpu_benchmark.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                records=list(pool.map(child,scenes))
        write(RUN/'execution.json',records)
        raise SystemExit(int(any(x['exit_code'] for x in records)))
