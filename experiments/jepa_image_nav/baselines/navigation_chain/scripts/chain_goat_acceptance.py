"""Full execution/trace acceptance and scene-cluster confidence intervals."""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

from instance_eval_goat import ROOT, OLD, write

panel = OLD / 'plans/hssd_image_nav_v1_test_instance_20perscene.jsonl'
episodes = [json.loads(x) for x in panel.read_text().splitlines() if x.strip()]
expected = {e['episode_id']:e for e in episodes}
rng = np.random.default_rng(20261016)
results = []
all_records = {}
parser=argparse.ArgumentParser()
parser.add_argument('--run', action='append')
args=parser.parse_args()
required=['goat_shared_hssd300','goat_native_camera_common_geometry300']
for run in (args.run or required):
    records, checked_steps, replay_requests = [], 0, []
    base = ROOT / 'results' / run
    for scene in sorted({e['scene_id'] for e in episodes}):
        summary = json.loads((base/scene/'summary.json').read_text())
        assert summary['status']=='complete' and summary['episodes']==20
        assert summary['panel_sha256']==hashlib.sha256(panel.read_bytes()).hexdigest()
        policy = json.loads((base/scene/'policy.json').read_text())
        assert policy['strict_all_state_keys'] and policy['checkpoint_keys']==349
        for e in [x for x in episodes if x['scene_id']==scene]:
            record = json.loads((base/scene/(e['episode_id']+'.json')).read_text())
            assert record['episode_id']==e['episode_id']
            records.append(record)
            if record['status']!='complete':
                continue
            with gzip.open(base/scene/(e['episode_id']+'.steps.jsonl.gz'),'rt') as stream:
                steps = [json.loads(x) for x in stream]
            assert len(steps)==record['steps']
            assert record['steps']<=500
            assert all(s['t']==i and s['action']==int(np.argmax(s['logits'])) for i,s in enumerate(steps))
            assert all(0<=s['action']<=6 and len(s['logits'])==7 for s in steps)
            assert record['goal_stop']==(steps[-1]['action']==6)
            assert record['global_abort']==(steps[-1]['action']==0)
            if record['goal_stop'] or record['global_abort']:
                assert abs(record['final_distance_m']-steps[-1]['distance_m'])<1e-6
                assert abs(record['target_fraction_at_end']-steps[-1]['target_fraction'])<1e-9
            correct = int(record['goal_stop'] and record['final_distance_m']<.5 and record['target_fraction_at_end']>=.025)
            assert correct==record['success']
            d0, length = record['initial_distance_m'],record['path_length_m']
            assert abs(record['spl']-correct*d0/max(d0,length,1e-8))<1e-8
            points=np.asarray([s['position'] for s in steps])
            logged_length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
            if record['goal_stop'] or record['global_abort'] or steps[-1]['action']!=1:
                assert abs(length-logged_length)<2e-4
            else:
                assert -2e-4<=length-logged_length<=.251
            for i,s in enumerate(steps):
                dx,dz=(np.asarray(s['position'])-np.asarray(e['start_position']))[[0,2]]
                c, sn=math.cos(e['start_yaw']),math.sin(e['start_yaw'])
                gps=[-(sn*dx+c*dz),c*dx-sn*dz]
                assert np.allclose(gps,s['gps'],atol=3e-5)
                if i+1<len(steps):
                    displacement=float(np.linalg.norm(points[i+1]-points[i]))
                    if s['action']!=1:
                        assert displacement<1e-5
                    elif displacement>.251:
                        replay_requests.append({'episode_id':e['episode_id'],'scene':scene,
                            'profile':summary['profile_name'],'t':i,'before':points[i].tolist(),
                            'after':points[i+1].tolist(),'rotation_xyzw':s['rotation_xyzw']})
            checked_steps+=len(steps)
    if replay_requests:
        replay_input=base/'movement_replay_requests.json'
        replay_output=base/'movement_replay_result.json'
        write(replay_input,replay_requests)
        if not replay_output.exists():
            subprocess.run(['/root/autodl-tmp/jepa_vln_data_20261014/env/bin/python',
                str(ROOT/'tools/chain_motion_replay.py'),'--requests',str(replay_input),
                '--output',str(replay_output)],check=True)
        replay=json.loads(replay_output.read_text())
        assert replay['passed'] and replay['cases']==len(replay_requests)
    assert {r['episode_id'] for r in records}==set(expected)
    by_scene=defaultdict(list)
    for r in records:
        by_scene[r['scene_id']].append(r)
    means=np.array([[np.mean([x.get('success',0) for x in rs]),np.mean([x.get('spl',0.) for x in rs])]
                    for rs in by_scene.values()])
    sampled=means[rng.integers(0,len(means),(10000,len(means)))].mean(axis=1)
    def group(key):
        groups=defaultdict(list)
        for r in records:
            groups[r.get(key,expected[r['episode_id']].get(key,'unknown'))].append(r)
        return {k:{'episodes':len(rs),'successes':sum(x.get('success',0) for x in rs),
                   'SR':float(np.mean([x.get('success',0) for x in rs])),
                   'SPL':float(np.mean([x.get('spl',0.) for x in rs]))} for k,rs in groups.items()}
    report={'run':run,'model':'GOAT SenseAct-NN Monolithic official pretrained policy',
        'scope':'HSSD cross-domain, cold-start single image subtask; not official lifelong GOAT-Bench scores',
        'episodes':len(records),'scenes':len(by_scene),'checked_trace_steps':checked_steps,
        'completed':sum(r['status']=='complete' for r in records),
        'invalid_under_profile':sum(r['status']!='complete' for r in records),
        'successes':sum(r.get('success',0) for r in records),
        'SR':float(np.mean([r.get('success',0) for r in records])),
        'SPL':float(np.mean([r.get('spl',0.) for r in records])),
        'SR_scene_cluster_bootstrap95':np.quantile(sampled[:,0],[.025,.975]).tolist() if any(r.get('success',0) for r in records) else None,
        'SPL_scene_cluster_bootstrap95':np.quantile(sampled[:,1],[.025,.975]).tolist() if any(r.get('success',0) for r in records) else None,
        'uncertainty_note':'All-zero scene resampling is degenerate, so no misleading [0,0] generalization interval is assigned' if not any(r.get('success',0) for r in records) else 'Scene cluster bootstrap, 10000 resamples',
        'goal_visible_ever':sum(r.get('target_visible_ever',False) for r in records),
        'reached_success_region':sum(r.get('reached_success_region',False) for r in records),
        'outcomes':dict(Counter(r.get('outcome',r['status']) for r in records)),
        'by_category':group('category'),'by_distance':group('distance_bin'),
        'acceptance':'passed'}
    results.append(report)
    all_records[run]=records
    write(base/'accepted_summary.json',report)
write(ROOT/'results/goat_instance_comparison.json',{'status':'passed',
    'all_expected_runs':set(args.run or required)==set(required),'results':results})
print(json.dumps(results,indent=2),flush=True)
