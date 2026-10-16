"""Validate official EB-Navigation task identity, actions and distance scores."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
OUT=ROOT/'results/eb_navigation_qwen25vl7b'
REPO=ROOT/'repos/EmbodiedBench'
SUBSETS=['base','common_sense','complex_instruction','visual_appearance','long_horizon']
parser=argparse.ArgumentParser()
parser.add_argument('--limit',type=int,default=60)
args=parser.parse_args()
assert 1<=args.limit<=60
assert json.loads((OUT/'simulator_gate.json').read_text())['passed']
assert json.loads((OUT/'transport_gate.json').read_text())['passed']
assert json.loads((ROOT/'results/qwen_integrity_gate.json').read_text())['safetensor_shards_publisher_verified']
records=[]
physical_groups=defaultdict(dict)
datasets=[]
for subset in SUBSETS:
    path=REPO/'embodiedbench/envs/eb_navigation/datasets'/(subset+'.json')
    tasks=json.loads(path.read_text())['tasks']
    assert len(tasks)==60
    datasets.append({'subset':subset,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tasks':len(tasks)})
    for index,task in enumerate(tasks[:args.limit]):
        value=json.loads((OUT/subset/f'{index:03d}.json').read_text())
        assert value['status']=='complete' and value['subset']==subset and value['index']==index
        assert value['instruction']==task['instruction']
        trace=value['trace']
        assert 1<=len(trace)<=20 and len(trace)==value['steps']
        length=0.
        for t,step in enumerate(trace,1):
            assert step['t']==t and 0<=step['executed_action']<8
            if isinstance(step['requested_action'],int) and 0<=step['requested_action']<8:
                assert step['requested_action']==step['executed_action']
            before,after=step['before'],step['after']
            a=step['executed_action']
            p0=np.array([before['position'][k] for k in ['x','z']])
            p1=np.array([after['position'][k] for k in ['x','z']])
            length+=float(np.linalg.norm(p1-p0))
            if a>=4:
                assert np.linalg.norm(p1-p0)<.003
            if a in {4,5} and step['action_success']:
                delta=(after['rotation']['y']-before['rotation']['y'])%360
                assert abs(delta-(90 if a==4 else 270))<.01
            target=task['target_position']
            d=math.hypot(after['position']['x']-target['x'],after['position']['z']-target['z'])
            assert abs(d-step['distance'])<1e-7
            assert int(d<=1)==int(step['task_success'])
        assert int(trace[-1]['task_success'])==int(value['success'])
        assert abs(trace[-1]['distance']-value['final_distance'])<1e-7
        assert value['fallback_steps']==sum(bool(s['fallback']) for s in trace)
        assert value['invalid_action_steps']==sum(not s['action_success'] for s in trace)
        original_pose=task['agentPose']['position']
        d0=math.hypot(original_pose['x']-task['target_position']['x'],original_pose['z']-task['target_position']['z'])
        lower_bound=math.ceil(max(0.,d0-1)/.25-1e-6)
        scene=task['scene']
        physical={k:task[k] for k in ['scene','agentPose','target_position','targetObjectIds']}
        physical_key=hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
        record={k:v for k,v in value.items() if k not in {'trace','responses','instruction'}}
        record.update(scene=scene,path_length_m=length,initial_target_distance_m=d0,
            euclidean_lower_bound_translation_actions=lower_bound,
            impossible_with_20_actions_by_distance_bound=lower_bound>20,
            inference_seconds=sum(r['seconds'] for r in value['responses']),
            physical_case_key=physical_key)
        records.append(record)
        physical_groups[physical_key][subset]=record['success']
groups={}
for subset in SUBSETS:
    rs=[r for r in records if r['subset']==subset]
    groups[subset]={'episodes':len(rs),'successes':sum(r['success'] for r in rs),
        'SR':float(np.mean([r['success'] for r in rs])),
        'mean_steps':float(np.mean([r['steps'] for r in rs])),
        'parser_errors':sum(r['parser_errors'] for r in rs),
        'fallback_steps':sum(r['fallback_steps'] for r in rs),
        'invalid_action_steps':sum(r['invalid_action_steps'] for r in rs),
        'model_calls':sum(r['planner_steps'] for r in rs),
        'distance_lower_bound_exceeds_budget':sum(r['impossible_with_20_actions_by_distance_bound'] for r in rs)}
result={'acceptance':'passed','complete_official_300':len(records)==300,
    'scope':'Official EB-Navigation with a local Qwen2.5-VL-7B backend, native 3-shot prompts/8 actions/20-step budget/1m horizontal-distance success',
    'episodes':len(records),'scenes':len({r['scene'] for r in records}),
    'unique_physical_initial_goal_cases':len(physical_groups),
    'physical_cases_present_in_all_five_instruction_sets':sum(len(v)==5 for v in physical_groups.values()),
    'successes':sum(r['success'] for r in records),
    'SR':float(np.mean([r['success'] for r in records])),
    'groups':groups,'datasets':datasets,'records':records,
    'sampling':'All 60 tasks per subset' if len(records)==300 else 'Engineering pilot: first '+str(args.limit)+' tasks per subset; not a robust performance estimate',
    'not_covered':'R2R/RxR long-route trajectory following, physical robot navigation, and policy-locomotion joint deployment'}
target=OUT/('accepted_summary.json' if len(records)==300 else 'accepted_pilot_summary.json')
target.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in {'records','datasets'}},indent=2),flush=True)
