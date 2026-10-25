"""Reconcile every native UniGoal execution with the frozen panel and trace."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--required',type=int,default=300)
args=p.parse_args();out=ROOT/'results'/args.run
assert not (out/'runtime_failure.json').exists(),'Failed execution cannot be accepted'
config=json.loads((out/'configuration.json').read_text())
panel_bytes=(out/'panel.jsonl').read_bytes()
assert hashlib.sha256(panel_bytes).hexdigest()==config['panel_sha256']
panel=[json.loads(line) for line in panel_bytes.decode().splitlines()]
assert len(panel)==args.required and len({e['episode_id'] for e in panel})==len(panel)
if args.required==300:
    source=Path('/root/autodl-tmp/jepa_vln_baselines_20261015/plans/hssd_image_nav_v1_test_instance_20perscene.jsonl')
    assert source.read_bytes()==panel_bytes
for gate in ['cuda_operator_gate.json','detector_weight_gate.json']:
    assert json.loads((out/gate).read_text())['passed'],gate
runtime=json.loads((ROOT/'results/unigoal_llm_runtime_gate.json').read_text())
assert runtime['passed'] and runtime['model']=='llama3.2-vision:11b'
records=json.loads((out/'records.json').read_text());assert len(records)==len(panel)
assert len({r['episode_id'] for r in records})==len(records)
total_steps=0
for episode,record in zip(panel,records):
    assert record['episode_id']==episode['episode_id'] and record['status']=='complete'
    trace=[json.loads(line) for line in (out/(episode['episode_id']+'.steps.jsonl')).read_text().splitlines()]
    assert len(trace)==record['steps'] and 1<=len(trace)<=config['max_episode_length']
    length=0.
    for t,step in enumerate(trace):
        assert step['t']==t and step['action'] in {0,1,2,3}
        before=np.asarray(step['before']);after=np.asarray(step['after'])
        displacement=float(np.linalg.norm(after-before));length+=displacement
        if t:assert np.linalg.norm(before-np.asarray(trace[t-1]['after']))<1e-5
        if step['action']!=1:assert displacement<1e-5
        else:assert np.linalg.norm(after-np.asarray(step['navmesh_expected_after']))<1e-4
        assert 0<=step['target_fraction_after']<=1
        assert step['action']!=0 or t==len(trace)-1
        if t+1<len(trace) and step['action'] in {2,3}:
            qa=np.asarray(step['rotation_xyzw_before']);qb=np.asarray(trace[t+1]['rotation_xyzw_before'])
            angle=2*math.acos(float(np.clip(abs(np.dot(qa,qb)),-1,1)))
            assert abs(angle-math.radians(15))<1e-4
    assert abs(length-record['path_length_m'])<1e-5
    assert bool(record['goal_stop'])==(trace[-1]['action']==0)
    assert abs(record['target_fraction_at_end']-trace[-1]['target_fraction_after'])<1e-10
    success=int(record['goal_stop'] and record['final_distance_m']<.5 and record['target_fraction_at_end']>=.025)
    assert success==record['success']
    spl=success*record['initial_distance_m']/max(record['initial_distance_m'],length,1e-8)
    assert abs(spl-record['spl'])<1e-7
    assert record['category_supplied']==config['category_assisted']
    total_steps+=len(trace)
result={'acceptance':'passed','episodes':len(records),'full_300':len(records)==300,
    'successes':sum(r['success'] for r in records),'SR':float(np.mean([r['success'] for r in records])),
    'SPL':float(np.mean([r['spl'] for r in records])),'trace_steps_checked':total_steps,
    'category_assisted':config['category_assisted'],'panel_sha256':config['panel_sha256'],
    'scope':'Released UniGoal main/agent/graph/BEV/FMM with calibrated sensor interfaces and frozen common HSSD geometry; not native HM3D scores',
    'records':records}
(out/'accepted_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
