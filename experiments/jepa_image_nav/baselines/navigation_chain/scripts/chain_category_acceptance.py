"""Full trace acceptance for class goals, separate from instance-image scoring."""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
DATA=ROOT/'datasets/hssd_objectnav_v3'
RUN=ROOT/'results/goat_category_hssd300_v3'
manifest=json.loads((DATA/'manifest.json').read_text())
episodes=[json.loads(x) for x in (DATA/'test.jsonl').read_text().splitlines()]
assert len(episodes)==300 and len({e['episode_id'] for e in episodes})==300
assert len({e['scene_id'] for e in episodes})==15 and all(e['status']=='complete' for e in episodes)
assert manifest['completion_gate']['complete']
records=[]
traced_steps=0
unusual_translations=0
for scene in sorted({e['scene_id'] for e in episodes}):
    part=json.loads((DATA/'scenes'/(scene+'.json')).read_text())
    objects={g['semantic_id']:g for g in part['instances']}
    assert len(objects)==len(part['instances'])
    policy=json.loads((RUN/scene/'policy.json').read_text())
    assert policy['strict_all_state_keys'] and policy['checkpoint_keys']==349
    for e in [e for e in episodes if e['scene_id']==scene]:
        r=json.loads((RUN/scene/(e['episode_id']+'.json')).read_text())
        assert r['status']=='complete' and r['category']==e['category']
        with gzip.open(RUN/scene/(e['episode_id']+'.steps.jsonl.gz'),'rt') as stream:
            trace=[json.loads(x) for x in stream]
        assert len(trace)==r['steps'] and r['steps']<=500
        assert trace[-1]['action']==r['final_action']
        length=0.
        length_native=0.
        for i,step in enumerate(trace):
            assert step['t']==i and step['action']==int(np.argmax(step['logits']))
            delta=np.linalg.norm(np.asarray(step['after'])-step['before'])
            length+=delta
            length_native+=float(np.linalg.norm(np.asarray(step['after'],dtype=np.float32)-np.asarray(step['before'],dtype=np.float32)))
            if step['action']!=1:
                assert delta<1e-5
            elif delta>.251:
                unusual_translations+=1
            if i+1<len(trace):
                assert np.allclose(step['after'],trace[i+1]['before'],atol=1e-6)
            assert all(objects[g['semantic_id']]['category']==e['category'] and .025<=g['fraction']<=1
                for g in step['visible_category_evidence'])
        assert abs(length-r['path_length_m'])<1e-5
        assert abs(length_native-r['path_length_m'])<1e-10
        assert trace[-1]['action'] in {0,6} or len(trace)==500
        for g in r['final_category_evidence']:
            obj=objects[g['semantic_id']]
            assert obj['category']==e['category'] and .025<=g['fraction']<=1
            point=np.asarray(trace[-1]['after'])[[0,2]]
            lo=np.asarray(obj['aabb_min'])[[0,2]]
            hi=np.asarray(obj['aabb_max'])[[0,2]]
            gap=float(np.linalg.norm(np.maximum(np.maximum(lo-point,point-hi),0)))
            assert abs(gap-g['horizontal_aabb_gap_m'])<1e-6
        success=int(r['final_action']==6 and any(g['horizontal_aabb_gap_m']<=1 for g in r['final_category_evidence']))
        assert success==r['success']
        view_success=int(r['final_action']==6 and r['final_verified_view_geodesic_m']<.25)
        assert view_success==r['verified_view_success']
        d0=r['initial_verified_view_geodesic_m']
        assert abs(r['verified_view_spl']-view_success*d0/max(d0,length_native,1e-8))<1e-10
        records.append(r)
        traced_steps+=len(trace)
def groups(key):
    buckets=defaultdict(list)
    for r in records:
        buckets[r[key]].append(r)
    return {k:{'episodes':len(v),'successes':sum(r['success'] for r in v),
        'SR':float(np.mean([r['success'] for r in v]))} for k,v in buckets.items()}
result={'acceptance':'passed','model':'GOAT SenseAct-NN Monolithic official category policy',
    'scope':'HSSD derived cross-domain category goals, native camera/common geometry, cold-start single subtask',
    'episodes':len(records),'scenes':15,'trace_steps_checked':traced_steps,
    'successes':sum(r['success'] for r in records),'SR':float(np.mean([r['success'] for r in records])),
    'macro_category_SR':float(np.mean([v['SR'] for v in groups('category').values()])),
    'primary_success':'subtask STOP6 plus >=2.5% visible pixels for ANY requested-category instance with horizontal AABB gap<=1m',
    'SPL_reporting':'No sparse-view SPL is assigned to the primary AABB/visibility goal set',
    'verified_view_secondary_SR':float(np.mean([r['verified_view_success'] for r in records])),
    'verified_view_secondary_SPL':float(np.mean([r['verified_view_spl'] for r in records])),
    'goal_visible_ever':sum(r['category_visible_ever'] for r in records),
    'near_correct_class_ever':sum(r['near_correct_class_ever'] for r in records),
    'outcomes':dict(Counter(r['outcome'] for r in records)),
    'by_category':groups('category'),'by_scene':groups('scene'),
    'dataset_sha256':hashlib.sha256((DATA/'test.jsonl').read_bytes()).hexdigest(),
    'dataset_category_reassignments':manifest['category_reassignments'],
    'unusual_forward_3d_translations':unusual_translations,
    'path_precision':'Stored native float32 per-step norms recompute exactly; independent float64 accumulated path agrees within 1e-5 m',
    'movement_check_scope':'Trace continuity, fixed-position turns/STOP, and path recomputation; no new independent full visual replay claimed',
    'not_official_paper_scores':True}
(RUN/'accepted_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
