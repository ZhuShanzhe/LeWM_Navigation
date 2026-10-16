"""Freeze a complete category panel without assuming unverified goals solvable."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from instance_eval_goat import ROOT,DATA,OLD,PROFILES,opensim,write
sys.path.insert(0,str(DATA/'tools'))
from nav_build import pathlength

source=ROOT/'datasets/hssd_objectnav_v2'
dest=ROOT/'datasets/hssd_objectnav_v3'
frozen=OLD/'plans/hssd_image_nav_v1_test_instance_20perscene.jsonl'
original={e['episode_id']:e for e in (json.loads(x) for x in frozen.read_text().splitlines())}
changes=[]
parts=[]
for file in sorted((source/'scenes').glob('*.json')):
    part=json.loads(file.read_text())
    incomplete=[e for e in part['episodes'] if e['status']!='complete']
    if incomplete:
        sim=opensim(part['scene'],PROFILES['shared'])
        try:
            available=sorted({g['category'] for g in part['instances'] if g['valid_approach_views']})
            for e in incomplete:
                old=original[e['episode_id'].replace('hssd-object','hssd-instance')]
                category=available[int(hashlib.sha256(e['episode_id'].encode()).hexdigest()[:8],16)%len(available)]
                goals=[g for g in part['instances'] if g['category']==category and g['valid_approach_views']]
                views=[v['position'] for g in goals for v in g['valid_approach_views']]
                def distance(p):
                    values=[pathlength(sim,p,v) for v in views]
                    return min((v for v in values if v is not None and np.isfinite(v)),default=1e6)
                position=np.asarray(old['start_position'])
                d=distance(position)
                paired=True
                rng=np.random.default_rng(int(hashlib.sha256(e['episode_id'].encode()).hexdigest()[:8],16))
                for _ in range(3000):
                    if 1.5<=d<=30:
                        break
                    position=np.asarray(sim.pathfinder.get_random_navigable_point())
                    d=distance(position)
                    paired=False
                assert 1.5<=d<=30
                e.update(status='complete',category=category,source_category=old['category'],
                    source_episode_id=old['episode_id'],category_reassigned_for_verified_goal=True,
                    task='object_category_nav',split='test',start_position=position.tolist(),
                    start_yaw=old['start_yaw'] if paired else float(rng.uniform(-np.pi,np.pi)),
                    paired_start_with_instance_task=paired,geodesic_to_any_category_view_m=d,
                    goal_instances=len(goals),goal_viewpoints=len(views),
                    construction_note='Original category has no verified view in this scene; deterministic available-category reassignment before model testing')
                changes.append({'episode_id':e['episode_id'],'scene':part['scene'],
                    'from_category':old['category'],'to_category':category,'paired_start':paired})
        finally:
            sim.close()
    for e in part['episodes']:
        e.setdefault('source_category',e['category'])
        e.setdefault('category_reassigned_for_verified_goal',False)
        assert e['status']=='complete' and 1.5<=e['geodesic_to_any_category_view_m']<=30
    part['statistics']['paired_starts']=sum(e['paired_start_with_instance_task'] for e in part['episodes'])
    part['category_reassignment']='Unverified category pairs are replaced deterministically; no model outcomes used'
    write(dest/'scenes'/file.name,part)
    parts.append(part)
episodes=[e for p in parts for e in p['episodes']]
assert len(episodes)==300 and len({e['episode_id'] for e in episodes})==300 and len(parts)==15
(dest/'test.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
manifest=json.loads((source/'manifest.json').read_text())
manifest.update(version='hssd_objectnav_v3',source_version='hssd_objectnav_v2',
    category_reassignments=changes,statistics=[p['statistics'] for p in parts],
    test_panel_sha256=hashlib.sha256((dest/'test.jsonl').read_bytes()).hexdigest(),
    completion_gate={'complete':True,'episodes':300,'scenes':15,'no_unverified_category_episode':True},
    goal_inventory_scope='All 826 recognized instances retained; finite view sampling verifies a subset, not proof remaining instances unreachable')
write(dest/'manifest.json',manifest)
print(json.dumps({'ready':True,'episodes':300,'scenes':15,'category_reassignments':changes}),flush=True)
