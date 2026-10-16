"""Build exhaustive-category HSSD ObjectNav labels, not selected-instance proxies.

All scene objects are classified using the same publisher taxonomy as the
image-goal dataset. Category success can include ANY visible instance, including
objects not selected when constructing the earlier instance-image benchmark.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from instance_eval_goat import ROOT, DATA, OLD, opensim, PROFILES, write
sys.path.insert(0, str(DATA / 'tools'))
from nav_build import object_category, pathlength, setpose, classify

OUT = ROOT / 'datasets/hssd_objectnav_v2'
PANEL = OLD / 'plans/hssd_image_nav_v1_test_instance_20perscene.jsonl'

def scene_build(scene):
    import habitat_sim as hs
    from habitat_sim.utils.common import quat_from_angle_axis
    output = OUT / 'scenes' / (scene+'.json')
    if output.exists():
        return json.loads(output.read_text())['statistics']
    sim = opensim(scene, PROFILES['shared'])
    rng = np.random.default_rng(int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16))
    try:
        assets = DATA / 'raw/hssd-hab'
        metadata = json.loads((assets / 'metadata/objects.json').read_text())
        taxonomy = {r['id']: r['main_category'] for r in csv.DictReader((assets / 'metadata/fpmodels-with-decomposed.csv').open()) if r.get('main_category')}
        mapping = {'chair':'chair', 'armchair':'chair', 'stool':'chair', 'couch':'couch', 'sofa':'couch',
            'bed':'bed', 'bed_small':'bed', 'plant':'plant', 'plant_art':'plant', 'potted_plant':'plant',
            'toilet':'toilet', 'tv':'tv', 'television':'tv', 'led_tv':'tv'}
        source = [json.loads(line) for line in PANEL.read_text().splitlines() if line.strip()]
        source = [e for e in source if e['scene_id'] == scene]
        instances, inventory = [], []
        objs = sim.get_rigid_object_manager().get_objects_by_handle_substring('')
        for handle, obj in sorted(objs.items()):
            asset = handle.split('_:')[0]
            category = mapping.get(taxonomy[asset]) if asset in taxonomy else object_category(metadata.get(asset, {}).get('name', ''))
            if category is None:
                continue
            box = hs.geo.get_transformed_bb(obj.root_scene_node.cumulative_bb, obj.transformation)
            center = np.asarray(box.center())
            views = []
            candidates = [np.asarray(v['position']) for e in source if e['instance_id']==scene+':'+handle
                          for v in e['valid_approach_views']]
            candidates += [center+np.array([radius*math.sin(angle),0,radius*math.cos(angle)])
                for angle in np.linspace(-math.pi,math.pi,24,endpoint=False)
                for radius in [.5,.75,1.,1.5,2.]]
            seen = set()
            for sample in candidates:
                    key=tuple(np.round(sample,3))
                    if key in seen:
                        continue
                    seen.add(key)
                    position = np.asarray(sim.pathfinder.snap_point(sample))
                    if not np.isfinite(position).all() or np.linalg.norm(position[[0,2]]-sample[[0,2]]) > .4:
                        continue
                    if abs(float(position[1])-float(box.min.y)) > 3.:
                        continue
                    delta = np.maximum(np.asarray(box.min)[[0,2]]-position[[0,2]], 0)+np.minimum(np.asarray(box.max)[[0,2]]-position[[0,2]], 0)
                    if np.linalg.norm(delta) > 1.:
                        continue
                    yaw = math.atan2(-(center[0]-position[0]), -(center[2]-position[2]))
                    pitch = math.atan2(center[1]-position[1]-1.25, np.linalg.norm(center[[0,2]]-position[[0,2]]))
                    setpose(sim, position, yaw)
                    state = sim.get_agent(0).get_state()
                    # Pitch the sensor, NOT the body/camera offset.
                    for pose in state.sensor_states.values():
                        pose.position = position + np.array([0., 1.25, 0.])
                        pose.rotation = quat_from_angle_axis(yaw, np.array([0.,1.,0.]))*quat_from_angle_axis(pitch, np.array([1.,0.,0.]))
                    sim.get_agent(0).set_state(state, infer_sensor_states=False)
                    obs = sim.get_sensor_observations()
                    fraction = float((obs['semantic'] == obj.semantic_id).mean())
                    if fraction >= .025:
                        views.append({'position':position.tolist(), 'yaw':yaw, 'pitch':pitch, 'coverage':fraction,
                            'camera_position': (position+np.array([0.,1.25,0.])).tolist()})
            item = {'handle':handle, 'instance_id':scene+':'+handle, 'semantic_id':obj.semantic_id,
                'category':category, 'asset_id':asset, 'aabb_min':list(box.min), 'aabb_max':list(box.max),
                'category_source':'publisher_main_category' if asset in taxonomy else 'strict_asset_name_rule',
                'valid_approach_views':views}
            inventory.append(item)
            if views:
                instances.append(item)
        episodes = []
        for e in source:
            goals = [g for g in instances if g['category'] == e['category']]
            viewpoints = [v['position'] for g in goals for v in g['valid_approach_views']]
            if not viewpoints:
                episodes.append({'episode_id':e['episode_id'].replace('hssd-instance', 'hssd-object'),
                    'scene_id':scene, 'category':e['category'], 'status':'no_verified_category_viewpoint'})
                continue
            def distance(p):
                values = [pathlength(sim, p, q) for q in viewpoints]
                return min((d for d in values if d is not None and np.isfinite(d)), default=1e6)
            p = np.asarray(e['start_position'])
            d = distance(p)
            paired = True
            for _ in range(3000):
                if 1.5 <= d <= 30:
                    break
                p = np.asarray(sim.pathfinder.get_random_navigable_point())
                paired, d = False, distance(p)
            if not 1.5 <= d <= 30:
                raise RuntimeError('Cannot obtain a nontrivial category start')
            yaw = e['start_yaw'] if paired else float(rng.uniform(-math.pi, math.pi))
            episodes.append({'episode_id':e['episode_id'].replace('hssd-instance', 'hssd-object'),
                'source_episode_id':e['episode_id'], 'scene_id':scene, 'category':e['category'],
                'task':'object_category_nav', 'split':'test', 'start_position':p.tolist(), 'start_yaw':yaw,
                'paired_start_with_instance_task':paired, 'geodesic_to_any_category_view_m':d,
                'goal_instances':len(goals), 'goal_viewpoints':len(viewpoints), 'status':'complete'})
        statistics = {'scene':scene, 'episodes':len(episodes), 'all_category_instances':len(inventory),
            'verified_instances':len(instances), 'unverified_instances':len(inventory)-len(instances),
            'viewpoints':sum(len(g['valid_approach_views']) for g in instances),
            'paired_starts':sum(e.get('paired_start_with_instance_task',False) for e in episodes)}
        write(output, {'scene':scene, 'task':'object_category_nav', 'statistics':statistics,
            'instances':inventory, 'episodes':episodes,
            'labels':'All recognized scene instances retained; finite sampled-view coverage is explicit, not proof of unreachability',
            'success':'Explicit target STOP, distance_to_any_verified_category_approach<0.25m; visible-instance canonical score also reported',
            'camera':dict(PROFILES['shared'], pitch='sensor only, fixed optical center')})
        print(json.dumps(statistics), flush=True)
        return statistics
    finally:
        sim.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    (OUT / 'scenes').mkdir(parents=True, exist_ok=True)
    if args.scene:
        scene_build(args.scene)
    else:
        source = [json.loads(line) for line in PANEL.read_text().splitlines() if line.strip()]
        scenes = sorted({e['scene_id'] for e in source})
        def child(scene):
            with (ROOT / 'logs' / ('objectnav_build_'+scene+'.log')).open('w') as log:
                env = dict(os.environ, MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                proc = subprocess.run([sys.executable,__file__,'--scene',scene],stdout=log,stderr=subprocess.STDOUT,env=env)
            return {'scene':scene,'exit_code':proc.returncode}
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            result = list(pool.map(child, scenes))
        write(OUT / 'execution.json', result)
        if any(r['exit_code'] for r in result):
            raise SystemExit(1)
        parts = [json.loads((OUT / 'scenes' / (scene+'.json')).read_text()) for scene in scenes]
        episodes = [e for part in parts for e in part['episodes']]
        (OUT / 'test.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
        write(OUT / 'manifest.json', {'scenes':len(scenes),'episodes':len(episodes),
            'construction':'v2: seed candidate views from frozen instance labels plus 24 bearings x 5 radii, 2.5% pixel gate; no model-dependent filtering',
            'task':'category targets, not instance targets','source_panel_sha256':hashlib.sha256(PANEL.read_bytes()).hexdigest(),
            'source_scene_revision':'4369cb9876214c7fbebcf552eb532380e4d287e4',
            'statistics':[part['statistics'] for part in parts]})
