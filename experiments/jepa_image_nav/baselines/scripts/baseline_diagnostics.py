"""Asset overlap audit and privileged controller checks, separate from model SR."""
import argparse, collections, gzip, json, math, os, sys, time
from pathlib import Path
import numpy as np
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015');DATA=Path('/root/autodl-tmp/jepa_vln_data_20261014')
sys.path.insert(0,str(DATA/'tools'))

def assets():
    ds=DATA/'datasets/hssd_image_nav_v1';splits=json.loads((ds/'split_manifest.json').read_text())['scenes']
    all_assets=collections.defaultdict(set);targets=collections.defaultdict(list);records=[]
    for f in sorted((ds/'instances').glob('*.json')):
        scene=f.stem;items=json.loads(f.read_text());group=collections.Counter(i['asset_id'] for i in items)
        for i in items:
            all_assets[splits[scene]].add(i['asset_id']);targets[splits[scene]].append((scene,i['asset_id']))
            records.append({'scene':scene,'split':splits[scene],'instance_id':i['instance_id'],'asset_id':i['asset_id'],
                            'same_asset_other_targets_in_scene':group[i['asset_id']]-1})
    result={'scope':'selected target objects only, not all scene furniture','split_unique_assets':{s:len(a) for s,a in all_assets.items()},
            'split_target_instances':dict(collections.Counter(r['split'] for r in records)),
            'test_target_instances_with_asset_in_train':sum(a in all_assets['train'] for _,a in targets['test']),
            'test_target_instances':len(targets['test']),
            'test_target_instances_with_same_asset_other_selected_instance_in_scene':sum(r['split']=='test' and r['same_asset_other_targets_in_scene']>0 for r in records),
            'meaning':'Scene-disjoint is not asset-disjoint. Same asset reuse is a diagnostic, not proof identical goal photographs or unsolvable identity.'}
    (BASE/'results/asset_overlap_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

def oracle():
    from nav_build import opensim,setpose,pathlength
    import habitat_sim
    result=[]
    for name in ['hssd_image_nav_v1_test_place_20perscene','hssd_image_nav_v1_test_instance_20perscene']:
        eps=[json.loads(l) for l in (BASE/'plans'/(name+'.jsonl')).read_text().splitlines()]
        groups=collections.defaultdict(list)
        for e in eps:groups[e['scene_id']].append(e)
        for scene,items in sorted(groups.items()):
            out=BASE/'results/oracle_controls'/f'{name}_{scene}.json'
            if out.exists():result.extend(json.loads(out.read_text()));continue
            sim=opensim(scene,DATA/'raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json')
            rows=[]
            try:
                objs=sim.get_rigid_object_manager().get_objects_by_handle_substring('')
                for i,(h,o) in enumerate(sorted(objs.items()),1):o.semantic_id=i
                meta={i['instance_id']:i for i in json.loads((DATA/'datasets/hssd_image_nav_v1/instances'/f'{scene}.json').read_text())}
                # Two fixed episodes per scene: controller check, not a dataset-wide upper bound.
                for e in items[:2]:
                    instance=e['task']=='instance_image_nav';views=e.get('valid_approach_views',[])
                    goal=min(views,key=lambda v:pathlength(sim,e['start_position'],v['position']) or 1e6)['position'] if instance else e['goal_position']
                    setpose(sim,e['start_position'],e['start_yaw']);follower=habitat_sim.GreedyGeodesicFollower(sim.pathfinder,sim.get_agent(0),goal_radius=.2)
                    stopped=False;length=0.;err=None
                    for t in range(500):
                        pos=np.array(sim.get_agent(0).get_state().position)
                        try:a=follower.next_action_along(goal)
                        except Exception as ex:err=str(ex);break
                        if a is None:stopped=True;break
                        sim.step(a);length+=float(np.linalg.norm(np.array(sim.get_agent(0).get_state().position)-pos))
                    pos=sim.get_agent(0).get_state().position
                    d=min([pathlength(sim,pos,v['position']) for v in views if pathlength(sim,pos,v['position']) is not None],default=1e6) if instance else pathlength(sim,pos,goal)
                    visible=None
                    if instance:
                        sid=meta[e['instance_id']]['semantic_id'];visible=False
                        # The oracle may rotate to find the object but cannot pitch the camera.
                        for k in range(12):
                            obs=sim.get_sensor_observations()
                            if (obs['semantic']==sid).mean()>=.025:visible=True;break
                            sim.step('turn_left')
                    rows.append({'episode_id':e['episode_id'],'task':e['task'],'scene':scene,'steps':t+1,'stopped':stopped,
                                 'distance_m':d,'yaw_only_target_visible':visible,'success':bool(stopped and d is not None and d<(.5 if instance else 1.) and (visible is None or visible)),
                                 'follower_error':err,'privileged_control_not_model':True})
            finally:sim.close()
            out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(rows,indent=2));result.extend(rows)
    report={'episodes':len(result),'successes':sum(r['success'] for r in result),'by_task':{task:{'n':sum(r['task']==task for r in result),
             'successes':sum(r['task']==task and r['success'] for r in result)} for task in ['place_image_nav','instance_image_nav']},
             'failures':[r for r in result if not r['success']],
             'scope':'2 fixed episodes per task per test scene; truth navigation grid and goal used; not learned navigation performance'}
    (BASE/'results/oracle_control_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    assets();oracle()
