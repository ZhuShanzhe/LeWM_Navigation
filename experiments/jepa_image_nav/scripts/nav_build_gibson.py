"""A derived Gibson place benchmark; preserve author's val scene isolation."""
import json, numpy as np
from nav_build import ROOT, dump, opensim, photo, startpoints, stable, classify, writeepisodes, expert
tree=json.loads((ROOT/'manifests/gibson_original_episode_tree.json').read_text())
train={x['path'].split('/')[1].split('.json')[0] for x in tree['tree'] if x['path'].startswith('train/') and x['path'].endswith('.json.gz')}
val={x['path'].split('/')[1].split('.json')[0] for x in tree['tree'] if x['path'].startswith('val/') and x['path'].endswith('.json.gz')}
test=set(sorted(train,key=lambda s:stable(s+'test-v1'))[:10])
out=ROOT/'datasets/gibson_place_nav_v1'
splits={s:('test' if s in test else 'train') for s in train};splits.update({s:'val' for s in val})
dump(out/'split_manifest.json',{'episode_source_commit':tree['sha'],'split_source':'facebookresearch/image-goal-nav-dataset','derived_resampled_episodes':True,
                               'official_val_preserved':True,'local_test_reserved_from_train':sorted(test),'scenes':splits})
dump(out/'protocol.json',{'task':'place_image_nav','derived_not_original_episodes':True,'rgb_resolution':[256,256],'hfov':90,'camera_height_m':1.25,
                         'agent_radius_m':.18,'agent_height_m':1.4,'forward_m':.25,'turn_deg':30,'success':'explicit STOP and geodesic <1m',
                         'default_policy_input':'RGB history, goal RGB, previous action; no goal coordinates or full map'})
for scene,split in sorted(splits.items()):
    if (out/'scene_statistics'/f'{scene}.json').exists():continue
    mesh=ROOT/'raw/scene_datasets/gibson'/f'{scene}.glb'
    if not mesh.exists():dump(out/'failures'/f'{scene}.json',{'reason':'scene_missing'});continue
    sim=None
    try:
        sim=opensim(str(mesh),None);rng=np.random.default_rng(stable(scene+'v1'));sim.seed(stable(scene+'v1'))
        images=out/'images'/scene;images.mkdir(parents=True,exist_ok=True);eps=[]
        ng=100 if split=='train' else 40;ns=30 if split=='train' else 20
        for g in range(ng):
            q=np.array(sim.pathfinder.get_random_navigable_point());yaw=float(rng.uniform(-np.pi,np.pi));ip=images/f'{g:04d}.jpg'
            cam=photo(sim,q,yaw,None,ip,None)
            if cam is None:continue
            for j,(p,d,sy) in enumerate(startpoints(sim,q,ns,rng)):
                eps.append({'episode_id':f'gibson-place-{scene}-{g}-{j}','task':'place_image_nav','scene_id':scene,'split':split,
                            'start_position':p.tolist(),'start_yaw':sy,'goal_position':q.tolist(),'goal_camera':cam,
                            'goal_image':str(ip.relative_to(out)),**classify(d,p,q)})
        writeepisodes(out/'episodes'/split/f'{scene}.jsonl.gz',eps)
        trajectories=[]
        if split=='train':
            for e in eps[:5]:trajectories.append(expert(sim,e,out/'trajectories'/scene/e['episode_id'],rng))
        stat={'scene':scene,'split':split,'episodes':len(eps),'goals':len({e['goal_image'] for e in eps}),
              'navigable_area_m2':sim.pathfinder.navigable_area,'expert_trajectories':trajectories,'status':'complete'}
        dump(out/'scene_statistics'/f'{scene}.json',stat);print(json.dumps(stat),flush=True)
    except Exception as e:dump(out/'failures'/f'{scene}.json',{'reason':str(e)});print('SCENE_FAILED',scene,str(e),flush=True)
    finally:
        if sim:sim.close()
stats=[json.loads(f.read_text()) for f in (out/'scene_statistics').glob('*.json')];done={s['scene'] for s in stats}
summary={'processed_scenes':len(stats),'episodes':sum(s['episodes'] for s in stats),'split_counts':{sp:sum(s['episodes'] for s in stats if s['split']==sp) for sp in ('train','val','test')},
         'failed_scenes':sum(f.stem not in done for f in (out/'failures').glob('*.json'))}
dump(out/'summary.json',summary);print('BUILD_SUMMARY',summary,flush=True)
