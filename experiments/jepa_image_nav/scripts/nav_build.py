"""Build place/instance image-goal data with scene isolation and visible approach sets.

This is a derived HSSD benchmark, not a replacement label for official HM3D-IIN.
Privileged coordinates, semantic IDs, navmesh and expert routes are labels only.
"""
import argparse, gzip, hashlib, json, math, traceback, csv, re
from pathlib import Path
import numpy as np, magnum as mn, habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs
from PIL import Image
ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014')
REV='4369cb9876214c7fbebcf552eb532380e4d287e4'
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def stable(s): return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def setpose(sim,p,yaw,pitch=0):
    st=habitat_sim.AgentState();st.position=np.asarray(p,dtype=np.float32)
    st.rotation=quat_from_angle_axis(yaw,np.array([0.,1.,0.]))*quat_from_angle_axis(pitch,np.array([1.,0.,0.]))
    sim.get_agent(0).set_state(st)
    return sim.get_sensor_observations()
def photo(sim,p,yaw,target,out,sid):
    pitch=0 if target is None else math.atan2(target[1]-(p[1]+1.25),np.linalg.norm(np.array(target)[[0,2]]-np.array(p)[[0,2]]))
    obs=setpose(sim,p,yaw,pitch)
    if np.std(obs['rgb'][:,:,:3])<5: return None
    if sid is not None:
        mask=obs['semantic']==sid
        if mask.mean()<.025: return None
        Image.fromarray(mask.astype('uint8')*255).save(out.with_suffix('.mask.png'))
    Image.fromarray(obs['rgb'][:,:,:3]).save(out,quality=90)
    # Quality is measured on the saved observation, not only the pre-JPEG RGB.
    with Image.open(out) as encoded:
        if np.asarray(encoded.convert('RGB')).std()<5:return None
    return {'position':list(map(float,p)),'yaw':float(yaw),'pitch':float(pitch),'height':1.25,'hfov':90,'resolution':[256,256]}
def pathlength(sim,p,q):
    req=habitat_sim.ShortestPath();req.requested_start=np.array(p,dtype=np.float32);req.requested_end=np.array(q,dtype=np.float32)
    if not sim.pathfinder.find_path(req): return None
    return float(req.geodesic_distance)
def classify(d,p,q):
    e=float(np.linalg.norm(np.array(p)-np.array(q)));ratio=d/max(e,.01)
    return {'distance_bin':'short' if d<3 else 'medium' if d<5 else 'long' if d<10 else 'extra',
            'detour_bin':'direct' if ratio<1.2 else 'detour','geodesic_m':d,'euclidean_m':e,'detour_ratio':ratio}
def viewgap(start_yaw,goal_yaw):
    gap=abs((start_yaw-goal_yaw+math.pi)%(2*math.pi)-math.pi)*180/math.pi
    return {'start_goal_yaw_gap_deg':gap,'view_gap_bin':'0-45' if gap<45 else '45-90' if gap<90 else '90-180'}
def object_category(name):
    if any(w in name.lower() for w in ['bedside','bed table','toilet brush','toilet paper','tv stand','chair cushion','chair pad']):return None
    return next((v for v,k in [('chair',('chair','armchair','stool')),('couch',('sofa','couch')),('bed',('bed',)),('plant',('plant',)),('toilet',('toilet',)),('tv',('tv','television'))] if any(re.search(r'\b'+w+r'\b',name.lower()) for w in k)),None)
def startpoints(sim,q,n,rng):
    found=[]
    for _ in range(n*120):
        p=np.array(sim.pathfinder.get_random_navigable_point());d=pathlength(sim,p,q)
        if d is not None and 1.5<=d<=30:
            found.append((p,d,float(rng.uniform(-math.pi,math.pi))))
            if len(found)==n: break
    return found
def opensim(scene,config):
    c=habitat_sim.SimulatorConfiguration()
    if config is not None:c.scene_dataset_config_file=str(config)
    c.scene_id=scene;c.enable_physics=True
    a=habitat_sim.agent.AgentConfiguration();a.height=1.4;a.radius=.18
    specs=[]
    for name,t in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH),('semantic',habitat_sim.SensorType.SEMANTIC)]:
        s=habitat_sim.CameraSensorSpec();s.uuid=name;s.sensor_type=t;s.resolution=[256,256];s.position=mn.Vector3(0,1.25,0);s.hfov=mn.Deg(90);specs.append(s)
    a.sensor_specifications=specs
    a.action_space={'move_forward':habitat_sim.agent.ActionSpec('move_forward',habitat_sim.agent.ActuationSpec(amount=.25)),
                    'turn_left':habitat_sim.agent.ActionSpec('turn_left',habitat_sim.agent.ActuationSpec(amount=30)),
                    'turn_right':habitat_sim.agent.ActionSpec('turn_right',habitat_sim.agent.ActuationSpec(amount=30))}
    sim=habitat_sim.Simulator(habitat_sim.Configuration(c,[a]));sim.seed(stable(scene))
    nav=habitat_sim.NavMeshSettings();nav.set_defaults();nav.agent_radius=.18;nav.agent_height=1.4;nav.include_static_objects=True
    if not sim.recompute_navmesh(sim.pathfinder,nav):sim.close();raise ValueError('Navmesh failed')
    return sim
def writeepisodes(path,eps):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as f:
        for e in eps:f.write(json.dumps(e)+'\n')
def expert(sim,ep,folder,rng):
    folder.mkdir(parents=True,exist_ok=True);setpose(sim,ep['start_position'],ep['start_yaw'])
    follower=habitat_sim.GreedyGeodesicFollower(sim.pathfinder,sim.get_agent(0),goal_radius=.5)
    frames=[]
    for t in range(350):
        st=sim.get_agent(0).get_state();obs=sim.get_sensor_observations()
        Image.fromarray(obs['rgb'][:,:,:3]).save(folder/f'{t:04d}.jpg',quality=85)
        try: action=follower.next_action_along(ep['goal_position'])
        except habitat_sim.errors.GreedyFollowerError: return {'success':False,'steps':t,'reason':'follower_failure'}
        before=np.array(st.position).copy();rot=quat_to_coeffs(st.rotation).tolist()
        if action is None:
            frames.append({'t':t,'image':f'{t:04d}.jpg','position':before.tolist(),'rotation_xyzw':rot,'command':'stop','collision':False})
            break
        perturbed=bool(rng.random()<.08)
        if perturbed: action=str(rng.choice(['move_forward','turn_left','turn_right']))
        step_obs=sim.step(action);after=sim.get_agent(0).get_state()
        frames.append({'t':t,'image':f'{t:04d}.jpg','position':before.tolist(),'rotation_xyzw':rot,'command':action,
                       'executed_translation_world':(np.array(after.position)-before).tolist(),
                       'next_rotation_xyzw':quat_to_coeffs(after.rotation).tolist(),
                       'collision':bool(step_obs['collided']),'perturbed':perturbed})
    d=pathlength(sim,sim.get_agent(0).get_state().position,ep['goal_position'])
    good=d is not None and d<1
    dump(folder/'trajectory.json',{'episode_id':ep['episode_id'],'split':'train','provenance':'privileged_geodesic_expert_with_8pct_action_perturbation',
                                'privileged_labels_not_policy_inputs':True,'success':good,'frames':frames})
    return {'success':good,'steps':len(frames)}
def buildscene(scene,split,assets,out,pilot=False):
    cfg=assets/'hssd-hab.scene_dataset_config.json';sim=opensim(scene,cfg);rng=np.random.default_rng(stable(scene+'v1'))
    images=out/'images'/scene;images.mkdir(parents=True,exist_ok=True)
    place=[];instances=[];targets=[]
    try:
        ng=3 if pilot else (50 if split=='train' else 30);ns=10 if pilot else (40 if split=='train' else 20)
        for g in range(ng):
            q=np.array(sim.pathfinder.get_random_navigable_point());yaw=float(rng.uniform(-math.pi,math.pi));ip=images/f'place_{g:04d}.jpg'
            cam=photo(sim,q,yaw,None,ip,None)
            if cam is None:continue
            for j,(p,d,sy) in enumerate(startpoints(sim,q,ns,rng)):
                place.append({'episode_id':f'hssd-place-{scene}-{g}-{j}','task':'place_image_nav','split':split,'scene_id':scene,
                              'dataset_revision':REV,'start_position':p.tolist(),'start_yaw':sy,'goal_position':q.tolist(),
                              'goal_camera':cam,'goal_image':str(ip.relative_to(out)),**classify(d,p,q),**viewgap(sy,yaw)})
        objs=sim.get_rigid_object_manager().get_objects_by_handle_substring('')
        for i,(handle,obj) in enumerate(sorted(objs.items()),1):obj.semantic_id=i
        meta=json.loads((assets/'metadata/objects.json').read_text())
        csv_path=assets/'metadata/fpmodels-with-decomposed.csv'
        taxonomy={r['id']:r['main_category'] for r in csv.DictReader(csv_path.open()) if r.get('main_category')} if csv_path.exists() else {}
        mapping={'chair':'chair','armchair':'chair','stool':'chair','couch':'couch','sofa':'couch','bed':'bed','bed_small':'bed','plant':'plant','plant_art':'plant','potted_plant':'plant','toilet':'toilet','tv':'tv','television':'tv','led_tv':'tv'}
        categories={h:(mapping.get(taxonomy[h.split('_:')[0]]) if h.split('_:')[0] in taxonomy else object_category(meta.get(h.split('_:')[0],{}).get('name',''))) for h in objs}
        for handle,obj in sorted(objs.items()):
            base=handle.split('_:')[0];name=meta.get(base,{}).get('name','').lower()
            category=categories[handle]
            if category is None:continue
            box=habitat_sim.geo.get_transformed_bb(obj.root_scene_node.cumulative_bb,obj.transformation)
            center=np.array(box.center());valid=[];views=[]
            for angle in np.linspace(-math.pi,math.pi,16,endpoint=False):
                for radius in [1.,1.6,2.4]:
                    sample=center+np.array([radius*math.sin(angle),0,radius*math.cos(angle)])
                    p=sim.pathfinder.snap_point(sample)
                    if not np.all(np.isfinite(p)) or np.linalg.norm(np.array(p)[[0,2]]-sample[[0,2]])>.4:continue
                    if abs(float(p[1])-float(box.min.y))>1.2:continue
                    yaw=math.atan2(-(center[0]-p[0]),-(center[2]-p[2]))
                    img=images/f'instance_{obj.semantic_id:04d}_{len(views):02d}.jpg'
                    cam=photo(sim,p,yaw,center,img,obj.semantic_id)
                    if cam is None:continue
                    near=np.maximum(np.array(box.min)[[0,2]]-np.array(p)[[0,2]],0)+np.minimum(np.array(box.max)[[0,2]]-np.array(p)[[0,2]],0)
                    if np.linalg.norm(near)<=1.0:valid.append(cam)
                    views.append({'camera':cam,'image':str(img.relative_to(out))})
            if not valid or len(views)<2:continue
            iid=f'{scene}:{handle}';target={'instance_id':iid,'semantic_id':obj.semantic_id,'category':category,'category_source':'author_main_category' if base in taxonomy else 'strict_asset_name_rule','asset_id':base,'views':views,
                                          'valid_approach_views':valid,'aabb_min':list(box.min),'aabb_max':list(box.max)}
            targets.append(target)
            for vi,view in enumerate(views[:3]):
                q=valid[0]['position'];n=5 if pilot else (30 if split=='train' else 15)
                for j,(p,d,sy) in enumerate(startpoints(sim,q,n,rng)):
                    distances=[pathlength(sim,p,v['position']) for v in valid]
                    best=min((k for k,distance in enumerate(distances) if distance is not None),key=lambda k:distances[k]);dd=distances[best]
                    if not 1.5<=dd<=30:continue
                    instances.append({'episode_id':f'hssd-instance-{scene}-{obj.semantic_id}-{vi}-{j}','task':'instance_image_nav','split':split,
                                      'scene_id':scene,'dataset_revision':REV,'instance_id':iid,'category':category,'start_position':p.tolist(),
                                      'start_yaw':sy,'goal_image':view['image'],'goal_camera':view['camera'],'valid_approach_views':valid,
                                      'same_category_distractors':[h for h,o in objs.items() if h!=handle and categories[h]==category],
                                      **classify(dd,p,valid[best]['position']),**viewgap(sy,view['camera']['yaw'])})
            if len(targets)>=(3 if pilot else 24):break
        writeepisodes(out/'episodes'/split/f'{scene}.place.jsonl.gz',place)
        writeepisodes(out/'episodes'/split/f'{scene}.instance.jsonl.gz',instances)
        dump(out/'instances'/f'{scene}.json',targets)
        trajectories=[]
        if split=='train' and not pilot:
            for ep in place[:5]:trajectories.append(expert(sim,ep,out/'trajectories'/scene/ep['episode_id'],rng))
        result={'scene':scene,'split':split,'place_episodes':len(place),'instance_episodes':len(instances),'target_instances':len(targets),
                'navigable_area_m2':sim.pathfinder.navigable_area,'objects':len(objs),'expert_trajectories':trajectories,'status':'complete'}
        dump(out/'scene_statistics'/f'{scene}.json',result);print(json.dumps(result),flush=True)
    finally:sim.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--force',action='store_true');p.add_argument('--assets',type=Path);args=p.parse_args()
    assets=args.assets or ROOT/('raw/hssd-pilot' if args.pilot else 'raw/hssd-hab');out=ROOT/'datasets'/('hssd_pilot' if args.pilot else 'hssd_image_nav_v1')
    scenes=sorted(x.name.split('.scene_instance')[0] for x in (assets/'scenes').glob('*.json'))
    if args.pilot:splits={x:'debug_only' for x in scenes}
    else:
        import yaml
        official=yaml.safe_load((assets/'scene_splits.yaml').read_text());print('OFFICIAL_SPLIT_KEYS',list(official),flush=True)
        train=set(map(str,official['train']));val=set(map(str,official['val']))
        test=set(sorted(train,key=lambda s:stable(s+'test-v1'))[:max(10,len(train)//8)])
        splits={s:('test' if s in test else 'train' if s in train else 'val' if s in val else 'excluded') for s in scenes}
        dump(out/'split_manifest.json',{'source_revision':REV,'official_val_preserved':True,'local_test_reserved_from_official_train':sorted(test),'scenes':splits})
    dump(out/'protocol.json',{'name':'Derived HSSD place and instance image navigation v1','not_official_HM3D_IIN':True,
                              'observation':'egocentric RGB 256x256; optional depth separate track','privileged_labels_only':['pose','semantic_id','navmesh','target_instance_id'],
                              'agent_radius_m':.18,'agent_height_m':1.4,'camera_height_m':1.25,'hfov_deg':90,
                              'actions':{'forward_m':.25,'turn_deg':30,'stop':'explicit'},'place_success':'STOP and geodesic distance <1m',
                              'instance_success':'STOP within 0.5m geodesic of a validated visible approach view; exact instance required',
                              'additional_metrics':['SR','SPL','SoftSPL','collision_rate','wrong_instance_stop_rate','view_gap_bins','detour_bins'],
                              'goal_masks':'supervision/diagnosis only; never policy input by default'})
    for scene,split in splits.items():
        if split=='excluded' or ((out/'scene_statistics'/f'{scene}.json').exists() and not args.force):continue
        try:buildscene(scene,split,assets,out,args.pilot)
        except Exception as e:
            dump(out/'failures'/f'{scene}.json',{'scene':scene,'error':str(e),'traceback':traceback.format_exc()});print('SCENE_FAILED',scene,str(e),flush=True)
    stats=[json.loads(f.read_text()) for f in (out/'scene_statistics').glob('*.json')]
    processed={s['scene'] for s in stats}
    summary={'processed_scenes':len(stats),'failed_scenes':sum(f.stem not in processed for f in (out/'failures').glob('*.json')),'place_episodes':sum(s['place_episodes'] for s in stats),
             'instance_episodes':sum(s['instance_episodes'] for s in stats),'target_instances':sum(s['target_instances'] for s in stats),'pilot':args.pilot}
    dump(out/'summary.json',summary);print('BUILD_SUMMARY',summary,flush=True)
    if not args.pilot and (summary['failed_scenes'] or summary['processed_scenes']!=sum(s!='excluded' for s in splits.values())):raise SystemExit('Incomplete scene coverage; inspect failures')
if __name__=='__main__':main()
