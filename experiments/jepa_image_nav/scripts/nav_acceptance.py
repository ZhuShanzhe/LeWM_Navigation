"""Full metadata/image audit plus every-episode geodesic replay and live instance visibility.

This checks dataset validity, not a learned policy's navigation performance.
Each scene runs in an isolated process; failures cannot be hidden by subset success.
"""
import os,sys,json,gzip,hashlib,subprocess,resource,traceback,argparse,time,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014');REPORT=ROOT/'reports/acceptance_20261015'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(dataset,scene):
    import numpy as np
    from PIL import Image
    from nav_build import opensim,pathlength,setpose
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    out=ROOT/'datasets'/dataset;issues=[];checked=0;geo=0;live=0;imgs=0;tasks={};image_cache=set()
    is_hssd=dataset=='hssd_image_nav_v1'
    assets=ROOT/'raw/hssd-hab-compat'
    sim=opensim(scene,assets/'hssd-hab.scene_dataset_config.json') if is_hssd else opensim(str(ROOT/'raw/scene_datasets/gibson'/f'{scene}.glb'),None)
    try:
        targets={}
        if is_hssd:
            objs=sim.get_rigid_object_manager().get_objects_by_handle_substring('')
            for sid,(handle,obj) in enumerate(sorted(objs.items()),1):obj.semantic_id=sid
            records=json.loads((out/'instances'/f'{scene}.json').read_text())
            targets={t['instance_id']:t for t in records}
            if len(targets)!=len(records):issues.append('duplicate target instance')
            for iid,t in targets.items():
                handle=iid.split(':',1)[1]
                if handle not in objs or objs[handle].semantic_id!=t['semantic_id']:issues.append('instance identity mismatch '+iid);continue
                if len(t['views'])<2 or not t['valid_approach_views']:issues.append('insufficient views '+iid)
                for view in t['views']:
                    p=out/view['image']
                    if p not in image_cache:
                        a=np.asarray(Image.open(p).convert('RGB'));m=np.asarray(Image.open(p.with_suffix('.mask.png')))>0
                        if a.shape!=(256,256,3) or a.std()<5:issues.append('invalid declared instance view '+view['image'])
                        if m.shape!=(256,256) or m.mean()<.025:issues.append('invalid declared instance mask '+view['image'])
                        image_cache.add(p);imgs+=1
                for cam in t['valid_approach_views']:
                    if not sim.pathfinder.is_navigable(np.asarray(cam['position'],dtype=np.float32)):issues.append('approach not navigable '+iid)
                    obs=setpose(sim,cam['position'],cam['yaw'],cam['pitch'])
                    live+=1
                    if (obs['semantic']==t['semantic_id']).mean()<.025:issues.append('live instance not visible '+iid)
        seen=set()
        for epfile in sorted((out/'episodes').glob(f'*/{scene}*.jsonl.gz')):
            for line in gzip.open(epfile,'rt'):
                e=json.loads(line);checked+=1;eid=e['episode_id'];tasks[e['task']]=tasks.get(e['task'],0)+1
                if eid in seen:issues.append('duplicate episode '+eid)
                seen.add(eid)
                if e['scene_id']!=scene:issues.append('scene mismatch '+eid)
                if not sim.pathfinder.is_navigable(np.asarray(e['start_position'],dtype=np.float32)):issues.append('start not navigable '+eid)
                if e['task']=='place_image_nav':
                    dist=pathlength(sim,e['start_position'],e['goal_position']);geo+=1
                    if not np.allclose(e['goal_position'],e['goal_camera']['position'],atol=1e-6):issues.append('goal camera mismatch '+eid)
                else:
                    t=targets.get(e['instance_id'])
                    if t is None:issues.append('unknown target '+eid);continue
                    if e['valid_approach_views']!=t['valid_approach_views']:issues.append('approach set mismatch '+eid)
                    if e['goal_image'] not in {v['image'] for v in t['views']}:issues.append('goal instance view mismatch '+eid)
                    ds=[pathlength(sim,e['start_position'],c['position']) for c in t['valid_approach_views']];geo+=len(ds)
                    ds=[d for d in ds if d is not None];dist=min(ds) if ds else None
                if dist is None or not 1.5-1e-5<=dist<=30+1e-5:issues.append('unreachable/out-of-range '+eid)
                elif abs(dist-e['geodesic_m'])>.02:issues.append('geodesic mismatch '+eid)
        stat=json.loads((out/'scene_statistics'/f'{scene}.json').read_text())
        expected=(stat['place_episodes']+stat['instance_episodes']) if is_hssd else stat['episodes']
        if checked!=expected:issues.append('scene count mismatch')
        if checked==0:issues.append('empty scene')
        result={'dataset':dataset,'scene':scene,'episodes_checked':checked,'task_counts':tasks,'shortest_path_queries':geo,
                'live_approach_views_checked':live,'declared_instance_images_checked':imgs,'issue_count':len(issues),'issues':issues[:80],'passed':not issues}
    finally:sim.close()
    dest=REPORT/dataset;dest.mkdir(parents=True,exist_ok=True);(dest/f'{scene}.json').write_text(json.dumps(result,indent=2))
    return result
def run(item):
    dataset,scene=item;logs=ROOT/'logs/acceptance';logs.mkdir(exist_ok=True)
    env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',PYTHONUNBUFFERED='1')
    with (logs/f'{dataset}_{scene}.log').open('w') as f:
        try:p=subprocess.run([sys.executable,__file__,'--worker',dataset,scene],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=1200)
        except subprocess.TimeoutExpired:return {'dataset':dataset,'scene':scene,'passed':False,'reason':'timeout'}
    path=REPORT/dataset/f'{scene}.json'
    if p.returncode or not path.exists():return {'dataset':dataset,'scene':scene,'passed':False,'reason':'worker_failed','returncode':p.returncode}
    return json.loads(path.read_text())
def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));REPORT.mkdir(parents=True,exist_ok=True)
    from nav_audit import audit
    metadata={}
    for dataset in ['gibson_place_nav_v1','hssd_image_nav_v1']:
        metadata[dataset]=audit(ROOT/'datasets'/dataset)
    pending=[]
    for dataset in metadata:
        splits=json.loads((ROOT/'datasets'/dataset/'split_manifest.json').read_text())['scenes']
        assert len(splits)==(168 if dataset.startswith('hssd') else 86)
        pending.extend((dataset,s) for s,sp in splits.items() if sp!='excluded')
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,item) for item in pending]
        for future in as_completed(futures):
            r=future.result();results.append(r)
            print('ACCEPT_SCENE',len(results),'/',len(pending),r['dataset'],r['scene'],r['passed'],flush=True)
    baseline=json.loads((ROOT/'manifests/recovery_baseline.json').read_text())['frozen_files']
    frozen_ok=all(digest(ROOT/'datasets/hssd_image_nav_v1'/p)==h for p,h in baseline.items())
    material=json.loads((ROOT/'manifests/material_compatibility.json').read_text())
    source_ok=digest(ROOT/'raw/hssd-hab'/material['asset'])==material['original_sha256']
    repair=json.loads((ROOT/'manifests/goal_repair.json').read_text())
    # Compare all original rows to ensure the twenty image references are the ONLY pack change.
    old=[json.loads(x) for x in gzip.open(Path(repair['backup'])/'105515307_173104317.place.jsonl.gz','rt')]
    new=[json.loads(x) for x in gzip.open(ROOT/'datasets/hssd_image_nav_v1/episodes/test/105515307_173104317.place.jsonl.gz','rt')]
    changed=[]
    assert len(old)==len(new)
    for a,b in zip(old,new):
        if a!=b:
            a=dict(a);b=dict(b);a.pop('goal_image');b.pop('goal_image');assert a==b;changed.append(b['episode_id'])
    report={'date':'2026-10-15','passed':all(x['metadata_image_audit_passed'] for x in metadata.values()) and len(results)==254 and all(r['passed'] for r in results) and frozen_ok and source_ok and len(changed)==20,
            'metadata_image_audits':metadata,'scene_replay_counts':{ds:sum(r['dataset']==ds and r['passed'] for r in results) for ds in metadata},
            'episodes_geometrically_rechecked':sum(r.get('episodes_checked',0) for r in results),
            'shortest_path_queries':sum(r.get('shortest_path_queries',0) for r in results),
            'live_instance_approach_views_checked':sum(r.get('live_approach_views_checked',0) for r in results),
            'declared_instance_images_checked':sum(r.get('declared_instance_images_checked',0) for r in results),
            'original_scene_stats_and_split_unchanged':frozen_ok,'pinned_original_material_unchanged':source_ok,'only_20_goal_references_changed':len(changed)==20,
            'failed_scene_reports':[r for r in results if not r['passed']],
            'scope':'All published episodes geometrically rechecked; all declared instance views image/mask checked; all valid instance approach views live-rendered. Not learned-policy evaluation or per-episode full action execution.'}
    (REPORT/'full_acceptance.json').write_text(json.dumps(report,indent=2));print('FINAL_ACCEPTANCE',report['passed'],flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],sys.argv[3])
    else:main()
