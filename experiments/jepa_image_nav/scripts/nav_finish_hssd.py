"""Isolated scene recovery; preserve completed scenes and the frozen split."""
import json,sys,subprocess,hashlib,time,os,resource
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014');OUT=ROOT/'datasets/hssd_image_nav_v1'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(item):
    scene,split=item
    code='from nav_build import buildscene,ROOT;buildscene('+repr(scene)+','+repr(split)+',ROOT/"raw/hssd-hab-compat",ROOT/"datasets/hssd_image_nav_v1")'
    env=dict(os.environ,PYTHONPATH=str(ROOT/'tools'),MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')
    with (ROOT/'logs'/f'recovery_{scene}.log').open('w') as log:
        try:
            p=subprocess.run([sys.executable,'-u','-c',code],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            result={'scene':scene,'returncode':p.returncode,'completed':p.returncode==0 and (OUT/'scene_statistics'/f'{scene}.json').exists()}
        except subprocess.TimeoutExpired:result={'scene':scene,'completed':False,'reason':'timeout_1800s'}
    print('RECOVERY_SCENE',json.dumps(result),flush=True);return result
def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    splits=json.loads((OUT/'split_manifest.json').read_text())['scenes']
    before={str(p.relative_to(OUT)):digest(p) for p in (OUT/'scene_statistics').glob('*.json')}
    before['split_manifest.json']=digest(OUT/'split_manifest.json')
    pending=[(s,sp) for s,sp in splits.items() if sp!='excluded' and not (OUT/'scene_statistics'/f'{s}.json').exists()]
    (ROOT/'manifests/recovery_baseline.json').write_text(json.dumps({'frozen_files':before,'pending_scenes':[s for s,sp in pending]},indent=2))
    print('PENDING_SCENES',len(pending),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,pending))
    unchanged=all(digest(OUT/p)==sha for p,sha in before.items())
    (ROOT/'manifests/recovery_execution.json').write_text(json.dumps({'scenes':results,'original_156_scene_stats_and_split_unchanged':unchanged},indent=2))
    assert unchanged
    if any(not r['completed'] for r in results):raise SystemExit('Recovery incomplete; inspect per-scene logs')
    stats=[json.loads(p.read_text()) for p in (OUT/'scene_statistics').glob('*.json')]
    summary={'processed_scenes':len(stats),'failed_scenes':0,'place_episodes':sum(s['place_episodes'] for s in stats),'instance_episodes':sum(s['instance_episodes'] for s in stats),
             'target_instances':sum(s['target_instances'] for s in stats),'split_task_counts':{sp:{'scenes':sum(s['split']==sp for s in stats),'place':sum(s['place_episodes'] for s in stats if s['split']==sp),'instance':sum(s['instance_episodes'] for s in stats if s['split']==sp)} for sp in ['train','val','test']},
             'material_compatibility_manifest':'material_compatibility.json','pilot':False}
    assert len(stats)==len(splits)==168
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('RECOVERY_COMPLETE',summary,flush=True)
if __name__=='__main__':main()
