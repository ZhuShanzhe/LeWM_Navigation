"""Fetch a complete rigid-scene dependency closure before the full snapshot."""
import json, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from nav_fetch import ROOT, DEST, REV, get
SID='102344115'
data=json.loads((ROOT/'manifests/hssd_api.json').read_text())
files=[f['rfilename'] for f in data['siblings']]
scene=DEST/f'scenes/{SID}.scene_instance.json'
get(f'scenes/{SID}.scene_instance.json')
d=json.loads(scene.read_text())
handles={x['template_name'] for x in d['object_instances']}
selected=[p for p in files if (p.startswith('objects/') and any(Path(p).name.startswith(h+'.') for h in handles)) or (p.startswith('stages/') and Path(p).name.startswith(SID+'.'))]
selected+=['semantics/hssd-hab_semantic_lexicon.json','metadata/objects.json','metadata/fpmodels-with-decomposed.csv',f'scenes/{SID}.scene_instance.json']
print('PILOT_FILES',len(selected),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(get,selected))
bad=[x for x in results if isinstance(x,dict)]
if bad: print(bad,flush=True);raise SystemExit(1)
pilot=ROOT/'raw/hssd-pilot'
for p in selected:
    dest=pilot/p;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(DEST/p,dest)
cfg={'stages':{'paths':{'.json':['stages']}},'objects':{'paths':{'.json':['objects/*','objects/decomposed/*']}},'scene_instances':{'paths':{'.json':['scenes']}},'semantic_scene_descriptor_instances':{'hssd_ssd_map':'semantics/hssd-hab_semantic_lexicon.json'}}
(pilot/'hssd-hab.scene_dataset_config.json').write_text(json.dumps(cfg,indent=2))
print('PILOT_READY',SID,flush=True)
