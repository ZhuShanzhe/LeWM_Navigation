"""Repair controlled dataset paths; retain original failed job and artifacts."""
import json,copy,time,py_compile,hashlib
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent
p=R/'jobs.json';jobs=json.loads(p.read_text());state=json.loads((R/'status.json').read_text())
assert state['active'] is None
for variant in ['single','multi']:
 d=ROOT/'data'/f'nav_{variant}_control.h5'
 assert d.is_file() and d.stat().st_size>1000000
 print('DATA',variant,d.stat().st_size)
old='geometry_smoke_single_s4001';new=old+'_v2'
assert next(x for x in state['jobs'] if x['tag']==old)['status']=='failed'
if new not in {x['tag'] for x in jobs}:
 retry=copy.deepcopy(next(x for x in jobs if x['tag']==old))
 retry['tag']=new;retry['argv']=[a.replace(old,new) for a in retry['argv']]
 retry['note']='Path-resolution retry; original failure preserved. 20-update interface smoke only.'
 jobs.insert(next(i for i,x in enumerate(jobs) if x['tag']=='geometry_smoke_single_eval'),retry)
seen={x['tag'] for x in state['jobs']}
for j in jobs:
 if j['tag'] in seen:continue
 j['argv']=[a.replace('data.dataset.name=nav_single_control.h5',f'data.dataset.name={ROOT}/data/nav_single_control.h5').replace('data.dataset.name=nav_multi_control.h5',f'data.dataset.name={ROOT}/data/nav_multi_control.h5') for a in j['argv']]
 if j['tag']=='geometry_smoke_single_eval':
  j['argv']=[a.replace('/'+old+'/', '/'+new+'/') for a in j['argv']]
backup=R/'jobs.before_geometry_path_repair.json'
if not backup.exists():backup.write_bytes(p.read_bytes())
tmp=R/'jobs.geometry_path_repair.tmp';tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(p)
a=R/'audit_geometry_smokes.py';s=a.read_text()
s=s.replace("tr=R/'runs'/f'geometry_smoke_{variant}_s4001';", "tr=R/'runs'/(f'geometry_smoke_{variant}_s4001'+('_v2' if variant=='single' else ''));")
s=s.replace("'full_training_not_yet_authorized_by_this_audit':True", "'full_training_requires_stage_review':True")
a.write_text(s);py_compile.compile(str(a),doraise=True)
q=R/'queue_geometry_smokes.py';s=q.read_text()
s=s.replace("f'data.dataset.name=nav_{variant}_control.h5'", "f'data.dataset.name={ROOT}/data/nav_{variant}_control.h5'")
q.write_text(s);py_compile.compile(str(q),doraise=True)
w=ROOT/'data/checkpoints/strict_tw_s3073/weights_final.pt'
assert w.is_file() and w.stat().st_size>1000000
print('SEED3073_CHECKPOINT',w.stat().st_size,hashlib.sha256(w.read_bytes()).hexdigest())
print('REPAIRED',new,'jobs',len(jobs))
