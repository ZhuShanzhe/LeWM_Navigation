"""Prepare evaluation statistics routing for controlled-data models; defaults unchanged."""
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
p=R/'eval_strict.py';s=p.read_text()
old="stats=json.loads((R/'splits/normalization.json').read_text())"
new="stats_path=Path(os.environ.get('LEWM_NORMALIZATION_FILE',str(R/'splits/normalization.json')))\nstats=json.loads(stats_path.read_text())"
if old in s:s=s.replace(old,new)
old="'normalization':'strict training set only','action_constraint':"
new="'normalization':'strict training set only','normalization_path':str(stats_path),'normalization_sha256':hashlib.sha256(stats_path.read_bytes()).hexdigest(),'action_constraint':"
if old in s:s=s.replace(old,new)
p.write_text(s)
import json
jobs=json.loads((R/'jobs.json').read_text());tag='audit_geometry_collections'
if not any(j['tag']==tag for j in jobs):
 idx=next(i for i,j in enumerate(jobs) if j['tag']=='strict_tw_s3073')+1
 jobs.insert(idx,{'tag':tag,'watchdog_seconds':1200,'argv':[str(R.parent/'venv/bin/python'),str(R/'audit_geometry_collections.py')],'env':{'CUDA_VISIBLE_DEVICES':''},'note':'Both finite CPU collectors must finish and pass full-data audit before controlled-data model training.'})
 tmp=R/'jobs.tmp';tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(R/'jobs.json')
print('normalization routing prepared; queue',len(jobs))
