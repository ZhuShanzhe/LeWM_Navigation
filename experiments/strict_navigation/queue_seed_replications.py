"""Append two independent training seeds and fixed baseline replications; no sweep."""
import json,copy,time
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
jobs=json.loads((R/'jobs.json').read_text());tags={j['tag'] for j in jobs};added=[]
for seed in [3073,3074]:
 tag=f'strict_tw_s{seed}'
 if tag not in tags:
  j=copy.deepcopy(next(j for j in jobs if j['tag']=='strict_tw_s3072_v2'))
  j.pop('retry_of',None);j.pop('retry_reason',None);j['tag']=tag
  change={'seed':str(seed),'subdir':tag,'output_model_name':tag}
  j['argv']=[a.split('=',1)[0]+'='+change[a.split('=',1)[0]] if a.split('=',1)[0] in change else a for a in j['argv']]
  added.append(j)
 for split,count in [('validation',2),('test',6)]:
  for v in ['short150','long_base']:
   for b in range(count):
    newtag=f'strict{seed}_{split}_{v}_b{b}'
    if newtag in tags:continue
    j=copy.deepcopy(next(j for j in jobs if j['tag']==f'strict3072_{split}_{v}_b{b}'));j['tag']=newtag
    j['argv']=[('policy='+str(R.parent/'data/checkpoints'/tag/'weights_final.pt')) if a.startswith('policy=') else ('output.filename='+str(R/'runs'/newtag/'result.txt')) if a.startswith('output.filename=') else a for a in j['argv']]
    added.append(j)
tag='summarize_three_seed_baseline'
if tag not in tags:added.append({'tag':tag,'watchdog_seconds':300,'argv':[str(R.parent/'venv/bin/python'),str(R/'summarize_navigation.py')],'env':{}})
jobs.extend(added);tmp=R/'jobs.tmp';tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(R/'jobs.json')
print(json.dumps({'added':len(added),'queue':len(jobs),'note':'Independent seeds use identical strict split, objectives and fixed baseline cases.'}))
