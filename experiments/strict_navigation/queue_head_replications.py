"""Extend only validated temporal-head mechanism: independent models and validation geometries."""
import json,copy
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
jobs=json.loads((R/'jobs.json').read_text());tags={x['tag'] for x in jobs}
def clone(base,tag):
 j=copy.deepcopy(next(x for x in jobs if x['tag']==base));j['tag']=tag
 j['argv']=[('output.filename='+str(R/'runs'/tag/'result.txt')) if a.startswith('output.filename=') else a for a in j['argv']]
 return j
# Temporal-only replacement: do not choose hybrid merely for +1 validation success.
maps=[]
for base in list(jobs):
 if base['tag'].startswith('strict3072_map_validation_'):
  tag=base['tag'].rsplit('_b',1)[0]+'_temporal_replace_b'+base['tag'].rsplit('_b',1)[1]
  if tag not in tags:
   j=clone(base['tag'],tag);j['env'].update(LEWM_TEMPORAL_HEAD=str(R/'heads/temporal3072/temporal.pt'),LEWM_HEAD_COST='replace');maps.append(j)
ix=next(i for i,j in enumerate(jobs) if j['tag']=='summarize_stage1');jobs[ix:ix]=maps
extra=[]
for seed in [3073,3074]:
 tag=f'fit_temporal{seed}'
 if tag not in tags:
  extra.append({'tag':tag,'watchdog_seconds':7200,'argv':[str(R.parent/'venv/bin/python'),str(R/'fit_temporal_head.py'),f'temporal{seed}',str(R.parent/f'data/checkpoints/strict_tw_s{seed}/weights_final.pt')],'env':{}})
 for kind,mode,variant in [('temporal','replace','long_base'),('temporal','hybrid','long_base'),('shuffled','replace','long_base'),('shuffled','hybrid','long_base'),('temporal','replace','short150')]:
  for b in [0,1]:
   tag=f'strict{seed}_validation_{variant}_{kind}_{mode}_b{b}'
   if tag in tags:continue
   j=clone(f'strict3072_validation_{variant}_{kind}_{mode}_b{b}',tag)
   j['argv']=[('policy='+str(R.parent/f'data/checkpoints/strict_tw_s{seed}/weights_final.pt')) if a.startswith('policy=') else a for a in j['argv']]
   j['env']['LEWM_TEMPORAL_HEAD']=str(R/f'heads/temporal{seed}/{kind}.pt')
   extra.append(j)
ix=next(i for i,j in enumerate(jobs) if j['tag']=='summarize_three_seed_baseline');jobs[ix:ix]=extra
assert len({j['tag'] for j in jobs})==len(jobs)
tmp=R/'jobs.tmp';tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(R/'jobs.json')
print({'map_jobs_added':len(maps),'replication_jobs_added':len(extra),'queue':len(jobs)})
