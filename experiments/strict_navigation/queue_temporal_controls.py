"""Queue validated lightweight temporal-head adaptation and independent-seed planning controls."""
import json,copy
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R.parent/'venv/bin/python'
jobs=json.loads((R/'jobs.json').read_text());tags={j['tag'] for j in jobs}
def clone(base,tag):
 j=copy.deepcopy(next(x for x in jobs if x['tag']==base));j['tag']=tag
 j['argv']=[('output.filename='+str(R/'runs'/tag/'result.txt')) if a.startswith('output.filename=') else a for a in j['argv']]
 return j
head=[]
tag='fit_temporal3072'
if tag not in tags:head.append({'tag':tag,'watchdog_seconds':7200,'argv':[str(P),str(R/'fit_temporal_head.py'),'temporal3072',str(R.parent/'data/checkpoints/strict_tw_s3072_v2/weights_final.pt')],'env':{}})
tag='smoke_temporal_formal'
if tag not in tags:
 j=clone('smoke_model_posttrain',tag);j['env']['LEWM_TEMPORAL_HEAD']=str(R/'heads/temporal3072/temporal.pt');head.append(j)
for kind in ['temporal','shuffled']:
 for mode,variant in [('replace','long_base'),('hybrid','long_base'),('replace','short150')]:
  for b in [0,1]:
   tag=f'strict3072_validation_{variant}_{kind}_{mode}_b{b}'
   if tag in tags:continue
   j=clone(f'strict3072_validation_{variant}_b{b}',tag);j['env'].update(LEWM_TEMPORAL_HEAD=str(R/'heads/temporal3072'/(kind+'.pt')),LEWM_HEAD_COST=mode);head.append(j)
ix=next(i for i,j in enumerate(jobs) if j['tag']=='smoke_map_model_posttrain');jobs[ix:ix]=head
# Add only pre-specified continuity contrasts for independent model seeds, before final summary.
extra=[]
for seed in [3073,3074]:
 for variant in ['long_feedback_extra','long_feedback_extra_warm','long_h10_budget','long_h10_budget_warm']:
  for b in [0,1]:
   tag=f'strict{seed}_validation_{variant}_b{b}'
   if tag in tags:continue
   j=clone(f'strict3072_validation_{variant}_b{b}',tag)
   j['argv']=[('policy='+str(R.parent/f'data/checkpoints/strict_tw_s{seed}/weights_final.pt')) if a.startswith('policy=') else a for a in j['argv']]
   extra.append(j)
# Ensure both checkpoints exist by placing these after the second new seed baseline jobs.
ix=next(i for i,j in enumerate(jobs) if j['tag']=='summarize_three_seed_baseline');jobs[ix:ix]=extra
tmp=R/'jobs.tmp';tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(R/'jobs.json')
print(json.dumps({'head_jobs_added':len(head),'continuity_replications_added':len(extra),'queue':len(jobs)}))
