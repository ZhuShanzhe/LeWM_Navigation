"""Test whether dynamic prediction adds value to retrieval on four validation geometries."""
import json,copy
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
j=json.loads((R/'jobs.json').read_text());tags={x['tag'] for x in j};extra=[]
for base in list(j):
 if not base['tag'].startswith('strict3072_map_validation_') or '_temporal_' in base['tag'] or '_retrieval_' in base['tag']:continue
 for mode in ['only','rerank','init']:
  tag=base['tag'].rsplit('_b',1)[0]+f'_retrieval_{mode}_b'+base['tag'].rsplit('_b',1)[1]
  if tag in tags:continue
  x=copy.deepcopy(base);x['tag']=tag
  x['argv']=[a.replace(str(R/'eval_strict.py'),str(R/'eval_retrieval.py')) for a in x['argv']]
  x['argv']=[('output.filename='+str(R/'runs'/tag/'result.txt')) if a.startswith('output.filename=') else ('solver.n_steps=1' if mode=='rerank' and a.startswith('solver.n_steps=') else a) for a in x['argv']]
  x['env'].update(LEWM_RETRIEVAL_LIBRARY=str(R/'priors/retrieval3072/library.pt'),LEWM_RETRIEVAL_MODE=mode);extra.append(x)
idx=next(i for i,x in enumerate(j) if x['tag']=='strict_tw_s3074');j[idx:idx]=extra
assert len({x['tag'] for x in j})==len(j)
tmp=R/'jobs.tmp';tmp.write_text(json.dumps(j,indent=2));tmp.replace(R/'jobs.json')
p=R/'geometry_result_report.py';s=p.read_text().replace("or 'temporal' in key:continue","or 'temporal' in key or '_retrieval_' in key:continue");p.write_text(s)
print({'added':len(extra),'queue':len(j)})
