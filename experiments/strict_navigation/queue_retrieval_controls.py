"""Finite validation-only retrieval controls, after initial map baseline."""
import json,copy
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
assert 'RETRIEVAL_CPU_SMOKES_PASSED' in (R/'logs/smoke_retrieval_driver_v2.log').read_text()
jobs=json.loads((R/'jobs.json').read_text());tags={x['tag'] for x in jobs};extra=[]
def clone(base,tag,mode=None,smoke=False):
 j=copy.deepcopy(next(x for x in jobs if x['tag']==base));j['tag']=tag
 replace={'output.filename':str(R/'runs'/tag/'result.txt')}
 if smoke:replace.update({'plan_config.horizon':'5','plan_config.receding_horizon':'5','eval.eval_budget':'25'})
 if mode=='rerank' or mode=='gaussian1':replace['solver.n_steps']='1'
 j['argv']=[a.split('=',1)[0]+'='+replace[a.split('=',1)[0]] if a.split('=',1)[0] in replace else a for a in j['argv']]
 if mode in ['only','rerank','init']:
  j['argv']=[a.replace(str(R/'eval_strict.py'),str(R/'eval_retrieval.py')) for a in j['argv']]
  j['env'].update(LEWM_RETRIEVAL_LIBRARY=str(R/'priors/retrieval3072/library.pt'),LEWM_RETRIEVAL_MODE=mode)
 return j
for mode in ['only','rerank','init']:
 tag='smoke_retrieval_gpu_'+mode
 if tag not in tags:extra.append(clone('smoke_model_posttrain',tag,mode,True))
for variant in ['long_base','short150']:
 for mode in ['only','rerank','init','gaussian1']:
  for b in [0,1]:
   tag=f'strict3072_validation_{variant}_retrieval_{mode}_b{b}'
   if tag not in tags:extra.append(clone(f'strict3072_validation_{variant}_b{b}',tag,mode))
ix=next(i for i,j in enumerate(jobs) if j['tag']=='summarize_stage1')
jobs[ix:ix]=extra
tmp=R/'jobs.tmp';tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(R/'jobs.json')
print({'added':len(extra),'queue':len(jobs)})
