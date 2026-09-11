import os,json,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
jobs=json.loads((R/'jobs.json').read_text())
base=next(x for x in jobs if x['tag']=='smoke_model_posttrain')
for mode in ['only','rerank','init']:
 tag='smoke_retrieval_cpu_v2_'+mode
 argv=[a.replace(str(R/'eval_strict.py'),str(R/'eval_retrieval.py')) for a in base['argv']]
 replace={'eval.eval_budget':'25','plan_config.horizon':'5','plan_config.receding_horizon':'5','output.filename':str(R/'runs'/tag/'result.txt')}
 argv=[a.split('=',1)[0]+'='+replace[a.split('=',1)[0]] if a.split('=',1)[0] in replace else a for a in argv]+['solver.device=cpu']
 env={**os.environ,**base['env'],'LEWM_RUN_TAG':tag,'LEWM_EVAL_DEVICE':'cpu','LEWM_RETRIEVAL_MODE':mode,'LEWM_RETRIEVAL_LIBRARY':str(R/'priors/retrieval3072/library.pt')}
 with (R/'logs'/(tag+'.log')).open('w') as f:
  p=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
 print(tag,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
print('RETRIEVAL_CPU_SMOKES_PASSED',flush=True)
