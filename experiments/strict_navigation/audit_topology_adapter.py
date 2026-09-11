"""Finite CPU-only random/no-op checks for custom topology -> strict evaluation interface."""
import json,os,subprocess,time,hashlib
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');D=R/'topology_dev_v2';PY=str(R.parent/'venv/bin/python')
catalog=json.loads((D/'catalog.json').read_text());assert catalog['passed']
records=[]
for spec in catalog['maps']:
 for mode in ['random','noop']:
  tag=f"topology_adapter_{spec['id']}_{mode}_v2";out=R/'runs'/tag
  log=R/'logs'/(tag+'.log');env=os.environ.copy()
  env.update(CUDA_VISIBLE_DEVICES='',LEWM_EVAL_DEVICE='cpu',LEWM_RUN_TAG=tag,LEWM_CASE_FILE=str(D/(spec['id']+'_b0.json')),LEWM_CASE_LIMIT='2',LEWM_VIDEO='0')
  if mode=='noop':env['LEWM_NOOP']='1'
  args=[PY,str(R/'eval_topology.py'),'--config-name','tworoom','policy=random','seed=42','eval.num_eval=2','eval.dataset_name=tworoom','eval.goal_offset_steps=75','eval.eval_budget=10','plan_config.horizon=1','plan_config.receding_horizon=1',f'output.filename={out}/result.txt']
  with log.open('x') as f:run=subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
  assert run.returncode==0,(tag,run.returncode)
  read=lambda name:json.loads((out/name).read_text())
  meta=read('case_metadata.json');rows=read('cases_results.json');a=read('map_adapter_audit.json')
  assert len(rows)==len(a)==len(meta['cases'])==2
  assert all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in a)
  trace=np.load(out/'trace.npz');assert all(np.isfinite(trace[k]).all() for k in trace.files)
  if mode=='noop':assert all(x['path_length']<1e-5 and not x['success'] for x in rows)
  records.append({'tag':tag,'passed':True,'cases':2,'mode':mode,'case_sha256':meta['case_sha256']})
  print('ADAPTER_PASSED',tag,flush=True)
payload={'passed':True,'updated_unix':time.time(),'checks':records,'limits':'Random/no-op CPU interface only; no learned-model topology result.'}
(D/'adapter_audit.json').write_text(json.dumps(payload,indent=2));print('TOPOLOGY_ADAPTER_AUDIT_PASSED',flush=True)
