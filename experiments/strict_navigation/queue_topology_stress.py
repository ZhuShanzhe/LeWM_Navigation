"""Queue finite three-topology stress tests after audited matched-engine bridge."""
import json,hashlib,time,copy
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent;D=R/'topology_dev_v2';P=R/'topology_stress_v1';PY=str(ROOT/'venv/bin/python')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bridge=read(R/'topology_bridge_v1/bridge_results.json');assert bridge['passed'] and read(R/'topology_bridge_v1/model_smoke_audit.json')['passed']
assert read(D/'catalog.json')['passed'];P.mkdir(exist_ok=False)
parent=read(R/'topology_bridge_v1/protocol.json');models=parent['models']
for v,m in models.items():assert sha(Path(m['weights']))==bridge['models'][v]['weights_sha256']
maps=['four_room_chain','four_room_cycle','six_room_branch'];jobs=[];files={}
for mapid in maps:
 for b in range(2):
  f=D/f'{mapid}_b{b}.json';files[str(f)]=sha(f)
 for v in ['single','multi','random','noop']:
  for b in range(2):
   tag=f'topology_stress_{v}_{mapid}_b{b}_v1';learned=v in models
   norm=models[v if learned else 'single']['normalization']
   args=[PY,str(R/'eval_topology.py'),'--config-name','tworoom','policy='+(models[v]['weights'] if learned else 'random'),'seed=42','eval.num_eval=50','eval.dataset_name=tworoom','eval.goal_offset_steps=75','eval.eval_budget=150','plan_config.horizon=5','plan_config.receding_horizon=5']
   if learned:args+=['solver.n_steps=10','solver.num_samples=300','solver.topk=30','+plan_config.warm_start=false']
   args+=[f'output.filename={R}/runs/{tag}/result.txt']
   env={'LEWM_CASE_FILE':str(D/f'{mapid}_b{b}.json'),'LEWM_CASE_LIMIT':'50','LEWM_VIDEO':'0','LEWM_NORMALIZATION_FILE':norm}
   if not learned:env.update(CUDA_VISIBLE_DEVICES='',LEWM_EVAL_DEVICE='cpu')
   if v=='noop':env['LEWM_NOOP']='1'
   jobs.append({'tag':tag,'argv':args,'env':env,'watchdog_seconds':2400 if learned else 600,'note':'Finite full-observation OOD geometry+topology stress, not pure topology causal effect, not final test.'})
jobs.append({'tag':'audit_topology_stress_v1','argv':[PY,str(R/'audit_topology_stress.py')],'env':{'CUDA_VISIBLE_DEVICES':''},'watchdog_seconds':600})
protocol={'created_unix':time.time(),'stage_review':{'bridge_paired':bridge['paired'],'decision':'Proceed only as standalone custom-simulator stress test, not equivalence or pure topology transfer. Bridge shifts -4pp and +6pp indicate simulator effects cannot be ignored.'},'maps':maps,'models':models,'frozen_model_sha256':{v:m['weights_sha256'] for v,m in bridge['models'].items()},'source_sha256':{str(R/f):sha(R/f) for f in ['eval_strict.py','eval_topology.py','topology_env.py','audit_executed_trace.py','audit_topology_stress.py']},'case_files_sha256':files,'bridge_results_sha256':sha(R/'topology_bridge_v1/bridge_results.json'),'jobs':jobs,'limits':['Custom full-observation simulator, not first-person or real robot.','Original and new simulators differ: bridge bounds observed combined collision/render shift, not equivalence.','New room graphs also change geometry, room scale and visual distribution; no isolated topology causal claim.','Each map100 fixed development cases20same/80cross; no topology data used to train either model.','One matched training seed4001, 8 training geometries for multi; map/seed uncertainty not captured by case bootstrap.','Low known-layout score means low topology score alone cannot localize topology reasoning failure.','No-op/random share150 physical steps, but not learned planning compute.','Reference paths feasible, not shortest; no SPL reported.','No memory or partial-observation experiment authorized by this protocol.']}
with (P/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
old=read(R/'jobs.json');known={x['tag'] for x in old};assert not known.intersection(x['tag'] for x in jobs)
state=read(R/'status.json');idx=next(i for i,x in enumerate(old) if x['tag']=='audit_topology_bridge_v1')
if state.get('active'):idx=max(idx,next(i for i,x in enumerate(old) if x['tag']==state['active']['tag']))
with (P/'jobs_before.json').open('x') as f:json.dump(old,f,indent=2)
new=old[:idx+1]+jobs+old[idx+1:];assert len(new)==len({x['tag'] for x in new})
tmp=R/'jobs.topology_stress.tmp';tmp.write_text(json.dumps(new,indent=2));tmp.replace(R/'jobs.json')
print(json.dumps({'added':len(jobs),'after':old[idx]['tag'],'total':len(new)}))
