"""Queue only four oracle-router integration smokes and audit, no full diagnostic yet."""
from pathlib import Path
import json,time,hashlib,copy
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'oracle_waypoint_v1';PY=str(R.parent/'venv/bin/python')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'topology_stress_v1/results.json')['passed'] and read(R/'reports/oracle_waypoint_cpu_audit.json')['passed']
P.mkdir(exist_ok=False);parent=read(R/'topology_bridge_v1/protocol.json');models=parent['models'];bridge=read(R/'topology_bridge_v1/bridge_results.json')
old=read(R/'jobs.json');jobs=[];specs=[];baseline={}
case=R/'topology_bridge_v1/four_room_chain_smoke.json'
for v in models:
 baseline[v]=f'topology_model_smoke_{v}_four_room_chain_v1'
 base=next(x for x in old if x['tag']==baseline[v])
 for mode in ['final','waypoint']:
  j=copy.deepcopy(base);tag=f'oracle_router_smoke_{v}_{mode}_v1';j['tag']=tag
  j['argv'][1]=str(R/'eval_oracle_waypoints.py')
  j['argv']=[f'output.filename={R}/runs/{tag}/result.txt' if x.startswith('output.filename=') else x for x in j['argv']]
  j['env']['LEWM_ROUTE_MODE']=mode;j['note']='Privileged oracle router interface; final mode must reproduce unchanged baseline.'
  jobs.append(j);specs.append({'tag':tag,'model':v,'mode':mode,'case_file':str(case)})
jobs.append({'tag':'audit_oracle_router_smokes_v1','argv':[PY,str(R/'audit_oracle_router_smokes.py')],'env':{'CUDA_VISIBLE_DEVICES':''},'watchdog_seconds':600})
q={'created_unix':time.time(),'models':models,'model_sha256':{v:m['weights_sha256'] for v,m in bridge['models'].items()},'baseline_tags':baseline,'case_sha256':sha(case),'source_sha256':{str(R/f):sha(R/f) for f in ['oracle_waypoint_router.py','eval_oracle_waypoints.py','audit_oracle_router_smokes.py','eval_topology.py','eval_strict.py','topology_env.py','audit_executed_trace.py']},'smoke_jobs':specs,'jobs':jobs,'formal_evaluation_not_queued':True,'next_protocol_proposal':'After interface review: global versus true doorway entry/exit waypoints at identical H5/R1,2 CEM iterations,300 samples,150 physical steps. True pose/graph is privileged, no memory claim. Actual work and extra render/encoding cost must be reported.'}
with (P/'protocol.json').open('x') as f:json.dump(q,f,indent=2)
assert not {x['tag'] for x in old}.intersection(x['tag'] for x in jobs)
state=read(R/'status.json');anchor=state['active']['tag'] if state.get('active') else old[-1]['tag']
i=next(i for i,x in enumerate(old) if x['tag']==anchor)
with (P/'jobs_before.json').open('x') as f:json.dump(old,f,indent=2)
new=old[:i+1]+jobs+old[i+1:];assert len(new)==len({x['tag'] for x in new})
tmp=R/'jobs.oracle_router.tmp';tmp.write_text(json.dumps(new,indent=2));tmp.replace(R/'jobs.json')
print(json.dumps({'added':len(jobs),'after':anchor,'total':len(new)}))
