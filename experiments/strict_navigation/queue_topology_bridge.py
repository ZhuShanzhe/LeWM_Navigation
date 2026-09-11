"""Finite learned-interface smoke and matched-simulator bridge; no full topology evaluation."""
import json,copy,time,hashlib
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');D=R/'topology_dev_v2';ROOT=R.parent;PY=str(ROOT/'venv/bin/python')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def create(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
assert read(D/'catalog.json')['passed'] and read(D/'adapter_audit.json')['passed']
P=R/'topology_bridge_v1';P.mkdir(exist_ok=False)
catalog=read(D/'catalog.json');jobs=[];casefiles={}
models={}
for v in ['single','multi']:
 w=ROOT/'data/checkpoints'/f'geometry_control_{v}_s4001'/'weights_final.pt'
 models[v]={'weights':str(w),'normalization':str(R/'collections'/v/'normalization.json'),'normalization_sha256':sha(R/'collections'/v/'normalization.json'),'training_run':f'geometry_control_{v}_s4001'}
# Each custom-map smoke includes one same-room and one cross-room case.
for spec in catalog['maps']:
 data=read(D/(spec['id']+'.json'));data['cases']=[data['cases'][0],data['cases'][99]]
 p=P/(spec['id']+'_smoke.json');create(p,data);casefiles[str(p)]=sha(p)
bridge=read(D/'bridge_two_room.json')
for c in bridge['cases']:c['goals']['75']['across_wall']=c['goals']['75']['different_room']
for engine in ['custom','original']:
 data=copy.deepcopy(bridge)
 if engine=='original':
  data.pop('topology_spec')
  data['init_value']={'wall.axis':1,'door.position':[49,49,49],'door.number':1,'door.size':[14,14,14]}
 data['simulator_bridge_engine']=engine
 for b in range(2):
  batch=copy.deepcopy(data);batch['cases']=batch['cases'][b*50:(b+1)*50]
  p=P/(engine+f'_b{b}.json');create(p,batch);casefiles[str(p)]=sha(p)
 if engine=='original':
  data['cases']=[data['cases'][0],data['cases'][99]]
  p=P/'original_smoke.json';create(p,data);casefiles[str(p)]=sha(p)
def job(tag,v,file,engine,smoke=False):
 n=2 if smoke else 50;budget=10 if smoke else 150;h=1 if smoke else 5
 return {'tag':tag,'watchdog_seconds':600 if smoke else 2400,'argv':[PY,str(R/('eval_topology.py' if engine=='custom' else 'eval_strict.py')),'--config-name','tworoom','policy='+models[v]['weights'],'seed=42',f'eval.num_eval={n}','eval.dataset_name=tworoom','eval.goal_offset_steps=75',f'eval.eval_budget={budget}',f'solver.n_steps={1 if smoke else 10}',f'solver.num_samples={8 if smoke else 300}',f'solver.topk={4 if smoke else 30}',f'plan_config.horizon={h}',f'plan_config.receding_horizon={h}','+plan_config.warm_start=false',f'output.filename={R}/runs/{tag}/result.txt'],'env':{'LEWM_CASE_FILE':str(file),'LEWM_CASE_LIMIT':str(n),'LEWM_VIDEO':'0','LEWM_NORMALIZATION_FILE':models[v]['normalization']},'note':'Learned interface only, not a performance estimate.' if smoke else 'Same 100 development start-goals, old/new collision-render semantics bridge; not a new-topology score.'}
for v in models:
 for spec in catalog['maps']:
  jobs.append(job(f"topology_model_smoke_{v}_{spec['id']}_v1",v,P/(spec['id']+'_smoke.json'),'custom',True))
 jobs.append(job(f'topology_model_smoke_{v}_original_v1',v,P/'original_smoke.json','original',True))
jobs.append({'tag':'audit_topology_model_smokes_v1','argv':[PY,str(R/'audit_topology_model_bridge.py'),'smoke'],'watchdog_seconds':600,'env':{'CUDA_VISIBLE_DEVICES':''}})
for v in models:
 for engine in ['original','custom']:
  for b in range(2):
   jobs.append(job(f'topology_bridge_{v}_{engine}_b{b}_v1',v,P/(engine+f'_b{b}.json'),engine))
jobs.append({'tag':'audit_topology_bridge_v1','argv':[PY,str(R/'audit_topology_model_bridge.py'),'bridge'],'watchdog_seconds':600,'env':{'CUDA_VISIBLE_DEVICES':''}})
protocol={'created_unix':time.time(),'stage':'learned interface and collision/render bridge only','models':models,'case_files_sha256':casefiles,'source_sha256':{str(R/name):sha(R/name) for name in ['topology_env.py','eval_topology.py','eval_strict.py','audit_topology_model_bridge.py']},'catalog_sha256':sha(D/'catalog.json'),'jobs':jobs,'limits':['Full observation custom simulator; not official TwoRoom or first-person navigation.','Each bridge uses same 100 cases, 20 same-room and 80 cross-room; not untouched final test.','One paired training seed4001 with matched data volume and behavior; not cross-seed conclusion.','Rendering and collision change together; bridge measures combined simulator shift, not isolated collision effect.','No full learned-model topology benchmark queued: review bridge before expanding.','Original and custom success radius both16px; goals22px inside room bounds avoid across-wall success in this set.','Room-center reference is feasible but not shortest path, so no SPL claim.','Equal nominal compute caps; early stops change actual work. Batched time not online latency.']}
create(P/'protocol.json',protocol)
old=read(R/'jobs.json');known={x['tag'] for x in old};assert not any(x['tag'] in known for x in jobs)
assert not any(x['tag']=='audit_geometry_control_pair4001' and x['status']=='complete' for x in read(R/'status.json')['jobs']),'review insertion anchor before proceeding'
anchor=next(i for i,x in enumerate(old) if x['tag']=='audit_geometry_control_pair4001')
create(P/'jobs_before.json',old)
new=old[:anchor+1]+jobs+old[anchor+1:];assert len({x['tag'] for x in new})==len(new)
tmp=R/'jobs.topology_v1.tmp';tmp.write_text(json.dumps(new,indent=2));tmp.replace(R/'jobs.json')
print(json.dumps({'added_jobs':len(jobs),'queued_after':'audit_geometry_control_pair4001','total_jobs':len(new),'protocol':str(P/'protocol.json')}))
