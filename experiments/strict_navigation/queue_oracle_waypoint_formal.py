"""Queue a finite oracle-route experiment after third-seed baseline evaluation."""
import json,hashlib,time
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'oracle_waypoint_formal_v1'
PY=str(R.parent/'venv/bin/python')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'oracle_waypoint_v1/model_smoke_audit.json')['passed']
assert read(R/'reports/oracle_waypoint_cpu_audit.json')['passed']
assert read(R/'topology_stress_v1/results.json')['passed']
parent=read(R/'oracle_waypoint_v1/protocol.json');models=parent['models']
for v,m in models.items():
 assert sha(Path(m['weights']))==parent['model_sha256'][v]
 assert sha(Path(m['normalization']))==m['normalization_sha256']
 assert read(R/'runs'/m['training_run']/'training_summary.json')['full_epoch_budget_completed']
maps=['four_room_chain','four_room_cycle','six_room_branch'];jobs=[];files={}
for mapid in maps:
 for v in ['single','multi']:
  for mode in ['final','waypoint']:
   for b in range(2):
    case=R/'topology_dev_v2'/f'{mapid}_b{b}.json';files[str(case)]=sha(case)
    tag=f'oracle_formal_{v}_{mode}_{mapid}_b{b}_v1'
    args=[PY,str(R/'eval_oracle_waypoints.py'),'--config-name','tworoom',
      'policy='+models[v]['weights'],'seed=42','eval.num_eval=50','eval.dataset_name=tworoom',
      'eval.goal_offset_steps=75','eval.eval_budget=150','solver.n_steps=2',
      'solver.num_samples=300','solver.topk=30','plan_config.horizon=5','plan_config.receding_horizon=1',
      '+plan_config.warm_start=false',f'output.filename={R}/runs/{tag}/result.txt']
    jobs.append({'tag':tag,'model':v,'mode':mode,'map':mapid,'argv':args,
     'env':{'LEWM_CASE_FILE':str(case),'LEWM_CASE_LIMIT':'50','LEWM_VIDEO':'0',
       'LEWM_NORMALIZATION_FILE':models[v]['normalization'],'LEWM_ROUTE_MODE':mode},
     'watchdog_seconds':2400,'note':'Privileged high-level route diagnostic; same H5/R1 CEM budget as final-goal control, not deployable learned hierarchy.'})
audit={'tag':'audit_oracle_waypoint_formal_v1','argv':[PY,str(R/'audit_oracle_waypoint_formal.py')],
 'env':{'CUDA_VISIBLE_DEVICES':''},'watchdog_seconds':600}
old=read(R/'jobs.json');known={x['tag'] for x in old}
assert not known.intersection(x['tag'] for x in jobs+[audit])
idx=next(i for i,x in enumerate(old) if x['tag']=='strict3074_test_long_base_b5')
state=read(R/'status.json')
assert state['active']['tag']=='strict_tw_s3074',state.get('active')
assert not any(x['tag']=='strict3074_test_long_base_b5' for x in state['jobs'])
q={'created_unix':time.time(),'maps':maps,'models':models,'model_sha256':parent['model_sha256'],
 'case_files_sha256':files,
 'source_sha256':{str(R/f):sha(R/f) for f in ['eval_strict.py','eval_topology.py','topology_env.py','oracle_waypoint_router.py','eval_oracle_waypoints.py','audit_executed_trace.py','audit_oracle_waypoint_formal.py']},
 'prerequisite_sha256':{str(R/f):sha(R/f) for f in ['oracle_waypoint_v1/model_smoke_audit.json','reports/oracle_waypoint_cpu_audit.json','topology_stress_v1/results.json']},
 'evaluation_jobs':jobs,'audit_job':audit,
 'budget':{'physical_steps':150,'horizon_model_steps':5,'receding_model_steps':1,'action_block_physical_steps':5,'cem_iterations':2,'candidates':300,'topk':30,'warm_start':False,'candidate_model_step_cap_per_case':90000},
 'frozen_choice':'Both arms switch to H5/R1 with2 CEM rounds; original H5/R5 permits only6 replans in150 steps and may not process7 subgoals for3 doors. Main inference only from matched H5/R1 arms.',
 'limits':['Development cases, not final reserved test. One matched seed4001 per training condition.',
 'Oracle uses true graph, door positions and exact pose plus synthetic waypoint images; not learned visual hierarchy or deployable policy.',
 'Target replacement never flushes buffers or changes true terminal goal. Switch threshold8px, door offsets20px fixed before formal outcomes.',
 'Equal candidate-model-step caps do not mean equal actual cost after early stop. Encoder work, route image count and batch time logged; batch time is not online latency.',
 'Compared to previous H5/R5, encoder and feedback counts differ despite same nominal candidate budget.',
 'OOD topology also shifts geometry, scale and appearance. Positive/negative results do not isolate topology causal mechanisms.',
 'No topology training, no memory, no partial observation, no robot deployment.'],
 'scheduled_after':old[idx]['tag']}
P.mkdir(exist_ok=False)
(P/'protocol.json').write_text(json.dumps(q,indent=2))
(P/'jobs_before.json').write_text(json.dumps(old,indent=2))
new=old[:idx+1]+jobs+[audit]+old[idx+1:]
assert len(new)==len({x['tag'] for x in new})
tmp=R/'jobs.oracle_formal.tmp';tmp.write_text(json.dumps(new,indent=2));tmp.replace(R/'jobs.json')
print(json.dumps({'added':25,'after':q['scheduled_after'],'total':len(new)}))
