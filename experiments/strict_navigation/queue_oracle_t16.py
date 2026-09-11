"""One finite tolerance sensitivity, after existing cross-seed interventions."""
from pathlib import Path
import json,copy,time,hashlib
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'oracle_waypoint_t16_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(R/'oracle_waypoint_formal_v1/results.json')['passed']
assert read(R/'reports/oracle_threshold_cpu_audit.json')['passed']
parent=read(R/'oracle_waypoint_formal_v1/protocol.json');q=copy.deepcopy(parent)
jobs=[];evaluation=[]
for job in parent['evaluation_jobs']:
 j=copy.deepcopy(job)
 if j['mode']=='waypoint':
  oldtag=j['tag'];j['tag']=oldtag.replace('oracle_formal_','oracle_t16_',1)
  j['argv']=[a.replace('/eval_oracle_waypoints.py','/eval_oracle_waypoints_t16.py').replace('/'+oldtag+'/', '/'+j['tag']+'/') for a in j['argv']]
  j['note']='Only switch threshold8->16px; matches final goal tolerance. Sensitivity diagnosis, no tuning sweep or learned hierarchy.'
  jobs.append(j)
 evaluation.append(j)
audit={'tag':'audit_oracle_waypoint_t16_v1','argv':[str(R.parent/'venv/bin/python'),str(R/'audit_oracle_waypoint_t16.py')],
 'env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},'watchdog_seconds':600}
q['created_unix']=time.time();q['evaluation_jobs']=evaluation;q['audit_job']=audit
q['source_sha256'].update({str(R/f):sha(R/f) for f in ['eval_oracle_waypoints_t16.py','audit_oracle_waypoint_t16.py']})
q['prerequisite_sha256'].update({str(R/f):sha(R/f) for f in ['reports/oracle_threshold_cpu_audit.json','oracle_waypoint_formal_v1/results.json','reports/oracle_failure_stage_breakdown.json']})
q['frozen_choice']='One sensitivity condition:16px matches final success radius; same8px global-control records reused. No outcome-adaptive further thresholds.'
q['limits']+=[
 'This16px sensitivity was proposed after observed8px failures approached within16 but not8; development diagnostic, not preregistered before8px outcomes.',
 'CPU actual switching reference succeeds all300 cases for each tolerance.16px yields some non-clear next straight segments, resolved by sliding control; not a perfect collision-free subgoal oracle.',
 'No action-buffer flush, same90000 candidate-model-step cap. Existing strict three-seed interventions retain priority.',
 'Old CPU audit proved nominal exact waypoint routes; new CPU audit explicitly executes8/16 switching rule.'
]
old=read(R/'jobs.json');known={x['tag'] for x in old}
assert not known.intersection(x['tag'] for x in jobs+[audit])
idx=next(i for i,x in enumerate(old) if x['tag']=='summarize_three_seed_baseline')
assert not any(x['tag']=='summarize_three_seed_baseline' for x in read(R/'status.json')['jobs'])
P.mkdir(exist_ok=False)
(P/'protocol.json').write_text(json.dumps(q,indent=2));(P/'jobs_before.json').write_text(json.dumps(old,indent=2))
new=old[:idx]+jobs+[audit]+old[idx:];assert len(new)==len({x['tag'] for x in new})
temp=R/'jobs.oracle_t16.tmp';temp.write_text(json.dumps(new,indent=2));temp.replace(R/'jobs.json')
print('QUEUED',json.dumps({'added':len(jobs)+1,'before':old[idx]['tag'],'total':len(new)}))
