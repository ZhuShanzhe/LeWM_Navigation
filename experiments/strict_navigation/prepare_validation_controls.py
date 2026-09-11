"""Reserve untouched episodes for confirmatory evaluation, and queue finite validation controls."""
import json,time,hashlib,copy
from pathlib import Path
import numpy as np,h5py,hdf5plugin
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');D=R/'splits'
def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
if not (D/'final_reserve_manifest.json').exists():
 m=json.loads((D/'manifest.json').read_text());dev=json.loads((D/'cases_test.json').read_text())
 used={c['episode'] for c in dev['cases']};rng=np.random.default_rng(9102042)
 with h5py.File(R.parent/'data/tworoom.h5','r') as f:
  lens=f['ep_len'][:];offset=f['ep_offset'][:];pos=f['proprio'][:]
  pool=[e for e in m['episodes']['test'] if e not in used and lens[e]>=81]
  chosen=rng.choice(pool,min(300,len(pool)),replace=False);cases=[]
  for e in chosen:
   t=int(rng.integers(0,int(lens[e])-75));i=int(offset[e])+t;x=pos[i]
   cases.append({'episode':int(e),'start_step':t,'start_xy':x.tolist(),'goals':{str(h):{'xy':pos[i+h].tolist(),'euclidean':float(np.linalg.norm(pos[i+h]-x)),'across_wall':bool((x[0]-112)*(pos[i+h,0]-112)<0),'initially_within_success':bool(np.linalg.norm(pos[i+h]-x)<16)} for h in [25,75]}})
 body={'split':'final_reserve','selection_seed':9102042,'sampling':'random unused heldout episodes; no outcome filtering; do not evaluate before method freezing','cases':cases}
 save(D/'cases_final_reserve.json',body)
 for j in range(0,len(cases),50):save(D/f'cases_final_reserve_{j//50}.json',{**body,'cases':cases[j:j+50]})
 save(D/'final_reserve_manifest.json',{'created_unix':time.time(),'eligible_pool':len(pool),'n':len(cases),'disjoint_development_test_episodes':True,'original_split_sha256':hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('cases_final_reserve*.json')},'rationale':'Original test results now used for mechanism exploration. Independent episodes remain unqueried for final comparison; shared map still limits external validity.'})
jobs=json.loads((R/'jobs.json').read_text());tags={j['tag'] for j in jobs};new=[]
for v in ['long_feedback_budget','long_feedback_extra','long_h10_budget']:
 for b in [0,1]:
  tag=f'strict3072_validation_{v}_warm_b{b}'
  if tag in tags:continue
  job=copy.deepcopy(next(j for j in jobs if j['tag']==f'strict3072_validation_{v}_b{b}'))
  job['tag']=tag;job['argv']=[a.replace('+plan_config.warm_start=false','+plan_config.warm_start=true') if not a.startswith('output.filename=') else 'output.filename='+str(R/'runs'/tag/'result.txt') for a in job['argv']]
  new.append(job)
for v,warm in [('long_base',False),('long_feedback_extra',False),('long_feedback_extra',True)]:
 for b in [0,1]:
  suffix='_warm_bounded' if warm else '_bounded';tag=f'strict3072_validation_{v}{suffix}_b{b}'
  if tag in tags:continue
  job=copy.deepcopy(next(j for j in jobs if j['tag']==f'strict3072_validation_{v}_b{b}'));job['tag']=tag
  job['env']['LEWM_BOUND_ACTIONS']='1'
  job['argv']=[('output.filename='+str(R/'runs'/tag/'result.txt')) if a.startswith('output.filename=') else a.replace('+plan_config.warm_start=false','+plan_config.warm_start=true') if warm else a for a in job['argv']]
  new.append(job)
# Add small bounded smoke before formal new controls, preserving all existing jobs and active process.
tag='smoke_bounded_posttrain'
if tag not in tags:
 smoke=copy.deepcopy(next(j for j in jobs if j['tag']=='smoke_model_posttrain'));smoke['tag']=tag;smoke['env']['LEWM_BOUND_ACTIONS']='1'
 smoke['argv']=[a if not a.startswith('output.filename=') else 'output.filename='+str(R/'runs'/tag/'result.txt') for a in smoke['argv']]
 new.insert(0,smoke)
ix=next(i for i,j in enumerate(jobs) if j['tag'].startswith('strict3072_test_'))
jobs[ix:ix]=new;save(R/'jobs.json',jobs)
save(R/'validation_controls_20260910.json',{'created_unix':time.time(),'new_tags':[j['tag'] for j in new],'observations_used':'completed validation groups; counterfactual audit on development-test only','hypotheses':['Warm-start plan continuity interacts with feedback and horizon.','Unbounded normalized proposals are clipped by the simulator but not by original world-model predictions.'],'not_novelty_claim':True,'final_reserve_not_queued':True})
print(json.dumps({'added_validation_controls_and_smoke':len(new),'queue':len(jobs),'final_reserve':json.loads((D/'final_reserve_manifest.json').read_text())['n']}))
