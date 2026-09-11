"""Freeze a final five-method, three-seed confirmation; no future adaptive sweep."""
import json,hashlib,time,copy
from pathlib import Path
from functools import lru_cache
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent;P=R/'final_confirmation_v1'
PY=str(ROOT/'venv/bin/python')
def read(p):return json.loads(p.read_text())
@lru_cache(None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ta=read(R/'reports/three_seed_transfer_audit.json');assert ta['passed']
assert read(R/'reports/three_seed_planning_controls.json')['passed']
assert read(R/'reports/final_holdout_preflight.json')['passed']
artifacts=ta['artifacts'];methods=['base','h10_warm','temporal','retrieval_only','retrieval_rerank']
domains={'known_layout':{'n':300,'files':[str(R/'splits'/f'cases_final_reserve_{b}.json') for b in range(6)]}}
for m in read(R/'final_geometry_v1/catalog.json')['maps']:
 domains[m['id']]={'n':100,'files':[str(R/'final_geometry_v1'/f"{m['id']}_b{b}.json") for b in range(2)]}
reserve=read(R/'splits/final_reserve_manifest.json')
for n,h in reserve['files'].items():assert sha(str(R/'splits'/n))==h
fm=read(R/'final_geometry_v1/freeze_manifest.json')
for n,h in fm['files_sha256'].items():assert sha(str(R/'final_geometry_v1'/n))==h
norm=str(R/'splits/normalization.json');evaluations=[]
for domain,domain_info in domains.items():
 for seed in [3072,3073,3074,'reference']:
  for method in (methods if seed!='reference' else ['random','noop']):
   for batch,casefile in enumerate(domain_info['files']):
    tag=f'finalv1_{domain}_s{seed}_{method}_b{batch}'
    learned=seed!='reference';a=artifacts[str(seed)] if learned else None
    modelpath=a['model']['weights'] if learned else 'random'
    driver=R/('eval_retrieval.py' if method.startswith('retrieval') else 'eval_strict.py')
    args=[PY,str(driver),'--config-name','tworoom','policy='+modelpath,f'seed={42+batch}',
      'eval.num_eval=50','eval.dataset_name=tworoom','eval.goal_offset_steps=75','eval.eval_budget=150',
      'solver.n_steps='+('1' if method=='retrieval_rerank' else '10'),
      'solver.num_samples='+('150' if method=='h10_warm' else '300'),'solver.topk=30',
      'plan_config.horizon='+('10' if method=='h10_warm' else '5'),'plan_config.receding_horizon=5',
      '+plan_config.warm_start='+('true' if method=='h10_warm' else 'false'),f'output.filename={R}/runs/{tag}/result.txt']
    env={'LEWM_CASE_FILE':casefile,'LEWM_CASE_LIMIT':'50','LEWM_VIDEO':'0','LEWM_NORMALIZATION_FILE':norm}
    inputs={casefile:sha(casefile),norm:sha(norm),str(R/'splits/manifest.json'):sha(str(R/'splits/manifest.json'))}
    if learned:inputs[modelpath]=a['model']['sha256']
    if method=='temporal':
     hp=a['head_paths']['temporal'];env.update(LEWM_TEMPORAL_HEAD=hp,LEWM_HEAD_COST='replace');inputs[hp]=a['head_sha256']['temporal']
    if method.startswith('retrieval'):
     lp=a['library_path'];env.update(LEWM_RETRIEVAL_LIBRARY=lp,LEWM_RETRIEVAL_MODE=method.split('_')[1]);inputs[lp]=a['library_sha256']
    if not learned:env.update(CUDA_VISIBLE_DEVICES='',LEWM_EVAL_DEVICE='cpu',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if method=='noop':env['LEWM_NOOP']='1'
    evaluations.append({'tag':tag,'domain':domain,'training_seed':seed,'method':method,'batch':batch,
     'eval_argv':args,'argv':[PY,str(R/'run_final_frozen.py')],'env':env,'frozen_inputs':inputs,
     'watchdog_seconds':2400 if learned else 600,
     'note':'Frozen final confirmation. No training or parameter selection from outcomes; actual compute and domain dependence must be reported.'})
assert len(evaluations)==238
hashes={sha(f) for d in domains.values() for f in d['files']}
for p in (R/'runs').glob('*/case_metadata.json'):assert read(p).get('case_sha256') not in hashes,('Final data already evaluated',str(p))
sources=[R/f for f in ['run_final_frozen.py','audit_final_confirmation.py','eval_strict.py','eval_retrieval.py','retrieval_prior.py','reachability_model.py','audit_executed_trace.py']]
sources += [ROOT/'le-wm'/f for f in ['eval.py','jepa.py','module.py']]
sources += [ROOT/'venv/lib/python3.12/site-packages/stable_worldmodel/policy.py']
assert all(f.exists() for f in sources)
audit={'tag':'audit_final_confirmation_v1','argv':[PY,str(R/'audit_final_confirmation.py')],
 'env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},'watchdog_seconds':1800}
q={'created_unix':time.time(),'methods':methods,'domains':domains,'artifacts':artifacts,
 'evaluations':evaluations,'audit_job':audit,'source_sha256':{str(f):sha(str(f)) for f in sources},
 'selection_evidence_sha256':{str(R/f):sha(str(R/f)) for f in ['reports/three_seed_transfer_audit.json','reports/three_seed_planning_controls.json','reports/final_holdout_preflight.json']},
 'selection_rationale':'Base; best-motivated no-training continuity control; simpler temporal replacement rather than negligible-gain hybrid; retrieval alone and dynamic rerank to isolate predictor contribution. No final outcomes used.',
 'readiness_gate':'Original fixed16px route sensitivity audit must pass before any final case; its performance is not used to choose these five non-oracle methods.',
 'budget':{'physical_steps':150,'learned_method_seed_count':3,'known_layout_unique_cases':300,'new_geometry_count':4,'new_geometry_cases_each':100,'gpu_batches':210,'cpu_reference_batches':28},
 'limits':['Known-layout reserve episodes are disjoint from training/validation/earlier development-test episodes; same environment remains a limitation.',
 'New door values73/169 and new coordinates, but still two-room full-observation synthetic navigation, not new topology or real robots.',
 'Three training seeds share fixed head pairs and evaluation seeds. Four maps reuse coordinate pairs; no pooling as independent trials.',
 'Base/temporal/H10 share90000 candidate-model-step caps, not total FLOPs. Rerank9000 plus retrieval/encoding; only0 dynamic steps plus retrieval/encoding. Early stopping changes actual costs.',
 'Temporal head uses behavioral temporal separation, not geodesic labels. Retrieval uses training trajectories, not expert shortest paths.',
 'Only primary long-goal task is final-confirmed; short-goal/partial-observation/topology/coverage claims retain separately labeled evidence.',
 'No new training or tuning from final results. Remaining unexpected findings become follow-up research hypotheses.',
 'Completion of this audit requires final report synthesis and explicit no-pending-job review before stopping monitoring; do not shut down server.']}
old=read(R/'jobs.json');assert not {j['tag'] for j in old}&{j['tag'] for j in evaluations+[audit]}
assert any(j['tag']=='audit_oracle_waypoint_t16_v1' for j in old)
P.mkdir(exist_ok=False);(P/'protocol.json').write_text(json.dumps(q,indent=2));(P/'jobs_before.json').write_text(json.dumps(old,indent=2))
new=old+evaluations+[audit];assert len(new)==len({j['tag'] for j in new})
tmp=R/'jobs.final_confirmation.tmp';tmp.write_text(json.dumps(new,indent=2));tmp.replace(R/'jobs.json')
print('FINAL_CONFIRMATION_QUEUED',json.dumps({'added':239,'gpu_batches':210,'cpu_batches':28,'total':len(new)}),flush=True)
