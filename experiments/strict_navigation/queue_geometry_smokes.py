"""Queue finite controlled-data smoke tests after the existing full collection audit."""
import json,copy
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent;PY=str(ROOT/'venv/bin/python')
p=R/'jobs.json';jobs=json.loads(p.read_text());known={j['tag'] for j in jobs}
add=[]
for variant in ['single','multi']:
 tag=f'geometry_smoke_{variant}_s4001'
 add.append({'tag':tag,'watchdog_seconds':900,'argv':[PY,str(R/'train_strict.py'),'data=tworoom',f'data.dataset.name={ROOT}/data/nav_{variant}_control.h5','seed=4001','history_size=1','loss.sigreg.weight=0.1','trainer.max_epochs=10',f'subdir={tag}',f'output_model_name={tag}','num_workers=6','+trainer.enable_progress_bar=false'],
 'env':{'LEWM_SPLIT_MANIFEST':str(R/'collections'/variant/'manifest.json'),'LEWM_STOP_STEPS':'20','LEWM_TRAIN_SECONDS':'850','SPT_CACHE_DIR':str(R/'spt_cache')},
 'note':'Interface smoke, exactly 20 updates; NOT a complete training seed or performance result.'})
 etag=f'geometry_smoke_{variant}_eval'
 add.append({'tag':etag,'watchdog_seconds':300,'argv':[PY,str(R/'eval_strict.py'),'--config-name','tworoom',f'policy={ROOT}/data/checkpoints/{tag}/weights_final.pt','seed=42','eval.num_eval=2','eval.dataset_name=tworoom','eval.goal_offset_steps=75','eval.eval_budget=10','solver.n_steps=1','solver.num_samples=8','solver.topk=2','plan_config.horizon=1','plan_config.receding_horizon=1','+plan_config.warm_start=false',f'output.filename={R}/runs/{etag}/result.txt'],
 'env':{'LEWM_CASE_FILE':str(R/'maps/validation_axis0_door65_b0.json'),'LEWM_CASE_LIMIT':'2','LEWM_VIDEO':'0','LEWM_NORMALIZATION_FILE':str(R/'collections'/variant/'normalization.json')},
 'note':'Two frozen validation cases, smoke only; uses this model dataset training-only scaling.'})
add.append({'tag':'audit_geometry_smokes','watchdog_seconds':300,'argv':[PY,str(R/'audit_geometry_smokes.py')],'env':{'CUDA_VISIBLE_DEVICES':''},'note':'Requires both 20-update checkpoints, disjoint training splits, matching statistics and valid model rollouts before full training is considered.'})
add=[j for j in add if j['tag'] not in known]
idx=next(i for i,j in enumerate(jobs) if j['tag']=='audit_geometry_collections')+1
jobs[idx:idx]=add
if add:
 tmp=p.with_name('jobs.geometry_smoke.tmp');tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(p)
print('CONTROLLED_SMOKES_QUEUED',len(add),'TOTAL',len(jobs))
