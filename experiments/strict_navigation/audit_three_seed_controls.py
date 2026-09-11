"""Three-seed development controls, audited separately from learned transfers."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from audit_executed_trace import audit_executed_trace
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseaudit=read(R/'reports/three_seed_baseline_audit.json');assert baseaudit['passed']
done={j['tag'] for j in read(R/'status.json')['jobs'] if j['status']=='complete'}
variants=['base','feedback_extra','feedback_extra_warm','h10_budget','h10_budget_warm']
results={};pairs={};checks={}
def paired(a,b):
 d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
 rng=np.random.default_rng(9106000);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
 return {'delta_pp':float(d.mean()*100),'ci95_conditional':np.quantile(boot,[.025,.975]).tolist(),'wins':int((d==1).sum()),'losses':int((d==-1).sum())}
for seed in [3072,3073,3074]:
 model=baseaudit['models'][str(seed)];assert sha(Path(model['weights']))==model['sha256']
 results[str(seed)]={};pairs[str(seed)]={};groups={};cases0=None
 for variant in variants:
  rows=[];cases=[];metrics=[];caps=[]
  for b in range(2):
   tag=f'strict{seed}_validation_long_{variant}_b{b}';assert tag in done
   p=R/'runs'/tag;m=read(p/'case_metadata.json');rr=read(p/'cases_results.json');metric=read(p/'metrics.json')
   assert m['case_sha256']==sha(R/'splits'/f'cases_validation_{b}.json')
   assert m['cases']==read(R/'splits'/f'cases_validation_{b}.json')['cases']
   assert read(p/'model_metadata.json')['model_path']==model['weights']
   assert [x['success'] for x in rr]==[bool(x) for x in metric['metrics']['episode_successes']]
   checks[tag]=audit_executed_trace(np.load(p/'trace.npz'),rr)
   assert m['budget']==150 and m['goal_offset']==75
   rows+=rr;cases+=m['cases'];metrics.append(metric);caps.append(m['compute_cap_per_case'])
  assert len(rows)==100 and caps[0]==caps[1]
  if cases0 is None:cases0=cases
  assert cases==cases0
  cap=caps[0];assert cap['candidate_model_steps']==(450000 if variant.startswith('feedback_extra') else 90000)
  assert cap['warm_start']==variant.endswith('_warm')
  assert cap['horizon']==(10 if variant.startswith('h10') else 5)
  assert cap['receding']==(1 if variant.startswith('feedback') else 5)
  assert cap['candidates']==(150 if variant.startswith('h10') else 300) and cap['iterations']==10
  groups[variant]=rows
  results[str(seed)][variant]={'sr':float(np.mean([x['success'] for x in rows])*100),
    'cap':cap,'actual_candidate_model_steps':sum(x['telemetry']['candidate_model_steps'] for x in metrics),
    'encoder_images':sum(x['telemetry']['encoder_images'] for x in metrics),
    'predictor_token_inputs':sum(x['telemetry']['predictor_token_inputs'] for x in metrics),
    'batch_seconds_not_online':sum(x['elapsed_seconds'] for x in metrics)}
 for v in variants[1:]:pairs[str(seed)][v+'_vs_base']=paired(groups['base'],groups[v])
 for v in ['feedback_extra','h10_budget']:pairs[str(seed)][v+'_warm_vs_cold']=paired(groups[v],groups[v+'_warm'])
payload={'passed':True,'updated_unix':time.time(),'results':results,'paired':pairs,'trace_audits':checks,
 'limits':'Same100 development cases reused across models and settings; paired case CI conditional on fixed seed, not training-seed CI. Same candidate-step cap is not same total FLOPs; extra-feedback cap is5x. Warm-start is existing planner functionality, not proposed novel architecture. Legacy seed3072 normalization metadata limitation remains as baseline audit.'}
(R/'reports/three_seed_planning_controls.json').write_text(json.dumps(payload,indent=2))
labels={'base':'原始H5/R5','feedback_extra':'高频反馈H5/R1，不保留计划','feedback_extra_warm':'高频反馈H5/R1，保留计划','h10_budget':'H10/R5，不保留计划','h10_budget_warm':'H10/R5，保留计划'}
lines=['## 三种子规划实现对照：延长时域必须同时检查计划连续性','',
'同一组100例开发长目标，均为150物理步。warm指保留上一轮尚未执行的动作后缀作为下一轮搜索初始化，不是新增训练或新的世界模型架构。','',
'| 设置 | seed3072 | seed3073 | seed3074 | 候选模型步上限/例 |','|---|---:|---:|---:|---:|']
for v in variants:
 lines.append('| '+labels[v]+' | '+' | '.join(f"{results[str(s)][v]['sr']:.1f}%" for s in [3072,3073,3074])+f" | {results['3072'][v]['cap']['candidate_model_steps']} |")
lines+=['',
'H10开启计划延续相对冷启动分别恢复38、41、38个百分点；但相对原始基线的净提升仅4、15、8个百分点，不能把恢复冷启动退化全部宣传为新方法提升。',
'相对原始基线，H10暖启动的病例配对95%区间依次[-5,13]、[7,24]、[-2,18]。三个种子点估计方向一致，但两个单种子区间包含零；需要未触碰病例确认，不能宣称已得到普遍显著提升。',
'高频反馈增加了5倍候选模型步上限，冷启动仍下降，暖启动相对原始基线仅+2/+4/+4个百分点，三个区间均包含零。更多计算不保证更高成功率；不能把高频反馈的收益直接归因于预测改善。',
'原基线和H10对照候选模型步上限相同，但候选数/时域不同，预测器token工作、编码和实际早停计算都需单独计量；不是严格总FLOPs或在线时延等价。',
'这说明规划连续性是解释模型局限前必须控制的实现因素，不证明LeWM动态本身无误。后续关键迁移比较应保留这个无额外训练的简单规划对照，而不是只与已退化的冷启动设置比较。',
'原标记为test的6个几何布局此前已被原模型评估并用于阶段解释，不能再称完全未查看的最终地图；它们仍是训练外布局，但不是方法开发过程未触碰的环境。最终轨迹预留300例仍未运行，另需冻结新几何病例以确认布局结论。','']
text='\n'.join(lines);(R/'reports/three_seed_planning_controls.md').write_text(text)
for f in [R/'reports/frontier_transfer_review.md',R.parent/'reports/navigation_capability_merged.md']:
 old=f.read_text()
 if lines[0] not in old:f.write_text(old+'\n\n'+text)
print('PLANNING_CONTROLS_AUDITED',flush=True)
