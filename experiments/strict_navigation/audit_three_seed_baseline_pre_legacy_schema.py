"""Full-budget three-seed baseline audit, with shared-case dependence explicit."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from audit_executed_trace import audit_executed_trace
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sr(rows):return float(np.mean([x['success'] for x in rows])*100) if rows else None
state=read(R/'status.json');done={j['tag'] for j in state['jobs'] if j.get('status')=='complete'}
manifest_sha=sha(R/'splits/manifest.json');norm_sha=sha(R/'splits/normalization.json')
models={};records={};audits={};identity={};config=None
for seed in [3072,3073,3074]:
 tag=f'strict_tw_s{seed}'+('_v2' if seed==3072 else '')
 assert tag in done
 run=R/'runs'/tag;t=read(run/'training_summary.json');inv=read(run/'invocation.json');split=read(run/'split_audit.json')
 assert inv['seed']==seed and inv['stop_steps']==0
 assert t['full_epoch_budget_completed'] and t['global_step']==51920 and t['epoch_counter']==10
 assert split['passed'] and split['manifest_sha256']==manifest_sha
 assert split['episode_counts']=={'train':8000,'validation':1000,'test':1000}
 assert split['clip_counts']=={'train':664684,'validation':83354}
 w=Path(t['weights']);assert w==ROOT/'data/checkpoints'/tag/'weights_final.pt' and w.stat().st_size==72291743
 cfg=read(w.parent/'config.json')
 if config is None:config=cfg
 assert cfg==config
 models[str(seed)]={'tag':tag,'sha256':sha(w),'weights':str(w),'steps':t['global_step'],'epochs':t['epoch_counter'],'elapsed_seconds':t['elapsed_seconds'],'manifest_sha256':manifest_sha}
 for part,batches,n in [('validation',2,100),('test',6,300)]:
  for task,h in [('short150',25),('long_base',75)]:
   key=f'{part}_{task}';allrows=[];allcases=[];telemetry=[]
   for b in range(batches):
    jt=f'strict{seed}_{key}_b{b}';assert jt in done
    p=R/'runs'/jt;meta=read(p/'case_metadata.json');rows=read(p/'cases_results.json');metric=read(p/'metrics.json')
    casepath=R/'splits'/f'cases_{part}_{b}.json'
    assert meta['cases']==read(casepath)['cases'] and meta['case_sha256']==sha(casepath)
    assert meta['normalization_sha256']==norm_sha and meta['normalization_path']==str(R/'splits/normalization.json')
    assert meta['budget']==150 and meta['goal_offset']==h
    assert read(p/'model_metadata.json')['model_path']==str(w)
    assert [(x['episode'],x['start_step']) for x in rows]==[(c['episode'],c['start_step']) for c in meta['cases']]
    assert [bool(x['success']) for x in rows]==[bool(x) for x in metric['metrics']['episode_successes']]
    cap=meta['compute_cap_per_case']
    assert cap=={'candidate_model_steps':90000,'horizon':5,'iterations':10,'candidates':300,'receding':5,'warm_start':False},cap
    tr=np.load(p/'trace.npz');audits[jt]=audit_executed_trace(tr,rows)
    pos=tr['proprio'].reshape(len(tr['proprio']),len(rows),-1)[:,:,:2]
    for i,(row,case) in enumerate(zip(rows,meta['cases'])):
     path=np.vstack([case['start_xy'],pos[:row['steps'],i]])
     assert np.isclose(np.linalg.norm(np.diff(path,axis=0),axis=1).sum(),row['path_length'],atol=1e-4)
     assert np.isclose(np.linalg.norm(path[-1]-case['goals'][str(h)]['xy']),row['endpoint_distance'],atol=1e-4)
    allrows+=rows;allcases+=meta['cases'];telemetry.append(metric)
   assert len(allrows)==n and len({(x['episode'],x['start_step']) for x in allrows})==n
   if key in identity:assert allcases==identity[key]
   else:identity[key]=allcases
   records[(seed,key)]={'rows':allrows,'telemetry':telemetry}
assert len({x['sha256'] for x in models.values()})==3
results={}
for key in identity:
 per={};matrix=[]
 for seed in [3072,3073,3074]:
  d=records[(seed,key)];rows=d['rows'];matrix.append([int(x['success']) for x in rows])
  per[str(seed)]={'n':len(rows),'sr':sr(rows),'excluding_initial_sr':sr([x for x in rows if not x['initial_success']]),
    'cross_wall_sr':sr([x for x in rows if x['across_wall']]),'cross_wall_n':sum(x['across_wall'] for x in rows),
    'same_side_sr':sr([x for x in rows if not x['across_wall']]),
    'mean_path_all':float(np.mean([x['path_length'] for x in rows])),
    'stalled_step_fraction':sum(x['stalled_steps_lt_0_5px'] for x in rows)/sum(x['steps'] for x in rows),
    'candidate_model_steps':sum(m['telemetry']['candidate_model_steps'] for m in d['telemetry']),
    'encoder_images':sum(m['telemetry']['encoder_images'] for m in d['telemetry']),
    'batch_elapsed_seconds_not_online':sum(m['elapsed_seconds'] for m in d['telemetry'])}
 matrix=np.array(matrix);values=np.array([p['sr'] for p in per.values()])
 results[key]={'per_seed':per,'mean_sr':float(values.mean()),'sample_sd_sr':float(values.std(ddof=1)),
  'minimum_sr':float(values.min()),'maximum_sr':float(values.max()),'n_unique_cases':matrix.shape[1],
  'cases_all_three_succeed':int((matrix.sum(0)==3).sum()),'cases_all_three_fail':int((matrix.sum(0)==0).sum()),
  'cases_with_seed_disagreement':int(((matrix.sum(0)>0)&(matrix.sum(0)<3)).sum()),
  'note':'Mean and sample SD across3 training seeds on same cases, not CI; no pooling as3N independent episodes.'}
payload={'passed':True,'updated_unix':time.time(),'models':models,'results':results,'trace_audits':audits,
 'limits':['Three independent training seeds, same architecture/split/normalization and evaluation cases; evaluation random seeds held fixed by batch.',
 'The former test300 has informed development and is not an untouched final test.',
 'Short versus long goals also differ in distance and crossing composition, not a pure causal effect of horizon.',
 'Training wall-time includes shared-host load; evaluation batch time is not single-robot latency.',
 'Temporal/retrieval intervention repetitions and oracle route outcomes are still pending.']}
(R/'reports/three_seed_baseline_audit.json').write_text(json.dumps(payload,indent=2))
lines=['## 三个严格独立训练种子：基础导航能力审计完成','',
'更新：2026-09-11。三个模型均从零完成10轮、51920更新，架构、轨迹划分与归一化一致，最终权重哈希互异；所有基础评估的病例身份、成功记录及终止前动作/状态检查通过。','',
'| 组别 | seed3072 | seed3073 | seed3074 | 三种子均值 ± 样本标准差 |','|---|---:|---:|---:|---:|']
labels={'validation_short150':'开发验证短目标100例','validation_long_base':'开发验证长目标100例','test_short150':'既定开发测试短目标300例','test_long_base':'既定开发测试长目标300例'}
for key,label in labels.items():
 g=results[key];p=g['per_seed']
 lines.append(f"| {label} | {p['3072']['sr']:.1f}% | {p['3073']['sr']:.1f}% | {p['3074']['sr']:.1f}% | {g['mean_sr']:.2f} ± {g['sample_sd_sr']:.2f} pp |")
lines+=['','这里的±是三个训练种子的样本标准差，不是置信区间。三模型复用同一批病例，不能视为900个独立任务。','']
for key in ['test_short150','test_long_base']:
 g=results[key];lines.append(f"- {labels[key]}：三模型全部成功{g['cases_all_three_succeed']}例，全部失败{g['cases_all_three_fail']}例，成功/失败随种子变化{g['cases_with_seed_disagreement']}例。")
lines+=['','排除初始成功后的成绩：','']
for seed in ['3072','3073','3074']:
 lines.append(f"- seed{seed}：短目标{results['test_short150']['per_seed'][seed]['excluding_initial_sr']:.2f}%，长目标{results['test_long_base']['per_seed'][seed]['excluding_initial_sr']:.2f}%。")
lines+=['',
'当前证据较稳定地支持：同布局短目标较可靠，长目标成功率约56%—59.3%。但长短目标距离与跨墙比例不同，不能把差异全部归于规划时域；未知几何/多房间结果须单独解释。',
'第三种子训练耗时7368.04秒（约2.05小时），比前两次更长；相同更新预算与结构已核验，不能把共享服务器运行时间差异当作算法复杂度差异。',
'关键评价头/检索干预的三种子结果仍未完成，尚不能宣称迁移改进稳定。当前已使用的300例是机制开发材料，最终确认仍须另用未触碰预留病例。正式路点同预算诊断正在执行；本基础阶段完成不代表整轮研究结束。','',
'完整模型身份、同/跨墙分层、路径/停滞与实际计算：strict_nav_20260910/reports/three_seed_baseline_audit.json。','']
text='\n'.join(lines)
(R/'reports/three_seed_baseline_results.md').write_text(text)
for path in [R/'reports/frontier_transfer_review.md',ROOT/'reports/navigation_capability_merged.md']:
 old=path.read_text()
 if lines[0] not in old:path.write_text(old+'\n\n'+text)
print('THREE_SEED_BASELINE_AUDITED',json.dumps({k:{x:v[x] for x in ['mean_sr','sample_sd_sr','cases_all_three_fail','cases_with_seed_disagreement']} for k,v in results.items()}),flush=True)
