"""Final descriptive supplements, no new evaluations or parameter selection."""
from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
d=json.loads((R/'final_confirmation_v1/results.json').read_text())
q=json.loads((R/'final_confirmation_v1/protocol.json').read_text())
assert d['passed'] and len(d['trace_audits'])==238
state=json.loads((R/'status.json').read_text());done={x['tag'] for x in state['jobs'] if x['status']=='complete'}
assert all(j['tag'] in done for j in q['evaluations'])
excluded={(c['episode'],c['start_step']) for tag,x in d['noop_replay_audits'].items() if 'known_layout' in tag for c in x['initial_projection_cases']}
assert excluded=={(3024,0),(6221,0),(3782,0)}
rows={}
for j in q['evaluations']:
 key=(j['domain'],str(j['training_seed']),j['method'])
 rows.setdefault(key,[]).extend(json.loads((R/'runs'/j['tag']/'cases_results.json').read_text()))
def rate(rr):return 100*float(np.mean([x['success'] for x in rr])) if rr else None
def pair(aa,bb):
 assert [(x['episode'],x['start_step']) for x in aa]==[(x['episode'],x['start_step']) for x in bb]
 delta=np.array([int(b['success'])-int(a['success']) for a,b in zip(aa,bb)])
 rng=np.random.default_rng(9106501);boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)*100
 return {'n':len(delta),'delta_pp':float(delta.mean()*100),'conditional_case_ci95':np.quantile(boot,[.025,.975]).tolist()}
clean={};removed={};pairclean={}
for seed in ['3072','3073','3074','reference']:
 clean[seed]={};removed[seed]={}
 for m in (q['methods'] if seed!='reference' else ['random','noop']):
  rr=rows[('known_layout',seed,m)];keep=[x for x in rr if (x['episode'],x['start_step']) not in excluded]
  assert len(rr)==300 and len(keep)==297
  clean[seed][m]={'n':297,'sr':rate(keep),'sr_excluding_initial':rate([x for x in keep if not x['initial_success']]),'original_sr':rate(rr)}
  removed[seed][m]=[{'episode':x['episode'],'start_step':x['start_step'],'success':x['success']} for x in rr if (x['episode'],x['start_step']) in excluded]
 if seed!='reference':
  filt=lambda m:[x for x in rows[('known_layout',seed,m)] if (x['episode'],x['start_step']) not in excluded]
  pairclean[seed]={m+'_minus_base':pair(filt('base'),filt(m)) for m in q['methods'] if m!='base'}
  pairclean[seed]['rerank_minus_only']=pair(filt('retrieval_only'),filt('retrieval_rerank'))
cross={};cost={}
for domain,g in d['results'].items():
 cross[domain]={};cost[domain]={}
 for m in q['methods']:
  vals=[g[str(seed)][m] for seed in [3072,3073,3074]]
  cross[domain][m]={k:float(np.mean([x[k] for x in vals])) for k in ['cross_wall_sr','same_side_sr','sr_excluding_initial','stalled_step_fraction','mean_path_all']}
  n=vals[0]['n']
  cost[domain][m]={k:float(np.mean([x[k]/n for x in vals])) for k in ['candidate_model_steps','encoder_images','predictor_token_inputs','batch_elapsed_seconds_not_online']}
payload={'passed':True,'updated_unix':time.time(),'source_final_results_sha256':hashlib.sha256((R/'final_confirmation_v1/results.json').read_bytes()).hexdigest(),'excluded_cases':sorted(excluded),'known_layout_boundary_sensitivity':clean,'removed_case_outcomes':removed,'boundary_sensitivity_pairs':pairclean,'cross_seed_descriptive_strata':cross,'per_case_compute_descriptive':cost,'limits':'Post-audit descriptive exclusion of the same three boundary-projected starts in every method; primary frozen 300-case result unchanged. Seeds and maps reuse cases. No new training/evaluation. Paired intervals conditional on fixed model and domain, not multiple-testing adjusted.'}
(R/'reports/final_supplementary_analysis.json').write_text(json.dumps(payload,indent=2))
labels={'base':'原始','h10_warm':'H10暖启动','temporal':'时间头','retrieval_only':'仅检索','retrieval_rerank':'检索+排序'}
lines=['## 最终补充统计：分层、计算和边界敏感性','','### 同侧与跨墙成功率（三训练种子描述均值）','','| 任务域 | 方法 | 同侧 SR | 跨墙 SR | 停滞步比例 |','|---|---|---:|---:|---:|']
for domain,g in cross.items():
 for m,x in g.items():lines.append(f"| {domain} | {labels[m]} | {x['same_side_sr']:.2f}% | {x['cross_wall_sr']:.2f}% | {100*x['stalled_step_fraction']:.2f}% |")
lines+=['','### 原布局最终300例：实际计算（逐例均值，再对三种子平均）','','| 方法 | 候选模型步 | 编码图像数 | 预测器 token 输入 | 批处理秒/例（非在线时延） |','|---|---:|---:|---:|---:|']
for m,x in cost['known_layout'].items():lines.append(f"| {labels[m]} | {x['candidate_model_steps']:.1f} | {x['encoder_images']:.2f} | {x['predictor_token_inputs']:.1f} | {x['batch_elapsed_seconds_not_online']:.3f} |")
lines+=['','上表是执行期间记录，不包含世界模型、辅助头训练与检索库构建/额外库搜索的全部资源。不能用候选模型步数或批次摊销时间代替总FLOPs、峰值内存或单机器人端到端延迟。H10和原版虽然同90000候选步上限，但成功早停不同，实际用量不同。','','### 共同排除三个边界投影起点（非主结果）','','三个起点为episode3024/6221/3782、start_step均为0；它们与三个“初始已经成功”病例不是同一个筛选条件。原300例结果原样保留，以下所有方法均使用共同297例。','','| 方法 | seed3072 | seed3073 | seed3074 | 三种子均值 |','|---|---:|---:|---:|---:|']
for m in q['methods']:
 v=[clean[s][m]['sr'] for s in ['3072','3073','3074']]
 lines.append('| '+labels[m]+' | '+' | '.join(f'{z:.2f}%' for z in v+[float(np.mean(v))])+' |')
lines+=['','此项是发现环境起点投影后的描述性敏感性分析，不是重新选测试集，不改变主结果或继续选参。逐种子配对区间及每个被排除病例的结果保存在final_supplementary_analysis.json。','']
(R/'reports/final_supplementary_analysis.md').write_text('\n'.join(lines))
print('SUPPLEMENT_PASSED')
print('BOUNDARY_MEANS',{m:float(np.mean([clean[s][m]['sr'] for s in ['3072','3073','3074']])) for m in q['methods']})
print('COST',cost['known_layout'])
