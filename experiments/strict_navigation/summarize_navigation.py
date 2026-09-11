"""Audit completed navigation outputs and update live research reports. No new experiments."""
import json,re,time,math,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
ROOT=R.parent
def read(p):return json.loads(p.read_text())
def save(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
state=read(R/'status.json');jobs=read(R/'jobs.json')
now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
records={j['tag']:j for j in state['jobs']}
groups={};audits=[];errors=[];maprows=[]
def metric(rows):
 a=np.array([r['success'] for r in rows],dtype=float);n=len(a)
 if not n:return {'n':0}
 p=float(a.mean());z=1.95996398454;den=1+z*z/n
 center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 non=[r for r in rows if not r['initial_success']]
 return {'n':n,'success_rate':100*p,'wilson_95':[100*(center-half),100*(center+half)],
 'excluding_initial_success_n':len(non),
 'excluding_initial_success_rate':100*float(np.mean([r['success'] for r in non])) if non else None,
 'mean_path_length':float(np.mean([r['path_length'] for r in rows])),
 'mean_stalled_steps':float(np.mean([r['stalled_steps_lt_0_5px'] for r in rows]))}
for job in jobs:
 tag=job['tag']
 if records.get(tag,{}).get('status')!='complete' or 'smoke' in tag:continue
 out=R/'runs'/tag
 if not (out/'cases_results.json').exists():continue
 rows=read(out/'cases_results.json');meta=read(out/'case_metadata.json');m=read(out/'metrics.json')
 official=m['metrics']['episode_successes'];ids=[(x['episode'],x['start_step']) for x in rows]
 checks={'tag':tag,'n':len(rows),'unique_cases':len(set(ids))==len(ids),
 'recorded_and_official_success_match':[bool(x['success']) for x in rows]==[bool(x) for x in official],
 'requested_cases_match':ids==[(x['episode'],x['start_step']) for x in meta['cases']],
 'n_consistent':len(rows)==m['n_cases']}
 audits.append(checks)
 if not all(v for k,v in checks.items() if k not in ['tag','n']):errors.append(checks);continue
 if '_map_' in tag:
  ma=read(out/'map_adapter_audit.json')
  if len(ma)!=len(rows) or any(x['start_render_max_error'] or x['goal_render_max_error'] or x['start_state_error']>1e-5 for x in ma):
   errors.append({'tag':tag,'map_adapter':'mismatch'});continue
  key=re.sub(r'_b\d+$','',tag);maprows.append((key,rows))
 elif re.match(r'strict(?:\d+|_(?:random|noop))_(validation|test)_',tag):
  key=re.sub(r'_b\d+$','',tag);g=groups.setdefault(key,{'rows':[],'tags':[],'elapsed':0,'candidate_steps':0,'caps':[]})
  g['rows'].extend(rows);g['tags'].append(tag);g['elapsed']+=m['elapsed_seconds']
  g['candidate_steps']+=m['telemetry']['candidate_model_steps'];g['caps'].append(meta.get('compute_cap_per_case',{}))
summary={}
for key,g in groups.items():
 expected=100 if '_validation_' in key else 300
 ids=[(x['episode'],x['start_step']) for x in g['rows']]
 if len(ids)!=len(set(ids)):errors.append({'group':key,'duplicate_ids':True});continue
 summary[key]={**metric(g['rows']),'expected_n':expected,'complete':len(ids)==expected,
 'across_wall':metric([r for r in g['rows'] if r['across_wall']]),
 'same_side':metric([r for r in g['rows'] if not r['across_wall']]),
 'elapsed_seconds':g['elapsed'],'candidate_model_steps':g['candidate_steps'],
 'compute_caps':g['caps'],'source_tags':g['tags']}
paired=[]
rng=np.random.default_rng(9102040)
for key,g in groups.items():
 if not key.endswith('_long_base') or not summary.get(key,{}).get('complete'):continue
 for alt in sorted(k for k in groups if k.startswith(key[:-len('long_base')]) and k!=key and '_long_' in k):
  if not summary.get(alt,{}).get('complete'):continue
  aa={(r['episode'],r['start_step']):r for r in g['rows']}
  bb={(r['episode'],r['start_step']):r for r in groups[alt]['rows']}
  if aa.keys()!=bb.keys():errors.append({'pair':[key,alt],'ids_mismatch':True});continue
  for stratum in ['all','across_wall','same_side']:
   keys=sorted(k for k in aa if stratum=='all' or bool(aa[k]['across_wall'])==(stratum=='across_wall'))
   if not keys:continue
   delta=np.array([int(bb[k]['success'])-int(aa[k]['success']) for k in keys])
   pair_seed=int.from_bytes(hashlib.sha256((key+'|'+alt+'|'+stratum).encode()).digest()[:8],'little')
   pair_rng=np.random.default_rng(pair_seed)
   boots=delta[pair_rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)*100
   paired.append({'baseline':key,'variant':alt,'stratum':stratum,'n':len(keys),
    'delta_percentage_points':100*float(delta.mean()),'paired_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
    'wins':int((delta==1).sum()),'losses':int((delta==-1).sum()),
    'interpretation':'Episode bootstrap conditional on this trained model and known layout; not cross-training-seed uncertainty.'})
maps={}
for key,rows in maprows:maps.setdefault(key,[]).extend(rows)
maps={key:{**metric(rows),'complete':len(rows)==100,'across_wall':metric([r for r in rows if r['across_wall']]),'same_side':metric([r for r in rows if not r['across_wall']])} for key,rows in maps.items()}
payload={'updated':now,'stage':state['stage'],'queue_records':len(jobs),'completed_queue_records':sum(j['status']=='complete' for j in state['jobs']),
 'groups':summary,'paired':paired,'maps':maps,'audits':audits,'errors':errors,
 'cautions':['Exclude smoke tests. No pooled independent-trial CI across reused maps.',
 'Single-seed episode uncertainty is not training-seed uncertainty.',
 'Synthetic map goals have no logged-trajectory horizon even when adapter key is 75.',
 'Known-layout heldout trajectories do not establish unknown-map navigation.']}
save(R/'reports/navigation_results_audit.json',payload)
if (R/'heads/temporal3072/summary.json').exists():
 import runpy
 runpy.run_path(str(R/'head_result_report.py'))
 runpy.run_path(str(R/'geometry_result_report.py'))
 runpy.run_path(str(R/'transfer_result_report.py'))
lines=['# 严格导航实验：阶段结果','',f'更新时间：{now}','',f"当前阶段：{state['stage']}。有限队列 {len(jobs)} 条记录，已完成 {payload['completed_queue_records']} 条；记录数包括冒烟/训练/报告任务，不等于正式独立实验数量。",
 '',f"结果一致性检查：已核对 {len(audits)} 个正式评估输出，发现 {len(errors)} 项需处理问题。"]
if errors:lines+=['','存在审计错误，相关结果已从汇总排除；修复前不作为研究结论。']
lines+=['','## 已知布局、保留轨迹','']
if summary:
 lines+=['| 模型与设置 | 病例/计划数 | SR (%) | 排除初始成功 SR (%) | 状态 |','|---|---:|---:|---:|---|']
 for k,v in summary.items():lines.append(f"| {k} | {v['n']}/{v['expected_n']} | {v['success_rate']:.1f} | {v['excluding_initial_success_rate']} | {'完整组' if v['complete'] else '部分组，勿定论'} |")
else:lines+=['尚无完成并通过审计的正式组；不从冒烟测试推断性能。']
lines+=['', '原始无热启动协议与新增热启动/动作边界对照分开列出。后者用于验证计划延续性和预测—执行动作一致性；尚不能将任何下降单独归因于模型表示。']
lines+=['','## 配对差值','']
for p in paired:
 lo,hi=p['paired_bootstrap_95'];lines.append(f"- {p['variant']} 相对 {p['baseline']}，{p['stratum']}，n={p['n']}：{p['delta_percentage_points']:+.1f} 个百分点，病例配对 bootstrap 95% 区间 [{lo:+.1f}, {hi:+.1f}]。")
if not paired:lines+=['等待同病例、完整组的对照完成。']
lines+=['','## 几何布局变化','']
for k,v in maps.items():lines.append(f"- {k}：{v['n']}/100，SR {v['success_rate']:.1f}%；{'完整组' if v['complete'] else '尚未完成'}。")
if not maps:lines+=['18 布局环境及适配已检查；尚无通过审计的正式模型地图结果。']
lines+=['','## 解释边界与后续','',
 '以上区间仅衡量固定训练模型下的病例不确定性，未包含训练种子不确定性。不同地图复用起终点，不能把所有地图病例视为独立抽样。这里的门位置与朝向变化不构成新连通拓扑；单地图训练的新布局下降也不能单独证明模型架构无法泛化。',
 '', '剩余工作以报告末尾带日期的阶段审阅和队列为准，勿将历史计划当作当前未完成项；部分观测是否加入由证据决定。整轮研究尚未完成。前沿方法及旧结果参见 frontier_transfer_review.md 和根目录 reports/navigation_capability_merged.md。']
(R/'reports/detailed_navigation_report.md').write_text('\n'.join(lines)+'\n')
for path in [R/'reports/frontier_transfer_review.md',ROOT/'reports/navigation_capability_merged.md']:
 original=path.read_text()
 history_marker='<!-- PRESERVED_STAGE_HISTORY_START -->'
 generated,sep,history=original.partition(history_marker)
 assert sep, 'Missing preserved stage history marker; refusing to rewrite '+str(path)
 body=generated.split('### 当前执行快照')[0].split('## 阶段机制审计记录')[0].rstrip()
 note=R/'reports/protocol_diagnostic_note.md'
 if note.exists():body+='\n\n## 阶段机制审计记录\n\n'+note.read_text().split('\n',1)[1].strip()
 controls=R/'reports/continuity_controls.json'
 if controls.exists():
  body+='\n\n### 计划连续性与动作边界：完整验证对照\n\n'
  for c in read(controls):body+=f"- {c['variant']} 相对 {c['baseline']}：{c['delta_pp']:+.1f} 个百分点，配对病例95%区间 {c['ci95']}。\n"
  body+='\n这些是单模型开发验证，不是跨种子结论。热启动主要恢复关闭计划后缀造成的退化；相对默认长目标62%，上述规划参数对照的最佳验证值66%只高4个百分点，不应将冷启动对照的38个百分点称为新模型改进幅度。动作投影没有稳定收益，负结果保留。\n'
 adaptation=R/'reports/temporal_head_adaptation.md'
 if adaptation.exists():body+='\n\n### 可达性评价头迁移\n\n'+adaptation.read_text().split('\n',1)[1].strip()
 for filename,title in [('head_mechanism_results.md','时间评价头：阶段验证结果'),('retrieval_adaptation.md','训练轨迹检索对照'),('geometry_results.md','几何布局能力结果'),('transfer_mechanism_results.md','迁移机制与研究方向判断'),('geometry_collection_protocol.md','匹配采集与多布局训练准备')]:
  extra_note=R/'reports'/filename
  if extra_note.exists():body+='\n\n### '+title+'\n\n'+extra_note.read_text().split('\n',1)[1].strip()
 body+='\n\n### 当前执行快照\n\n'
 for k in ['strict3072_test_short150','strict3072_test_long_base','strict_random_test_long','strict_noop_test_long']:
  if k in summary and summary[k]['complete']:
   v=summary[k];body+=f"- {k}：n={v['n']}，SR={v['success_rate']:.1f}%，排除初始成功后={v['excluding_initial_success_rate']:.1f}%。\n"
 td=R/'reports/trajectory_diagnostics.json'
 if td.exists():
  g=read(td)['groups'].get('strict3072_test_long_base')
  if g and g['n']==300:
   c=g['failure_categories'];body+=f"\n长目标默认组失败分解：跨墙任务未曾越过墙中线 {c['cross_wall_failed_before_crossing']} 例；曾到达目标一侧但仍失败 {c['cross_wall_failed_after_crossing']} 例；同侧任务失败 {c['same_side_failure']} 例。这是轨迹描述，不是单一原因的因果归因。执行动作分量超界率为 {100*g['executed_action_component_clipping_fraction']:.1f}%，与候选池超界率不同。\n"
 body+='\n'+f"更新时间：{now}。自动回查已获用户明确授权并启用，无本轮12小时时限；完成约定研究后停止，不额外租用或扩容。当前服务器阶段为 {state['stage']}；正式结果持续写入 strict_nav_20260910/reports/detailed_navigation_report.md 与 navigation_results_audit.json。有限队列完成仅表示阶段结束，不代表所有导航研究已完成。\n"
 path.write_text(body+'\n\n'+history_marker+history)
print(json.dumps({'stage':state['stage'],'formal_evals_audited':len(audits),'errors':len(errors),'groups':len(summary),'maps':len(maps)}))
if errors:raise SystemExit(2)
