"""Paired and stratified results; no claims of unseen-scene generalization."""
from pathlib import Path
import json,time,csv,numpy as np
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909');O=R/'reports';O.mkdir(exist_ok=True)
def read(p):
 try:return json.loads(p.read_text())
 except:return {}
groups={};audit=[];raw=[]
for p in sorted((R/'runs').glob('focus_*/metrics.json')):
 d=read(p);meta=read(p.parent/'case_metadata.json');inv=read(p.parent/'invocation.json')
 if not meta or not d:continue
 name=p.parent.name;series=name.rsplit('_s',1)[0];cases=meta['cases'];success=d['metrics']['episode_successes']
 if len(cases)!=len(success):audit.append({'tag':name,'error':'case/result count mismatch'});continue
 if [c['episode'] for c in cases]!=inv['episodes'] or [c['start_step'] for c in cases]!=inv['start_steps']:audit.append({'tag':name,'error':'invocation/case mismatch'});continue
 h=str(meta['goal_offset']);entry={'tag':name,'seconds':d['elapsed_seconds'],'n':len(cases),'success_rate':d['metrics']['success_rate']}
 try:
  tr=np.load(p.parent/'trace.npz');term=tr['terminated'].reshape(len(tr['terminated']),len(cases),-1).any(-1)
  entry['termination_agrees']=bool(np.array_equal(term.any(0),success))
  if meta['noop']:
   entry['noop_max_action']=float(np.abs(tr['actions']).max())
   xy=tr['proprio'].reshape(len(tr['proprio']),len(cases),-1)[...,:2];start=np.array([c['start_xy'] for c in cases])
   entry['noop_max_displacement']=float(np.linalg.norm(xy-start[None],axis=-1).max())
 except Exception as e:entry['trace_error']=repr(e)
 audit.append(entry)
 for c,ok in zip(cases,success):
  goal=c['goals'][h];row={'series':series,'tag':name,'episode':c['episode'],'start_step':c['start_step'],'goal_offset':int(h),'budget':meta['budget'],'success':bool(ok),'distance':goal['euclidean'],'across_wall':goal['across_wall'],'initially_success':goal['initially_within_success']}
  groups.setdefault(series,[]).append(row);raw.append(row)
(O/'focused_cases_results.json').write_text(json.dumps(raw,indent=2));(O/'focused_audit.json').write_text(json.dumps(audit,indent=2))
if raw:
 with (O/'focused_cases_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
def rate(items):return f"{100*np.mean([r['success'] for r in items]):.1f} (n={len(items)})" if items else '—'
lines=['# 导航局限诊断：配对实验与选题依据','',
'本报告服务于定位问题，不再以全论文复现率为目标。原截止时间不变：北京时间 2026-09-10 06:11:34。第三个完整训练种子与 PushT 从零训练已取消；保留两个独立训练模型、公开检查点及四任务公开评估作为背景。','',
'## 实验设计与解释边界','',
'- 固定 150 条不同轨迹，每轨迹随机选择一个能同时提供 25/75 步目标的起点，分成三个互不重叠的 50 实例批次。所有方法使用相同病例；不按成功失败挑选。',
'- 数据仍来自作者公开 TwoRooms 数据集和同一布局，不是未知地图泛化测试。长轨迹筛选改变了总体分布；不能与官方默认分数直接当成同分布比较。',
'- short：目标间隔 25，执行预算 50；short150：同样短目标、预算 150；long：目标间隔 75、预算 150。short150 与 long 固定预算，但目标位置/路径本身不同，不能把差异全部归因于“时间跨度”。',
'- 长目标规划对照保持起终点和执行预算不变。h10_n150 用更长预测、更少候选；replan1_iter2 用更频繁重规划、更少迭代。这是近似计算量控制，不是严格 FLOPs 匹配；同时报告实际耗时。',
'- 不动基线用于检查初始位置已在成功阈值内的比例。两室默认成功阈值为严格小于 16 像素；过门标签来自已核对的垂直隔墙 x=112。过门与距离相关，分层差异不等于隔墙的独立因果效应。',
'- 以下区间是同一固定布局内、按轨迹重采样的描述性配对 bootstrap（5,000 次），不覆盖训练种子不确定性、环境变化或多重比较误差。所有规划变体是探索性对照，不做挑最好结果后宣称显著。',
'','## 配对集结果','',
'表格为成功率 %；n 是实例数。只在完整 150 例后比较最终均值，未满样本的项目保留 n。',
'',
'| 系列 | 全部 | 排除初始已成功 | 需过门且非初始成功 | 同侧且非初始成功 | 平均每例运行秒数 |',
'|---|---:|---:|---:|---:|---:|']
for key,items in groups.items():
 subset=[x for x in items if not x['initially_success']];runs=[a for a in audit if a['tag'].rsplit('_s',1)[0]==key and 'seconds' in a]
 secs=sum(a['seconds'] for a in runs)/max(1,sum(a['n'] for a in runs))
 lines.append(f"| {key.removeprefix('focus_')} | {rate(items)} | {rate(subset)} | {rate([r for r in subset if r['across_wall']])} | {rate([r for r in subset if not r['across_wall']])} | {secs:.3f} |")
lines+=['','平均每例运行秒数是批量总耗时除以例数，不等于单环境在线延迟。没有并行运行其他正式评估；文件写入和视频记录也会占用时间。','',
'## 相同病例上的成功率差值','',
'下面 A→B 表示 B−A，单位为百分点；病例按轨迹 ID 与起点一一匹配。',
'',
'| 模型 | 对照 | 配对 n | 差值 [95% bootstrap 区间] | B 改善 / B 退步例数 |','|---|---|---:|---|---|']
strata=[]
for series,items in groups.items():
 for cross in [False,True]:
  for lo,hi in [(16,40),(40,80),(80,float('inf'))]:
   sub=[r for r in items if not r['initially_success'] and r['across_wall']==cross and lo<=r['distance']<hi]
   if sub:strata.append({'series':series,'across_wall':cross,'distance_bin':f'{lo}-{hi}','n':len(sub),'success_rate':float(np.mean([r['success'] for r in sub])*100)})
(O/'distance_wall_strata.json').write_text(json.dumps(strata,indent=2))
paired=[];rng=np.random.default_rng(20260910)
for model in ['public','trained3072','trained3073']:
 for a,b in [('short','short150'),('short150','long'),('long','long_cem30'),('long','long_h10_n150'),('long','long_replan1_iter2'),('long','long_replan1_iter10')]:
  left=groups.get(f'focus_{model}_{a}',[]);right=groups.get(f'focus_{model}_{b}',[])
  la={(x['episode'],x['start_step']):x for x in left};rb={(x['episode'],x['start_step']):x for x in right};keys=sorted(la.keys()&rb.keys())
  if not keys:continue
  diff=np.array([int(rb[k]['success'])-int(la[k]['success']) for k in keys]);draw=rng.integers(0,len(diff),(5000,len(diff)));lo,hi=np.percentile(diff[draw].mean(1)*100,[2.5,97.5])
  row={'model':model,'a':a,'b':b,'n':len(keys),'delta_pp':float(diff.mean()*100),'ci_low':float(lo),'ci_high':float(hi),'improved':int((diff>0).sum()),'worsened':int((diff<0).sum())};paired.append(row)
  lines.append(f"| {model} | {a} → {b} | {len(keys)} | {row['delta_pp']:.1f} [{lo:.1f}, {hi:.1f}] | {row['improved']} / {row['worsened']} |")
(O/'paired_comparisons.json').write_text(json.dumps(paired,indent=2))
lines+=['','## 选题如何由证据决定','',
'1. **长时预测可靠性与自适应规划**：若固定病例下长预测变差，而更频繁反馈能改善，优先研究预测误差/不确定性驱动的规划跨度或重规划。需要确认两个独立训练模型趋势一致，并控制额外计算开销。单纯“多步误差累积”不是新贡献。',
'2. **面向可达性的表征或拓扑子目标**：若真实模拟后的 latent 成本排序仍与过门路径不一致，且增加搜索量不能修复，才更支持目标度量/拓扑方向。现有固定候选审计只是观察性分解，不是 oracle 规划介入，因此目前只能提出机制假说。',
'3. **训练目标与控制效用的协调**：已有 3,000 步正则化消融显示表示有效秩/方差与短期控制成绩不一一对应。可以研究任务相关正则化或课程，但当前只有一个训练种子的早期消融，优先级低于得到重复验证的导航问题。',
'4. **记忆、语言与真实室内导航**：本轮未测试遮挡、多视角和未见布局，不能从 TwoRooms 结果推断这些方向已被验证。它们是下一阶段的外部验证，不应包装成本轮已发现的模型通用缺陷。',
'',
'## 数据质量与状态','']
bad=[a for a in audit if 'error' in a or 'trace_error' in a or a.get('termination_agrees') is False or a.get('noop_max_action',0)>0 or a.get('noop_max_displacement',0)>1e-4]
lines.append(f"已收集 {len(audit)} 次正式配对评估；自动一致性异常 {len(bad)} 次。详见 focused_audit.json。若有异常，相关结果应先隔离核对。")
for a in bad:lines.append('- '+json.dumps(a,ensure_ascii=False))
lines.append('距离与过门的二维分层（16–40、40–80、≥80 像素）见 distance_wall_strata.json；小样本分层只作线索。')
state=read(R/'status.json');lines.append('当前队列阶段：'+state.get('stage','未知'))
lines+=['','其余基础实验、协议差异与失败记录见 reproduction_report.md；本文件更新方向优先级，不删除不利或不显著结果。最终选题建议须在全部可用结果核对后另行总结。']
(O/'focused_navigation_report.md').write_text('\n'.join(lines),encoding='utf-8')
print('FOCUSED_REPORT_UPDATED',len(audit),len(paired),len(bad))
