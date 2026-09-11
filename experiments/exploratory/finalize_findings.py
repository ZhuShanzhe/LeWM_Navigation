import json,time,subprocess,hashlib
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909');O=R/'reports'
s=json.loads((R/'status.json').read_text());assert s['stage']=='complete' and s['active'] is None
aud=json.loads((O/'focused_audit.json').read_text());assert len(aud)==75
bad=[a for a in aud if 'error' in a or 'trace_error' in a or a.get('termination_agrees') is False or a.get('noop_max_action',0)>0 or a.get('noop_max_displacement',0)>1e-4];assert not bad
rows=json.loads((O/'focused_cases_results.json').read_text());groups={}
for row in rows:groups.setdefault(row['series'],[]).append(row)
assert len(groups)==25 and all(len(v)==150 and len({(r['episode'],r['start_step']) for r in v})==150 for v in groups.values())
pairs=json.loads((O/'paired_comparisons.json').read_text());assert len(pairs)==18 and all(x['n']==150 for x in pairs)
def sr(k):return float(np.mean([x['success'] for x in groups['focus_'+k]])*100)
def seconds(k):
 a=[x for x in aud if x['tag'].rsplit('_s',1)[0]=='focus_'+k]
 return sum(x['seconds'] for x in a)/sum(x['n'] for x in a)
formal=[p for p in (R/'runs').glob('*/metrics.json') if not p.parent.name.startswith('smoke')]
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
diags={t:json.loads((R/'diagnostics'/t/'diagnostics.json').read_text()) for t in ['public' if False else 'official_tworoom','trained3072','trained3073']}
summary={'checked_unix':time.time(),'queue_finished_unix':s['finished_unix'],'elapsed_hours':(s['finished_unix']-json.loads((R/'start.json').read_text())['start_unix'])/3600,'formal_runs':len(formal),'paired_runs':len(aud),'paired_conditions':len(groups),'unique_trajectory_cases':150,'paired_rollouts':len(rows),'paired_comparisons':len(pairs),'audit_failures':bad,'gpu_processes_at_check':gpu,'two_complete_training_seeds':{}}
for seed in [3072,3073]:
 d=json.loads((R/f'runs/r12_tw_s{seed}/training_summary.json').read_text());assert d['full_epoch_budget_completed'] and d['global_step']==58410
 summary['two_complete_training_seeds'][str(seed)]={'steps':d['global_step'],'epochs':d['epoch_equivalent'],'seconds':d['elapsed_seconds']}
summary['models']={}
for t in ['public','trained3072','trained3073']:
 summary['models'][t]={v:{'success_rate':sr(t+'_'+v),'batch_seconds_per_case':seconds(t+'_'+v)} for v in ['short','short150','long','long_cem30','long_h10_n150','long_replan1_iter2','long_replan1_iter10']}
(O/'final_audit.json').write_text(json.dumps(summary,indent=2))
lines=['# 实验结论与后续选题建议','',
'## 结论先说','',
'我建议优先验证“预测可靠性驱动的自适应反馈与规划预算分配”，将“有数据支持、可执行的分层子目标”作为第二方向。暂不优先投入单纯增加 CEM 搜索量、只调 SIGReg 权重、直接加入语言大模型。这个排序是下一轮实验优先级，不是已证明的新方法或论文创新点。','',
f"本轮队列已完成，耗时约 {summary['elapsed_hours']:.2f} 小时（包括准备与下载），早于 12 小时上限。共 {len(formal)} 次正式评估，其中 75 次为固定 150 条轨迹上的配对评估（25 种组合×3 批次），并非 3,750 个独立测试场景。两个从零训练种子均完成 10 epochs、58,410 步。另有三个 3,000 步早期消融，不称为完整收敛实验。自动病例/轨迹检查无异常，GPU 实验进程已退出；服务器没有自动关机。",
'','## 1. 哪些现象有比较稳定的证据','',
'### 1.1 短目标控制较好，长目标下降在两个训练种子中重复出现','',
'| 固定150例 | 公开检查点 | 训练seed3072 | 训练seed3073 |','|---|---:|---:|---:|']
for v,label in [('short','短目标，预算50'),('short150','短目标，预算150'),('long','长目标，预算150'),('long_cem30','长目标，CEM迭代30'),('long_h10_n150','长目标，预测跨度10/候选150'),('long_replan1_iter2','长目标，每5环境步重规划/迭代2'),('long_replan1_iter10','长目标，每5环境步重规划/迭代10')]:
 lines.append(f"| {label} | {sr('public_'+v):.1f}% | {sr('trained3072_'+v):.1f}% | {sr('trained3073_'+v):.1f}% |")
lines+=['',
'默认每次执行 5 个动作块，每块 5 个环境步，故约每 25 环境步重规划；receding=1 则约每 5 环境步重规划。目标间隔25/75是数据轨迹间隔，不是机器人到目标的最短距离。',
'',
'将短目标预算从50增加到150几乎不改变结果；同预算下换成长目标，两个训练模型分别下降39.3和32.0个百分点。配对起点相同，但目标位置、路径长度与障碍关系同时改变，所以不能断言“下降完全由预测时间跨度导致”。',
'',
'不动基线短目标4.7%、长目标0%；随机策略短目标28.0%、长目标2.7%。排除初始即满足16像素成功阈值的病例后，短目标高成功率仍然存在。过门与同侧长目标均有失败；当前不支持简单归因于“只是不知道怎么过门”。',
'',
'### 1.2 多步预测误差增长，但尚未证明它是全部闭环失败的原因','',
'在64条固定轨迹的开放环预测诊断中，模型步=5个环境步。这里使用专家数据中的动作，不等同于 CEM 搜索得到的分布外动作。']
for t in ['trained3072','trained3073']:
 d=diags[t];err=d['rollout_mse'];p=d['probes']['linear']
 lines.append(f"- {t}：位置探针R²={p['r2']:.4f}；开环MSE从1步的{err['1']['open_loop']:.5f}增至15步的{err['15']['open_loop']:.5f}，15步处每步真值纠正MSE为{err['15']['teacher_forced']:.5f}。")
lines+=['',
'位置可读性好不保证能完成复杂规划；开环/真值纠正差距支持研究误差与反馈的关系，但它只是相关诊断。跨模型 latent 的坐标和尺度不同，不能直接按原始 MSE 高低评定模型优劣。',
'',
'### 1.3 加搜索量收益弱，频繁反馈有收益但有成本','',
'CEM迭代10→30仅改善1.3～2.0个百分点，各模型的配对95%区间均覆盖0。这不证明增加搜索绝对无效，但不足以支持将其作为主攻方向。',
'',
'| 更频繁反馈、迭代仍为10 | 成功率增量/百分点 | 配对95%描述区间 | 实测批量耗时倍率 |','|---|---:|---|---:|']
for t in ['public','trained3072','trained3073']:
 p=next(x for x in pairs if x['model']==t and x['b']=='long_replan1_iter10')
 lines.append(f"| {t} | {p['delta_pp']:+.1f} | [{p['ci_low']:.1f}, {p['ci_high']:.1f}] | {seconds(t+'_long_replan1_iter10')/seconds(t+'_long'):.2f}× |")
lines+=['',
'在两个训练模型上，频繁反馈的增益方向一致，但公开检查点没有平均增益；训练历史长度、权重来源等不同，不能把差异完全归因于训练质量。把每次迭代数同步从10减到2后，收益变得不一致。说明“如何选择何时反馈、如何花计算量”值得检验，但“更多反馈必然更好且免费”不成立。',
'',
'增加预测跨度且减半候选数带来4.7～6.7个百分点的正向点估计，但区间均覆盖0；不能宣布它显著优于默认。该设置也不严格等计算量，真实耗时已在详细报告中列出。',
'',
'所有区间均是固定布局内按轨迹重采样的探索性区间，未做多重比较校正，不能据个别区间排除0就声称普适显著结果。',
'',
'## 2. 最值得先做的方向：预测可靠性驱动的反馈与预算分配','',
'**研究问题**：在相同环境步数和严格约束的规划计算预算下，能否只在模型不可靠时增加观测纠正、缩短执行前缀或调整预测跨度，而不是所有状态都高频重规划？',
'',
'最小可行方案（建议，不在本轮继续训练）：',
'1. 冻结已有世界模型，用独立校准轨迹估计未来预测误差或动作候选的不确定性。区分每步真实观测后的预测残差与执行前可用的风险预测，避免用未来真值作弊。',
'2. 先检验风险指标能否预测失败、卡住、明显偏离；比较简单距离阈值、固定频率、随机触发、误差触发。若简单阈值同样好，不引入复杂模型。',
'3. 再让风险控制执行前缀/采样预算，在相同总模型调用量及环境步数下比较，并另外测单环境在线时延。当前批量吞吐量不能替代机器人在线延迟。',
'4. 最后再增加未见门位置、布局与局部观测；训练、校准与测试按完整轨迹/地图拆分。每个模型至少增加独立训练重复后再评估稳定性。',
'',
'**否定条件**：若风险不比距离等简单指标更能预测失败，或收益在等预算后消失，应降低此方向优先级。当前只支持做这项验证，不保证创新性。',
'',
'## 3. 第二方向：可执行且有数据支持的子目标与可达性评分','',
'短程控制强、远程仍困难，为分层分解提供动机，但不是充分证据。首先做隔离诊断：保持同一组候选动作，分别使用模型预测终点、真实模拟终点以及替代目标代价评分；再用真实轨迹中的中间观测作为 oracle 子目标，看低层是否能顺利追踪。特权状态只能用于诊断上界，不能混入纯视觉主方法。',
'',
'只有确认目标评分或子目标可执行性确实造成可恢复的失败，才考虑学习轻量可达性代价，或从历史观测节点选取子目标。现有固定候选审计只是观察性线索，没有完成上述 oracle 闭环介入，因此目前不能声称已定位唯一瓶颈。',
'',
'**相关工作必须避开重复**：',
'- [RC-aux](https://arxiv.org/html/2605.07278) 已将多步开环预测与预算条件可达性监督加入 LeWM；“加多步损失+可达性头”不是空白。',
'- [TRM](https://arxiv.org/html/2605.22164v2) 已研究冻结表征上的时间成对代价及候选排序；必须与其同病例、同预算比较。其论文分数与本轮测试分布不同，不能直接拼表。',
'- [Hi-LeWM](https://arxiv.org/html/2607.12547v2) 已指出高层无约束搜索可能产生难执行子目标，并研究数据支持约束；简单“LeWM+分层”不构成新意。',
'',
'更有研究空间的候选是未知布局/部分观测下的可靠性与可执行性，但本轮未验证这些场景。三篇当前引用的是预印本版本，本文不替它们认定主会录用状态。',
'',
'## 4. 暂时降低优先级的方向','',
'- **只调正则化权重**：3,000步时λ=0.01成功率46%，λ=0.1为20%，无正则24%；这是单训练种子的早期速度差异。默认设置完整训练后很强，不能由短训练宣称默认正则化错误。',
'- **只追求更高表征有效秩**：低秩但非塌缩的表示可能正好编码低维任务状态；必须结合尺度、探针和闭环表现。',
'- **直接加入LLM、语言与复杂记忆**：本轮没有语言、遮挡或真实机器人证据。此时增加模块可能让因果定位更困难，建议待图像目标导航主问题明确后再扩展。',
'',
'## 5. 可信度边界与交付清单','',
'- 两个独立训练种子而非三个；75次配对运行复用150条不同轨迹，而非3750个独立样本。只有同一布局，不能称为泛化结果。',
'- 原训练代码随机划分滑动窗口，可能共享相邻帧；本轮不是严格未见轨迹/地图泛化评估。后续发表实验需重做拆分。',
'- 论文10epochs与仓库100默认、history长度、SIGReg权重和CEM迭代存在差异，均已记录；本地rollout按检查点历史容量绑定。',
'- PushT公开数据量与论文文字不完全一致；Cube/Reacher完成公开权重评估，未从零训练。没有完成全部论文基线、原版全套物理违例和真实导航。',
'- 首次诊断字段名错误已经修复补跑；原失败日志不删除。配对轨迹/终止信号/零动作一致性均通过，但自动检查不是对所有实现细节的数学证明。',
'',
'主建议：research_recommendation.md（本文件）；配对表与区间：focused_navigation_report.md；全部历史记录：reproduction_report.md；逐例数据：focused_cases_results.csv；质量审计：final_audit.json、focused_audit.json；原始权重、日志、轨迹与视频均保留在服务器。',
'',
'本轮实验已结束，不再自动启动新作业。服务器仍开机；若暂不使用，需由用户决定是否在平台停机，以免继续产生租用费用。']
(O/'research_recommendation.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,indent=2))
