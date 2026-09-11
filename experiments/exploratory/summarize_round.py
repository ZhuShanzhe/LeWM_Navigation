"""Generate an evidence-only Chinese living report and machine-readable tables."""
import json,time,csv,math,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909');O=R/'reports';O.mkdir(exist_ok=True)
def read(p,default=None):
 try:return json.loads(p.read_text())
 except Exception:return default if default is not None else {}
def date(t):return datetime.fromtimestamp(t,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
def fmt(x,n=3):
 try:return f'{float(x):.{n}f}'
 except:return str(x)
start=read(R/'start.json');state=read(R/'status.json')
rows=[];trains=[];diags={}
for p in sorted((R/'runs').glob('*/metrics.json')):
 if p.parent.name.startswith('smoke'):continue
 m=read(p);i=read(p.parent/'invocation.json');meta=read(p.parent/'model_metadata.json');a=m.get('argv',[])
 def arg(k,default=''):
  return next((v.split('=',1)[1] for v in reversed(a) if v.startswith(k+'=')),default)
 env=a[a.index('--config-name')+1] if '--config-name' in a else 'unknown'
 ep=m.get('metrics',{}).get('episode_successes',[]);n=len(ep);k=sum(bool(v) for v in ep)
 z=1.96;ph=k/max(n,1);den=1+z*z/max(n,1);cen=(ph+z*z/(2*max(n,1)))/den;half=z*math.sqrt(ph*(1-ph)/max(n,1)+z*z/(4*max(n,1)**2))/den
 rows.append(dict(tag=p.parent.name,env=env,policy=arg('policy'),seed=i.get('seed'),n=n,success_rate=m.get('metrics',{}).get('success_rate'),wilson_low=100*(cen-half),wilson_high=100*(cen+half),goal_offset=i.get('goal_offset'),budget=i.get('eval_budget'),cem_steps=arg('solver.n_steps',30),horizon=arg('plan_config.horizon',5),receding=arg('plan_config.receding_horizon',5),samples=arg('solver.num_samples',300),history=meta.get('history'),seconds=m.get('elapsed_seconds')))
for p in sorted((R/'runs').glob('*/training_summary.json')):
 if p.parent.name.startswith('smoke'):continue
 t=read(p);t['tag']=p.parent.name;trains.append(t)
for p in sorted((R/'diagnostics').glob('*/diagnostics.json')):diags[p.parent.name]=read(p)
(O/'experiment_table.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
if rows:
 with (O/'experiment_table.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'training_table.json').write_text(json.dumps(trains,indent=2))
lines=['# LeWorldModel：12 小时限时复现记录','','**范围已调整**：用户现在优先要求可信的导航局限诊断，不再追求完整复现。已取消第三个训练种子与 PushT 从零训练；当前计划见 ../FOCUSED_PLAN.md，主诊断见 focused_navigation_report.md。以下保留历史实验，不表示继续执行旧计划。','',
f"更新时间：{date(time.time())}（北京时间）。预算起点：{date(start['start_unix'])}；截止：{date(start['deadline_unix'])}。",
f"队列状态：{state.get('stage','尚未启动监督队列')}。这是自动更新的实验记录，不代表未完成实验已有结果。",
'','## 1. 范围与原文协议','',
'原论文：[LeWorldModel，arXiv:2603.19312v3](https://arxiv.org/html/2603.19312v3)。使用作者公开代码提交 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac，单张 RTX 5090（32 GB）。本轮以导航 TwoRooms 的完整从零训练和失效诊断为重点，同时争取覆盖 PushT、Reacher、Cube 的公开检查点评估。随机策略仅是 sanity baseline，不替代 PLDM、DINO-WM、GCBC 等论文对照。',
'',
'| 项目 | 论文/公开实现差异 | 本轮处理 |','|---|---|---|',
'| 训练轮次 | 论文附录 D 为 10 epochs；仓库默认 100 | 以 10 epochs 为完整训练预算；同时报告实际步数与 epoch 等效值 |',
'| 历史长度 | 论文 TwoRooms 为 1；公开 HF 检查点为 3；本地 rollout 默认硬编码 3 | TwoRooms 从零训练为 1，检查点评估绑定各自历史容量，不能把两者当成完全相同设置 |',
'| SIGReg 权重 | 论文为 0.1；仓库默认 0.09 | 主训练使用 0.1；消融使用 0 / 0.01 |',
'| CEM | 附录 B 泛称 30；附录 D 为 PushT 30，其他 10 | 主评估按 D，额外检查 TwoRooms 的 30 次迭代 |',
'| 评估 | goal offset 25，budget 50；horizon/receding/action block 均 5 | 默认每配置 50 个实例，评估种子 42/43/44；长目标实验另标注 |',
'| 模型规模 | 论文约 15M；公开权重实际参数量存在差异 | 原样保留；逐次 model_metadata.json 记录实际量，不暗中改模型凑数 |',
'',
'公开 PushT HDF5 实测为 18,685 条轨迹、2,336,736 帧（均长 125.06），与论文所述 20k、均长约 196 不同。原因未确认，见 ../DATASET_AUDIT.md；不能宣称完全相同数据条件。',
'训练种子与评估种子是不同层级：同一模型的三个评估种子不等于三次独立训练。复用同一数据集采样起终点不构成未知场景泛化证明。随机训练划分沿用官方代码，未宣称是未见地图测试。',
'','## 2. 实际训练覆盖','',
'| 训练 | 实际步数 | epoch 等效值 | 完成 10 epochs | 训练耗时（分钟） |','|---|---:|---:|---|---:|']
for t in trains:lines.append(f"| {t['tag']} | {t.get('global_step')} | {fmt(t.get('epoch_equivalent'))} | {'是' if t.get('full_epoch_budget_completed') else '否（限时/限步）'} | {fmt(t.get('elapsed_seconds',0)/60,1)} |")
if not trains:lines.append('| 暂无完整保存的训练结果 | — | — | — | — |')
for p in sorted((R/'runs').glob('*/progress.json')):
 if p.parent.name.startswith('smoke') or (p.parent/'training_summary.json').exists():continue
 d=read(p);lines+=['',f"尚无训练结束记录：{p.parent.name}，最近进度 {d.get('step')} 步 / 每轮 {d.get('batches_per_epoch')} 步，更新时间 {date(d.get('updated_unix',time.time()))}。"]
lines+=['','消融只在主模型 step=3000 与消融确实达到 3000 步时作等训练步数比较。64 维实验同时改变预测器/动作嵌入宽度，并非仅改变一层投影；不能据此孤立归因于潜空间维度。限步结果只是早期训练现象，不代表收敛表现。',
'','## 3. 实际闭环控制结果','',
'成功率采用库返回的定义；括号区间为逐次运行的二项 Wilson 95% 描述区间。多个起点可能来自同一轨迹，实例独立性近似并不严格，不能把区间当成严格的地图泛化置信区间。',
'',
'| 实验 | 环境 | 评估 seed | n | 成功率 % [95% 区间] | 目标间隔 / 执行预算 | CEM 次数 | 秒 |','|---|---|---:|---:|---|---|---:|---:|']
for r in rows:lines.append(f"| {r['tag']} | {r['env']} | {r['seed']} | {r['n']} | {fmt(r['success_rate'],1)} [{fmt(r['wilson_low'],1)}, {fmt(r['wilson_high'],1)}] | {r['goal_offset']} / {r['budget']} | {r['cem_steps']} | {fmt(r['seconds'],1)} |")
if not rows:lines.append('| 等待队列评估；冒烟测试不计入正式结果 | — | — | — | — | — | — | — |')
lines+=['','同名系列仅在环境、预算、规划参数、训练检查点全部一致时聚合。规划参数扫描不等计算量；成功率变化不能直接称为无成本改进。长目标采样会改变可用轨迹集合，短/长差异不是严格单变量因果比较。','',
'## 4. 表征、长预测与候选排序诊断','',
'不同模型的 latent 坐标和尺度不相同，不能直接按原始 latent MSE 跨模型排名；尤其表征塌缩会让真实表征间距离很小。须联合逐维方差、有效秩、探针与闭环成功率判断。',
'位置探针采用 500/100/100 个轨迹的训练/验证/测试划分，每轨迹取 8 帧；探针测试轨迹不等于世界模型从未见过的训练数据。标准化 MSE 使用探针训练集位置标准差；论文未充分明确的归一化细节不强行对齐。']
for tag,d in diags.items():
 pr=d.get('probes',{});rep=d.get('representation',{})
 lines+=['',f'### {tag}','',f"线性探针：标准化 MSE {fmt(pr.get('linear',{}).get('normalized_mse'),6)}，R² {fmt(pr.get('linear',{}).get('r2'),6)}；MLP 探针 MSE {fmt(pr.get('mlp',{}).get('normalized_mse'),6)}。表征有效秩 {fmt(rep.get('effective_rank'),2)}，平均逐维标准差 {fmt(rep.get('mean_dimension_std'),4)}。",'',
 '| 预测跨度（模型步；1 步=5 环境步） | 开环 latent MSE | 每步真值纠正 MSE | 保持首帧 MSE |','|---:|---:|---:|---:|']
 for h,v in d.get('rollout_mse',{}).items():lines.append(f"| {h} | {fmt(v.get('open_loop'),6)} | {fmt(v.get('teacher_forced'),6)} | {fmt(v.get('persistence'),6)} |")
 c=d.get('fixed_candidates',[])
 if c:
  vals=lambda key:[v[key] for v in c if isinstance(v.get(key),(int,float)) and math.isfinite(v[key])]
  lines+=['',f"固定候选审计：{len(c)} 组起终点、每组 24 个候选动作序列。预测成本与真实模拟后 latent 成本的组内 Spearman 中位数：{fmt(np.median(vals('predicted_vs_actual_latent_rank')),4)}；真实 latent 成本与真实欧氏目标距离的排序相关中位数：{fmt(np.median(vals('actual_latent_vs_actual_distance_rank')),4)}。门口路径长度只是几何代理，不是精确测地线。候选集不是完整 CEM 搜索空间。"]
 sur=d.get('surprise',{})
 lines+=['',f"合成时间跳跃检测 AUROC：{fmt(sur.get('synthetic_jump_auroc'),4)}（{sur.get('n_pairs')} 对）。这只是显著时间不连续的替代诊断，不等同论文完整物理违例/穿墙测试，更不能由高分推断已学会物理规律。"]
lines+=['','### 等步数消融的解释','',
'| 设置（3,000 步，同训练 seed） | 三个评估 seed 的平均成功率 % | 表征有效秩 | 平均逐维标准差 |',
'|---|---:|---:|---:|']
for label,prefix,dtag in [('默认正则化 0.1','base3000_eval_','base3000'),('无正则化','r12_noreg_eval_','r12_noreg'),('较弱正则化 0.01','r12_lowreg_eval_','r12_lowreg'),('64 维容量变体','r12_dim64_eval_','r12_dim64')]:
 subset=[v for v in rows if v['tag'].startswith(prefix)]
 if len(subset)==3 and dtag in diags:
  rep=diags[dtag]['representation'];lines.append(f"| {label} | {fmt(np.mean([v['success_rate'] for v in subset]),2)} | {fmt(rep['effective_rank'],2)} | {fmt(rep['mean_dimension_std'],5)} |")
lines+=['','有效秩低不自动等于表征塌缩；任务相关的低维表示可能有用。须同时看逐维标准差、位置可读性和控制表现。现有结果最多说明早期学习速度与正则化/容量有关；无独立训练重复、无完整收敛消融，不作显著性或最终优劣结论。64 维变体同时改变网络容量。']
lines+=['','## 5. 不足与科研方向：证据等级','',
'1. **多步预测稳定性（可检验的优先方向）**：比较开环与每步真值纠正误差随跨度的差距，再结合长目标闭环成功率；若两者一致恶化，可提出多步一致性训练、不确定性约束或自适应规划。误差增长本身不新颖，须用等算力基线证明具体改进。',
'2. **目标距离不等于可达路径（待验证机制）**：检查真实模拟端点的 latent 排序是否仍偏离通行路径，而不只检查模型预测端点。若真实端点也排错，优先尝试拓扑子目标/可达性价值；若仅预测排错，优先改动力学。候选审计只是诊断，不能直接声称因果证明。',
'3. **历史与部分可观测性（当前任务尚未检验）**：TwoRooms 是简化可视环境；导航中的遮挡、回访和记忆需要新实验。可研究有记忆的 JEPA + 拓扑图，但本轮成功率不能支持真实室内 VLN 泛化。',
'4. **表征防塌缩与控制效用（早期消融）**：同时看有效秩、位置探针、预测误差和成功率，避免仅凭 loss 下降评价好坏。等 3000 步消融不足以否定最终收敛能力。',
'5. **部署计算与性能权衡**：记录实际耗时、规划次数和成功率。只有把重规划/采样改动的算力成本算进去，才能判断是否是实用改进。',
'',
'本轮不能替代完整论文全表复现：所有环境从零多种子训练、全部对比方法、多种数据规模、全部物理违例、真实导航迁移均可能未覆盖。没有实际结果的项目一律保留为待测，不以“预计”填表。',
'','## 6. 阻碍、失败与待完成项目','']
jobs=state.get('jobs',[])
bad=[j for j in jobs if j.get('status') not in ('complete','running')]
if bad:
 for j in bad:lines.append(f"- {j.get('tag')}：{j.get('status')}；{j.get('error','详情见 logs/ 对应日志')}。")
else:lines.append('目前无监督队列记录的失败；尚未执行的计划项目不代表已通过。')
if (R/'diagnostics/trained3072/diagnostics.json').exists():lines.append('- trained3072 诊断已有完整输出；首次失败记录保留，随后在串行诊断阶段修复字段名并成功补跑。')
for env in ['pusht','reacher','cube']:
 ready=read(R/f'ready_datasets_{env}.json');lines.append(f"- {env} 数据集：{'解压完成' if ready.get('h5') else '未就绪/下载中'}。")
if not (R/'baseline_drive_index.json').exists():lines.append('- 作者 Google Drive 基线列表尚未成功获取；不能声称完成 PLDM/DINO-WM/GCBC 比较。')
lines+=['','## 7. 文件与可复查性','',
'本目录上级保留全部脚本、时间预算、队列状态和日志；runs/ 包含检查点对应的调用参数、成功标志、动作轨迹、视频；diagnostics/ 保留探针数据与候选成本；data/checkpoints/ 在科研根目录下保存权重和训练配置。experiment_table.csv/json 为机器可读结果。',
'',
'已另有上一轮的导航论文综述：../../reports/navigation_research.md；其研究建议仍需本轮证据约束。服务器不会被自动关机，只停止本轮自己启动的作业。']
(O/'reproduction_report.md').write_text('\n'.join(lines),encoding='utf-8')
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for tag,d in diags.items():
  v=d.get('rollout_mse',{});h=list(map(int,v))
  axes[0].plot(h,[v[str(k)]['open_loop'] for k in h],marker='o',label=tag)
 axes[0].set(xlabel='Model steps (5 environment steps each)',ylabel='Open-loop latent MSE');axes[0].legend(fontsize=7)
 for p in R.glob('spt_cache/**/*.csv'):
  if 'metrics' not in p.name:continue
  import pandas as pd
  try:
   df=pd.read_csv(p);key='fit/loss'
   if key in df and 'step' in df:
    q=df[['step',key]].dropna()
    if len(q):axes[1].plot(q['step'],q[key],alpha=.7,label=p.parent.name[-25:])
  except Exception:pass
 axes[1].set(xlabel='Training step',ylabel='Logged training loss');axes[1].set_yscale('log')
 if axes[1].lines:axes[1].legend(fontsize=6)
 fig.tight_layout();fig.savefig(O/'diagnostics_overview.png',dpi=150);plt.close(fig)
except Exception as e:(O/'plot_warning.txt').write_text(repr(e))
print('REPORT_UPDATED',len(rows),len(trains),len(diags))
