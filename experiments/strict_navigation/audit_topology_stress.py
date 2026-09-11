"""Audit frozen topology stress results, including baselines and conditional paired intervals."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from audit_executed_trace import audit_executed_trace
from topology_env import TopologyEnv
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent;P=R/'topology_stress_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
q=read(P/'protocol.json');done={x['tag'] for x in read(R/'status.json')['jobs'] if x['status']=='complete'}
for f,h in q['source_sha256'].items():assert sha(Path(f))==h,f
for f,h in q['case_files_sha256'].items():assert sha(Path(f))==h,f
assert sha(R/'topology_bridge_v1/bridge_results.json')==q['bridge_results_sha256']
for v,m in q['models'].items():assert sha(Path(m['weights']))==q['frozen_model_sha256'][v]
records={};trace_audits={}
for j in q['jobs']:
 tag=j['tag']
 if tag.startswith('audit_'):continue
 assert tag in done,tag
 path=R/'runs'/tag;rows=read(path/'cases_results.json');meta=read(path/'case_metadata.json');metric=read(path/'metrics.json')
 expected=read(Path(j['env']['LEWM_CASE_FILE']))
 assert meta['cases']==expected['cases'] and len(rows)==50
 assert meta['case_sha256']==q['case_files_sha256'][j['env']['LEWM_CASE_FILE']]
 assert meta['normalization_path']==j['env']['LEWM_NORMALIZATION_FILE']
 assert meta['normalization_sha256']==sha(Path(meta['normalization_path']))
 assert all((x['episode'],x['start_step'])==(c['episode'],c['start_step']) for x,c in zip(rows,meta['cases']))
 adapter=read(path/'map_adapter_audit.json')
 assert len(adapter)==50 and all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in adapter)
 tr=np.load(path/'trace.npz');trace_audits[tag]=audit_executed_trace(tr,rows)
 env=TopologyEnv(meta['topology_spec']);pos=tr['proprio'].reshape(len(tr['proprio']),50,-1)
 for i,row in enumerate(rows):
  assert not row['initial_success']
  assert row['different_room']==meta['cases'][i]['goals']['75']['different_room']
  assert all(env.valid(xy[:2]) for xy in pos[:row['steps'],i]),(tag,i)
  assert np.isfinite(row['path_length']) and np.isfinite(row['endpoint_distance'])
  if row['success']:assert row['endpoint_distance']<16.0001
 env.close()
 assert np.isclose(np.mean([x['success'] for x in rows])*100,metric['metrics']['success_rate'])
 v=next(x for x in ['single','multi','random','noop'] if tag.startswith('topology_stress_'+x+'_'))
 tel=metric['telemetry']
 if v in q['models']:
  model=read(path/'model_metadata.json');assert model['model_path']==q['models'][v]['weights']
  assert tel['candidate_model_steps']>0 and tel['encoder_images']>0 and tel['predictor_token_inputs']>0
 else:
  assert tel['candidate_model_steps']==0
  if v=='noop':assert all(x['path_length']<1e-5 and not x['success'] for x in rows)
 records[tag]={'rows':rows,'cases':meta['cases'],'metric':metric}
results={};paired={}
for mapid in q['maps']:
 results[mapid]={};group={}
 for v in ['single','multi','random','noop']:
  data=[records[f'topology_stress_{v}_{mapid}_b{b}_v1'] for b in range(2)]
  rows=sum([x['rows'] for x in data],[]);cases=sum([x['cases'] for x in data],[])
  assert len(rows)==100 and len({x['episode'] for x in rows})==100 and sum(x['different_room'] for x in rows)==80
  group[v]=(rows,cases)
  def sr(rs):return float(np.mean([x['success'] for x in rs])*100) if rs else None
  results[mapid][v]={'n':100,'sr':sr(rows),'sr_excluding_initial':sr(rows),'same_room_sr':sr([x for x in rows if not x['different_room']]),'cross_room_sr':sr([x for x in rows if x['different_room']]),'mean_path_all':float(np.mean([x['path_length'] for x in rows])),'mean_path_success':float(np.mean([x['path_length'] for x in rows if x['success']])) if any(x['success'] for x in rows) else None,'mean_endpoint_distance':float(np.mean([x['endpoint_distance'] for x in rows])),'stalled_step_fraction':sum(x['stalled_steps_lt_0_5px'] for x in rows)/sum(x['steps'] for x in rows),'candidate_model_steps':sum(x['metric']['telemetry']['candidate_model_steps'] for x in data),'encoder_images':sum(x['metric']['telemetry']['encoder_images'] for x in data),'batch_elapsed_seconds_not_online_latency':sum(x['metric']['elapsed_seconds'] for x in data)}
  results[mapid][v]['room_graph_hops']={str(h):{'n':sum(c['room_graph_hops']==h for c in cases),'sr':sr([x for x,c in zip(rows,cases) if c['room_graph_hops']==h])} for h in sorted({c['room_graph_hops'] for c in cases})}
 a,ca=group['single'];b,cb=group['multi'];assert all(ca==group[v][1] for v in group)
 d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
 rng=np.random.default_rng(9104301);boot=d[rng.integers(0,100,(10000,100))].mean(1)*100
 paired[mapid]={'multi_minus_single_pp':float(d.mean()*100),'ci95_cases_conditional_fixed_models_map':np.quantile(boot,[.025,.975]).tolist(),'gained':int((d==1).sum()),'lost':int((d==-1).sum())}
payload={'passed':True,'updated_unix':time.time(),'results':results,'paired':paired,'trace_audits':trace_audits,'protocol':q,'next_stage_requires_review':True}
(P/'results.json').write_text(json.dumps(payload,indent=2))
lines=['## 多房间未知拓扑压力测试：冻结模型与简单基线','','两模型均仅在两房间数据训练；每个新环境100例，20同房间/80跨房间。以下是自建全观测环境的几何与拓扑联合分布外测试，不是纯拓扑因果实验，不是第一人称导航或最终预留测试。','','| 图结构 | 单布局模型SR | 多布局模型SR | 随机动作SR | 零动作SR |','|---|---:|---:|---:|---:|']
for mapid,g in results.items():lines.append(f"| {mapid} | {g['single']['sr']:.1f}% | {g['multi']['sr']:.1f}% | {g['random']['sr']:.1f}% | {g['noop']['sr']:.1f}% |")
lines+=['','两模型配对差值（固定地图/模型条件下病例bootstrap95%区间）：']
for mapid,p in paired.items():
 lo,hi=p['ci95_cases_conditional_fixed_models_map'];lines.append(f"- {mapid}：多布局减单布局 {p['multi_minus_single_pp']:+.1f} 个百分点，[{lo:.1f}, {hi:.1f}]。")
lines+=['','所有初始位置均不满足成功条件，因此排除初始成功后的SR相同。跨房间、图距离分组、路径、停滞、候选展开与编码量详见 topology_stress_v1/results.json；参考路线不是最短路，不报告SPL。','',
'解释限制：新环境同时改变墙结构、房间数量/尺度和图像分布，桥接只量化相同两房间的新旧碰撞/渲染影响；不能将全部差异归因于拓扑理解。原布局基础成功率偏低时，新图低分更不能单独证明模型缺少长程拓扑推理。单个训练种子配对不足以形成跨种子鲁棒性结论；随机策略亦仅一个固定随机种子。','',
'本阶段只审计能力边界，未加入记忆。下一步优先结合局部控制与目标评分对照定位瓶颈，再决定是否值得进入部分观测；不直接承诺更大模型或层级结构会提升。']
text='\n'.join(lines)+'\n';(R/'reports/topology_stress_results.md').write_text(text)
merged=ROOT/'reports/navigation_capability_merged.md';old=merged.read_text()
if lines[0] not in old:merged.write_text(old+'\n\n'+text)
print('TOPOLOGY_STRESS_AUDITED',json.dumps({m:{v:x['sr'] for v,x in g.items()} for m,g in results.items()}),flush=True)
