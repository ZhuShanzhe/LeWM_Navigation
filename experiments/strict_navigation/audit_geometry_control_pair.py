"""Audit the finite controlled-geometry pair and publish paired development results."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from audit_executed_trace import audit_executed_trace
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(R/'geometry_control_protocol.json');state=read(R/'status.json')
completed={j['tag'] for j in state['jobs'] if j['status']=='complete'}
models={};results={};raw={};trace_audits={}
maps=[('train',1,49),('validation',1,65),('validation',1,129),('validation',0,65),('validation',0,129)]
for v in ['single','multi']:
 tag=f'geometry_control_{v}_s4001';assert tag in completed
 tr=R/'runs'/tag;s=read(tr/'training_summary.json');a=read(tr/'split_audit.json')
 assert s['full_epoch_budget_completed'] and s['global_step']==51920
 assert a['passed'] and a['clip_counts']=={'train':664684,'validation':83354}
 assert a['manifest_sha256']==protocol['files'][v]['manifest_sha256']
 w=Path(s['weights']);assert w.is_file() and w.stat().st_size>1000000
 models[v]={'summary':s,'weights_sha256':sha(w),'split_audit':a}
 results[v]={};raw[v]={}
 for split,axis,door in maps:
  key=f'{split}_a{axis}_d{door}';rows=[];cases=[];work=0;elapsed=0
  for b in range(2):
   tag=f'geometry_{v}4001_map_{key}_b{b}';assert tag in completed
   p=R/'runs'/tag;rr=read(p/'cases_results.json');meta=read(p/'case_metadata.json');metric=read(p/'metrics.json')
   assert len(rr)==len(meta['cases'])==50
   modelmeta=read(p/'model_metadata.json');assert modelmeta['model_path']==str(w)
   expected=read(R/'maps'/f'{split}_axis{axis}_door{door}_b{b}.json')['cases']
   assert meta['cases']==expected
   assert all((x['episode'],x['start_step'])==(c['episode'],c['start_step']) for x,c in zip(rr,expected))
   assert metric['telemetry']['candidate_model_steps']>0 and metric['telemetry']['encoder_images']>0
   assert meta['normalization_sha256']==protocol['files'][v]['normalization_sha256']
   assert meta['normalization_path']==str(R/'collections'/v/'normalization.json')
   assert np.isclose(np.mean([x['success'] for x in rr])*100,metric['metrics']['success_rate'])
   audit=read(p/'map_adapter_audit.json')
   assert len(audit)==50 and all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in audit)
   trace=np.load(p/'trace.npz');trace_audits[tag]=audit_executed_trace(trace,rr)
   assert all(np.isfinite(x['endpoint_distance']) and np.isfinite(x['path_length']) for x in rr)
   rows+=rr;cases+=meta['cases'];work+=metric['telemetry']['candidate_model_steps'];elapsed+=metric['elapsed_seconds']
  def sr(items):return float(np.mean([x['success'] for x in items])*100) if items else None
  assert len(rows)==100
  results[v][key]={'n':len(rows),'sr':sr(rows),'sr_excluding_initial':sr([x for x in rows if not x['initial_success']]),'cross_sr':sr([x for x in rows if x['across_wall']]),'same_sr':sr([x for x in rows if not x['across_wall']]),'mean_path_all':float(np.mean([x['path_length'] for x in rows])),'mean_endpoint_distance':float(np.mean([x['endpoint_distance'] for x in rows])),'stalled_step_fraction':sum(x['stalled_steps_lt_0_5px'] for x in rows)/sum(x['steps'] for x in rows),'candidate_model_steps':work,'batch_elapsed_seconds_not_online_latency':elapsed}
  raw[v][key]=(rows,cases)
paired={}
for split,axis,door in maps:
 key=f'{split}_a{axis}_d{door}';a,ca=raw['single'][key];b,cb=raw['multi'][key];assert ca==cb
 d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
 rng=np.random.default_rng(4001);boot=d[rng.integers(0,100,(10000,100))].mean(1)*100
 paired[key]={'delta_pp':float(d.mean()*100),'paired_case_ci95_conditional_fixed_map_seed':np.quantile(boot,[.025,.975]).tolist(),'gained':int((d==1).sum()),'lost':int((d==-1).sum())}
payload={'passed':True,'updated_unix':time.time(),'protocol':protocol,'models':models,'trace_audits':trace_audits,'results':results,'paired':paired,'limits':protocol['limits']}
(R/'reports/geometry_control_pair4001.json').write_text(json.dumps(payload,indent=2))
lines=['## 单布局／多布局匹配训练：seed4001 阶段结果','', '两组均从零完成10轮51920步；以下是固定开发病例结果，不是最终测试。单布局与多布局共享回合长度、采集行为混合、模型初始化与优化配置；归一化分别由各自训练数据确定。','', '| 地图 | 单布局SR | 多布局SR | 差值百分点 | 配对95%区间 |','|---|---:|---:|---:|---|']
for key,p in paired.items():lines.append(f"| {key} | {results['single'][key]['sr']:.1f}% | {results['multi'][key]['sr']:.1f}% | {p['delta_pp']:+.1f} | [{p['paired_case_ci95_conditional_fixed_map_seed'][0]:.1f}, {p['paired_case_ci95_conditional_fixed_map_seed'][1]:.1f}] |")
lines+=['','区间仅条件于一个训练种子与对应固定地图；多个地图复用同一组坐标，不合并为独立试验。跨墙／同侧、排除初始成功、轨迹长度、停滞和实际候选展开详见同名JSON。批量评估耗时不是单机器人在线时延。','', '这一步用于辨认数据覆盖效应，不能独立定位编码器、动力学与目标评价的因果作用。尚需关键迁移对照、跨种子确认和新连通拓扑；不因两个数据组训练完成就宣称研究结束。']
text='\n'.join(lines)+'\n';(R/'reports/geometry_control_pair4001.md').write_text(text)
merged=ROOT/'reports/navigation_capability_merged.md';old=merged.read_text()
if lines[0] not in old:merged.write_text(old+'\n\n'+text)
print('GEOMETRY_CONTROL_PAIR_AUDITED',json.dumps(paired))
