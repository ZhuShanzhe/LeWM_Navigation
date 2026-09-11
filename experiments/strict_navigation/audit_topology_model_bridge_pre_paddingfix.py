"""Verify every learned interface, then report matched-engine bridge without topology claims."""
import json,hashlib,time,sys
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'topology_bridge_v1';ROOT=R.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(P/'protocol.json');mode=sys.argv[1];assert mode in ['smoke','bridge']
state=read(R/'status.json');completed={x['tag'] for x in state['jobs'] if x['status']=='complete'}
for f,h in protocol['case_files_sha256'].items():assert sha(Path(f))==h,f
for f,h in protocol['source_sha256'].items():assert sha(Path(f))==h,f
models={}
for v,m in protocol['models'].items():
 assert m['training_run'] in completed
 summary=read(R/'runs'/m['training_run']/'training_summary.json')
 audit=read(R/'runs'/m['training_run']/'split_audit.json')
 assert summary['full_epoch_budget_completed'] and summary['global_step']==51920 and audit['passed']
 assert summary['weights']==m['weights'] and Path(m['weights']).stat().st_size>1000000
 assert sha(Path(m['normalization']))==m['normalization_sha256']
 models[v]={'weights_sha256':sha(Path(m['weights'])),'weights_bytes':Path(m['weights']).stat().st_size,'training_summary':summary,'normalization_sha256':m['normalization_sha256']}
if mode=='bridge':
 smoke=read(P/'model_smoke_audit.json');assert smoke['passed']
 assert {v:m['weights_sha256'] for v,m in models.items()}=={v:m['weights_sha256'] for v,m in smoke['models'].items()}
records=[]
for j in protocol['jobs']:
 tag=j['tag']
 if not tag.startswith('topology_model_smoke_' if mode=='smoke' else 'topology_bridge_'):continue
 assert tag in completed,tag
 v=next(v for v in models if f'_{v}_' in tag);p=R/'runs'/tag
 rows=read(p/'cases_results.json');meta=read(p/'case_metadata.json');metric=read(p/'metrics.json');model=read(p/'model_metadata.json');adapter=read(p/'map_adapter_audit.json')
 expected=read(Path(j['env']['LEWM_CASE_FILE']))['cases']
 assert meta['cases']==expected and len(rows)==len(adapter)==len(expected)
 assert meta['case_sha256']==protocol['case_files_sha256'][j['env']['LEWM_CASE_FILE']]
 assert meta['normalization_sha256']==models[v]['normalization_sha256']
 assert model['model_path']==protocol['models'][v]['weights'] and model['parameters']>1000000
 assert all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in adapter)
 assert all((x['episode'],x['start_step'])==(c['episode'],c['start_step']) for x,c in zip(rows,expected))
 assert all(not x['initial_success'] for x in rows)
 assert np.isclose(np.mean([x['success'] for x in rows])*100,metric['metrics']['success_rate'])
 trace=np.load(p/'trace.npz');assert all(np.isfinite(trace[k]).all() for k in trace.files)
 tel=metric['telemetry'];assert tel['cost_calls']>0 and tel['candidate_model_steps']>0 and tel['encoder_images']>0 and tel['predictor_token_inputs']>0
 # Confirm all executed custom states obey collision geometry, not only reset images.
 if 'topology_spec' in meta:
  from topology_env import TopologyEnv
  env=TopologyEnv(meta['topology_spec'])
  pos=trace['proprio'].reshape(len(trace['proprio']),len(rows),-1)
  for i,x in enumerate(rows):
   assert all(env.valid(xy[:2]) for xy in pos[:x['steps'],i]),(tag,i)
  env.close()
 assert all(np.isfinite(x['path_length']) and np.isfinite(x['endpoint_distance']) and (x['endpoint_distance']<16.0001 if x['success'] else True) for x in rows)
 records.append({'tag':tag,'model':v,'n':len(rows),'case_sha256':meta['case_sha256'],'candidate_model_steps':tel['candidate_model_steps'],'encoder_images':tel['encoder_images'],'batch_elapsed_seconds_not_online_latency':metric['elapsed_seconds'],'rows':rows,'cases':expected})
if mode=='smoke':
 assert len(records)==10 and all(x['n']==2 for x in records)
 result={'passed':True,'updated_unix':time.time(),'models':models,'checks':[{k:v for k,v in x.items() if k not in ['rows','cases']} for x in records],'limits':'20 brief learned interface checks; no topology performance estimate.'}
 (P/'model_smoke_audit.json').write_text(json.dumps(result,indent=2))
 print('TOPOLOGY_MODEL_INTERFACE_PASSED',len(records),flush=True)
else:
 assert len(records)==8
 results={};paired={}
 for v in models:
  results[v]={};group={}
  for engine in ['original','custom']:
   rr=sorted([x for x in records if x['tag'].startswith(f'topology_bridge_{v}_{engine}_')],key=lambda x:x['tag'])
   rows=sum([x['rows'] for x in rr],[]);cases=sum([x['cases'] for x in rr],[])
   assert len(rows)==100 and sum(c['goals']['75']['different_room'] for c in cases)==80
   def sr(mask):return 100*float(np.mean([x['success'] for x,c in zip(rows,cases) if mask(c)]))
   results[v][engine]={'n':100,'sr':sr(lambda c:True),'sr_excluding_initial':sr(lambda c:True),'same_room_sr':sr(lambda c:not c['goals']['75']['different_room']),'cross_room_sr':sr(lambda c:c['goals']['75']['different_room']),'mean_path_all':float(np.mean([x['path_length'] for x in rows])),'stalled_step_fraction':sum(x['stalled_steps_lt_0_5px'] for x in rows)/sum(x['steps'] for x in rows),'candidate_model_steps':sum(x['candidate_model_steps'] for x in rr),'encoder_images':sum(x['encoder_images'] for x in rr),'batch_elapsed_seconds_not_online_latency':sum(x['batch_elapsed_seconds_not_online_latency'] for x in rr)}
   group[engine]=(rows,cases)
  a,ca=group['original'];b,cb=group['custom'];assert ca==cb
  d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
  rng=np.random.default_rng(9104201);boot=d[rng.integers(0,100,(10000,100))].mean(1)*100
  paired[v]={'custom_minus_original_pp':float(d.mean()*100),'paired_case_ci95_fixed_model_and_map':np.quantile(boot,[.025,.975]).tolist(),'gained':int((d==1).sum()),'lost':int((d==-1).sum())}
 result={'passed':True,'updated_unix':time.time(),'models':models,'results':results,'paired':paired,'limits':protocol['limits'],'next_stage_requires_review':True}
 (P/'bridge_results.json').write_text(json.dumps(result,indent=2))
 lines=['## 新拓扑前置桥接：相同两房间的新旧模拟器对照','','本节不是多房间能力结果；只量化改变渲染边界与碰撞规则后的闭环影响。两种训练数据条件各自使用同一最终模型、归一化、100个起终点和规划预算比较新旧模拟器。','','| 训练条件 | 原模拟器SR | 新模拟器SR | 新减旧百分点 | 配对95%区间 |','|---|---:|---:|---:|---|']
 for v,p in paired.items():
  lo,hi=p['paired_case_ci95_fixed_model_and_map'];lines.append(f"| {v} | {results[v]['original']['sr']:.1f}% | {results[v]['custom']['sr']:.1f}% | {p['custom_minus_original_pp']:+.1f} | [{lo:.1f}, {hi:.1f}] |")
 lines+=['','区间条件于固定模型与地图，并非跨模型或跨拓扑可靠性。即使差值接近零也不证明两个仿真器等价；原性能接近零时尤其无法据此排除仿真器影响。需要结合单步、跨门与轨迹审计解释。完整分层/路径/停滞/计算记录见 strict_nav_20260910/topology_bridge_v1/bridge_results.json。','',
 '下一阶段须审阅本对照后才能开展冻结病例的多房间模型评估。新环境没有用于模型训练，参考控制器可解也不代表模型可解。']
 text='\n'.join(lines)+'\n';(R/'reports/topology_bridge_results.md').write_text(text)
 merged=ROOT/'reports/navigation_capability_merged.md';old=merged.read_text()
 if lines[0] not in old:merged.write_text(old+'\n\n'+text)
 print('TOPOLOGY_BRIDGE_AUDITED',json.dumps(paired),flush=True)
