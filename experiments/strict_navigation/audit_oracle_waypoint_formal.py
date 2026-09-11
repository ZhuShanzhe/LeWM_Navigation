"""Frozen paired oracle waypoint diagnostic; no extra replans or changed terminal goals."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from audit_executed_trace import audit_executed_trace
from topology_env import TopologyEnv
from oracle_waypoint_router import OracleWaypointRouter
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
P=R/'oracle_waypoint_formal_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sr(rows):return float(np.mean([x['success'] for x in rows])*100) if rows else None
def paired(a,b):
 assert len(a)==len(b)
 d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
 rng=np.random.default_rng(9105101)
 boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
 return {'n':len(d),'waypoint_minus_final_pp':float(d.mean()*100),
 'ci95_conditional_fixed_model_map':np.quantile(boot,[.025,.975]).tolist(),
 'gained':int((d==1).sum()),'lost':int((d==-1).sum())}
q=read(P/'protocol.json')
done={x['tag'] for x in read(R/'status.json')['jobs'] if x['status']=='complete'}
for group in ['source_sha256','case_files_sha256','prerequisite_sha256']:
 for f,h in q[group].items():assert sha(Path(f))==h,f
for v,m in q['models'].items():
 assert sha(Path(m['weights']))==q['model_sha256'][v]
 assert sha(Path(m['normalization']))==m['normalization_sha256']
records={};audits={}
for job in q['evaluation_jobs']:
 tag=job['tag'];v=job['model'];mode=job['mode']
 assert tag in done,tag
 p=R/'runs'/tag;meta=read(p/'case_metadata.json');rows=read(p/'cases_results.json')
 metric=read(p/'metrics.json');route=read(p/'oracle_waypoint_metadata.json')
 expected=read(Path(job['env']['LEWM_CASE_FILE']))
 assert meta['cases']==expected['cases'] and meta['topology_spec']==expected['topology_spec']
 assert len(rows)==len(route['cases'])==50
 assert meta['case_sha256']==q['case_files_sha256'][job['env']['LEWM_CASE_FILE']]
 assert meta['normalization_path']==q['models'][v]['normalization']
 assert meta['normalization_sha256']==q['models'][v]['normalization_sha256']
 assert read(p/'model_metadata.json')['model_path']==q['models'][v]['weights']
 assert meta['budget']==150 and meta['goal_offset']==75
 cap=meta['compute_cap_per_case']
 assert cap=={'candidate_model_steps':90000,'horizon':5,'iterations':2,'candidates':300,'receding':1,'warm_start':False},cap
 assert meta['argv'][1:]==job['argv'][2:]
 assert route['privileged'] and route['mode']==mode and route['threshold_px']==8
 assert not route['buffer_flushing'] and route['final_environment_goal_unchanged']
 adapter=read(p/'map_adapter_audit.json')
 assert len(adapter)==50 and all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in adapter)
 tr=np.load(p/'trace.npz');ta=audit_executed_trace(tr,rows)
 pos=tr['proprio'].reshape(len(tr['proprio']),50,-1)[:,:,:2]
 assert route['calls']==len(pos)<=150
 env=TopologyEnv(meta['topology_spec'])
 ref=OracleWaypointRouter(meta['topology_spec'],meta['cases'],mode=mode)
 events=[]
 for i,(row,case,rr,targets) in enumerate(zip(rows,meta['cases'],route['cases'],ref.routes)):
  assert (row['episode'],row['start_step'])==(case['episode'],case['start_step'])
  assert rr['episode']==row['episode']
  assert rr['route']==[t.tolist() for t in targets]
  assert np.allclose(rr['route'][-1],case['goals']['75']['xy'],atol=1e-6)
  assert row['different_room']==case['goals']['75']['different_room'] and not row['initial_success']
  assert all(env.valid(xy) for xy in pos[:row['steps'],i])
  assert np.isfinite(row['path_length']) and np.isfinite(row['endpoint_distance'])
  enddist=np.linalg.norm(pos[row['steps']-1,i]-np.array(case['goals']['75']['xy']))
  assert abs(enddist-row['endpoint_distance'])<1e-4
  if row['success']:assert enddist<16.0001
  # Reconstruct switches from pre-action true poses, not just router self-reports.
  idx=0
  for t in range(row['steps']):
   xy=np.array(case['start_xy']) if t==0 else pos[t-1,i]
   before=idx
   while idx<len(targets)-1 and np.linalg.norm(xy-targets[idx])<=8:idx+=1
   if idx!=before:events.append({'call':t,'episode':case['episode'],'before':before,'after':idx})
  assert rr['index']==rr['intermediate_reached']==idx
  assert rr['intermediate_total']==len(targets)-1
 env.close()
 assert sorted(events,key=lambda x:(x['call'],x['episode']))==sorted(route['events'],key=lambda x:(x['call'],x['episode']))
 assert [bool(x['success']) for x in rows]==[bool(x) for x in metric['metrics']['episode_successes']]
 assert np.isclose(sr(rows),metric['metrics']['success_rate'])
 tel=metric['telemetry'];assert tel['candidate_model_steps']>0 and tel['encoder_images']>0
 assert tel['candidate_model_steps']<=50*90000
 # Exactly one solve per live case every five physical steps.
 expected_steps=sum(int(np.ceil(x['steps']/5))*2*300*5 for x in rows)
 assert tel['candidate_model_steps']==expected_steps,(tag,tel['candidate_model_steps'],expected_steps)
 audits[tag]={'passed':True,'trace':ta,'route_events_reconstructed':len(events),'nominal_cap_verified':90000,'actual_candidate_steps_verified':expected_steps}
 records[tag]={'rows':rows,'cases':meta['cases'],'metric':metric,'route':route}
results={};pairs={}
for mapid in q['maps']:
 results[mapid]={};pairs[mapid]={}
 for v in ['single','multi']:
  groups={}
  for mode in ['final','waypoint']:
   data=[records[f'oracle_formal_{v}_{mode}_{mapid}_b{b}_v1'] for b in range(2)]
   rows=sum([d['rows'] for d in data],[]);cases=sum([d['cases'] for d in data],[])
   assert len(rows)==100 and len({x['episode'] for x in rows})==100
   groups[mode]=(rows,cases)
   routes=sum([d['route']['cases'] for d in data],[])
   results[mapid][v+'_'+mode]={'n':100,'sr':sr(rows),'sr_excluding_initial':sr(rows),
    'same_room_sr':sr([x for x in rows if not x['different_room']]),
    'cross_room_sr':sr([x for x in rows if x['different_room']]),
    'mean_path_all':float(np.mean([x['path_length'] for x in rows])),
    'mean_path_success':float(np.mean([x['path_length'] for x in rows if x['success']])) if any(x['success'] for x in rows) else None,
    'mean_endpoint_distance':float(np.mean([x['endpoint_distance'] for x in rows])),
    'stalled_step_fraction':sum(x['stalled_steps_lt_0_5px'] for x in rows)/sum(x['steps'] for x in rows),
    'physical_steps':sum(x['steps'] for x in rows),
    'candidate_model_steps':sum(d['metric']['telemetry']['candidate_model_steps'] for d in data),
    'encoder_images':sum(d['metric']['telemetry']['encoder_images'] for d in data),
    'predictor_token_inputs':sum(d['metric']['telemetry']['predictor_token_inputs'] for d in data),
    'batch_elapsed_seconds_not_online':sum(d['metric']['elapsed_seconds'] for d in data),
    'cached_synthetic_goal_images':sum(d['route']['cached_goal_images'] for d in data),
    'intermediate_targets_reached':sum(x['intermediate_reached'] for x in routes),
    'intermediate_targets_total':sum(x['intermediate_total'] for x in routes),
    'room_graph_hops':{str(h):{'n':sum(c['room_graph_hops']==h for c in cases),
      'sr':sr([r for r,c in zip(rows,cases) if c['room_graph_hops']==h])} for h in sorted({c['room_graph_hops'] for c in cases})}}
  aa,ca=groups['final'];bb,cb=groups['waypoint'];assert ca==cb
  pairs[mapid][v]={}
  for label in ['all','same_room','cross_room','hops_ge2']:
   idx=[i for i,c in enumerate(ca) if label=='all' or (label=='same_room' and c['room_graph_hops']==0) or (label=='cross_room' and c['room_graph_hops']>0) or (label=='hops_ge2' and c['room_graph_hops']>=2)]
   pairs[mapid][v][label]=paired([aa[i] for i in idx],[bb[i] for i in idx])
payload={'passed':True,'updated_unix':time.time(),'results':results,'paired':pairs,'audits':audits,'protocol':q,'requires_stage_review':True}
(P/'results.json').write_text(json.dumps(payload,indent=2))
lines=['## 正确高层路线的特权诊断：同预算全局目标与路点目标','',
'两种方法均为 H5/R1、2轮×300候选、150物理步、最多90000候选模型步/例；目标替换不清空动作缓存，也不更改环境真实终点。路点方案使用真实房间图、门坐标和位姿，不是纯视觉方法或已学会的分层策略。','',
'| 地图 | 训练条件 | 全局目标SR | 路点目标SR | 配对差值及病例95%区间 |','|---|---|---:|---:|---|']
for mapid in q['maps']:
 for v in ['single','multi']:
  a=results[mapid][v+'_final'];b=results[mapid][v+'_waypoint'];c=pairs[mapid][v]['all']
  lines.append(f"| {mapid} | {v} | {a['sr']:.1f}% | {b['sr']:.1f}% | {c['waypoint_minus_final_pp']:+.1f} pp; {c['ci95_conditional_fixed_model_map']} |")
lines+=['','跨房间、多连接距离、路点到达数量、路径/停滞和实际计算详见 oracle_waypoint_formal_v1/results.json。',
'上限相同不表示成功早停后的实际计算相同；路点额外使用特权输入、合成目标渲染与路由计算。批处理耗时不是在线时延。当前 H5/R1 与早先 H5/R5 对照仅候选模型步上限相同，编码与反馈次数不同，不能作全部资源等价比较。',
'每种训练条件仅一个seed4001；图结构变化也改变房间尺度和视觉分布。改进表示提供正确高层路线可能挽救控制，不证明模型自行具备拓扑推理；无改进也不能单独排除层级方法，需检查局部路点跟踪与目标支持。','']
text='\n'.join(lines)
(R/'reports/oracle_waypoint_formal_results.md').write_text(text)
for f in [R/'reports/frontier_transfer_review.md',R.parent/'reports/navigation_capability_merged.md']:
 old=f.read_text()
 if lines[0] not in old:f.write_text(old+'\n\n'+text)
print('ORACLE_FORMAL_AUDIT',json.dumps({m:{v:r['sr'] for v,r in g.items()} for m,g in results.items()}),flush=True)
