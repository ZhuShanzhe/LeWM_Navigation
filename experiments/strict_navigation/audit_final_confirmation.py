"""Final frozen confirmation audit; no adaptive method selection or new training."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from audit_executed_trace import audit_executed_trace
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'final_confirmation_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sr(rows):return float(np.mean([r['success'] for r in rows])*100) if rows else None
q=read(P/'protocol.json');protocol_sha=sha(P/'protocol.json')
done={x['tag'] for x in read(R/'status.json')['jobs'] if x['status']=='complete'}
for f,h in q['source_sha256'].items():assert sha(f)==h,f
records={};audits={}
for job in q['evaluations']:
 tag=job['tag'];assert tag in done,tag
 path=R/'runs'/tag;meta=read(path/'case_metadata.json');rows=read(path/'cases_results.json');metric=read(path/'metrics.json')
 guard=read(path/'final_frozen_inputs.json')
 assert guard['protocol_sha256']==protocol_sha and guard['inputs_sha256']==job['frozen_inputs']
 for f,h in job['frozen_inputs'].items():assert sha(f)==h
 casefile=Path(job['env']['LEWM_CASE_FILE']);expected=read(casefile)
 assert len(rows)==50 and meta['cases']==expected['cases'] and meta['case_sha256']==sha(casefile)
 assert meta['budget']==150 and meta['goal_offset']==75
 assert meta['normalization_path']==job['env']['LEWM_NORMALIZATION_FILE']
 assert meta['normalization_sha256']==sha(meta['normalization_path'])
 assert meta['argv']==[str(R/'eval_strict.py')]+job['eval_argv'][2:]
 assert [(x['episode'],x['start_step']) for x in rows]==[(x['episode'],x['start_step']) for x in meta['cases']]
 assert [bool(x['success']) for x in rows]==[bool(x) for x in metric['metrics']['episode_successes']]
 assert np.isclose(sr(rows),metric['metrics']['success_rate'])
 tr=np.load(path/'trace.npz');audits[tag]=audit_executed_trace(tr,rows)
 pos=tr['proprio'].reshape(len(tr['proprio']),50,-1)[:,:,:2]
 for i,(row,case) in enumerate(zip(rows,meta['cases'])):
  assert row['initial_success']==case['goals']['75']['initially_within_success']
  assert row['across_wall']==case['goals']['75']['across_wall']
  pts=np.vstack([case['start_xy'],pos[:row['steps'],i]])
  assert np.isclose(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum(),row['path_length'],atol=1e-4)
  assert np.isclose(np.linalg.norm(pts[-1]-case['goals']['75']['xy']),row['endpoint_distance'],atol=1e-4)
  if row['success']:assert row['endpoint_distance']<16.0001
 if job['domain']!='known_layout':
  ad=read(path/'map_adapter_audit.json')
  assert len(ad)==50 and all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in ad)
 tel=metric['telemetry'];method=job['method'];seed=str(job['training_seed'])
 if method in ['random','noop']:
  assert tel['candidate_model_steps']==0
  if method=='noop':assert all(x['path_length']<1e-5 for x in rows)
 else:
  a=q['artifacts'][seed];assert read(path/'model_metadata.json')['model_path']==a['model']['weights']
  cap=meta['compute_cap_per_case']
  expectedcap=0 if method=='retrieval_only' else 9000 if method=='retrieval_rerank' else 90000
  assert cap['candidate_model_steps']==expectedcap
  if method=='h10_warm':assert cap['horizon']==10 and cap['warm_start'] and cap['candidates']==150
  else:assert cap['horizon']==5 and not cap['warm_start']
  if method=='temporal':
   hm=read(path/'metric_head_metadata.json')
   assert hm['sha256']==a['head_sha256']['temporal'] and hm['model_sha256']==a['model']['sha256'] and hm['cost']=='replace'
  if method.startswith('retrieval'):
   rm=read(path/'retrieval_metadata.json');assert rm['library_sha256']==a['library_sha256'] and rm['mode']==method.split('_')[1]
  else:assert tel['candidate_model_steps']>0
 records[tag]={'rows':rows,'cases':meta['cases'],'metric':metric,'cap':meta.get('compute_cap_per_case')}
groups={}
for job in q['evaluations']:
 key=(job['domain'],str(job['training_seed']),job['method'])
 groups.setdefault(key,[]).append(records[job['tag']])
stats={};joined={}
for key,data in groups.items():
 domain,seed,method=key;rows=sum([x['rows'] for x in data],[]);cases=sum([x['cases'] for x in data],[])
 n=q['domains'][domain]['n'];assert len(rows)==n and len({(r['episode'],r['start_step']) for r in rows})==n
 joined[key]=(rows,cases)
 summary={'n':n,'sr':sr(rows),'sr_excluding_initial':sr([x for x in rows if not x['initial_success']]),
  'initial_success_count':sum(x['initial_success'] for x in rows),
  'cross_wall_n':sum(x['across_wall'] for x in rows),'cross_wall_sr':sr([x for x in rows if x['across_wall']]),
  'same_side_sr':sr([x for x in rows if not x['across_wall']]),
  'mean_path_all':float(np.mean([x['path_length'] for x in rows])),
  'mean_path_success':float(np.mean([x['path_length'] for x in rows if x['success']])) if any(x['success'] for x in rows) else None,
  'stalled_step_fraction':sum(x['stalled_steps_lt_0_5px'] for x in rows)/sum(x['steps'] for x in rows),
  'candidate_model_steps':sum(x['metric']['telemetry']['candidate_model_steps'] for x in data),
  'encoder_images':sum(x['metric']['telemetry']['encoder_images'] for x in data),
  'predictor_token_inputs':sum(x['metric']['telemetry']['predictor_token_inputs'] for x in data),
  'batch_elapsed_seconds_not_online':sum(x['metric']['elapsed_seconds'] for x in data),
  'caps':[x['cap'] for x in data]}
 stats.setdefault(domain,{}).setdefault(seed,{})[method]=summary
def paired(a,b):
 assert len(a)==len(b)
 d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
 rng=np.random.default_rng(9106201);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
 return {'n':len(d),'delta_pp':float(d.mean()*100),'ci95_conditional_fixed_model_domain':np.quantile(boot,[.025,.975]).tolist(),
 'gained':int((d==1).sum()),'lost':int((d==-1).sum())}
pairs={};means={}
for domain in q['domains']:
 pairs[domain]={};means[domain]={}
 for seed in ['3072','3073','3074']:
  pairs[domain][seed]={}
  for base,alt in [('base',m) for m in q['methods'] if m!='base']+[('retrieval_only','retrieval_rerank')]:
   aa,ca=joined[(domain,seed,base)];bb,cb=joined[(domain,seed,alt)];assert ca==cb
   pairs[domain][seed][alt+'_minus_'+base]={}
   for st in ['all','cross_wall','same_side','excluding_initial']:
    ix=[i for i,x in enumerate(aa) if st=='all' or (st=='cross_wall' and x['across_wall']) or (st=='same_side' and not x['across_wall']) or (st=='excluding_initial' and not x['initial_success'])]
    if ix:pairs[domain][seed][alt+'_minus_'+base][st]=paired([aa[i] for i in ix],[bb[i] for i in ix])
 for method in q['methods']:
  values=np.array([stats[domain][str(s)][method]['sr'] for s in [3072,3073,3074]])
  means[domain][method]={'mean_sr':float(values.mean()),'sample_sd_sr':float(values.std(ddof=1)),'per_seed_sr':values.tolist()}
payload={'passed':True,'updated_unix':time.time(),'protocol_sha256':protocol_sha,'results':stats,'three_seed_descriptive':means,
 'paired':pairs,'trace_audits':audits,'limits':q['limits'],'requires_final_research_synthesis':True}
(P/'results.json').write_text(json.dumps(payload,indent=2))
lines=['## 最终冻结确认：预留轨迹与全新几何结果','',
'方法与参数在此批模型结果之前冻结；全部模型/辅助头/检索库保持不变，不按最终结果再调参。以下均值和样本标准差来自三个训练种子，不是置信区间。','',
'| 任务域 | 原始方法 | H10暖启动 | 时间评价头 | 检索单独 | 检索+动态排序 |','|---|---:|---:|---:|---:|---:|']
for domain,g in means.items():
 lines.append('| '+domain+' | '+' | '.join(f"{g[m]['mean_sr']:.2f} ± {g[m]['sample_sd_sr']:.2f}%" for m in q['methods'])+' |')
lines+=['',
'每个任务域的三模型复用病例；四个新几何还复用同一组坐标，不能合并成大量独立试验。逐种子、同侧/跨墙、排除初始成功、配对病例区间、路径/停滞、实际计算与简单基线均保存于 final_confirmation_v1/results.json。',
'原始方法、时间头和H10采用相同候选模型步上限，但总FLOPs不等价；检索排序上限较少且额外遍历训练库。批处理耗时不是单机器人在线时延。CPU参考路线非精确最短路，不报告伪SPL。',
'此阶段仍是两房间俯视全观测导航，不能外推到第一人称、部分观测或真实机器人。多房间/路点诊断与多布局训练负结果另见既有阶段，不混合排名。最终方向需要结合上述机制证据综合解释，完成此评估不等于已完成报告交付。','']
text='\n'.join(lines);(R/'reports/final_confirmation_results.md').write_text(text)
for p in [R/'reports/frontier_transfer_review.md',R.parent/'reports/navigation_capability_merged.md']:
 old=p.read_text()
 if lines[0] not in old:p.write_text(old+'\n\n'+text)
print('FINAL_CONFIRMATION_AUDITED',json.dumps(means),flush=True)
