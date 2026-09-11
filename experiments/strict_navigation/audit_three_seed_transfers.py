"""Audit three-seed temporal and retrieval transfers before freezing final confirmation."""
from pathlib import Path
import json,hashlib,time
import numpy as np,torch
from audit_executed_trace import audit_executed_trace
torch.set_num_threads(1)
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ba=read(R/'reports/three_seed_baseline_audit.json');assert ba['passed']
state=read(R/'status.json');done={j['tag'] for j in state['jobs'] if j['status']=='complete'}
manifest=read(R/'splits/manifest.json');pairbase=np.load(R/'heads/temporal3072/pair_indices.npz')
librarybase=torch.load(R/'priors/retrieval3072/library.pt',map_location='cpu',weights_only=False)
results={};audits={};artifacts={};paired_results={}
suffixes=['temporal_replace','temporal_hybrid','shuffled_replace','shuffled_hybrid','retrieval_only','retrieval_rerank']
def compare(a,b):
 d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
 rng=np.random.default_rng(9106101);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
 return {'delta_pp':float(d.mean()*100),'ci95_cases_conditional':np.quantile(boot,[.025,.975]).tolist(),'gained':int((d==1).sum()),'lost':int((d==-1).sum())}
for seed in [3072,3073,3074]:
 head=R/'heads'/f'temporal{seed}';libpath=R/'priors'/f'retrieval{seed}'/'library.pt'
 s=read(head/'summary.json');model=ba['models'][str(seed)]
 assert s['complete'] and not s['smoke'] and s['checkpoint_sha256']==sha(model['weights'])==model['sha256']
 p=np.load(head/'pair_indices.npz');assert all(np.array_equal(p[k],pairbase[k]) for k in p.files)
 assert set(p['train'][:,3])<=set(manifest['episodes']['train'])
 assert set(p['validation'][:,3])<=set(manifest['episodes']['validation'])
 lib=torch.load(libpath,map_location='cpu',weights_only=False)
 assert lib['checkpoint_sha256']==model['sha256'] and lib['split_sha256']==sha(R/'splits/manifest.json')
 assert torch.equal(lib['rows'],librarybase['rows']) and torch.equal(lib['actions'],librarybase['actions'])
 assert set(lib['rows'][:,2].tolist())<=set(manifest['episodes']['train'])
 hashes={}
 for kind in ['temporal','shuffled']:
  ck=torch.load(head/(kind+'.pt'),map_location='cpu',weights_only=False)
  hs=read(head/(kind+'_summary.json'));assert ck['checkpoint_sha256']==model['sha256'] and hs['epochs']==20
  hashes[kind]=sha(head/(kind+'.pt'))
 artifacts[str(seed)]={'model':model,'head_paths':{k:str(head/(k+'.pt')) for k in hashes},'head_sha256':hashes,
  'library_path':str(libpath),'library_sha256':sha(libpath),'library_entries':len(lib['rows']),'same_pairs_actions_across_seeds':True}
 results[str(seed)]={};groups={};cases0=None
 for suffix in suffixes:
  rows=[];cases=[];metrics=[]
  for b in range(2):
   tag=f'strict{seed}_validation_long_base_{suffix}_b{b}';assert tag in done
   out=R/'runs'/tag;meta=read(out/'case_metadata.json');rr=read(out/'cases_results.json');metric=read(out/'metrics.json')
   assert meta['cases']==read(R/'splits'/f'cases_validation_{b}.json')['cases']
   assert meta['case_sha256']==sha(R/'splits'/f'cases_validation_{b}.json')
   assert read(out/'model_metadata.json')['model_path']==model['weights']
   assert [x['success'] for x in rr]==[bool(x) for x in metric['metrics']['episode_successes']]
   audits[tag]=audit_executed_trace(np.load(out/'trace.npz'),rr)
   if suffix.startswith('retrieval'):
    rm=read(out/'retrieval_metadata.json');mode=suffix.split('_')[1]
    assert rm['mode']==mode and rm['library_sha256']==sha(libpath) and rm['train_only'] and rm['no_geometry_input']
    assert (metric['telemetry']['candidate_model_steps']==0)==(mode=='only')
   else:
    kind,cost=suffix.split('_');hm=read(out/'metric_head_metadata.json')
    assert hm['sha256']==hashes[kind] and hm['model_sha256']==model['sha256'] and hm['kind']==kind and hm['cost']==cost
    assert metric['telemetry']['metric_head_pairs']>0
   rows+=rr;cases+=meta['cases'];metrics.append(metric)
  assert len(rows)==100 and len({x['episode'] for x in rows})==100
  if cases0 is None:cases0=cases
  assert cases==cases0
  groups[suffix]=rows
  results[str(seed)][suffix]={'n':100,'sr':float(np.mean([x['success'] for x in rows])*100),
   'candidate_model_steps':sum(m['telemetry']['candidate_model_steps'] for m in metrics),
   'encoder_images':sum(m['telemetry']['encoder_images'] for m in metrics)}
 paired_results[str(seed)]={'rerank_vs_retrieval_only':compare(groups['retrieval_only'],groups['retrieval_rerank']),
 'temporal_vs_shuffled_replace':compare(groups['shuffled_replace'],groups['temporal_replace']),
 'hybrid_vs_temporal_replace':compare(groups['temporal_replace'],groups['temporal_hybrid'])}
payload={'passed':True,'updated_unix':time.time(),'results':results,'paired':paired_results,'artifacts':artifacts,'trace_audits':audits,
 'limits':'All scores are same100 known-layout development cases; three independent WM seeds, same head pair/head optimization and evaluation seeds. Temporal targets are logged behavioral separation, not true geodesic reachability. Shuffled hybrid retains real latent cost and is not a no-information control. No final reserve data used.'}
(R/'reports/three_seed_transfer_audit.json').write_text(json.dumps(payload,indent=2))
lines=['## 关键迁移三种子验证：代价学习强，动态重排序尚无附加收益','',
 '| 设置 | seed3072 | seed3073 | seed3074 |','|---|---:|---:|---:|']
for v in suffixes:lines.append('| '+v+' | '+' | '.join(f"{results[str(s)][v]['sr']:.1f}%" for s in [3072,3073,3074])+' |')
lines+=['',
'已核对三模型最终权重、辅助头对应关系、训练/验证状态对、检索库以及全部36个评估批次的真实执行轨迹。各库67240项，三个种子使用相同训练状态对和动作块，只由各自编码器重新编码；测试数据不进入头训练或检索库。',
'时间标签头替换代价为97/100/100%，乱序标签替换为6/9/4%，支持有意义的时间关系监督参与改进；并不证明学到精确测地距离或能跨新地图泛化。混合乱序头仍保留原latent代价，其76/67/58%不能解释成纯随机评价也有效。',
'检索单独执行98/98/99%，LeWM动态重排序97/96/97%，三个种子均未观察到成功率附加收益。任务接近饱和、病例仅100、模式计算和候选规则不同，不能推断动态预测普遍无用；最终新几何对照更有区分价值。',
'最终确认选取五个固定方法：原始代价、H10暖启动、时间头直接替换、训练检索单独执行、检索加LeWM重排序。选择直接替换而非混合头，是为了减少额外混合规则、清楚隔离目标代价；不按最终结果再选赢家或调参。乱序头和冷启动已作为开发机制对照，不扩展为所有最终测试方法。',
'这些是开发结果，尚不是最终预留确认。两房间移动门不是新拓扑；既有多房间路点诊断另外报告。','']
text='\n'.join(lines);(R/'reports/three_seed_transfer_results.md').write_text(text)
for p in [R/'reports/frontier_transfer_review.md',R.parent/'reports/navigation_capability_merged.md']:
 old=p.read_text()
 if lines[0] not in old:p.write_text(old+'\n\n'+text)
print('THREE_SEED_TRANSFERS_AUDITED',json.dumps(paired_results),flush=True)
