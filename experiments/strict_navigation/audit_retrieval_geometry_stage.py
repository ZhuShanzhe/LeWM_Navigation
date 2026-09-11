"""Audit completed geometry retrieval outputs and record conditional paired evidence."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from audit_executed_trace import audit_executed_trace
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sr(rows):return float(np.mean([x['success'] for x in rows])*100) if rows else None
def pair(a,b):
 assert [(x['episode'],x['start_step']) for x in a]==[(x['episode'],x['start_step']) for x in b]
 d=np.array([int(y['success'])-int(x['success']) for x,y in zip(a,b)])
 rng=np.random.default_rng(9105111);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100
 return {'n':len(d),'delta_pp':float(d.mean()*100),'ci95_conditional':np.quantile(boot,[.025,.975]).tolist(),'wins':int((d==1).sum()),'losses':int((d==-1).sum())}
model=R.parent/'data/checkpoints/strict_tw_s3072_v2/weights_final.pt'
lib=R/'priors/retrieval3072/library.pt';norm=R/'splits/normalization.json'
assert sha(lib)==read(lib.parent/'audit.json')['library_sha256']
assert sha(model)==read(lib.parent/'audit.json')['checkpoint_sha256']
done={j['tag'] for j in read(R/'status.json')['jobs'] if j['status']=='complete'}
results={};pairs={};checks={}
for axis,door in [(1,65),(1,129),(0,65),(0,129)]:
 key=f'a{axis}_d{door}';groups={};results[key]={};pairs[key]={}
 for mode in ['only','rerank','init']:
  allrows=[];allcases=[];tel=[];calls=[]
  for b in range(2):
   tag=f'strict3072_map_validation_{key}_retrieval_{mode}_b{b}';assert tag in done
   p=R/'runs'/tag;rows=read(p/'cases_results.json');meta=read(p/'case_metadata.json');m=read(p/'metrics.json');retr=read(p/'retrieval_metadata.json')
   casepath=R/'maps'/f'validation_axis{axis}_door{door}_b{b}.json'
   assert meta['case_sha256']==sha(casepath) and meta['cases']==read(casepath)['cases']
   assert len(rows)==50 and [(x['episode'],x['start_step']) for x in rows]==[(x['episode'],x['start_step']) for x in meta['cases']]
   assert meta['normalization_sha256']==sha(norm) and meta['normalization_path']==str(norm)
   assert read(p/'model_metadata.json')['model_path']==str(model)
   assert retr['mode']==mode and retr['library_sha256']==sha(lib) and retr['train_only'] and retr['no_geometry_input']
   assert [bool(x['success']) for x in rows]==[bool(x) for x in m['metrics']['episode_successes']]
   assert (m['telemetry']['candidate_model_steps']==0)==(mode=='only')
   checks[tag]=audit_executed_trace(np.load(p/'trace.npz'),rows)
   adapter=read(p/'map_adapter_audit.json')
   assert len(adapter)==50 and all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in adapter)
   allrows+=rows;allcases+=meta['cases'];tel.append(m);calls+=retr['calls']
  assert len(allrows)==100 and len({x['episode'] for x in allrows})==100
  groups[mode]=(allrows,allcases)
  results[key][mode]={'n':100,'sr':sr(allrows),'cross_sr':sr([x for x in allrows if x['across_wall']]),
    'same_sr':sr([x for x in allrows if not x['across_wall']]),
    'excluding_initial_sr':sr([x for x in allrows if not x['initial_success']]),
    'mean_path':float(np.mean([x['path_length'] for x in allrows])),
    'stalled_step_fraction':sum(x['stalled_steps_lt_0_5px'] for x in allrows)/sum(x['steps'] for x in allrows),
    'candidate_model_steps':sum(m['telemetry']['candidate_model_steps'] for m in tel),
    'encoder_images':sum(m['telemetry']['encoder_images'] for m in tel),
    'retrieval_pair_distances':sum(x['pair_distances'] for x in calls),
    'batch_seconds_not_online':sum(m['elapsed_seconds'] for m in tel)}
 for mode in ['rerank','init']:
  a,ca=groups['only'];b,cb=groups[mode];assert ca==cb
  pairs[key][mode]={s:pair([x for x in a if s=='all' or x['across_wall']], [x for x in b if s=='all' or x['across_wall']]) for s in ['all','cross']}
payload={'passed':True,'updated_unix':time.time(),'results':results,'paired_against_only':pairs,
 'trace_audits':checks,'model_sha256':sha(model),'library_sha256':sha(lib),
 'limits':'Single trained model seed3072, four development geometries reuse coordinates; conditional per-map case bootstrap only, no pooled independent-trial interpretation. Retrieval+CEM differs in compute and candidate rules.'}
(R/'reports/retrieval_geometry_complete_audit.json').write_text(json.dumps(payload,indent=2))
lines=['## 检索先验在新几何中的边界：完整开发组','',
'固定seed3072编码器与训练轨迹库，每图100例（50同侧/50跨墙）。only直接执行检索轨迹；rerank由LeWM动态预测重排序300条候选；init以检索动作初始化CEM。','',
'| 布局 | only总SR / 跨墙 | rerank总SR / 跨墙 | init总SR / 跨墙 |','|---|---:|---:|---:|']
for key,g in results.items():
 cells=[f"{g[m]['sr']:.1f}% / {g[m]['cross_sr']:.1f}%" for m in ['only','rerank','init']]
 lines.append('| '+key+' | '+' | '.join(cells)+' |')
lines+=['','相对only的同病例配对差值（总SR百分点，条件于固定模型/地图的病例95%区间）：','']
for key,g in pairs.items():
 for mode,p in g.items():
  x=p['all'];lines.append(f"- {key}，{mode}：{x['delta_pp']:+.1f}，区间{x['ci95_conditional']}。")
lines+=['',
'当前未见动态重排序在四个变化布局带来一致改善。横墙两图三个模式的跨墙成功率均为0%，高总分若有也主要来自同侧控制；竖墙门129的跨墙同样很弱。竖墙门65检索已有较强跨墙支持，进一步CEM反而下降。该结果提示动作先验与评分均可能受原布局支持限制，而不是证明动态预测在任何导航任务中都无用。',
'这是一个训练种子、开发地图结果，且没有假设检索/重排序/CEM完全等计算；检索额外编码与约6.7万项匹配开销、实际模型展开保留在JSON。四图复用坐标，不合并病例当独立样本。跨种子稳定性与最终预留评估仍待完成。','']
text='\n'.join(lines)
(R/'reports/retrieval_geometry_complete.md').write_text(text)
for path in [R/'reports/frontier_transfer_review.md',R.parent/'reports/navigation_capability_merged.md']:
 old=path.read_text()
 if lines[0] not in old:path.write_text(old+'\n\n'+text)
print('RETRIEVAL_GEOMETRY_COMPLETE',json.dumps(pairs),flush=True)
