"""Paired evidence for transfer mechanisms, separate source of prior and WM gains."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');a=json.loads((R/'reports/navigation_results_audit.json').read_text())
def rows(key):
 out=[]
 for b in [0,1]:out+=json.loads((R/'runs'/f'{key}_b{b}'/'cases_results.json').read_text())
 return {x['episode']:x for x in out}
def compare(base,alt,stratum='all'):
 aa=rows(base);bb=rows(alt);assert aa.keys()==bb.keys()
 ids=[i for i in sorted(aa) if stratum=='all' or bool(aa[i]['across_wall'])==(stratum=='across')]
 d=np.array([int(bb[i]['success'])-int(aa[i]['success']) for i in ids])
 rng=np.random.default_rng(int.from_bytes(hashlib.sha256((base+alt+stratum).encode()).digest()[:8],'little'))
 ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1)*100,[.025,.975])
 return {'baseline':base,'variant':alt,'stratum':stratum,'n':len(d),'delta_pp':float(d.mean()*100),'ci95':ci.tolist(),'wins':int((d==1).sum()),'losses':int((d==-1).sum())}
pairs=[];lines=['# 迁移机制的直接对照','',f"快照：{a['updated']}；首个独立训练模型，开发验证阶段。",'',
'## 检索先验与世界模型的附加价值','',
'检索使用冻结LeWM图像编码器，但“单独执行”不调用动作动态预测器。因此它不是完全不使用LeWM，而是将表示/数据先验的价值与动态预测规划的价值分开。库仅含训练轨迹，验证与测试轨迹均排除。','',
'| 设置 | 长目标100例 | 短目标100例 |','|---|---:|---:|']
for mode in ['only','rerank','init','gaussian1']:
 v=a['groups'].get('strict3072_validation_long_base_retrieval_'+mode,{})
 s=a['groups'].get('strict3072_validation_short150_retrieval_'+mode,{})
 if v.get('complete') and s.get('complete'):lines.append(f"| {mode} | {v['success_rate']:.1f}% | {s['success_rate']:.1f}% |")
base='strict3072_validation_long_base_retrieval_only'
if a['groups'].get(base,{}).get('complete'):
 for mode in ['rerank','init']:
  alt='strict3072_validation_long_base_retrieval_'+mode
  if a['groups'].get(alt,{}).get('complete'):
   p=compare(base,alt);pairs.append(p);lines+=['',f"{mode} 相对检索单独执行：{p['delta_pp']:+.1f}个百分点，配对95%区间[{p['ci95'][0]:.1f}, {p['ci95'][1]:.1f}]，新增成功{p['wins']}例、失去成功{p['losses']}例。"]
lines+=['',
'当前已知布局任务上，加入动态预测排序或CEM没有表现出相对简单检索先验的成功率增益；高成功率不能单独归功于世界模型想象。也不能据此断言动态预测无用：任务接近上限、样本量有限，且目前只有一个训练种子。需要更换布局与拓扑、比较路径代价和时延后再判断。',
'',
'检索匹配遍历约6.7万条训练状态对，需要额外内存与距离计算；原始CEM和检索排序的候选预算与输出规则并非完全相同，详见retrieval_adaptation.md。三个正式检索模式及短长目标均已完成且通过病例审计；高斯单次迭代CEM只作为资源参照，不用于纯粹隔离候选分布。',
'',
'## 时间评价头的几何迁移边界','',
'| 布局 | 原始评价 | 时间头 | 时间头跨墙成功率 |','|---|---:|---:|---:|']
for axis in [1,0]:
 for door in [65,129]:
  b=f'strict3072_map_validation_a{axis}_d{door}';v=b+'_temporal_replace'
  old=a['maps'].get(b,{});new=a['maps'].get(v,{})
  if old.get('complete') and new.get('complete'):
   lines.append(f"| {'竖墙' if axis==1 else '横墙'} 门{door} | {old['success_rate']:.1f}% | {new['success_rate']:.1f}% | {new['across_wall']['success_rate']:.1f}% |")
   for st in ['all','across','same']:
    p=compare(b,v,st);pairs.append(p)
lines+=['','相同地图和病例上的完整组配对差值：','']
for p in pairs:
 if '_map_' in p['baseline'] and p['stratum']=='all':lines.append(f"- {p['variant']}：{p['delta_pp']:+.1f}个百分点，配对95%区间[{p['ci95'][0]:.1f}, {p['ci95'][1]:.1f}]。")
lines+=['',
'移动门的竖墙布局得到改善，但横墙布局仍表现很低；竖墙门129的总成功率54%也主要由同侧成功支撑，跨墙仅14%。因此不能把已知布局97%概括为通用路径可达性。观察与训练布局相关性较强，但编码器、动态模型、评价头及数据覆盖仍共同变化，尚不能定位唯一原因。',
'',
'## 对研究方向的当前影响','',
'优先问题从“能否把已知布局成功率提高到接近100%”转向：目标评价如何利用新地图连通结构，以及动作动态预测何时能提供超过数据先验的实际收益。单/多布局匹配训练用于区分覆盖不足；检索的变化布局对照用于验证动态模型是否在分布变化时才体现价值。新拓扑实验需要另行通过环境与可解性审计。',
'',
'这些是本轮数据支持的研究假设，不是创新性声明或最终方向。独立训练种子仍在运行；最终预留300例未使用。部分观测暂不加入，因为当前全观测几何外推问题尚未厘清。']
(R/'reports/transfer_mechanism_pairs.json').write_text(json.dumps({'updated':a['updated'],'paired':pairs},indent=2))
(R/'reports/transfer_mechanism_results.md').write_text('\n'.join(lines)+'\n')
