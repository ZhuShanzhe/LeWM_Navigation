"""Paired geometry results; no independent-trial pooling across maps."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
a=json.loads((R/'reports/navigation_results_audit.json').read_text())
maps=a['maps'];anchor='strict3072_map_train_a1_d49'
def cases(key):
 rows=[]
 for b in [0,1]:
  path=R/'runs'/f'{key}_b{b}'/'cases_results.json'
  rows+=json.loads(path.read_text())
 return {r['episode']:r for r in rows}
lines=['# 几何布局能力：阶段结果','',f"快照：{a['updated']}。模型仍只训练原始单布局数据；地图名称中的train只是未来多布局组标签。",'',
'所有布局均采用同一组100个起终点（50同侧、50跨墙）；水平墙条件对坐标转置。起终点欧氏距离75—125，预算150步。它们不是原轨迹未来75步目标，因此不能直接把该成功率与旧长目标59.3%相减作为地图泛化损失。参考控制器已证明每例在预算内可解，但最短路线难度仍随门位置变化。','',
'| 布局 | 病例 | 成功率 | 跨墙成功率 | 同侧成功率 |','|---|---:|---:|---:|---:|']
for k,v in maps.items():
 if v['complete']:lines.append(f"| {k} | {v['n']} | {v['success_rate']:.1f}% | {v['across_wall']['success_rate']:.1f}% | {v['same_side']['success_rate']:.1f}% |")
paired=[]
if anchor in maps and maps[anchor]['complete']:
 aa=cases(anchor)
 lines+=['','相对原始几何（竖墙、门49）的同起终点配对差值：','']
 for key,v in maps.items():
  if key==anchor or not v['complete'] or 'temporal' in key or '_retrieval_' in key:continue
  bb=cases(key);assert aa.keys()==bb.keys()
  delta=np.array([int(bb[i]['success'])-int(aa[i]['success']) for i in sorted(aa)])
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
  boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)*100;ci=np.quantile(boot,[.025,.975]).tolist()
  p={'reference':anchor,'variant':key,'n':len(delta),'delta_pp':float(delta.mean()*100),'paired_95':ci,'wins':int((delta==1).sum()),'losses':int((delta==-1).sum())};paired.append(p)
  lines.append(f"- {key}：{p['delta_pp']:+.1f}个百分点，病例配对95%区间 {ci}。")
lines+=['',
'上述区间条件于单个训练模型和固定地图，不代表跨训练种子或地图总体不确定性；不同布局复用病例，不汇总成独立的千次试验。两房间移动门或旋转墙不是新连通拓扑。当前低成功率可能包含布局外推、目标分布和数据覆盖等因素；不能单独归因于表示、动态、规划器或记忆。',
'',
'时间评价头的已知布局97%—98%结果尚不能外推到这些布局。其4个验证几何布局已完成，具体差值与解释见transfer_mechanism_results.md。匹配的单布局/多布局采集方案与额外采集先验见geometry_collection_protocol.md，尚未完成相应训练。']
(R/'reports/geometry_results.json').write_text(json.dumps({'updated':a['updated'],'maps':maps,'paired':paired},indent=2))
(R/'reports/geometry_results.md').write_text('\n'.join(lines)+'\n')
