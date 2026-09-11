"""Reuse frozen full trajectory audit with explicit per-arm tolerance and add8vs16 comparisons."""
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
def threshold_tag(v,mode,mapid,b):
 return f"{'oracle_t16' if mode=='waypoint' else 'oracle_formal'}_{v}_{mode}_{mapid}_b{b}_v1"
src=(R/'audit_oracle_waypoint_formal.py').read_text()
changes={
 "P=R/'oracle_waypoint_formal_v1'":"P=R/'oracle_waypoint_t16_v1'",
 "route['threshold_px']==8":"route['threshold_px']==(16 if mode=='waypoint' else 8)",
 "np.linalg.norm(xy-targets[idx])<=8":"np.linalg.norm(xy-targets[idx])<=(16 if mode=='waypoint' else 8)",
 "records[f'oracle_formal_{v}_{mode}_{mapid}_b{b}_v1']":"records[threshold_tag(v,mode,mapid,b)]",
 "reports/oracle_waypoint_formal_results.md":"reports/oracle_waypoint_t16_results.md",
 "## 正确高层路线的特权诊断：同预算全局目标与路点目标":"## 路点切换容差敏感性：16像素路点与原全局目标对照",
 "oracle_waypoint_formal_v1/results.json":"oracle_waypoint_t16_v1/results.json"
}
for a,b in changes.items():
 assert a in src,a
 src=src.replace(a,b)
exec(compile(src,str(R/'audit_oracle_waypoint_t16.py'),'exec'),globals())
#8px uses the exact original waypoint policy; shared global control is not rerun.
threshold_pairs={}
for mapid in q['maps']:
 threshold_pairs[mapid]={}
 for v in ['single','multi']:
  a=[];b=[];ca=[];cb=[]
  for batch in range(2):
   old=R/'runs'/f'oracle_formal_{v}_waypoint_{mapid}_b{batch}_v1'
   new=R/'runs'/threshold_tag(v,'waypoint',mapid,batch)
   a+=read(old/'cases_results.json');b+=read(new/'cases_results.json')
   ca+=read(old/'case_metadata.json')['cases'];cb+=read(new/'case_metadata.json')['cases']
  assert ca==cb
  d=paired(a,b)
  d['threshold16_minus_threshold8_pp']=d.pop('waypoint_minus_final_pp')
  d['sr8']=sr(a);d['sr16']=sr(b);threshold_pairs[mapid][v]=d
payload['threshold16_vs8']=threshold_pairs
(P/'results.json').write_text(json.dumps(payload,indent=2))
heading='## 8与16像素路点切换的直接配对比较'
lines=[heading,'','只改变路点切换半径；同模型、病例、规划和物理预算，16像素对齐最终目标成功半径。此项是诊断敏感性分析，不按结果选择新的最优半径。','',
 '| 地图 | 模型 | 8像素SR | 16像素SR | 配对差值及病例95%区间 |','|---|---|---:|---:|---|']
for m,g in threshold_pairs.items():
 for v,x in g.items():
  lines.append(f"| {m} | {v} | {x['sr8']:.1f}% | {x['sr16']:.1f}% | {x['threshold16_minus_threshold8_pp']:+.1f} pp; {x['ci95_conditional_fixed_model_map']} |")
lines+=['','CPU真值反馈控制器按真实8/16阈值切换时，两组均在150步内解决三个图各100例。16阈值的部分切换点到下一路点直线会与膨胀墙相交，虽可由碰撞滑动完成，不能称所有后续路点均可直线无碰撞到达。',
 '阈值放宽同时影响精确对齐和切换后的局部路径难度；不能把性能变化归于模型结构或单一高层规划原因。只测试8和16两个事先说明的阈值，不继续扫参。单训练种子配对，固定开发地图，无部署或跨种子泛化结论。','']
text='\n'.join(lines)
(R/'reports/oracle_waypoint_t16_results.md').write_text((R/'reports/oracle_waypoint_t16_results.md').read_text()+'\n\n'+text)
for f in [R/'reports/frontier_transfer_review.md',R.parent/'reports/navigation_capability_merged.md']:
 old=f.read_text()
 if heading not in old:f.write_text(old+'\n\n'+text)
print('THRESHOLD_SENSITIVITY_AUDITED',json.dumps(threshold_pairs),flush=True)
