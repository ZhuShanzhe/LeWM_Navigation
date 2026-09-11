"""Render current temporal-head evidence from audited outputs."""
import json
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
a=json.loads((R/'reports/navigation_results_audit.json').read_text())
h=json.loads((R/'heads/temporal3072/summary.json').read_text())
lines=['# 时间评价头阶段结果','',f"快照：{a['updated']}；仅限开发验证，不是最终测试。",'',
'首个严格训练模型保持编码器和动作动态预测器冻结，只改变终点—目标评价；增加约26.3万参数的时间间隔头。训练状态对全部来自训练轨迹，独立验证状态对用于选择训练轮次。导航验证病例也来自验证轨迹组，因此属于方法开发验证，不能当作未经选择的最终泛化证据。','',
'状态对离线验证：时间头的时间差平均绝对误差为 %.2f 步、Spearman %.3f；相同架构但打乱训练标签的对照为 %.2f 步、Spearman %.3f。时间间隔不是最短路长度，相关性不等于导航有效性。'%(h['heads']['temporal']['validation_temporal_mae_steps'],h['heads']['temporal']['validation_spearman'],h['heads']['shuffled']['validation_temporal_mae_steps'],h['heads']['shuffled']['validation_spearman']),'',
'| 设置 | 病例 | 成功率 | 状态 |','|---|---:|---:|---|']
for key,v in a['groups'].items():
 if key.startswith('strict3072_validation_') and (key.endswith('_long_base') or key.endswith('_short150') or 'temporal' in key or 'shuffled' in key):
  lines.append(f"| {key.removeprefix('strict3072_validation_')} | {v['n']} | {v['success_rate']:.1f}% | {'完整100例' if v['complete'] else '尚未完整，不定论'} |")
lines+=['','配对病例差值（条件于这个训练模型，而非跨训练种子）：','']
for p in a['paired']:
 if p['stratum']=='all' and ('temporal' in p['variant'] or 'shuffled' in p['variant']) and p['variant'].startswith('strict3072_validation'):
  lines.append(f"- {p['variant']} 相对默认长目标：{p['delta_percentage_points']:+.1f} 个百分点，配对95%区间 {p['paired_bootstrap_95']}；新增成功{p['wins']}例、失去成功{p['losses']}例。")
lines+=['',
'当前观察支持优先研究目标评价与动态预测之间的匹配：同一冻结世界模型在替换评价后表现变化，说明原有导航失败不能全部归因于动作动态预测本身。但本实验不能证明时间头学到了普适可达性，也不构成新颖性证据；已有TRM等工作正研究这一接口。',
'',
'搜索时域、候选数和迭代数保持一致，最多90000候选展开步/病例；新增评价头计算不能忽略。成功早停使实际展开数量不同，批量评估耗时不能替代单机器人时延。现有一次候选池诊断也不能替代全程因果分解。',
'',
'已安排另两个独立LeWM训练种子的时间头及打乱标签比较，并在4个验证几何布局检查迁移。当前不使用未触碰的300例最终预留集。多布局训练、新连通拓扑及条件性的部分观测仍未完成。原始结果、训练和数据审计保留在heads/temporal3072及各runs目录。']
(R/'reports/head_mechanism_results.md').write_text('\n'.join(lines)+'\n')
print('HEAD_REPORT_UPDATED',a['updated'])
