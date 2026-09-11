"""Record repaired interface, finite next stage, and a sourced PLDM correction."""
import json,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent
now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
a=json.loads((R/'collections/model_smoke_audit.json').read_text());assert a['passed']
section=f"""## 配对数据接口通过与完整训练排队（{now}）

严格模型 strict_tw_s3073 已完成10轮、51920步，训练用时6009秒（约100分钟）；最终权重实际存在且SHA256为52593ab1e0fd4487007ceddbba9aa01facce102fac65607b8dd4d5ec9f3b4849。它是本轮第二个独立完整训练种子，不能仅凭验证预测损失断言导航改善。seed3074仍在有限队列中。

配对采集完整审计已通过：每组9000回合，其中8000训练、1000验证，未采集测试地图；两组回合长度、行为混合及帧数匹配。每组384次图像重渲染及384次状态转移检查通过。数据和归一化审计见 collections/audit.json。

首次 geometry_smoke_single_s4001 在读取数据前因使用相对名称失败，原始失败记录及日志保留。已将新数据输入改为实际绝对路径，并以 geometry_smoke_single_s4001_v2 重试。两组都完成20次更新和2例模型闭环接口检查；训练划分、对应训练统计、起点/目标渲染和有限预测审计通过，见 collections/model_smoke_audit.json。20步冒烟不计为完整训练或导航性能证据。

已向唯一串行队列追加有限23条作业：单布局和多布局各一次从零完整训练、每组在原始几何与4个验证几何上各100例共同病例评估，以及最终配对审计。两组使用seed4001、原架构/目标、相同10轮51920更新预算；不接续20步冒烟权重，不根据导航验证分数选择训练检查点。评估固定原始代价、150物理步、H5/R5、300候选×10次CEM、关闭热启动，各用对应训练数据归一化。具体冻结协议见 geometry_control_protocol.json。

这仍是一对独立数据条件、同初始化的探索性训练，尚不能构成跨训练种子结论；采集器的几何引导先验两组共有，应计入额外数据条件。不同地图复用坐标病例，区间按固定地图和训练种子条件计算，不池化为独立千例。两房间几何变化不是新连通拓扑。预留最终病例不用于此阶段选择。新拓扑及条件性部分观测尚未完成。

下一阶段依据配对结果决定应优先修复数据支持、目标评价还是动作预测；保留检索无动力学对照并补足关键种子，不默认扩展到全部大模型或真实机器人。
"""
for p in [R/'reports/controlled_collection_stage.md',R/'reports/geometry_collection_protocol.md',ROOT/'reports/navigation_capability_merged.md']:
 old=p.read_text() if p.exists() else ''
 if '## 配对数据接口通过与完整训练排队（' not in old:p.write_text(old+'\n\n'+section+'\n')
oldclaim='PLDM 将联合表示预测与分层规划结合，在不同时间尺度上组织预测和规划，避免只靠短期动作搜索处理遥远目标。它是理解“预测模型不等于完整导航系统”的重要基础；NeurIPS 2025 官方论文集可核验。[2]'
replacement='PLDM 联合学习视觉表示和动作条件潜动力学，使用多步预测、VICReg类正则及逆动力学辅助目标；通过MPPI在潜空间滚动规划，可结合预测器集合的不确定性惩罚。原始PLDM不是这里所述的两级分层模型；时间抽象和高低层宏动作扩展应归到后续HWM。NeurIPS 2025官方论文集可核验。[2]'
correction=f"""## 文献校正与新增机制线索（{now}）

校正：早期工作稿把PLDM描述为分层规划模型，混淆了原始方法与后续HWM；原始PLDM是潜动力学与MPPI滚动规划，HWM才加入高低时间尺度。技术依据：[PLDM v4，第3节与附录](https://arxiv.org/html/2502.14819v4)。后续讨论中的“为LeWM引入高层子目标”属于HWM风格适配，不能称为仅复现PLDM即已实现。

近期MCR预印本直接塑造候选动作的终点代价排序：在多步训练后，对参考动作的分级扰动增加排序监督。与TRM状态对时间距离、RC-aux预算条件可达性是不同干预点。参考轨迹偏差不等于真实通行代价，适合列为候选级机制对照，而不是默认优于当前方法。[原文2608.09073v1](https://arxiv.org/html/2608.09073v1)

NavWAM预印本将未来视觉、动作块与目标进展值联合生成，默认直接策略而非CEM搜索；其24回合实机结果仅是小样本系统证据。本轮可借鉴轻量动作提议与进展学习，不计划运行其完整大型视频骨干或真实机器人。[原文2606.13494v1](https://arxiv.org/html/2606.13494v1)

这两项目前为文献机制线索，不表示已在服务器实现或验证，亦不构成新增大规模实验授权。
"""
for p in [R/'reports/frontier_transfer_review.md',ROOT/'reports/navigation_capability_merged.md']:
 s=p.read_text()
 s=s.replace(oldclaim,replacement)
 if '## 文献校正与新增机制线索（' not in s:s+='\n\n'+correction+'\n'
 p.write_text(s)
print('REPORTS_UPDATED',now)
