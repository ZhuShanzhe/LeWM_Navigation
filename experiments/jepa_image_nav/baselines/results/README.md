# 基线评测结果

`baseline_comparison.json`包含各面板的完成状态、SR、SPL、SoftSPL、场景bootstrap区间、距离和绕行分组、停止与碰撞结果。只有`complete_frozen_panel=true`且符合所述接口设置的结果进入主比较。原始回合和逐步轨迹保留在服务器，不放入仓库。

## 主结果与接口对照

| 结果 | 完成回合 | SR | SPL | SR场景bootstrap 95%区间 |
| --- | --- | --- | --- | --- |
| FGPrompt，Gibson地点 | 280/280 | 82.1% | 0.554 | 75.0%–88.9% |
| DEBiT-S 10°，Gibson地点 | 280/280 | 66.4% | 0.464 | 56.1%–76.1% |
| FGPrompt，HSSD地点 | 300/300 | 22.0% | 0.143 | 15.0%–28.3% |
| FGPrompt地点权重迁移，HSSD实例 | 300/300 | 3.0% | 0.012 | 1.3%–5.3% |
| DEBiT-S 30°失配对照，Gibson地点 | 280/280 | 21.4% | 0.145 | 13.9%–28.9% |

两种预训练策略的转向粒度不同，均为500次决策，不按同等物理转向距离或同等推理时间归一化，不能直接给出算法优劣排名。区间按场景均值进行5,000次bootstrap，反映现有14或15个场景的不确定性，不是多训练种子置信度。贪心动作选择、派生回合和本地传感器配置与论文设置存在差异，不作为论文原表格复现成绩。

`actuator_pair_audit.json`比较相同280个Gibson回合：对齐10°相对失配30°提升45.0个百分点，场景bootstrap区间为37.9–52.1个百分点。

## 未完成的辅助结果

SuperPoint＋LightGlue＋在线frontier仅完成2个场景40/300回合，SR为0；12/40曾进入可见成功区域而未正确停止。该系统为本项目适配实现，不是Mod-IIN、IEVE或UniGoal官方复现，也不是完整场景面板，不能用来否定这些方法。其权重兼容按照上游加载流程检查全部当前可学习参数；`confidence_thresholds`为确定性计算缓冲区，上游旧权重的9个`log_assignment.*.r`字段按官方兼容路径不参与当前网络。

DEBiT-S在HSSD的地点和实例迁移面板均尚无完整结果。CUDA/EGL初始化失败为执行错误，不转换为SR=0。`runtime_status.json`记录GPU状态及待补项目。

## 评分

- 地点：策略明确STOP，且到拍摄点的测地距离小于1米。
- 实例：策略明确STOP，距某个已验证接近视点的测地距离小于0.5米，且当前RGB相机中正确实例像素占比至少2.5%。真值实例图只用于评分。
- 严格地点位姿：距离小于0.25米且朝向差小于15°，作为额外分析。
- SPL使用实际执行平移路径；SoftSPL使用最终相对距离进展。
- 碰撞率是每回合碰撞前进次数除以前进尝试次数，再对回合取均值。它不是动态四足机器人的跌倒率。
- 错误实例停止仅在其他已收录同类实例的可见接近区域统计；范围有限。

`outcome_tag`是可观测结果分类，不是对神经网络内部失败原因的因果证明。多模型并发期间记录的推理耗时只用于运行审计，不能作为独占延迟或等时间预算比较。

## 来源与诊断文件

- `panel_index.json`：固定面板及SHA-256。场景内按距离和绕行分层，各场景20回合；样本不是自然频率总体。
- `fgprompt_inference.json`、`debit_inference.json`：检查点SHA-256、参数量、推理适配与严格状态加载。
- `source_debit.json`、`source_LightGlue.json`：官方源码树及Git blob校验结果。
- `fetch_debit.json`、`fetch_fgprompt.json`、`matcher_weights.json`：公开权重下载大小和本地SHA-256；无作者摘要时不声称与独立官方校验和比对。
- `policy_environment.json`、`habitat_inference_sources.json`：推理环境和Habitat依赖来源。
- `oracle_control_audit.json`：每任务每HSSD测试场景固定2回合，共60回合；真值控制器56次成功。它不是学习模型分数或全数据上界。
- `asset_overlap_audit.json`：测试实例与训练目标资产重叠，以及场景内重复资产诊断。只覆盖已选目标，不包括所有家具。
- `runtime_status.json`：当前GPU故障、已完成任务和待补项目。

原始Gibson训练场景中保留的本地测试集可能已被公开导航模型见过，因此主评测使用原作者验证场景。HSSD采用固定本地测试场景；该面板不是官方HM3D InstanceImageNav。
