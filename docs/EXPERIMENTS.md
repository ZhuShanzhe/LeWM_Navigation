# 实验代码与证据链

历史模块位于 [experiments/strict_navigation](../experiments/strict_navigation/README.md)，下表路径相对于该目录。`pre_*`版本、修复脚本、旧smoke和失败记录保留用于解释演进，不作为默认运行入口。

| 阶段 | 主要入口/定义 | 对应证据 |
|---|---|---|
| 回合级划分与训练归一化 | prepare_split.py / splits/ | split manifest、normalization、训练split_audit |
| 三种子训练 | train_strict.py | results/strict_navigation/runs/strict_tw_s*/training_summary.json |
| 基础规划与闭环评估 | eval_strict.py | runs中metrics、cases_results、case_metadata |
| 失败与候选后果诊断 | diagnose_navigation.py / trajectory_diagnostics.py | candidate_diagnostic_summary / trajectory_diagnostics |
| 时间评价头 | fit_temporal_head.py / reachability_model.py | heads摘要；three_seed_transfer_audit |
| 检索及动态排序 | retrieval_prior.py / eval_retrieval.py / build_retrieval_seed.py | priors审计、retrieval_metadata、三种子对照 |
| 几何变化 | prepare_maps.py / prepare_final_geometry.py | maps/、final_geometry_v1/、地图适配审计 |
| 匹配数据覆盖训练 | collect_geometry_control.py | geometry_control_protocol、单/多布局配对报告 |
| 多房间结构 | topology_env.py / eval_topology.py | topology_bridge_v1、topology_stress_v1 |
| 特权路点 | oracle_waypoint_router.py / eval_oracle_waypoints*.py | oracle_waypoint_formal_v1、oracle_waypoint_t16_v1 |
| 独立最终确认 | run_final_frozen.py | final_confirmation_v1/protocol.json、results.json |
| 完整核验及统计 | audit_final_confirmation_noop_replay.py / summarize_final_supplements.py | final_results、supplementary_analysis、completion_audit |

时间头借鉴时间/可达性评价，检索借鉴数据支持的动作先验，路点使用真实图和位姿。它们不是完整论文基线复现，不可直接与不同数据、模型和机器人系统横向排名。

旧探索在 [exploratory](../experiments/exploratory/README.md)，更早的依赖包装器在 [bootstrap](../experiments/bootstrap/README.md)。原始strict训练包装器依赖旧轮train_bounded.py，因此不能只复制一个新阶段文件就宣称训练可运行。
