# 核心实现与证据

下表代码均位于 `experiments/strict_navigation/`，结果报告位于 `results/strict_navigation/reports/`。

| 内容 | 入口 | 主要证据 |
|---|---|---|
| 严格划分与训练 | prepare_split.py、train_strict.py | splits/、训练记录、模型配置 |
| 目标图像导航 | eval_strict.py | final_confirmation_v1/、final_cases.json |
| 时间评价头 | fit_temporal_head.py、reachability_model.py | three_seed_transfer_results.md |
| 检索与动态排序 | retrieval_prior.py、eval_retrieval.py、build_retrieval_seed.py | three_seed_transfer_results.md、transfer_mechanism_results.md |
| 失败诊断 | diagnose_navigation.py、trajectory_diagnostics.py | candidate_diagnostic_summary.json、trajectory_diagnostics.json |
| 几何变化及匹配数据覆盖 | prepare_maps.py、collect_geometry_control.py | geometry_control_pair4001.md、geometry_results.md |
| 多房间与路点 | topology_env.py、eval_topology.py、eval_oracle_waypoints*.py | topology_stress_results.md、oracle_waypoint_*_results.md |
| 冻结最终确认 | run_final_frozen.py、audit_final_confirmation*.py | final_confirmation_v1/、final_research_synthesis.md |

历史审计源用于解释评估机制，依赖完整实验资源；日常数字复算使用 `scripts/`，不要直接启动历史运行器。

时间头、检索和特权路点是轻量迁移/诊断，不是完整复现 TRM、RC-aux、PiJEPA、HWM 或 Hi-LeWM。
