# 紧凑测试结果

| 文件 | 内容 |
|---|---|
| `official_source_model_gate.json` | 官方推荐参数、源码、权重与资源哈希核验，不代表闭环通过 |
| `official_profile_negative_tests.json` | 7 种错误配置被拒绝的检查 |
| `official_sensor_build_audit.json` | 仅排除 ROS 分支的构建适配哈希，ZMQ 感知代码保持不变 |
| `official_sensor_live_test.json` | C++ ZMQ 感知进程的两轮合成输入测试与实测发布频率 |
| `official_trt_fixture_test.json` | ORT 1.22.0 / TensorRT 10.9 官方样本误差，包含实际执行 provider |
| `official_scene_probe.txt` | 官方地形切换器启用后的几何、摩擦和物理步长 |
| `official_runtime_inventory.json` | 实测 CUDA/cuDNN/TensorRT 版本和共享库哈希 |
| `official_simulator_build_audit.json` | 可选 RGB-depth 构建适配的源文件哈希 |
| `official_native_build_manifest.json` | 原生可执行文件、机器人 XML、配置和未改控制器源码哈希 |
| `official_native_probes.json` | 6 条原生闭环记录及 4 次环境/切换故障；完成运行不等于任务成功 |
| `official_native_trajectories.jsonl` | 原生记录的机身位姿、朝向和竖直度，用于复核统计与翻倒阶段 |
| `official_native_guard_test.json` | 启动入口的端口占用、哈希漂移拒绝测试 |
| `results.jsonl` / `summary.json` | 基础运动、箱体楼梯及扰动，共 230 次 |
| `extended_results.jsonl` / `extended_summary.json` | 15 类扩展地形，共 135 次 |
| `ablation_results.jsonl` / `ablation_summary.json` | 平滑与髋关节半幅分项对照，共 20 次 |
| `interface_audit.json` | 模型输入输出、官方样本误差与接口检查 |
| `export_comparison.json` | 部署与训练导出目录的模型差异 |
| `contract_tests.json` | 10 项离线辅助接口测试，不是真实通信测试 |
| `deployment_validation.json` | 外部完整记录的病例数、轨迹数和完成状态 |
| `replay_checks.json` | 3 个视频回放与对应原试验终点的一致性 |
| `portable_replay_validation.json` | 可移植路径入口的 4 个配对回放和 10 项辅助接口检查；不增加独立测试数 |

JSONL 是每次试验的紧凑记录，不是逐步轨迹。`scene_file` 保留原服务器路径作为来源信息，不能假定该绝对路径在其他机器上存在。场景可根据代码与清单在外部工作目录重新构建。

平地的 `success=true` 表示完成指令序列、没有触发终止；地形组表示达到规定终点。两种成功定义不能混用。推理耗时来自服务器 CPU，不是 Jetson 端到端延迟。

统计区间是小样本描述，不代表独立环境数量或真机可靠性。旧结果不因论文对照尚未完成而被删除，但解释以[官方结果对照](../docs/official_comparison.md)为准。

原生运行的完整日志、原始 LiDAR 位姿与构建差异保存在服务器外部归档，见[原生证据索引](../configs/official_external_artifacts.json)。

[返回上一级](../README.md)
