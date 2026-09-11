# 复现与归档恢复指南

## 1. 三个可复现层级

1. **阅读与数字复算**：克隆仓库后运行 `python scripts/verify_repository.py` 和 `python scripts/recompute_final_table.py`。仅需Python标准库；验证快照SHA256、每级README、JSON/Python语法、敏感信息规则及238批最终逐例结果的统计一致性。
2. **恢复历史文件树**：运行 `python scripts/restore_workspace.py --destination /tmp/lewm-restore --apply`，从来源清单逐字节复制被归档文件。目标有不同内容时拒绝覆盖。这一步不下载数据、权重或依赖，也不运行历史代码。
3. **重新训练/闭环评估**：还需原始数据、模型/辅助头/库、完整环境及存储。历史脚本硬编码 `/root/autodl-tmp/lewm_research`，原样执行需恢复至该根目录；不能直接从本仓库的新分类目录运行。若移植路径，另做运行副本并更新新协议，不能继续冒称与旧源哈希完全一致。

## 2. 环境

实验环境为Linux、Python3.12、RTX5090、PyTorch/CUDA12.8配置。准确当前版本见 [core-versions.json](../environment/core-versions.json)，完整实际freeze见 [requirements.server-freeze.txt](../environment/requirements.server-freeze.txt)。最早的original-freeze另存，不要混作最终环境。

建议在独立虚拟环境中安装与设备匹配的PyTorch构建，再按core版本补齐依赖。完整freeze可能含本机路径、无关预装包和特定平台构建，**不是跨机器可直接安装的lock文件**。旧MuJoCo/图形兼容步骤和runtime库不是本轮TwoRoom的充分依赖说明；不要覆盖宿主环境或自动安装/修改系统库。

恢复后检查 `env.sh`、`round12h_20260909/env.sh` 和 `strict_nav_20260910/env.sh` 的原始关系，设置 `STABLEWM_HOME` 至实际数据根目录。本仓库未承诺在全新机器端到端重跑全部GPU实验已经通过。

## 3. 数据与外部模型

在恢复目录中按原下载脚本和资产清单获得作者数据；先确认上游许可和访问条件。训练、验证、测试必须使用已冻结的episode分割及训练集归一化，不重新随机生成“相似”划分覆盖旧文件。

需要的资源与SHA记录来自 [最终协议](../experiments/strict_navigation/final_confirmation_v1/protocol.json)、[三种子迁移审计](../results/strict_navigation/reports/three_seed_transfer_audit.json)、[外部资源目录](../artifacts/README.md)。权重、库和trace未公开上传；没有原始trace时，只能核对逐例统计，不能重新执行完整的轨迹审计。未提供公开资源下载链接时不得声称第三方已能完全复现。

## 4. 如何选择命令而不误跑全部历史队列

`experiments/strict_navigation/jobs.json` 保存每项任务的argv、环境覆盖和tag；`status.json` 是历史执行状态，不是现在需要继续跑的任务。先按tag阅读目标命令、依赖与资源需求，再使用**新的输出tag**运行。不能直接启动 `experiment_queue.py` 或批量执行 `queue_*.py`：它们会根据磁盘状态添加/恢复作业，并可能改写报告。

主要训练入口为 `train_strict.py`；评价入口为 `eval_strict.py`、`eval_retrieval.py`、`eval_topology.py`；辅助头为 `fit_temporal_head.py`。准确Hydra参数以冻结argv为准，不能将近似的教程命令当作当时的实际设置。

最终确认 `run_final_frozen.py` 还检查原模型/头/库/病例/源文件SHA及 `LEWM_RUN_TAG`；它会拒绝已有输出。它适用于冻结归档验证，不适用于无审查地启动下一轮实验。旧审计的零动作假设修订见 `audit_final_confirmation_noop_replay.py` 与对应amendment JSON。

## 5. 本次整理验证的范围

验证原文件拷贝哈希、全部目录说明、最终逐例数字复算、敏感信息筛查，以及恢复工具的dry-run/写入/冲突拒绝；没有重跑训练或修改既有实验结论。因公开上传排除了大文件，仓库是可追溯研究归档，不是完整数据镜像。
