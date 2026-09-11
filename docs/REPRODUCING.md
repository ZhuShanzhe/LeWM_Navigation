# 复现说明

## 数字复算（无需 GPU）

在仓库根目录运行 `python scripts/verify_repository.py` 与 `python scripts/recompute_final_table.py`。后者从 [final_cases.json](../results/final_cases.json) 复算 238 批、85 个域/种子/方法组合，再对照冻结结果及 25 行最终汇总表，包括跨墙、同侧和排除初始成功分层。

紧凑记录保留病例标识、成功、初始成功及跨墙标记；布尔值没有重新推断。生成时还与原始 metrics 中的逐例成功列表核对。来源文件哈希和转换说明见 [compact_transformation.json](../provenance/compact_transformation.json)。这不是重新执行原始轨迹审计，也不提供全部轨迹、延迟与诊断细节的独立复算。

## 核心文件恢复与闭环运行

`python scripts/restore_workspace.py --destination /tmp/lewm-core --apply` 按来源清单恢复当前保留的原文件，默认不带 `--apply` 只检查，有不同内容时拒绝覆盖。精简版仅恢复核心子集，不会恢复删除的历史运行目录。

核心脚本保留实验时原文及哈希，硬编码 `/root/autodl-tmp/lewm_research`。`train_strict.py` 依赖 `round12h_20260909/train_bounded.py`，环境依次引用根目录及旧轮的 `env.sh`；必要包装器已经保留。不能直接在 GitHub 分类目录下启动这些脚本。

闭环重跑还需原数据、模型权重、辅助头、检索库、完整运行环境，以及[最终协议](../experiments/strict_navigation/final_confirmation_v1/protocol.json)中指定的资源和哈希。模型配置在 [artifacts](../artifacts/README.md)，训练实参和完成证据在 [训练记录](../results/strict_navigation/runs/README.md)。环境版本见 [environment](../environment/README.md)，完整 freeze 是原机记录，不是跨机器安装锁文件。

`run_final_frozen.py` 与历史审计脚本还会读取完整归档的资源。需要重做完整轨迹审计时，应使用精简前提交 `cb0d22e894546b4ff99abbd120a88bfc9e59a54c` 加原服务器资源；该提交本身也不含数据、权重或原始 trace。请不要绕过哈希检查、重启旧队列或覆盖旧输出。新实验应另建阶段和输出标签。

本轮整理仅验证文件完整性和数值一致性，未声称全新机器端到端训练已经测试。
