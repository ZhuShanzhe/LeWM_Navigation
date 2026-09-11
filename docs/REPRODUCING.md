# 复现说明

## 数值复算

```bash
python scripts/verify_repository.py
python scripts/recompute_final_table.py
```

两项工具仅依赖 Python 标准库，无需 GPU。`verify_repository.py` 检查文件 SHA256、目录说明、语法及最终结果摘要。`recompute_final_table.py` 从 [final_cases.json](../results/final_cases.json) 复算 238 批、85 个任务域/种子/方法组合，并对照冻结结果及 25 行最终汇总表，包括跨墙、同侧和排除初始成功分层。

逐例 JSON 包含病例标识、成功、初始成功及跨墙标记。字段定义和来源文件哈希见 [compact_transformation.json](../provenance/compact_transformation.json)。该记录用于成功率复算，不包含完整轨迹、时延与全部诊断信息。

## 文件路径恢复

```bash
python scripts/restore_workspace.py --destination /tmp/lewm-core --apply
```

工具按 [publication_manifest.json](../provenance/publication_manifest.json) 中的路径映射复制文件；省略 `--apply` 时只检查，目标存在不同内容时拒绝覆盖。恢复范围为清单中的仓库文件，不包含数据、模型权重、检索库或原始轨迹。

## 训练与闭环评估

实验脚本使用 `/root/autodl-tmp/lewm_research` 作为根目录。`train_strict.py` 依赖 `round12h_20260909/train_bounded.py`，环境依次引用根目录及旧轮的 `env.sh`；对应文件位于 `experiments/bootstrap/` 和 `experiments/exploratory/`。这些脚本需要按路径映射恢复后运行。

训练配置见 [artifacts](../artifacts/README.md)，实际训练参数和完成记录见 [训练记录](../results/strict_navigation/runs/README.md)。[最终协议](../experiments/strict_navigation/final_confirmation_v1/protocol.json) 包含评估参数、病例、模型、辅助头和检索库的资源标识及哈希。

环境版本见 [environment](../environment/README.md)。完整 freeze 是实验环境的依赖记录，不是跨平台安装锁文件。

闭环评估和原始轨迹审计还依赖外部数据、权重、辅助头、检索库及完整运行记录。`run_final_frozen.py` 检查指定资源哈希和输出路径；只有仓库文件不足以执行完整闭环复现。数值复算不能替代全新环境中的端到端验证。
