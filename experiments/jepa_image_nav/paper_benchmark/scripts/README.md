# 评测代码

冻结与任务接口使用 `paper_freeze.py`、`paper_compact_freeze.py`、`paper_scope.py`、`paper_common.py`、`paper_sim.py`。`paper_eval.py` 与 `paper_unigoal.py` 调用原生预训练模型；其依赖还包括上级[基线代码](../../baselines/README.md)和服务器上的官方仓库、数据及权重。

`paper_unigoal_resume_technical.py` 只负责固定顺序续跑与严格原版异常隔离，不裁剪目标、扩图或新增决策回退。mmap、连接复用与诊断适配须先通过源码中规定的参数、状态、随机数及输出等价检查。

`paper_official_gibson.py` 保留作者原始ImageGoal回合和原生Habitat计分；`paper_official_accept.py` 检查原回合覆盖及原生运动证据。

完整性、独立物理重放、统计和最终验收依次为 `paper_exception_audit.py`、`paper_exception_replay.py`、`paper_exception_stats.py`、`paper_exception_acceptance.py`。原 `paper_compact_acceptance.py` 的430个UniGoal有效回合门槛保留为未通过；异常知情验收不伪造该旧门槛通过。`test_paper_exception_audit.py` 检查异常识别及重放内核不变。

源码保留实际运行服务器路径常量，没有为公开目录改写推理逻辑。闭环重跑需要还原目录、合法资产和各隔离环境；下载此目录并不构成开箱即用安装。仅复核公开数字可运行：

```bash
python scripts/verify_public_results.py
```

该命令不加载GPU、不调用API、不重新运行策略，也不能替代私有轨迹的物理重放。

[返回评测入口](../README.md)
