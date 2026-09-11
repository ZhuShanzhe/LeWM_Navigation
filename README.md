# LeWM × Navigation

研究 LeWM 在导航中的能力、失败原因和可迁移改进。当前版本只保留核心实现、实验配置、关键结果与分析、WM × Navigation 调研，不是 LeWM 全任务或真实机器人复现。

## 从这里开始

- [最终结论与研究方向建议](results/strict_navigation/reports/final_research_synthesis.md)：优先阅读，包含主要发现、对照与限制。
- [最终结果表](results/final_summary.csv)与[补充分析](results/strict_navigation/reports/final_supplementary_analysis.md)：三种子、分层成功率、配对比较和预算解释。
- [WM × Navigation 前沿调研](docs/literature/README.md)：各项工作的目的、思路、效果、不足及迁移线索。
- [核心代码与协议](experiments/strict_navigation/README.md)和[复现说明](docs/REPRODUCING.md)。

## 目录

| 目录 | 保留内容 |
|---|---|
| [experiments](experiments/README.md) | 训练、导航评估、时间头、检索、几何/拓扑/路点诊断及冻结协议 |
| [results](results/README.md) | 最终逐例紧凑记录、汇总、关键分析与训练完成证据 |
| [docs](docs/README.md) | 调研报告、复现入口、实验范围与局限 |
| [artifacts](artifacts/README.md) / [environment](environment/README.md) | 核心模型配置与依赖版本，不含权重或数据 |
| [third_party](third_party/README.md) | 固定版本的 LeWM 原始实现及许可 |
| [scripts](scripts/README.md) / [provenance](provenance/README.md) | 数字复算、完整性检查、来源与归档恢复 |

## 结果解读

时间评价头在原布局达到约 98.89% 成功率，仅检索也达到约 97.33%；这不代表未知地图规划已经解决。墙方向改变后仍明显失败，且时间头在较远门位置的跨墙成功率仅约 3.33%。原布局的高成功率不能替代绕障和泛化评估。

全部最终评估覆盖 238 批、11900 次环境回合；各方法和种子共享病例，11900 不是独立样本数。详细解释以最终报告为准。当前未开展部分观测、实机及全部大型论文基线；见[局限](docs/LIMITATIONS.md)。

## 检查

```bash
python scripts/verify_repository.py
python scripts/recompute_final_table.py
```

只用 Python 标准库即可复算最终成功率，不需要 GPU。闭环重跑还需未上传的数据、权重和原环境，不能把数字复算等同于从零复现。

本次精简没有更改实验观察或重跑训练。中间运行目录、历史队列、修补脚本和重复报告已从当前树移除；[精简前完整归档](https://github.com/ZhuShanzhe/LeWM_Navigation/tree/cb0d22e894546b4ff99abbd120a88bfc9e59a54c)与服务器原始档案仍保留。后续阶段继续更新本仓库，保留同样的核心范围与每级 README。
