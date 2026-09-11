# bootstrap

最早依赖、下载与训练包装器。严格训练仍引用部分旧包装器；保留用于恢复依赖关系，不默认启动下载或训练。

[返回上一级](../README.md)

## 内容

| 文件或目录 | 说明 |
|---|---|
| [audit_results.py](audit_results.py) | Audit recorded evaluation instances and training logs; never changes experiments. |
| [discover_assets12.py](discover_assets12.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [download_assets.py](download_assets.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [download_papers.py](download_papers.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [env.sh](env.sh) | 历史环境/运行包装器；先审查路径与资源。 |
| [extract_data.py](extract_data.py) | 历史实现/辅助模块；保留原始依赖关系。 |
| [make_experiment_report.py](make_experiment_report.py) | Create Chinese experiment report from measured artifacts, not guessed results. |
| [run_eval.py](run_eval.py) | Record official evaluation results; keep the model and planning algorithm unchanged. |
| [run_evaluation_suite.sh](run_evaluation_suite.sh) | 历史环境/运行包装器；先审查路径与资源。 |
| [run_seeded_random_baselines.sh](run_seeded_random_baselines.sh) | 历史环境/运行包装器；先审查路径与资源。 |
| [run_train.py](run_train.py) | Run official train.py with reproducible seeds and local CSV logging. |
| [save_provenance.py](save_provenance.py) | 历史实现/辅助模块；保留原始依赖关系。 |
