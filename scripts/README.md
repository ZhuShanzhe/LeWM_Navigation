# 验证与路径恢复工具

- [verify_repository.py](verify_repository.py)：文件 SHA256、目录说明、Python/JSON 语法、敏感信息规则、逐例记录哈希及最终审计摘要检查。
- [recompute_final_table.py](recompute_final_table.py)：从逐例 JSON 复算最终成功率、分层结果及三种子汇总，无需 GPU。
- [restore_workspace.py](restore_workspace.py)：按来源清单恢复实验文件路径。默认只检查，使用 `--apply` 复制；拒绝覆盖内容不同的文件。

数值检查不执行训练或闭环评估。完整轨迹审计需要外部实验资源。

[返回项目](../README.md)
