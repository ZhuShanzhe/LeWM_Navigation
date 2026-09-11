# 检查与恢复工具

- `verify_repository.py`：保留原文件 SHA256、目录说明、语法、敏感信息、紧凑记录哈希及最终审计摘要检查。
- `recompute_final_table.py`：从单个紧凑逐例 JSON 复算全部最终成功率、分层结果及三种子汇总；无需 GPU。
- `restore_workspace.py`：恢复当前保留原文件的历史相对路径，默认 dry-run，拒绝覆盖不同文件。只恢复核心子集，不恢复权重、数据、已删中间记录或原始 trace。

前两个工具不会启动训练或闭环评估。原始完整轨迹审计依赖服务器资源，不由此处检查替代。

[返回项目](../README.md)

## 目录索引

- [recompute_final_table.py](recompute_final_table.py)
- [restore_workspace.py](restore_workspace.py)
- [verify_repository.py](verify_repository.py)
