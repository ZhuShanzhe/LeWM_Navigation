# 仓库辅助工具（默认不运行GPU）

- `verify_repository.py`：标准库静态验证，检查逐字节快照、每级README、JSON/Python语法、文件类型/大小、常见凭据模式和最终完成标记。自动模式不是完整安全保证，上传前还需人工审阅差异。
- `recompute_final_table.py`：从238批逐例cases_results重新计算各方法SR、跨墙/同侧分层和三种子均值，校对冻结results.json及发布CSV。
- `restore_workspace.py`：按来源清单恢复原树，默认dry-run，`--apply`写入；不同内容已有文件会触发拒绝，绝不强制覆盖。

工具在仓库根执行，Python标准库即可。历史实验入口不在这里，见experiments及复现指南。任何命令都不应自动启动旧supervisor。
