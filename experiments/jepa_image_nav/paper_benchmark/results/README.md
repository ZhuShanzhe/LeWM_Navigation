# 评测结果

`compact_statistics.json` 包含43场景的场景加权及回合加权 SR、SPL和场景簇 bootstrap 区间，实例对照仅使用共同有效的429个ID。`episode_metrics.jsonl` 保存1719条紧凑标量记录，不含图像、动作序列、坐标或模型提示。

`diagnostics.json` 汇总终止类型、碰撞、距离及绕行分组和实测回合耗时。终止类型描述失败表现，不直接证明视觉表征或规划模块是原因。`implementation_exceptions.json` 单列原版越界异常，没有性能分数。

`official_gibson_fgprompt.json` 和 `official_gibson_episode_metrics.jsonl` 是原始4901个验证回合的独立结果，不与HSSD混合排名。`acceptance.json` 为服务器私有完整证书的脱敏摘要，保留原证书哈希；不是包含原始轨迹的可公开审计包。

[返回评测入口](../README.md)
