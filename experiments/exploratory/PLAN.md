# LeWorldModel 12 小时复现计划与交接
预算：2026-09-09 18:11:34 至 2026-09-10 06:11:34，北京时间。开始时间包含本轮环境准备和数据下载。只使用当前服务器；不自动续租或关机。

优先级：
1. 以论文附录 D 的 10 epochs、TwoRooms history=1、SIGReg=0.1 完成从零训练。主 seed=3072；争取 3073/3074，但以训练结束摘要为准。
2. 四环境公开检查点与相同起终点的随机策略，3 个评估 seed（42/43/44），每次 50 个实例。先确保 TwoRooms、PushT、Reacher；Cube 大数据下载可能延后。默认 solver 次数按附录 D。
3. 训练模型默认目标及长目标评估；位置探针、有效秩、开环误差、固定候选模拟审计与合成时间跳跃诊断。
4. 主模型第 3000 步对比无 SIGReg、0.01 SIGReg、64 维容量的等步数消融；这些只是早期消融。检查规划迭代、采样量、预测跨度、重规划频率。
5. 预算允许时额外完整训练种子；最后剩余时间用于 PushT 从零训练，明确记录未满 10 epochs。
6. 下载失败可断点重试；论文其他基线需要独立权重与兼容性核对，没有成功运行就标未复现。

关键解释边界：
- 官方本地 JEPA rollout 固定 history=3，包装器按检查点 positional embedding 长度绑定，官方仓库源文件不改。
- 不把同一模型三个评估种子称为三次独立训练。
- 默认短目标与长目标可选起始轨迹分布不同，不作严格因果比较。
- 位置探针测试拆分不是未知地图/OOD 测试。
- action_embedding_corrupted 控制是打乱嵌入维度的非物理控制，不是正常动作重排。较早的 official_tworoom diagnostics.json 内此键误标 action_shuffled；解读以本说明为准，主报告不使用此项归因。
- 合成时间跳跃检测不等于论文完整穿墙/物理违例协议。
- 固定候选诊断不是新方法结果，也不是完整 CEM 或真实导航验证。
- 不运行额外购买/扩容，不触碰 apt 锁和其他已有进程。

运行：
supervisor.py 串行管理 GPU 作业，记录 status.json；训练有时间/步数回调、外层任务超时，服务器监督器有总截止时间。监督器仅中止自身子进程组。
asset_recovery.py 仅重试本轮数据下载，最多两次/环境，保留 SHA 校验和下载修订号。
summarize_round.py 在每个作业结束时更新 reports/reproduction_report.md 和 CSV/JSON；最终需结合日志人工核对失败、方法局限与复现差异。

主目录：/root/autodl-tmp/lewm_research/round12h_20260909
原论文：https://arxiv.org/html/2603.19312v3
权重：/root/autodl-tmp/lewm_research/data/checkpoints/
上一轮实测与导航综述：/root/autodl-tmp/lewm_research/reports/

自动回查不影响服务器队列独立运行；桌面端回查需要本机开机且 Codex 运行。凭据不写入本目录或自动任务提示。
