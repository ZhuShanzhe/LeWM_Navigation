# LeWorldModel 首轮实验报告
实验日期：2026-09-09。所有模型训练、评估、数据下载、论文资料和报告生成均在用户租用服务器执行。
本报告是指定配置下的工程基线和短程训练验证，不是论文全部实验的完整复现。

## 1. 已完成的工作
- 克隆官方仓库，读取训练、评估、模型与规划代码；没有修改官方受版本控制的源文件。
- 下载并校验官方 TwoRooms 权重及完整数据：10,000 条轨迹，920,809 帧，图像为 224×224 RGB。展开后的 HDF5 为 12,775,849,984 字节。
- 在 RTX 5090 上完成预训练模型 3 个评估种子、较远目标评估，以及从零训练 500 个优化器步并评估其检查点。
- 保存逐任务成败、起点/目标索引、视频、完整调用参数、训练 CSV、检查点、环境版本和论文调研笔记。

## 2. 实测结果

| 实验 | 成功数 / 任务数 | 成功率 | 目标时间偏移 / 执行步数上限 |
|---|---:|---:|---:|
| pretrained_default_s42 | 43 / 50 | 86.0% | 25 / 50 |
| pretrained_default_s43 | 41 / 50 | 82.0% | 25 / 50 |
| pretrained_default_s44 | 44 / 50 | 88.0% | 25 / 50 |
| random_seeded_default_s42 | 16 / 50 | 32.0% | 25 / 50 |
| random_seeded_default_s43 | 13 / 50 | 26.0% | 25 / 50 |
| random_seeded_default_s44 | 14 / 50 | 28.0% | 25 / 50 |
| pretrained_long75_s42 | 17 / 50 | 34.0% | 75 / 150 |
| random_long75_s42 | 2 / 50 | 4.0% | 75 / 150 |
| pilot500_default_s42 | 22 / 50 | 44.0% | 25 / 50 |

默认任务：预训练模型三种子平均 **85.33%**，种子间样本标准差 **3.06 个百分点**；固定动作随机种子的随机策略平均 **28.67%**，标准差 **3.06 个百分点**。这里是同一官方权重的三个“评估种子”，不是三个独立训练模型。每种方法共 150 个评估实例；同一评估种子的起点和目标已逐一核对相同。

较远目标实验只测试了 seed=42，不能当成稳定的多种子结论。目标由轨迹中起点后第 25 步改为第 75 步，执行预算由 50 改为 150，规划 horizon 保持 5。两种任务采样的轨迹子集也会变化，不能将成功率差全部归因于某一个因素。

早期 random_default_s42 等探索性运行没有显式固定 Gym 动作空间的 RNG，26% 是当时那次真实运行结果，不作为最终可复现实验主表的随机对照。后续封装补齐 RandomPolicy.set_seed，主表默认任务统一使用 random_seeded_default_*；原结果保留，不覆盖。较远目标随机对照也显式固定了动作 RNG。

## 3. 任务难度与分组审计
成功由环境判断：智能体与目标的欧氏距离小于 16 像素；不是 SPL、碰撞率或真实机器人成功率。
“墙两侧”分组仅按起点/目标在固定竖直墙 x=112 两侧判断，不等于所有实例均具有相同绕行难度。

- pretrained_default_s42：初始已经落在成功半径内 4/50；排除这些起点后成功 39/46；墙两侧任务成功 10/12。

- random_seeded_default_s42：初始已经落在成功半径内 4/50；排除这些起点后成功 12/46；墙两侧任务成功 3/12。

- pretrained_long75_s42：初始已经落在成功半径内 0/50；排除这些起点后成功 17/50；墙两侧任务成功 13/36。

- random_long75_s42：初始已经落在成功半径内 0/50；排除这些起点后成功 2/50；墙两侧任务成功 0/36。

更完整的分组和逐任务起终点见 evaluation_summary.csv、evaluation_summary.json，以及 runs/<实验名>/episode_audit.json。每次独立运行的二项 Wilson 区间也已记录；小分组不要过度解读，数据轨迹之间还可能有关联。

## 4. 500 步训练试跑
- 完成 500 个优化器步，5 个受限 epoch，每个 epoch 100 个 batch，batch size 128，每 epoch 验证 8 个 batch。
- 约 64,000 个训练窗口呈现次数；这不是 64,000 条独立轨迹，也不代表完整遍历数据集。
- 用时 143.48 秒（包含训练初始化后的 fit、验证和保存开销）。
- 首个记录点 step=9：总损失 3.0015，预测损失 0.2047，SIGReg 31.0000。
- 末个记录点 step=499：总损失 0.9493，预测损失 0.5079，SIGReg 4.9062。
- 末次验证总损失 1.5736，预测损失 0.6029。
- 检查点：data/checkpoints/train_pilot500/weights_epoch_5.pt；完整恢复文件：runs/train_pilot500/last.ckpt。
- 曲线：training_loss.png；逐步数值：runs/train_pilot500/metrics.csv。

总损失下降不等于导航能力提升；预测误差还受 latent 方差与正则平衡影响。此轮主要验证数据→前向→梯度→优化→保存→重新加载→规划的全链路。官方配置默认 100 个完整 epoch，这里只跑 500 步；学习率调度也随受限 trainer 长度缩短，不能称为相同训练预算复现。

沿用官方训练代码的随机窗口切分（train 90%、validation 10%），没有改成按整条轨迹或地图划分。因此验证 loss 不是严格的未见轨迹/布局泛化指标；预训练权重使用过哪些具体评估轨迹也未额外确认。这些评估属于官方数据分布内的控制基线，不能宣称完全独立的测试集泛化。

## 5. 配置和兼容性
- 官方代码 SHA：8edfeb336732b5f3ce7b8b210d0ba370a09e2cac。
- GPU：NVIDIA GeForce RTX 5090，显存约 32 GiB；Python 3.12.3；PyTorch 2.8.0+cu128，CUDA 12.8。
- 官方下载检查点按当前实现统计含 18,034,478 个参数；这是实际加载对象的总参数数，不强行等同于论文中的近似模型规模表述。
- CEM：每轮 300 条候选、30 次更新、top-k=30；horizon=5，receding_horizon=5，action_block=5；保留官方默认 batch_size=1。
- 训练：AdamW，lr=5e-5，weight_decay=1e-3，bf16，梯度裁剪 1.0，SIGReg 权重 0.09，训练历史长度 3。
- pin Transformers=4.57.6、huggingface-hub=0.36.2、tokenizers=0.22.2，以兼容公开权重；最新 Transformers 5 的参数命名不匹配。
- pin Hydra=1.3.2、OmegaConf=2.3.0、antlr4-python3-runtime=4.9.3，补齐 hdf5plugin、imageio 和 imageio-ffmpeg。
- 官方 HDF5 cache 期望 data/datasets/tworoom.h5，使用指向实际 HDF5 的符号链接，不复制 12GB 数据。
- 日志和 WandB 均在本地/offline 模式；未上传实验结果到外部账号。
- 不安装与 TwoRooms 无关的完整环境 extras，避免不必要的 Box2D/MuJoCo 构建和覆盖已有 GPU PyTorch。
- 运行时长包含环境和视频编码；部分随机基线与其他评估有 CPU 并行，不作为严格推理速度 benchmark。

完整依赖见 requirements.freeze.txt，硬件/文件哈希见 provenance.json，下载哈希及版本见 data/download_manifest.json。官方源码只有运行自动生成的未跟踪目录，没有受版本控制源码改动。

## 6. 与导航研究的关系
目前 TwoRooms 使用完整俯视图，是可控的视觉规划诊断环境，不是第一人称、部分可观测的真实室内导航。
下一步最值得验证的不是单纯再降预测 loss，而是：
1. 固定 LeWM 与动作候选，比较 latent MSE、时间/预算可达性度量对候选轨迹的排序；把“预测错”与“评分错”分开。
2. 增加局部观测和历史记忆，在相似走廊、隔墙目标、未见门洞位置下测失败原因；这里尚未做实验证明。
3. 仅把数据/记忆中可支持的状态作为高层子目标，再检查低层可达性，避免高层优化钻模型误差的空子。
4. 做整轨迹与未见地图划分，固定规划预算，至少三训练种子，报告 SR、SPL、碰撞、时间和显存。

相关论文、已有方法与潜在重复点见 [navigation_research.md](navigation_research.md)。可达性和分层方向已经有 RC-aux、TRM、HWM、Hi-LeWM 等，不能把模块简单组合当作已成立的创新。

## 7. 如何继续使用
所有路径均相对 /root/autodl-tmp/lewm_research/。在服务器终端运行：
```bash
source /root/autodl-tmp/lewm_research/env.sh
LEWM_RUN_TAG=my_eval python /root/autodl-tmp/lewm_research/run_eval.py --config-name tworoom policy=/root/autodl-tmp/lewm_research/data/hf_tworoom seed=42 output.filename=/root/autodl-tmp/lewm_research/runs/my_eval_results.txt
```
每次使用新的 LEWM_RUN_TAG 和输出文件名，避免覆盖之前的视频与指标。训练命令和各次完整参数已保存在 invocation.json；完整评估队列是 run_evaluation_suite.sh。
