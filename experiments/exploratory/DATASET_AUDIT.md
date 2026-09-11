# 公开 PushT 数据核对
核对时间：北京时间 2026-09-09 19:41。
来源：quentinll/lewm-pusht，revision 655cd446b9929369d7d406001da85c15d1457850。
压缩文件 pusht_expert_train.h5.zst 已通过发布方 LFS SHA-256 校验：
7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318

HDF5 实际记录：
- 18,685 条轨迹。
- 2,336,736 帧。
- 平均轨迹长度 125.05946。
- 像素 224×224×3，action 2 维，proprio 4 维，state 7 维。
- episode_idx 字段而非 TwoRooms 的 ep_idx，官方 eval.py 支持此差异。
- 2 个实例低计算量检查已完成；不计入正式成功率结果。

论文附录先前核对记录为 PushT 20k 条轨迹、平均约 196 步。发布数据与该描述不完全相同；是否经过筛选、版本或 train/test 拆分差异仍需核查，不能臆断原因。当前仅能称为作者公开数据与检查点复现，不声称完全相同的数据量。主报告需保留这一限制。
