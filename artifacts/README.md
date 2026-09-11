# 外部数据与模型资源

此目录只发布模型配置，不发布权重。原研究工作根目录为 `/root/autodl-tmp/lewm_research`（路径用于定位，不包含服务器登录信息）。

| 未上传资源 | 原位置 | 验证依据 |
|---|---|---|
| 数据集 | data/、round12h_20260909/datasets/ | 数据资产清单、splits/manifest、训练划分审计 |
| 三严格模型权重 | data/checkpoints/strict_tw_s3072_v2、strict_tw_s3073、strict_tw_s3074 | final_confirmation_v1/protocol.json artifacts |
| 时间头 | strict_nav_20260910/heads/temporal*/ | temporal_summary、three_seed_transfer_audit、protocol头SHA |
| 训练轨迹库 | strict_nav_20260910/priors/retrieval*/library.pt | 库audit、protocol库SHA |
| 原始动作/状态trace | strict_nav_20260910/runs/*/trace.npz | 最终trace_audits与对应逐例JSON |
| 训练恢复状态、缓存、视频 | 原runs、spt_cache等 | 按实际需要另行恢复，不从仓库下载 |

完整来源文件映射见 [publication_manifest.json](../provenance/publication_manifest.json)；大文件排除清单见 [excluded-artifacts.json](../provenance/excluded-artifacts.json)。数据访问或权重分发须核对上游许可及项目维护者授权。不要把README中的路径当作公开下载地址。
