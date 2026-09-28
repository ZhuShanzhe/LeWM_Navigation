# 清单与环境

- `official_readme_profile.json`：官方推荐仿真启动参数、模型哈希及尚未确认的论文协议字段。
- `native_probe_protocol.json`：已记录原生短测试的指令、时长与自定义登顶停止条件，不是论文预注册 benchmark。
- `official_external_artifacts.json`：原生实验完整日志和位姿归档的服务器位置、大小及校验值。
- `main_manifest.json`：230 个基础与扰动病例。
- `extended_manifest.json`：135 个扩展地形病例及场景哈希。
- `ablation_manifest.json`：20 个追加转向诊断病例。
- `requirements.txt`：原运行中已记录的核心依赖版本；视频依赖单独安装。
- `external_artifacts.json`：不上传 Git 的视频与完整归档的路径、大小、SHA256。

清单的初态、种子、动作模式和条件用于追溯原测试。绝对路径是原服务器位置；跨机器重跑应通过 `JEPLO_WORKSPACE` 和 `JEPLO_REPO` 设置外部路径并重新生成相同版本场景，不修改已发布结果。

官方提交：`460e2272907af053d6e66b57fec71fdbe6ca1774`。部署模型哈希见[接口审计](../results/interface_audit.json)。

[返回上一级](../README.md)
