# 评测代码

| 文件 | 功能 |
|---|---|
| `deployment_eval_v2.py` | 冻结 ONNX、LiDAR raycast、机体观测、动作处理与同步 MuJoCo 闭环 |
| `deployment_suite.py` | 官方样本接口审计与 230 次主测试 |
| `build_dataset.py` | 固定种子生成独立程序化场景，不下载真实环境数据 |
| `deployment_extended.py` | 135 个 validation 地形病例 |
| `deployment_ablation.py` | 20 次平滑/髋关节半幅诊断 |
| `deployment_contract.py` | 离线格式与新鲜度辅助接口测试；不发送机器人指令 |
| `compare_exports.py` | 比较官方两个模型导出位置 |
| `deployment_videos.py` | 3 段确定性回放；需额外图形和视频依赖 |
| `verify_results.py` | 无仿真依赖的发布记录与汇总核验 |

## 数字复算

```bash
python experiments/jeplo_locomotion/src/verify_results.py
```

## 闭环重跑

Linux、Python 3.11，另需固定版本官方资源。下面的路径应放在仓库之外；命令仅运行仿真，不连接机器人。

```bash
export JEPLO_WORKSPACE=/absolute/path/to/new_jeplo_runs
export JEPLO_REPO=/absolute/path/to/JEPLO
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# JEPLO_REPO 应是官方提交 460e2272907af053d6e66b57fecc71fdbe6ca1774 的检出目录。
# 确保 deployment/go2_deploy/model/pretrained、机器人资产和扫描模式文件完整。
python -m pip install -r experiments/jeplo_locomotion/configs/requirements.txt

python experiments/jeplo_locomotion/src/deployment_suite.py \
  --audit-only --out "$JEPLO_WORKSPACE/runs/interface_audit"
python experiments/jeplo_locomotion/src/build_dataset.py \
  --output "$JEPLO_WORKSPACE/datasets/procedural_v1"
python experiments/jeplo_locomotion/src/deployment_suite.py \
  --trials 5 --out "$JEPLO_WORKSPACE/runs/deployment_v2_230"
python experiments/jeplo_locomotion/src/deployment_extended.py
python experiments/jeplo_locomotion/src/deployment_ablation.py
python experiments/jeplo_locomotion/src/deployment_contract.py
python experiments/jeplo_locomotion/src/compare_exports.py
```

主测试、扩展组、诊断组和场景生成均拒绝覆盖已存在的输出目录。重跑采用新的外部 workspace。所有数字记录中的原服务器路径仅供追溯，不要求复用同一个绝对路径。

视频需要可用的 EGL/渲染驱动及 `imageio`、`imageio-ffmpeg`、`Pillow`，不是无头数字评测的必要依赖。运行时设置 `MUJOCO_GL=egl`。图形环境未就绪时仍可运行无视频评测。

源码的外部路径通过环境变量指定；评测数值逻辑保留。原始执行快照保存在外部归档，正式记录不因可移植入口而重写。跨机器的推理耗时不可要求一致。

## 边界

模型参考误差虽通过官方容差，严格数值一致性仍未完全解决。没有执行官方 C++/DDS/ROS 链路，也没有实现可上机认证的安全控制器。禁止把 `deployment_contract.py` 的离线超时测试当作真实执行进程已经具备失联保护。

[返回上一级](../README.md)
