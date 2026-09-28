# 评测代码

| 文件 | 功能 |
|---|---|
| `audit_official_profile.py` | 只读核验官方提交、推荐命令、权重和资源；不代表闭环或论文性能验收 |
| `official_profile_negative_tests.py` | 验证错误配置被拒绝的 7 个反例 |
| `prepare_official_sensor.py` | 在独立文件中排除 ROS 专用分支，保留原始 C++ ZMQ 感知实现并生成差异和哈希 |
| `official_sensor_unit.cpp` | 调用官方函数验证下采样、归一化、累积和遮挡 |
| `official_sensor_live_test.py` | 合成点云驱动 C++ ZMQ 感知进程，核验无遮挡/保护罩两条件及发布频率 |
| `official_scene_probe.cpp` | 调用官方地形切换器，打印实际启用的楼梯和箱体几何 |
| `official_trt_fixture_test.py` | 在 ORT 1.22.0 / TensorRT 上验证官方样本，检查实际执行 provider，拒绝静默退回 CPU |
| `run_official_sensor_checks.sh` | 在全新外部目录依次运行来源、反例、C++ 函数和异步感知测试；不启动控制器 |
| `prepare_official_simulator.py` | 在独立源码副本中排除可选 RGB-depth 回调，保存差异；不改物理、LiDAR 和地形代码 |
| `official_native_smoke.py` | 在虚拟显示中启动原生仿真、感知和控制，强制本机回环/domain 1，记录零指令、前进或旋转的短测试 |
| `collect_official_native.py` | 根据外部原始位姿和运行日志生成紧凑原生记录与轨迹 |
| `verify_official_results.py` | 不运行仿真，仅复核发布的来源、运行时、TensorRT 证据和原生轨迹统计 |
| `official_native_guard_test.py` | 验证端口占用和构建哈希漂移会在启动仿真前被拒绝 |
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
python experiments/jeplo_locomotion/src/verify_official_results.py
```

## 官方配置源文件核验

```bash
python experiments/jeplo_locomotion/src/audit_official_profile.py --repo /absolute/path/to/JEPLO
```

返回成功仅代表源文件与模型配置符合冻结清单。运行时、完整闭环及论文协议必须分别验收，见[官方配置基准](../docs/official_alignment.md)。

感知测试需 C++17、ZeroMQ 开发头文件/库，以及 Python NumPy、pyzmq。将官方源码适配输出为外部测试目录内的 `lidar_depth_pub_sim.cpp`，再编译测试文件；不要覆盖官方源文件。`official_sensor_unit.cpp` 从编译 include 路径查找该适配文件。实时测试只启动感知进程，使用临时通信端口，不启动机器人控制器。

```bash
bash experiments/jeplo_locomotion/src/run_official_sensor_checks.sh /absolute/path/to/JEPLO /absolute/path/to/new-sensor-check
```

## 原生短测试

原生入口针对服务器独立构建目录布局，不是完整安装器。`--root` 下需预先准备 `source/`（官方源码副本）、`controller_output/deploy`、`controller_output/config.json` 与 `model/`、`lidar_depth_pub_sim`、`dependencies/prefix/usr`、`dependencies/mujoco-3.3.7`、`ort_env` 和 `runtime_env`。构建适配及版本差异见[基准说明](../docs/official_alignment.md)。

```bash
python experiments/jeplo_locomotion/src/official_native_smoke.py \
  --root /absolute/path/to/official_alignment \
  --output /absolute/path/to/new-native-smoke --case stand --duration 10
```

`--case` 可选 `stand`、`forward`、`rotate`、`stairs`、`boxes`；前三项指令分别为零、前进 0.25 m/s、原地转向 1 rad/s。楼梯和箱体使用前进 0.75 m/s；箱体通过官方 Ctrl+2 交互切换，需 xdotool 及其依赖，并检查切换确认。就绪后前 2 秒为零指令。程序独占本机端口 5560/5562/5590 和虚拟显示 `:97`，已有占用时拒绝启动；结束时只停止自身启动的进程。所有配置仅限仿真，不能用于真机。

`complete=true` 只表示完成规定时长，不等于运动成功、越障成功或安全认证；姿态、位移、OOD 和停机字段需要单独检查。公开短测试不替代每条件 500 次的论文实验。

启动前按 `--root` 中的 `native_build_manifest.json` 和 `official_readme_profile.json` 校验可执行文件、关键资源与权重哈希；发现漂移时拒绝启动。清单和程序需一起保存，不能在无记录的情况下覆盖权重或更换依赖。

地形测试可加 `--stop-at-ascent-goal`，在自定义登顶条件满足后切换为零指令并观察 2 秒。阈值和局限见[协议配置](../configs/native_probe_protocol.json)。未加此参数的持续前进测试可能继续走下平台，不能与仅登顶任务混为一谈。当前入口为后续运行保存自身源码快照；初期联调记录没有逐次入口快照，其原生程序哈希、指令、轨迹和日志哈希已保留。

## 闭环重跑

Linux、Python 3.11，另需固定版本官方资源。下面的路径应放在仓库之外；命令仅运行仿真，不连接机器人。

```bash
export JEPLO_WORKSPACE=/absolute/path/to/new_jeplo_runs
export JEPLO_REPO=/absolute/path/to/JEPLO
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# JEPLO_REPO 应是官方提交 460e2272907af053d6e66b57fec71fdbe6ca1774 的检出目录。
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
