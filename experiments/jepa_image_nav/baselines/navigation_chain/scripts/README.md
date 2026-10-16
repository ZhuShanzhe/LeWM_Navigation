# 资源准备与评测脚本

脚本使用服务器目录`/root/autodl-tmp/jepa_instance_bench_20261016`，依赖已有场景目录`/root/autodl-tmp/jepa_vln_data_20261014`和地点评测目录`/root/autodl-tmp/jepa_vln_baselines_20261015`。执行前按实际部署路径调整常量，禁止混用Python环境的二进制扩展。

资源版本以`results/resource_audit.json`中的commit、revision和摘要为准。部分获取脚本可以解析上游HEAD；重新部署时必须使用已记录版本，不把新版本下载结果与这里的固定检查点成绩混用。ROS依赖的完整解析版本保存在`pixi_ros.lock`。

## 脚本分工

| 脚本组 | 内容 |
| --- | --- |
| `instance_fetch_*`、`instance_download_models.py`、`chain_download_*` | 获取官方代码、模型及运行时，保存版本与哈希；已有资源记录用于核对固定版本。 |
| `instance_checkpoint_audit.py`、`instance_cache_audit.py` | 数据型受限检查点和类别缓存读取，不执行任意pickle对象。 |
| `instance_policy_goat.py`、`instance_eval_goat.py` | 官方GOAT策略推理RPC与冻结实例面板评测；区分共同几何、原生相机和原生身体配置。 |
| `instance_sensor_gate.py`、`chain_motion_replay.py`、`chain_goat_acceptance.py` | 相对位姿接口对照、异常前进动作重放与实例全量验收。 |
| `instance_build_objectnav_v2.py`、`chain_finalize_objectnav.py` | 构建类别实例集合和验证视点，形成模型测试前固定的完整300回合。 |
| `chain_goat_objectnav.py`、`chain_category_acceptance.py`、`chain_category_numeric_diagnostic.py` | 原生类别输入评测、任意同类实例评分和数值精度核查。 |
| `chain_reject_native_geometry.py` | 对改变navmesh后不满足转向位置不变性的执行明确标为无效。 |
| `chain_unigoal_import.py`、`chain_unigoal_sim.py`、`chain_unigoal_run.py` | 原生UniGoal模块与跨Python版本仿真接口，显式区分纯图像和类别辅助协议。 |
| `chain_setup_unigoal.py`、`chain_unigoal_build.sh`、`chain_prefetch_unigoal_matcher.py` | Grounded-SAM兼容依赖、CUDA编译、数据型Mask R-CNN转换及官方匹配器准备。 |
| `chain_ollama_compatible.py`、`chain_unigoal_runtime_gate.py`、`chain_unigoal_ready_smoke.py` | 独立的Llama 3.2 Vision兼容运行时、文字/图像调用核对及完整链路预检。 |
| `chain_provision_eb.sh`、`chain_ai2thor_asset.py`、`chain_verify_model_assets.py` | Qwen推理环境、原生AI2-THOR二进制与发布方摘要校验。 |
| `chain_eb_navigation.py`、`chain_run_eb_full.sh`、`chain_eb_acceptance.py` | 原生EB-Navigation闭环、分批续跑和原始300任务验收，格式错误不计作有效导航成绩。 |
| `chain_jeplo_batch.py`、`jeplo_native_smoke.py`、`chain_motion_acceptance.py` | 原生JEPLO部署的指令组、端口和位姿接口核查、15次运动统计。 |
| `pixi_ros.toml`、`chain_download_pixi.py`、`chain_build_jev.sh` | 隔离Noetic环境与原生C++规划模块；OSQP 0.6.3及osqp-eigen 0.8.1。 |
| `chain_prepare_jev_models.py`、`chain_provision_jev_vision.sh`、`chain_download_jev_blip.py` | 原生视觉服务的源码、依赖和EVA/BLIP2权重；下载、编译不等于闭环评测。 |
| `chain_convert_jev_yolo.py`、`chain_jev_native_gate.py` | 受限数据型YOLO检查点转换并严格载入1,450个权重键；原生ROS图结构fixture及动态链接核查，不替代闭环导航。 |

## 运行与隔离

Habitat-Sim使用已有Python 3.9环境；策略及视觉模块使用对应Python 3.12环境；JEPLO部署使用其独立运行环境；ROS使用Pixi环境。Open3D兼容所需的系统`libstdc++`预加载仅限对应策略进程，不能套到GCC 15的ROS二进制。

GPU评测共用`gpu_benchmark.lock`，原生适配器内部已经加锁，不要再在外层加同一把锁。验收子进程需要原生动作重放时也遵循此规则，防止父进程持锁等待子进程的死锁。

EB全量入口为`bash chain_run_eb_full.sh`，分批释放GPU并跳过已经完成的冻结任务；不替换失败任务，不改变20步预算。只有300任务全部完成且验收通过，才生成完整成绩。

UniGoal的`--base-url`用于选择隔离的本地Ollama端口。默认不提供真值类别；`--category-assisted`启用的结果必须单独报告。`--limit 1 --max-steps 20`仅为工程预检，不能与500步性能面板混用。

Jev的API凭据只从服务器私有配置提供，不写入脚本、命令行参数、日志或Git。四个视觉服务与官方决策服务均通过后，才启动完整评测；不能把演示或回退响应计作模型输出。
