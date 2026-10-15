# 基线代码

代码使用两个隔离进程：Habitat-Sim 0.3.3在Python 3.9数据环境中执行仿真；策略在支持RTX 5090的PyTorch 2.8.0 / CUDA 12.8环境中推理，通过共享内存图像文件和JSON行通信。RGB策略接口不接收目标坐标、实例ID、语义图或导航网格。

## 文件

- `requirements-policy.txt`：实际推理环境的依赖版本；PyTorch使用官方CUDA 12.8构建。
- `baseline_prepare.py`：独立推理环境与官方仓库准备，不修改数据环境。
- `baseline_source_fetch.py`：按官方Git树取得必要源码，并核对每个Git blob。
- `baseline_fetch_fast.py`、`baseline_fetch_matcher.py`：官方权重分段下载、长度验证和本地SHA-256。
- `baseline_fetch_habitat.py`：Habitat v0.2.3推理依赖来源。
- `baseline_plan.py`：固定场景均衡、距离和绕行分层面板；已冻结清单不允许静默更改。
- `baseline_policy.py`：FGPrompt与DEBiT推理适配。保留官方计算类和函数体，替代注册、未使用传感器及训练依赖；所有策略状态键严格匹配。仅使用受限权重反序列化，旧配置作为数据容器读取，不执行序列化的任意外部类。
- `baseline_modular.py`：预训练局部匹配与在线RGB-D探索辅助系统。额外使用仿真真值形成的理想相对里程计；不提供目标坐标或完整地图。不是官方模块化论文复现。
- `baseline_eval.py`：隔离场景闭环执行、STOP评分、轨迹记录和执行错误重试。完整场景结果支持断点恢复。
- `baseline_run_stage1.py`：三个固定面板的顺序执行；CUDA预检失败时不启动测试。
- `baseline_aggregate.py`：完整性核对、分组结果和场景bootstrap。未完成面板单独标注。
- `baseline_pair_audit.py`：相同检查点的转向接口配对测试和依赖版本记录。
- `baseline_diagnostics.py`：资产重叠和真值控制器诊断。
- `baseline_horizontal_audit.py`：水平相机在已存接近位置的可见性核查，不删除主面板失败样本。
- `baseline_capture_status.py`：CUDA与驱动状态记录，不重置GPU。

## 服务器运行

```bash
BENCH_ROOT=/root/autodl-tmp/jepa_vln_baselines_20261015
DATA_ROOT=/root/autodl-tmp/jepa_vln_data_20261014

# 网络加速仅用于官方代码和模型资源下载。
source /etc/network_turbo

# 固定面板
/root/miniconda3/bin/python "$BENCH_ROOT/tools/baseline_plan.py"

# 原网络计算与权重加载检查
"$BENCH_ROOT/policy_env/bin/python" "$BENCH_ROOT/tools/baseline_policy.py" --model fgprompt --smoke
"$BENCH_ROOT/policy_env/bin/python" "$BENCH_ROOT/tools/baseline_policy.py" --model debit --smoke

# 主策略。已完成场景按面板校验后跳过，不重新覆盖。
/root/miniconda3/bin/python "$BENCH_ROOT/tools/baseline_run_stage1.py" --model fgprompt --workers 2
/root/miniconda3/bin/python "$BENCH_ROOT/tools/baseline_run_stage1.py" --model debit --prefix debit_native10 --turn-deg 10 --workers 2

# 同一实例任务的辅助系统，独立RGB-D输入条件。
"$DATA_ROOT/env/bin/python" "$BENCH_ROOT/tools/baseline_eval.py" \
  --model lg_frontier \
  --panel "$BENCH_ROOT/plans/hssd_image_nav_v1_test_instance_20perscene.jsonl" \
  --run lg_frontier_hssd_test_instance --workers 1 --max-steps 500

"$BENCH_ROOT/policy_env/bin/python" "$BENCH_ROOT/tools/baseline_aggregate.py"
```

仓库的`scripts`对应服务器的`tools`目录；首次部署需按该目录名放置脚本。根目录的准备和下载脚本应放在服务器根目录，使辅助下载脚本可导入`baseline_fetch_fast`。配置和代码中的固定服务器路径与数据版本必须一并保存。

## 适配条件

FGPrompt接收256像素RGB后按官方编码器流程缩放到128；DEBiT单目输入缩放到224、双目分支按官方Small配置缩放到112。动作映射为STOP、前进、左转、右转；FGPrompt为30°，DEBiT原生为10°。当前DEBiT公开检查点属于ImageNav，不含实例专用的抬头、低头动作。

模型不从头训练。传感器和静态家具导航网格使用现有派生数据配置，保留HSSD材质兼容层。没有把旧版本依赖安装到数据生成环境，也没有修改原始场景资产。

若`cuInit`非零，或驱动要求`GPU Recovery Action: Reboot`，先恢复平台GPU；不要把启动失败当作导航失败。当前剩余项目需要GPU恢复后执行，详见结果目录。
