# 原生源码兼容补丁

补丁限定运行兼容和协议校正，不作为JEPA算法收益。原始仓库版本与资源哈希记录在[资源校验](../results/README.md)。补丁按各自上游许可证使用，不在仓库复制整套第三方源码。

| 文件 | 上游文件与作用 |
| --- | --- |
| `unigoal_slam_classes.patch` | UniGoal `src/graph/utils/slam_classes.py`：接受原生调用方传入的`device`关键字，保留list及匹配方法。 |
| `unigoal_llm.patch` | UniGoal `src/utils/llm.py`：OpenAI兼容接口的`image_url`使用`{"url": ...}`对象，保留提示和模型。 |
| `groundingdino_torch_dispatch.patch` | GroundingDINO `models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu`：两个类型分派参数由`value.type()`改为`value.scalar_type()`；RTX 5090上与原生PyTorch参考算子的最大误差5.96×10⁻⁸。 |
| `jev_geo_utils.patch` | Jev `jev_obj/src/planner/trajectory_manager/include/gcopter/geo_utils.hpp`：为集合比较器增加`const`限定，兼容GCC 15，比较逻辑不变。 |

UniGoal的相机内参宽高和图分支深度量纲在评测适配器中显式校正，BEV仍接收原生归一化深度。此条件标为校准后的公开框架，不宣称精确复现论文数字。完整推理预检仍需通过后才能形成导航性能结论。

UniGoal使用的Llama 3.2 Vision在Ollama新版本中发生架构加载不兼容，隔离运行时使用维护者建议的0.24.0，不更换模型。上下文固定为4,096 token、单请求并行，避免自动32K上下文导致大量CPU卸载；这属于显式的推理预算设置。参见[上游兼容说明](https://github.com/ollama/ollama/issues/16490)。
