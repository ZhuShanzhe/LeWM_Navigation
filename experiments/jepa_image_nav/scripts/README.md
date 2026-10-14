# 数据处理脚本

以下脚本默认使用服务器工作目录 `/root/autodl-tmp/jepa_vln_data_20261014`。

| 文件 | 功能 |
| --- | --- |
| nav_provision.sh | 建立独立Habitat-Sim 0.3.3 headless环境。 |
| nav_fetch.py | 从公开HSSD仓库下载固定版本，保存来源清单，重试限流／连接失败。 |
| nav_fetch_pilot.py | 下载一个完整场景的依赖，用于实例传感器与采样检查。 |
| nav_probe.py | 检查RGB、语义实例渲染及含家具的导航网格。 |
| nav_build.py | 构建HSSD地点和物体实例图像目标回合；只在训练场景采集专家／扰动轨迹。 |
| nav_build_gibson.py | 按原始作者场景列表构建派生Gibson地点导航回合。 |
| nav_complete_hssd.py | 资源完整后接续正式生成与验收，最多等待24小时。 |
| nav_audit.py | 验证回合唯一性、场景隔离、元数据、目标图像和实例掩码。 |
| nav_metrics.py | 成功、SPL、SoftSPL、碰撞率、错误实例停止率计算及基本单元检查。 |

## 构建顺序

先下载场景，完成压缩包大小／完整性检查并解压。环境准备完成后运行`nav_probe.py`，再运行`nav_build.py --pilot`和`nav_audit.py`检查调试数据。正式数据运行`nav_build.py`与`nav_build_gibson.py`，然后对每个输出目录执行`nav_audit.py DATASET_DIRECTORY`。

`--force`只用于明确重新生成该脚本的数据；正常运行按已完成场景统计恢复。下载、生成日志保存在服务器，不写入仓库。

## 验收范围

生成器在真实Habitat导航网格上逐回合验证路径；保存后的验收检查元数据、照片与场景隔离，不重新回放全部路径，也不代替策略闭环评测。指标函数要求调用者提供实际STOP、执行路径、最终距离和目标可见性，不能用生成器的专家结果冒充模型性能。

角度差分类表示起点与目标的朝向差，不是已经实测的图像重叠比例。物体类别优先采用作者资产表main_category，缺失时才使用严格名称规则，并记录来源；实例身份以场景与物体handle标识。旧资产配置的semantic_id不能直接套用新词表，运行时重新分配唯一实例ID用于掩码渲染。
