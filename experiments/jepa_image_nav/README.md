# Image-target 导航数据与模型

面向 JEPA 图像目标导航的仿真数据构建与模型调研。场景使用 Gibson Habitat 与 HSSD，不依赖 HM3D 或 Matterport 账号。

## 任务

- 地点图像导航：输入一张目标照片，到达拍摄位置。
- 物体实例图像导航：输入包含目标物体的照片，到达同一个实例的可见接近位置，而非任意同类别物体。

机器人使用第一视角 RGB；深度可作为独立比较设置。坐标、语义实例 ID、目标掩码和完整地图只作监督与诊断，不作为默认策略输入。

## 文件

- [scripts](scripts/README.md)：下载、场景加载检查、两类任务生成、数据验收与指标实现。
- [reports](reports/README.md)：22 项图像理解、目标定位、路线生成与世界模型工作的调研报告。
- [results](results/README.md)：已验收数据的实际规模与状态，不包含模型性能结论。
- [baselines](baselines/README.md)：预训练导航基线筛选、RTX 5090接口适配、两类任务的闭环评测与不足分析；完整结果和待补项目分别标注。

## 数据来源与划分

Gibson 场景包包含492个场景模型；构建任务优先使用原始 ImageGoal 数据作者公开的72个训练、14个验证场景列表。从训练场景中固定留出10个作为本地测试，保留作者验证场景不参与训练。重新采样的回合标为派生数据，不能宣称完全复现作者原始回合。

HSSD 固定公开版本 `4369cb9876214c7fbebcf552eb532380e4d287e4`，该版本原始刚体场景有168个，作者划分125个训练、43个验证场景。保留作者验证场景，从训练场景中按固定哈希留出本地测试。HSSD 物体图像目标回合是本项目派生任务，不冒充官方 HM3D InstanceImageNav。调试集独立存放，不计入正式规模。

HSSD正式数据包含168个场景：训练110、验证43、本地测试15。地点回合237,760个，物体实例回合228,176个，目标实例3,766个。与Gibson合计664,218个任务回合。统计与验收范围见[数据结果](results/README.md)。

## 渲染兼容配置

HSSD运行时使用服务器`raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json`，原始固定版本仍保存在`raw/hssd-hab`。兼容副本只修改一个资产的金属度／粗糙度贴图坐标，使其与已有颜色及法线贴图一致，解决渲染器共同纹理变换断言。几何、碰撞及纹理图像二进制保持不变，但该材质的金属度／粗糙度采样位置会改变，不能宣称与官方原版逐像素一致。具体路径和SHA-256见[材质兼容清单](results/material_compatibility.json)。

目标图像通常为JPEG；一个低对比度目标在相同相机位姿重新渲染为无损PNG，20个回合仅更新目标文件引用。读取端应遵循`goal_image`字段，不能写死JPEG扩展名。原始回合包和图像在服务器保留，见[目标图像记录](results/goal_repair.json)。

## 仿真与质量条件

Habitat-Sim 0.3.3 headless；相机256×256、水平视场90度、高度1.25米；机器人半径0.18米、高度1.4米；前进0.25米、旋转30度。HSSD 可通行区域包含静态家具。回合要求有限绕障路径；物体目标要求独立实例标识、多视角、可见像素和可达停靠点。

观测相机使用固定内参与1.25米高度。目标照片的`goal_camera.position`表示机器人原点，`height`是旋转前的局部安装偏移；物体目标带俯仰时，实际光心高度与水平位置会改变，不能直接当作固定世界高度。服务器`camera_extrinsics_v1`为168个HSSD场景的79,036张目标图像保存实际相机外参，不改变原有图像、回合或成功区域。相机约束核查见[基线诊断](baselines/results/README.md)。视场角、光照和纹理扰动尚不属于该版本的系统覆盖。

实例成功标准采用派生协议：明确STOP、距离有效接近视点小于0.5米，并能看到正确实例。地点成功标准是明确STOP且到拍摄点的测地距离小于1米。正式实验同时报告SR、SPL、SoftSPL、碰撞率和错误实例停止率，不能与不同传感器／阈值的论文数字直接比较。

## 服务器资源

服务器工作目录：`/root/autodl-tmp/jepa_vln_data_20261014`。场景、回合图像、实例掩码和时序轨迹保留在服务器，不上传GitHub。仓库只保存代码、来源、协议、聚合验收统计和调研材料。Gibson 与 HSSD 使用须遵循各自科研及非商业许可。

## 来源

- [Gibson 场景获取说明](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md)
- [ImageGoal 原始回合与场景划分](https://github.com/facebookresearch/image-goal-nav-dataset)
- [HSSD 场景](https://huggingface.co/datasets/hssd/hssd-hab)
- [HSSD 论文与方法](https://github.com/3dlg-hcvc/hssd)

- [paper_benchmark](paper_benchmark/README.md)：43个HSSD验证场景的四组受控测试和Gibson原始4901回合验证。
