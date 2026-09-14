# JEPA-VLA 与 Image-Goal Visual Navigation 调研

> 研究目标：面向后续 JEPA-VLN 项目，锁定图像目标导航（Image-Goal Navigation），梳理 JEPA-VLA、视觉语言导航、图像目标导航和导航世界模型的关键工作，并给出可验证的研究路线。
>
> 文献核对时间：2026-09-14。正式发表论文与预印本、Workshop 工作分别标注；不同数据集、传感器、动作空间和成功判定下的数字不可直接横向排名。

## 1. 结论先行

1. **任务名称需要严格区分。** 如果系统只接收“当前视觉历史 + 目标图像”并输出导航动作，正式任务是 **Image-Goal Visual Navigation（ImageNav）**，不属于狭义的 Vision-and-Language Navigation（VLN）。项目可以继续使用“JEPA-VLN”作为方向名，但第一篇工作的任务表述更适合写成 **JEPA-based ImageNav** 或 **JEPA for Image-Goal Navigation**。只有在输入中加入自然语言指令时，才应将任务正式称为 VLN。
2. **现有 JEPA-VLA 的直接证据主要来自机械臂操作，而不是第一视角长程导航。** JEPA-VLA、VLA-JEPA 和 JEPA-WAM 分别证明了视频预测表征、无信息泄漏的未来潜变量监督、以及共享预测器中的时序监督能改善 VLA，但它们没有解决未知环境探索、跨视角目标匹配、碰撞规避和长时记忆。
3. **ImageNav 的核心不是“让两个全局向量接近”。** 目标图像可能来自完全不同的朝向、相机高度和视场角；智能体必须建立局部区域对应、估计相对方向与可达性，并判断目标尚未出现、已经出现还是只是相似干扰物。ICLR 2024 的 DEBiT 和 3DV 2026 的系统研究都表明，早期、局部、patch 级交互以及与跨视角定位有关的预训练非常关键。
4. **JEPA 最有价值的切入点是“可达性与对应关系感知的潜在世界模型”。** 仅以预测终点和目标图像的全局余弦距离作为规划代价，很容易把“看起来相似”误当成“可以到达”。更合理的模型应同时学习：patch 对应、动作条件未来表征、时间/测地可达性、不确定性和碰撞风险。
5. **策略先验与世界模型需要等预算对照。** PiJEPA 的结果显示，使用策略分布初始化规划明显优于无先验采样，但其中大部分收益可能来自策略先验和世界模型打分，迭代 MPPI 的额外收益有限。因此后续实验必须比较 Policy-only、WM-only、Policy + WM reranking 和 Policy + iterative planning，并固定候选数、滚动长度和推理时延。
6. **推荐先做标准 ImageNav，再扩展到 InstanceImageNav。** 标准 ImageNav 更适合验证动作条件预测与跨视角相对位姿；InstanceImageNav 还叠加开放环境探索、特定物体重识别、相似干扰物排除和目标验证，适合作为第二阶段。

## 2. 任务边界与研究对象

| 任务 | 目标输入 | 成功含义 | 主要困难 | 与 JEPA-VLN 的关系 |
|---|---|---|---|---|
| PointNav | 目标坐标/相对位姿 | 到达坐标附近 | 几何规划、避障 | 适合验证世界模型动力学，但缺少图像目标匹配 |
| ObjectNav | 物体类别，如“chair” | 找到任一该类别实例 | 语义探索、常识、定位 | 可借鉴探索机制，但不要求实例级视觉对应 |
| 传统 VLN | 自然语言路线指令 | 按指令到达终点 | 语言落地、路径历史、指令进度 | 只有加入语言输入后才是严格意义的 JEPA-VLN |
| 标准 ImageNav | 某个目标位置拍摄的图像 | 到达目标图像对应的位置/视点 | 宽基线匹配、相对位姿、长程记忆 | **建议作为第一阶段主任务** |
| InstanceImageNav（IIN） | 特定物体实例的照片 | 找到同一物体实例并停在可交互位置 | 探索、实例重识别、干扰物排除、验证 | 建议作为第二阶段，更接近真实用户给图找物 |

Habitat 2023 的 ImageNav track 实际采用 InstanceImageNav 定义：智能体在未见场景中获得一个特定物体实例的 RGB 图像，目标相机与机器人相机解耦，相机高度、俯仰角和视场角可不同。该设定比早期“到达拍摄目标图像的精确视点”更接近真实应用，但也显著增加了实例识别难度。官方任务说明见 [Habitat Navigation Challenge 2023](https://aihabitat.org/challenge/2023/)。

### 2.1 本项目建议的正式定义

第一阶段采用标准 ImageNav：

- 输入：最近若干帧单目 RGB 观测 $o_{t-h:t}$、目标图像 $g$、历史动作 $a_{t-h:t-1}$；可设置 RGB-D 作为上界对照，但主模型优先 RGB-only。
- 输出：离散动作（前进、左转、右转、停止）或局部连续路点。
- 环境：未知的室内 3D 场景，训练、验证和测试按场景严格划分。
- 目标：在有限步数内到达目标图像拍摄位置附近，并正确触发停止。
- 主指标：Success、SPL、SoftSPL/Distance-to-Success、碰撞率、最终目标距离与推理时延。

第二阶段再转向 InstanceImageNav，加入目标实例重识别、探索—验证—利用切换和相机参数变化。

## 3. JEPA-VLA 相关工作

### 3.1 V-JEPA 2 与 V-JEPA 2-AC：表征和规划基础

[V-JEPA 2](https://arxiv.org/abs/2506.09985) 先在超过一百万小时的互联网视频上进行无动作标签的掩码潜在特征预测，再使用不足 62 小时的 DROID 机器人视频训练动作条件预测器 V-JEPA 2-AC。规划时给定当前图像、候选动作序列和目标图像，模型预测候选动作后的未来潜在状态，以目标潜在距离作为能量选择动作。它证明了大规模视频预训练可以向机器人规划迁移，但机器人实验是单目机械臂的 reaching、grasping 和 pick-and-place，并未验证移动机器人未知场景导航。

[V-JEPA 2.1](https://arxiv.org/abs/2603.14482) 进一步强调密集、时空一致的 patch 特征，通过 dense predictive loss、深层自监督和图像/视频联合 tokenizer 改善局部表征。它对 ImageNav 的启示很直接：目标定位依赖局部区域的几何和外观对应，使用 V-JEPA 2.1 的 dense token 比只使用全局 CLS token 更有希望。

### 3.2 三类名称相近但机制不同的 JEPA-VLA

| 工作 | 状态与场景 | 核心机制 | 主要效果 | 对 ImageNav 可迁移内容 | 关键不足 |
|---|---|---|---|---|---|
| [JEPA-VLA: Video Predictive Embedding is Needed for VLA Models](https://arxiv.org/html/2602.11832) | 2026 预印本；LIBERO、LIBERO-plus、RoboTwin2.0、单机械臂 | 冻结 V-JEPA 2，把最近视频帧的预测表征通过 early fusion 或 gated cross-attention 注入现有 VLA | Basic VLA 的 LIBERO 平均成功率 61.65→69.05；复现版 OpenVLA-OFT 90.30→96.40；RoboTwin clean 54.8→73.5、hard 9.3→17.7 | 视频表征可补充任务相关状态与策略先验；轻量融合便于先做验证 | 第三人称、多相机机械臂；没有目标图像匹配、未知地图、导航记忆和显式规划；提升不等于学到可干预的世界模型 |
| [VLA-JEPA: Enhancing VLA Model with Latent World Model](https://arxiv.org/html/2602.10098) | 2026 预印本；LIBERO、LIBERO-plus、SimplerEnv、实机操作 | target encoder 看未来帧生成监督目标；student 只看当前观测，避免未来信息泄漏；JEPA 预训练后微调动作头 | 报告在操作基准和分布偏移下有稳定提升；支持 Something-Something-v2 人类视频和 DROID 机器人视频 | **无泄漏未来监督**、动作相关潜变量和两阶段训练可直接借鉴 | 仍是操作任务；大规模预训练成本高；人类视频消融主要改善鲁棒性，未清楚增加新技能；未验证长程导航 |
| [JEPA-WAM: Learning VLA Policies with Joint-Embedding World Modeling](https://arxiv.org/html/2608.09381) | 2026 预印本；LIBERO-plus、RoboTwin2.0、双臂实机 | 在 V-JEPA 2.1 空间构造“当前—未来联合目标”；保留 patch 空间结构；时序预测与连续动作生成共享预测器 | LIBERO-plus 79.2；加入同类监督的 $\pi_{0.5}$ 84.5→86.3；消融显示 joint target、patch 结构和共享末端预测器均有效 | **最值得迁移的训练原则**：空间结构不能被全局池化；世界模型损失要真正作用于动作骨干，而不是成为旁路辅助头 | 最新预印本；依赖大模型和大规模操作数据；没有导航闭环、探索和目标可达性评估 |

这三篇论文共同支持“预测表征能够改善动作模型”，但不能直接推出“JEPA 已经解决导航”。移动机器人 ImageNav 的观测是第一视角，动作导致强烈相机自运动；目标可能长期不可见；墙体会造成视觉接近但几何不可达；一步误差还会在数百步闭环中累积。因此必须增加导航专用的归纳偏置和评估。

### 3.3 PiJEPA：离目标最近的现有原型

[PiJEPA: Policy-Guided World Model Planning for Language-Conditioned Visual Navigation](https://arxiv.org/abs/2603.25981) 是当前与 JEPA-VLN 最接近的公开工作。其输入包含第一视角当前图像、目标图像和语言指令：

1. 以冻结 DINOv2 或 V-JEPA 2 增强 Octo-Small，并在 CAST 上微调，得到动作分布；
2. 单独训练 JEPA 潜在世界模型，预测候选动作序列后的未来视觉 token；
3. 由策略样本估计 MPPI 的初始均值和方差，再用目标潜在距离优化候选轨迹。

它的贡献是把“策略擅长给出可行动作”和“世界模型擅长比较反事实结果”组合起来。论文比较 Default MPPI、PiJEPA、Octo-WM（只用世界模型重排策略样本）和 Octo。报告中 V-JEPA 2 配置的轨迹位置 ATE RMSE 从 Octo 的约 1.72 m 降至 PiJEPA 的约 1.65 m；DINOv2 配置从约 1.98 m 降至约 1.78 m。

但需要谨慎解读：

- 评价基于 CAST validation 上预测轨迹与记录真值轨迹的 ATE/RPE，而不是 Habitat 式未知环境闭环成功率、SPL 和碰撞率；
- 语言只进入策略，世界模型本身 language-agnostic；
- Octo-WM 与 PiJEPA 很接近，说明主要收益可能来自策略候选 + 世界模型打分，迭代 MPPI 的独立贡献有限；
- 目标代价仍是潜在空间终点距离，没有显式处理跨墙不可达、相似走廊、目标可见性和长时记忆。

因此，PiJEPA 应作为**最接近的结构基线**，而不是视为 ImageNav 已被解决。

## 4. VLN 与视觉导航的发展脉络

### 4.1 传统 VLN：语言指令、场景图与连续控制

| 工作 | 方向与改进 | 可迁移知识 | 与本项目的差距 |
|---|---|---|---|
| [R2R / Vision-and-Language Navigation](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html), CVPR 2018 | 建立 Matterport3D 上的 R2R 基准，将语言指令执行建模为视觉条件序列决策 | seen/unseen 场景划分、NE/SR/SPL 等评估范式 | 图拓扑离散动作，输入是文字而非目标图像 |
| [A New Path](https://openaccess.thecvf.com/content/CVPR2023/html/Kamath_A_New_Path_Scaling_Vision-and-Language_Navigation_With_Synthetic_Instructions_and_CVPR_2023_paper.html), CVPR 2023 | 用 500+ 室内环境的合成指令和模仿学习扩大 VLN 数据 | 数据规模与场景多样性往往比单纯加大模型更重要 | 语言增强不直接解决跨视角图像对应 |
| [NaviLLM](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Towards_Learning_a_Generalist_Model_for_Embodied_Navigation_CVPR_2024_paper.html), CVPR 2024 | 用 schema-based instruction 把多种导航/问答任务统一为生成问题 | 多任务统一、语言推理和迁移 | 高层语义强，低层几何、碰撞和实时控制仍需专门模块 |
| [NaVid](https://www.roboticsproceedings.org/rss20/p079.html), RSS 2024 | 只用单目 RGB 在线视频和语言指令预测下一步动作，不依赖地图、里程计或深度 | 第一视角视频历史可形成时空上下文，利于 Sim2Real | 生成式 VLM 的计算成本高；没有显式反事实世界模型 |
| [Uni-NaVid](https://www.roboticsproceedings.org/rss21/p013.html), RSS 2025 | 用统一 VLA 处理 VLN、ObjectNav、问答和跟随；在线 token merge 达到约 5 Hz | 压缩长视觉历史，跨任务共享导航技能 | 仍以语言任务为主；ImageNav 的细粒度目标对应不是核心 |
| [NaVILA](https://www.roboticsproceedings.org/rss21/p018.html), RSS 2025 | VLA 先产生“前进 75 cm”等中层语言动作，再由视觉运动 RL 策略执行四足控制 | 高层语义决策与低层安全控制解耦，适合实机 | 目标是语言指令；两层系统不能直接证明潜在世界模型有效 |
| [NavMorph](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.html), ICCV 2025 | RSSM 式确定/随机潜状态 + Contextual Evolution Memory + foresight action planner，在 R2R-CE/RxR-CE 预测未来视觉嵌入并修正 waypoint 分数 | 部分观测记忆、未来预测辅助策略、连续环境评估 | 潜在预测服务于语言 waypoint policy；没有目标图像的密集对应与实例验证 |

传统 VLN 对本项目最有价值的不是语言编码器，而是三点：历史状态必须压缩为可持续更新的记忆；高层目标/路径和低层避障最好分层；未见场景与连续动作评价比离散图上的结果更可信。

### 4.2 ImageNav：当前视角与目标图像的直接比较

| 工作 | 研究问题 | 主要改进与效果 | 不足与可迁移点 |
|---|---|---|---|
| [TSGM](https://proceedings.mlr.press/v205/kim23a.html), CoRL 2022 | 未知环境中如何保存长时视觉历史 | 增量构建 landmark topological semantic graph，使用 cross-graph mixer 和 memory decoder；SR 提升 5–9 个百分点，SPL 提升 7–23.5 个百分点，并做了移动机器人演示 | 依赖 RGB-D，图构建和策略模块较多；说明单帧/短历史 JEPA 不足，需要显式或可查询记忆 |
| [RNR-Map](https://openaccess.thecvf.com/content/CVPR2023/html/Kwon_Renderable_Neural_Radiance_Map_for_Visual_Navigation_CVPR_2023_paper.html), CVPR 2023 | 如何把环境外观保存在可定位的地图中 | 以网格 latent code 构建可渲染辐射场地图，用目标图像定位；NRNS curved 场景成功率 65.7%，比先前方法高 18.6 个百分点 | 需要 RGB-D 和里程计；构图/渲染复杂，但说明导航代价应包含几何定位而非纯语义相似度 |
| [Navigating to Objects Specified by Images](https://openaccess.thecvf.com/content/ICCV2023/html/Krantz_Navigating_to_Objects_Specified_by_Images_ICCV_2023_paper.html), ICCV 2023 | 在未知环境中寻找目标图像中的特定物体实例 | 把探索、实例重识别、目标定位和局部导航模块化；HM3D IIN 成功率 56% 对 25% 基线，实机 88% | 使用 RGB-D、pose 和多种现成组件；强有力地证明“目标匹配”和“环境探索”应分开诊断 |
| [DEBiT](https://openreview.net/forum?id=cphhnHjCvC), ICLR 2024 | RL 奖励是否足以学会宽基线图像比较 | cross-view completion + goal detection/finding 预训练；双编码器 binocular ViT 中自然出现 patch 对应；兼顾 ImageNav 与相机参数不同的 IIN | 模型较大，预训练流程较重；最重要启示是 early patch fusion 与相对位姿/可见性监督 |
| [IEVE](https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.html), CVPR 2024 | 如何避免把相似物体误认成目标 | 显式切换探索、靠近验证、利用三种模式；HM3D-SEM SR 由 0.561 提至 0.684/0.702 | 模块化且依赖检测/分割；JEPA-VLN 应预测匹配不确定性，并允许“靠近确认”而非一次性决策 |
| [GOAT-Bench](https://openaccess.thecvf.com/content/CVPR2024/html/Khanna_GOAT-Bench_A_Benchmark_for_Multi-Modal_Lifelong_Navigation_CVPR_2024_paper.html), CVPR 2024 | 连续执行类别、文字和实例图像目标时记忆是否有效 | 建立多模态、开放词汇、lifelong 导航基准，系统分析显式/隐式记忆和目标噪声 | 不是纯 ImageNav 首选训练基准；适合后期测长期记忆和多目标迁移 |
| [UniGoal](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.html), CVPR 2025 | 如何用一个零样本系统处理类别、实例图像和文本目标 | 统一 goal graph 与在线 scene graph；匹配状态驱动探索、定位、验证，并使用 blacklist | LLM/检测/深度/图推理链条较长；适合作为显式结构上界与 failure recovery 参考 |
| [IGL-Nav](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_IGL-Nav_Incremental_3D_Gaussian_Localization_for_Image-goal_Navigation_ICCV_2025_paper.html), ICCV 2025 | 目标图像来自任意相机、任意视角时如何做 3D 定位 | 前馈增量 3D Gaussian 场景表示，先离散粗定位，再可微渲染精定位；支持 free-view goal 和实机 | 3D 表示、深度/几何和优化成本较高；可作为“显式 3D 定位”强基线与几何监督来源 |
| [What Does Really Matter in Image Goal Navigation?](https://arxiv.org/abs/2507.01667), 3DV 2026 Oral | 低容量模型能否只靠 RL 奖励学会目标比较 | 系统比较 late fusion、channel stacking、space-to-depth 和 cross-attention；发现 realistic sliding=False 下，early/local fusion 与预训练仍关键，且导航与相对位姿 probe 相关 | 指出部分高分依赖墙边滑动的模拟器捷径；后续必须关闭 sliding 并单独报告碰撞与相对位姿能力 |

### 4.3 真实机器人视觉导航策略

| 工作 | 核心思想 | 对 JEPA-VLN 的作用 | 已知短板 |
|---|---|---|---|
| [ViNT](https://proceedings.mlr.press/v229/shah23a.html), CoRL 2023 | 多机器人、数百小时数据训练 goal-conditioned Transformer，预测局部导航动作与距离 | 适合作为强 policy-only 基线和动作候选生成器 | 长程任务常依赖预建拓扑图或外部远程启发式；不是显式世界模型 |
| [NoMaD](https://openreview.net/pdf?id=FhQRJW71h5), ICRA 2024 Best Paper | goal masking 在同一扩散策略中统一目标到达与无目标探索，建模多峰安全动作 | 最适合作为 Policy prior，与 JEPA 预测器组合做候选生成和重排 | 主要是行为克隆；世界动力学不显式，复杂几何和相似地点仍可能失败 |
| [Can Vision Foundation Models Navigate?](https://arxiv.org/abs/2603.25937), 2026 预印本 | 两种机器人、五类真实环境系统测 GNM/ViNT/NoMaD 等，加入碰撞、定位和视觉扰动指标 | 给出了非常重要的失败分层和实机评价协议 | 不是正式顶会论文；实验规模仍有限，但发现值得作为风险清单 |

2026 实机评测发现，即使 Transformer/扩散模型能沿大致路线行驶，也常在门框、桌椅和柱子处碰撞；在外观重复的地点会误判目标；运动模糊、眩光和环境分布变化会显著降低性能。这说明 JEPA-VLN 不能只报告“是否到达”，还要测碰撞、目标误识别、拓扑节点误差、最终距离和视觉扰动鲁棒性。

## 5. Navigation World Model 相关工作

| 工作 | 世界模型形式 | 规划方式 | 效果与价值 | 不足 |
|---|---|---|---|---|
| [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html), ICML 2025 | 冻结 DINOv2，预测未来 patch feature，不重建像素 | 测试时优化动作序列，使未来特征接近目标特征 | 在六类导航/操作环境上用离线轨迹实现零样本目标规划；验证 patch latent WM 的可行性 | 多数导航环境较简化；目标相似度未显式建模可达性；无长时未知室内探索 |
| [Navigation World Models](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html), CVPR 2025 | 最大 1B 参数 Conditional Diffusion Transformer 生成动作条件未来视频 | 从头搜索动作或重排 NoMaD 等策略候选 | 大规模人类/机器人第一视角视频带来强视觉想象，约束可在规划时动态加入 | 训练成本极高；像素生成慢；未知场景的“想象合理”不等于目标可达；难作为单服务器公平主基线 |
| [NavMorph](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.html), ICCV 2025 | RSSM 式随机/确定潜状态 + 上下文演化记忆 | future action planner 预测未来语义/轨迹并修正 waypoint 分数 | 证明连续 VLN 中潜在未来预测和跨情景记忆有价值 | 面向语言指令，不解决 image-goal correspondence；世界模型与最终动作的因果贡献需要更严格消融 |
| [PiJEPA](https://arxiv.org/abs/2603.25981), CVPR 2026 Workshop | 冻结 DINOv2/V-JEPA 2 的动作条件潜在预测器 | policy warm-start MPPI | 最接近 JEPA + 第一视角导航 + 目标图像的结构 | 主要为离线轨迹误差，不是未知室内闭环 ImageNav；迭代规划额外贡献较小 |

## 6. 现有工作留下的研究空白

### 6.1 表征空白：全局语义相似不等于导航对应

相似走廊、相同门、重复家具和纹理会让全局 embedding 接近，但对应位置可能相隔一堵墙。ImageNav 需要的是“哪些局部区域对应、目标相对当前视角在哪里、目标是否可见、经过什么动作能到达”，而不只是两个图像属于同一语义场景。

**可研究问题：** JEPA 的预测目标从 global latent 改成 spatial patch transition，并加入 binocular cross-attention、visibility 与 relative pose 辅助监督，是否能提高未见场景的 SR/SPL，并降低相似地点误停？

### 6.2 动力学空白：表征增强不等于可规划世界模型

JEPA-VLA 将 V-JEPA 表征注入策略，可能主要提供更好的视觉先验；只有当模型能在动作干预下区分不同未来，且这种预测确实改善动作选择，才能称为导航世界模型。

**可研究问题：** 在固定策略容量和候选预算下，action-conditioned latent prediction 相比只加 V-JEPA 特征，是否仍有独立收益？预测误差是否与闭环成功、碰撞和恢复能力相关？

### 6.3 代价函数空白：潜在距离缺乏可达性

欧氏/余弦 latent distance 容易偏好视觉上接近但不可通行的状态。导航还需要时序距离、测地可达性、碰撞风险和不确定性。

**可研究问题：** 学习 goal-conditioned temporal distance / reachability head，并用反例区分“同房间近”“隔墙近”“外观相似远”，能否使规划代价与真实测地距离更一致？

### 6.4 长时空白：短视频编码不能替代未知地图探索

目标在很长时间内不可见时，模型必须记住访问过的路口、失败分支和地标，并选择新的 frontier。单个滑动窗口无法稳定表达数百步历史。

**可研究问题：** 稀疏拓扑记忆存储 JEPA patch/landmark token，与循环 latent state 相比，是否能减少重复探索并提高长路径 SPL？

### 6.5 评价空白：离线轨迹误差不能替代闭环能力

低 ATE 可能来自复现数据集平均动作，不能证明碰到障碍后会恢复，也不能证明能在目标长期不可见时探索。ImageNav 必须在未见场景中闭环执行，并区分感知、动力学、搜索、碰撞、停止五类失败。

## 7. 推荐的 JEPA-ImageNav 方法框架

建议将模型暂称为 **CORA-JEPA**（Correspondence- and Reachability-Aware JEPA），论文题目层面可使用“JEPA for Image-Goal Navigation”。名称不是重点，关键是下列模块各自可消融。

### 7.1 输入与编码

- 历史观测编码器 $E_o$：编码最近 $H$ 帧第一视角 RGB，输出保持空间位置的 patch token；优先测试冻结 DINOv2-S、V-JEPA 2-L/V-JEPA 2.1-L，而不是直接训练超大 backbone。
- 目标编码器 $E_g$：编码目标图像，保留 patch token。
- 早期双目交互 $C(o_t,g)$：在当前 token 与目标 token 间做双向 cross-attention，输出局部对应、可见性和相对方向信息。
- 长时记忆 $M_t$：保存稀疏 landmark token、访问次数、动作连接和 reachability，不保存所有原始帧。

### 7.2 动作条件 JEPA 预测器

给定当前状态 token、记忆摘要和候选动作序列，预测未来 $k$ 步的空间潜在 token：

$$
\hat z_{t+k}=P_\theta(z_{t-H:t}, M_t, a_{t:t+k-1}).
$$

target encoder 只处理真实未来帧并 stop-gradient，student 不得看到未来输入，避免 information leakage。可比较 future-only target 与 JEPA-WAM 式 current-future joint target。

### 7.3 目标可达性与风险评分

候选轨迹总代价建议为：

$$
J(a)=\lambda_g D_{corr}(\hat z_{t+k},z_g)
+\lambda_r D_{reach}(\hat z_{t+k},z_g)
+\lambda_c R_{collision}(a)
+\lambda_u U(\hat z_{t+k})
-\lambda_p \log \pi(a\mid o,g,M_t).
$$

- $D_{corr}$：基于 patch correspondence 的目标匹配代价；
- $D_{reach}$：预测时间距离或测地可达性，惩罚隔墙近和视觉捷径；
- $R_{collision}$：由深度上界、occupancy head 或视觉碰撞分类器给出；
- $U$：多步预测不确定性，防止模型对分布外未来过度自信；
- $\pi$：NoMaD/轻量 Transformer 类策略先验，避免在高维动作空间盲目采样。

### 7.4 未知目标的探索模式

当目标匹配置信度低或所有候选的 reachability 都低时，不应继续朝某个相似区域硬走，而应切换到探索：

1. 在拓扑记忆中选择未访问或高信息增益 frontier；
2. 用 goal-masked policy 生成安全探索候选；
3. 通过 JEPA 预测碰撞与新颖性进行重排；
4. 一旦出现可靠目标对应，进入 goal-directed 模式；
5. 对 InstanceImageNav 增加“靠近验证”状态，若匹配下降则把该候选加入短期 blacklist 并恢复探索。

## 8. 关键实验协议

### 8.1 数据与划分

- **标准 ImageNav：** Habitat + Gibson 或 Matterport3D，严格按 scene 划分 train/val-seen/val-unseen/test-unseen；优先使用与 DEBiT 和 3DV 2026 工作可对齐的设置。
- **InstanceImageNav：** HM3D-Semantics v0.2，使用官方 InstanceImageNav episode，目标相机参数与 agent camera 解耦。
- 按 trajectory、goal image 和 scene 同时查重，避免同一场景相邻帧泄漏到训练与测试。
- 划分难度桶：起终点测地距离、转弯数、目标初始是否可见、跨房间/跨墙、视角差、相机 FOV 差、相似干扰物数量。
- 模拟器主结果必须关闭不真实的 wall sliding；若同时报告 sliding=True，只能作为兼容旧工作的附加结果。

### 8.2 第一轮必须包含的基线

| 编号 | 基线 | 目的 |
|---|---|---|
| B0 | 行为克隆/RL policy-only | 判断任务本身在当前数据上可学到何种程度 |
| B1 | DINOv2/V-JEPA 特征注入 policy，无未来预测 | 隔离“更好表征”带来的收益 |
| B2 | DINO-WM/LeWM 式 global latent WM + uninformed sampling | 对照简单潜在世界模型 |
| B3 | patch latent WM + uninformed MPPI/CEM | 判断空间 token 是否改善世界模型 |
| B4 | Policy + WM reranking（无迭代优化） | 对齐 PiJEPA 的 Octo-WM，量化打分收益 |
| B5 | Policy-guided MPPI/CEM | 判断迭代规划相对简单重排的独立价值 |
| B6 | DEBiT-style goal comparator + policy | 对照强跨视角对应，而非把所有收益归因于 JEPA |
| B7 | Oracle shortest-path / oracle map planner | 给出环境和动作接口的上界 |

如果进入 InstanceImageNav，再加入 Mod-IIN、IEVE 或 UniGoal 式 modular baseline；若做真实移动机器人，再加入 ViNT/NoMaD pretrained policy。

### 8.3 必做消融

1. global CLS 对比 spatial patch token；
2. late fusion 对比 early patch cross-attention；
3. 无未来损失、future-only target、current-future joint target；
4. 潜在余弦距离对比 temporal distance / reachability cost；
5. 无记忆、GRU/RSSM、稀疏拓扑记忆；
6. 无策略先验、policy reranking、iterative planning；
7. 单步预测对比多步 rollout；
8. 无 uncertainty/collision head 对比完整风险代价；
9. DINOv2、V-JEPA 2、V-JEPA 2.1 在相同参数和训练预算下对比；
10. RGB-only 主结果与 RGB-D/odometry 上界对照。

### 8.4 指标与失败诊断

| 层面 | 指标 |
|---|---|
| 最终任务 | SR、SPL、SoftSPL、Distance-to-Success、正确 STOP 率 |
| 路径质量 | 路径长度、相对最短路额外长度、重复访问率、平均转向次数 |
| 安全 | episode collision rate、每米碰撞数、最小障碍距离、卡死/恢复率 |
| 目标感知 | 目标可见性 AUC、相对方位/距离误差、patch matching precision、相似地点误停率 |
| 世界模型 | 1/3/5/10 步 latent prediction、action sensitivity、rank correlation with true geodesic progress |
| 规划 | 候选命中率、重排增益、MPPI 迭代增益、规划时延、显存和吞吐 |
| 鲁棒性 | 光照、模糊、遮挡、FOV/高度变化、动作噪声、未见场景和长距离分桶 |

最终结果需至少 3 个训练种子；episode 级配对比较；报告均值、置信区间和效应量。所有方法共享相同 episode、候选数、rollout horizon、重规划频率和 wall-clock 上限。

## 9. 最有价值的三条论文路线

### 路线 A：Patch-Correspondence JEPA（优先级最高）

**假设：** ImageNav 的首要瓶颈是跨视角局部对应，保留空间结构的 JEPA 预测比全局 latent distance 更可靠。

**最小贡献：** early patch cross-attention + action-conditioned future patch prediction + relative pose/visibility auxiliary objective。

**成功证据：** 未见场景 SR/SPL 提升，同时相似地点误停率和 relative pose error 降低；B1 与 B6 对照能证明收益不是单纯换了更强视觉 backbone。

### 路线 B：Reachability-Aware Planning（与 A 可组合）

**假设：** 世界模型失败的关键不是未来 token 不够准，而是目标代价不懂“能不能到”。

**最小贡献：** 从离线轨迹学习 temporal/geodesic reachability，把同房间、隔墙、回环和视觉相似远处构造成 hard negatives；用于候选轨迹重排。

**成功证据：** 跨墙和长绕行桶的成功率显著上升；预测评分与真实 geodesic progress 的相关性提高；在固定候选预算下优于余弦 latent cost。

### 路线 C：Memory-and-Verification JEPA（第二阶段）

**假设：** 部分观测和目标长期不可见时，稀疏拓扑记忆与显式验证比单纯扩大视频窗口更有效。

**最小贡献：** JEPA landmark memory + exploration/goal-seeking/verification 状态机 + uncertainty-aware rollback。

**成功证据：** 长距离、跨房间和多干扰物 IIN 中重复探索减少、SPL 上升；错误候选能够被验证后拒绝并恢复搜索。

## 10. 推荐实施顺序

### 阶段 0：任务和接口验证

- 复现一个 Habitat ImageNav policy baseline；
- 固定 scene split、sliding=False、动作空间、成功半径和评估脚本；
- 建立失败标签：碰撞、目标未发现、错误匹配、隔墙捷径、停止错误、长程漂移。

### 阶段 1：最小 JEPA-ImageNav

- 冻结 DINOv2-S 或 V-JEPA 2-L，训练小型 patch predictor；
- 只做 B0–B5，优先判断“世界模型是否比 policy/retrieval 有独立价值”；
- 如果 iterative MPPI 不优于 reranking，不继续堆规划迭代，转向改进代价和表征。

### 阶段 2：对应与可达性

- 加入 DEBiT 风格 early patch interaction；
- 训练 visibility、relative pose 和 temporal reachability heads；
- 构造跨墙与相似场景 hard negative，做按难度桶的严格分析。

### 阶段 3：未知地图与部分观测

- 引入稀疏拓扑记忆和 frontier exploration；
- 比较固定窗口、RSSM 和记忆图；
- 转向 HM3D InstanceImageNav，加入验证/回退机制。

### 阶段 4：语言扩展

- 只有在 ImageNav 主线已经证明世界模型贡献后，再加入语言约束或路线指令；
- 语言可以先只约束 policy prior，再比较语言同时进入 world model 是否有额外收益；
- 此时“JEPA-VLN”才成为严格的任务名称。

## 11. 风险判断

- **最大科学风险：** 结果提升全部来自 V-JEPA 视觉表征，而不是动作条件预测。解决办法是 B1 与 B2–B5 的严格对照。
- **最大工程风险：** Habitat、HM3D、V-JEPA 2 和大模型策略同时引入后，训练链条过重。第一阶段应使用冻结小/中型 backbone、短 horizon、小 predictor，先证明假设。
- **最大评价风险：** 使用 sliding=True、相邻帧泄漏或只报离线 ATE 会制造虚假进步。必须以 unseen-scene closed-loop SR/SPL 和碰撞为主。
- **最大投稿风险：** 把纯图像目标任务称为 VLN。论文标题和任务定义应使用 ImageNav，语言扩展作为后续版本。
- **最大泛化风险：** 视觉相似地点、相机参数变化和动作噪声会同时破坏目标匹配与动力学预测，需在设计初期就纳入测试桶。

## 12. 最终研究建议

首篇工作不应追求一个同时包含 VLM、LLM、3DGS、扩散策略、世界模型和拓扑图的庞大系统。最清晰、最可能形成可信论文证据的主张是：

> **在未见室内场景的 Image-Goal Navigation 中，空间对应与可达性感知的动作条件 JEPA，能够比全局潜在距离和纯策略更可靠地选择可执行、无碰撞、朝目标取得真实测地进展的动作。**

围绕这一主张，最小可发表结构是“patch correspondence + action-conditioned JEPA + reachability cost + policy reranking”。只有当长距离结果表明主要瓶颈转向重复探索，再加入拓扑记忆；只有当目标定义需要自然语言约束，再扩展成严格意义的 JEPA-VLN。

## 参考论文入口

### JEPA / VLA / World Model

- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482)
- [JEPA-VLA: Video Predictive Embedding is Needed for VLA Models](https://arxiv.org/abs/2602.11832)
- [VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model](https://arxiv.org/abs/2602.10098)
- [JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling](https://arxiv.org/abs/2608.09381)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://proceedings.mlr.press/v267/zhou25t.html)
- [Navigation World Models](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html)
- [PiJEPA: Policy-Guided World Model Planning for Language-Conditioned Visual Navigation](https://arxiv.org/abs/2603.25981)

### VLN / ImageNav

- [R2R: Vision-and-Language Navigation](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html)
- [NaviLLM](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Towards_Learning_a_Generalist_Model_for_Embodied_Navigation_CVPR_2024_paper.html)
- [NaVid](https://www.roboticsproceedings.org/rss20/p079.html)
- [Uni-NaVid](https://www.roboticsproceedings.org/rss21/p013.html)
- [NaVILA](https://www.roboticsproceedings.org/rss21/p018.html)
- [NavMorph](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.html)
- [Topological Semantic Graph Memory for Image-Goal Navigation](https://proceedings.mlr.press/v205/kim23a.html)
- [Renderable Neural Radiance Map for Visual Navigation](https://openaccess.thecvf.com/content/CVPR2023/html/Kwon_Renderable_Neural_Radiance_Map_for_Visual_Navigation_CVPR_2023_paper.html)
- [Navigating to Objects Specified by Images](https://openaccess.thecvf.com/content/ICCV2023/html/Krantz_Navigating_to_Objects_Specified_by_Images_ICCV_2023_paper.html)
- [End-to-End (Instance)-Image Goal Navigation through Correspondence](https://openreview.net/forum?id=cphhnHjCvC)
- [Instance-aware Exploration-Verification-Exploitation](https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.html)
- [GOAT-Bench](https://openaccess.thecvf.com/content/CVPR2024/html/Khanna_GOAT-Bench_A_Benchmark_for_Multi-Modal_Lifelong_Navigation_CVPR_2024_paper.html)
- [UniGoal](https://openaccess.thecvf.com/content/CVPR2025/html/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.html)
- [IGL-Nav](https://openaccess.thecvf.com/content/ICCV2025/html/Guo_IGL-Nav_Incremental_3D_Gaussian_Localization_for_Image-goal_Navigation_ICCV_2025_paper.html)
- [What Does Really Matter in Image Goal Navigation?](https://arxiv.org/abs/2507.01667)
- [ViNT](https://proceedings.mlr.press/v229/shah23a.html)
- [NoMaD](https://openreview.net/pdf?id=FhQRJW71h5)
- [Can Vision Foundation Models Navigate?](https://arxiv.org/abs/2603.25937)
