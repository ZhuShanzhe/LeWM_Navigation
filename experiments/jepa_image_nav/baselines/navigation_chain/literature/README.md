# 分任务模型选择

各方法承担不同环节，按原生任务建立对照。相同HSSD场景可以复用，但不能把地点、实例、类别、语言和运动任务转换成一个分数排名。

| 工作 | 架构与主要创新 | 对应数据与评测用途 | 需要关注的限制 |
| --- | --- | --- | --- |
| [GOAT Bench](https://github.com/Ram81/goat-bench) | 多模态目标与视觉编码结合循环策略，处理持续的图像、类别和语言子目标。实际下载的SenseAct-NN Monolithic检查点为两层LSTM。 | 作为有原生图像与类别输入的官方预训练策略；在HSSD分别测试单目标冷启动。 | 官方强调终身混合目标任务，HSSD冷启动不能替代其原生任务评价；传感器、动作6的子任务STOP和隐藏状态必须对齐。 |
| [UniGoal](https://github.com/bagh2178/UniGoal) | 目标与在线场景转换为图结构，结合Grounded-SAM、Mask R-CNN、DISK/LightGlue、本地LLM/VLM、BEV和FMM执行。 | 完整实例图像方法是HSSD实例任务的优先对照；图匹配与探索可作为JEPA替换模块的定位依据。 | 图构建和多模型推理成本高，受检测、深度标定与LLM输出影响。纯图片协议须禁止目标分类失败时回退到数据集真值类别；类别辅助需单独标注。 |
| [IEVE](https://github.com/XiaohanLei/IEVE) | 分离探索、候选验证和利用，针对同一物体实例进行主动确认。 | 与实例任务直接对口；保留作为第二个实例专用对照。 | 官方模型下载入口核查时未成功获取；不能用自己的匹配器系统冒充其完整复现。 |
| [Jev ObjNav](https://github.com/owlet914/jev-objnav) | RGB-D与位姿构建占据、目标和值地图及区域图；官方Jev API输出高层决策，ROS/C++模块规划与执行。原生视觉服务包含DINO、BLIP2 ITM、MobileSAM和YOLOv7。 | 适合类别与开放词汇目标导航；HSSD类别任务按任意同类实例评分。 | 不是图片唯一实例任务；API连通不等于完整模型部署。必须禁止demo或fallback输出替代真实视觉/决策服务，并记录远程模型版本与费用。 |
| [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench) | 统一多模态模型与具身任务接口；EB-Navigation在AI2-THOR内根据视觉与语言预测动作序列。 | 官方5组共300任务，测试基础指令、常识、复杂指令、外观描述与长时程条件；本地Qwen2.5-VL-7B为预训练后端。 | 只有20步，且距离足够近自动成功，不等同于R2R/RxR的长路线VLN或实例确认。受限JSON解码、上下文与反馈设置须明确固定。 |
| [VLFM](https://github.com/rai-opensource/vlfm) | BLIP2图文匹配给探索区域打分，结合占据地图、frontier与PointNav局部控制。 | 可作为相同类别场景下的局部视觉语言价值图方法对照。 | 地图质量、图文相关性和局部控制共同决定表现；类别相似不保证实例身份。当前未形成该方法的验收成绩。 |
| [PoliFormer](https://github.com/allenai/poliformer) | 基于Transformer的预训练物体导航策略，在大规模程序化室内环境中学习探索和目标到达。 | 若使用它，优先保留官方ProcTHOR/Objaverse资产、机器人与双相机设置，作为另一类别导航路线。 | 官方AI2-THOR版本和机器人传感器不同，不能仅把相机接到HSSD就称原生复现；当前未运行。 |
| [GC VLN](https://github.com/bagh2178/GC-VLN) | 通过图约束组织语言目标和空间关系，提供不依赖额外训练的VLN路线。 | 作为多目标关系与语言约束的延伸；与图结构记忆结合具有参考价值。 | 原生路线评测依赖相应MP3D/R2R场景权限，当前未运行；EB-Navigation成绩不能替代。 |
| [UniWM](https://github.com/UWMILab/UniWM) | 以多模态自回归预测与分层记忆统一视觉前瞻和动作建模。 | 作为世界模型与导航动作联合建模的延伸；关注跨数据域的轨迹与未来视觉预测。 | 公开README要求用户提供检查点；离线预测或使用真值动作的rollout不等于仿真闭环到达率，当前未运行。 |
| [JEPLO](https://github.com/ASIG-X/JEPLO) | 利用LiDAR和本体信息的潜空间预测学习形成运动表征，供四足策略执行复杂地形运动。 | 原生MuJoCo部署测速度跟踪、姿态和越障；[真机记录数据](https://huggingface.co/datasets/ASIG-Lab/JEPLO-Dataset)用于部署和里程计延伸。 | 不负责照片理解、实例辨识或语义路径选择，不能作为VLN成功率模型。仿真运动成绩不代表真机表现。 |

## JEPA 对照位置

实例路线优先围绕UniGoal或其他完整实例方法固定感知和执行模块，再替换历史表示、候选图匹配或动作条件未来预测；类别路线可在Jev或VLFM的区域选择与验证处接入；语言路线单独检验指令约束、历史信息和动作后果预测；低层运动保留独立的稳定性与速度跟踪指标。

不同输入预算分别比较。RGB策略、RGB-D地图系统、理想里程计系统和远程API系统不混为同一条件；相同回合之外，还需记录动作次数、墙钟时间和API调用成本。
