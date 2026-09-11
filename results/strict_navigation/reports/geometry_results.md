# 几何布局能力：阶段结果

快照：2026-09-11T04:55:51+08:00。模型仍只训练原始单布局数据；地图名称中的train只是未来多布局组标签。

所有布局均采用同一组100个起终点（50同侧、50跨墙）；水平墙条件对坐标转置。起终点欧氏距离75—125，预算150步。它们不是原轨迹未来75步目标，因此不能直接把该成功率与旧长目标59.3%相减作为地图泛化损失。参考控制器已证明每例在预算内可解，但最短路线难度仍随门位置变化。

| 布局 | 病例 | 成功率 | 跨墙成功率 | 同侧成功率 |
|---|---:|---:|---:|---:|
| strict3072_map_train_a1_d49 | 100 | 30.0% | 32.0% | 28.0% |
| strict3072_map_train_a1_d81 | 100 | 18.0% | 12.0% | 24.0% |
| strict3072_map_train_a1_d113 | 100 | 10.0% | 0.0% | 20.0% |
| strict3072_map_train_a1_d145 | 100 | 15.0% | 0.0% | 30.0% |
| strict3072_map_train_a0_d49 | 100 | 11.0% | 0.0% | 22.0% |
| strict3072_map_train_a0_d81 | 100 | 10.0% | 0.0% | 20.0% |
| strict3072_map_train_a0_d113 | 100 | 9.0% | 0.0% | 18.0% |
| strict3072_map_train_a0_d145 | 100 | 9.0% | 0.0% | 18.0% |
| strict3072_map_validation_a1_d65 | 100 | 24.0% | 20.0% | 28.0% |
| strict3072_map_validation_a1_d129 | 100 | 9.0% | 0.0% | 18.0% |
| strict3072_map_validation_a0_d65 | 100 | 8.0% | 0.0% | 16.0% |
| strict3072_map_validation_a0_d129 | 100 | 10.0% | 0.0% | 20.0% |
| strict3072_map_test_a1_d97 | 100 | 10.0% | 0.0% | 20.0% |
| strict3072_map_test_a1_d161 | 100 | 15.0% | 0.0% | 30.0% |
| strict3072_map_test_a1_d177 | 100 | 15.0% | 0.0% | 30.0% |
| strict3072_map_test_a0_d97 | 100 | 11.0% | 0.0% | 22.0% |
| strict3072_map_test_a0_d161 | 100 | 12.0% | 0.0% | 24.0% |
| strict3072_map_test_a0_d177 | 100 | 11.0% | 0.0% | 22.0% |
| strict3072_map_validation_a1_d65_temporal_replace | 100 | 88.0% | 80.0% | 96.0% |
| strict3072_map_validation_a1_d129_temporal_replace | 100 | 54.0% | 14.0% | 94.0% |
| strict3072_map_validation_a0_d65_temporal_replace | 100 | 7.0% | 2.0% | 12.0% |
| strict3072_map_validation_a0_d129_temporal_replace | 100 | 13.0% | 2.0% | 24.0% |
| geometry_single4001_map_train_a1_d49 | 100 | 30.0% | 34.0% | 26.0% |
| geometry_single4001_map_validation_a1_d65 | 100 | 23.0% | 30.0% | 16.0% |
| geometry_single4001_map_validation_a1_d129 | 100 | 13.0% | 4.0% | 22.0% |
| geometry_single4001_map_validation_a0_d65 | 100 | 11.0% | 2.0% | 20.0% |
| geometry_single4001_map_validation_a0_d129 | 100 | 14.0% | 10.0% | 18.0% |
| geometry_multi4001_map_train_a1_d49 | 100 | 9.0% | 6.0% | 12.0% |
| geometry_multi4001_map_validation_a1_d65 | 100 | 6.0% | 4.0% | 8.0% |
| geometry_multi4001_map_validation_a1_d129 | 100 | 9.0% | 8.0% | 10.0% |
| geometry_multi4001_map_validation_a0_d65 | 100 | 11.0% | 12.0% | 10.0% |
| geometry_multi4001_map_validation_a0_d129 | 100 | 11.0% | 12.0% | 10.0% |
| strict3072_map_validation_a1_d65_retrieval_only | 100 | 83.0% | 80.0% | 86.0% |
| strict3072_map_validation_a1_d65_retrieval_rerank | 100 | 81.0% | 70.0% | 92.0% |
| strict3072_map_validation_a1_d65_retrieval_init | 100 | 70.0% | 58.0% | 82.0% |
| strict3072_map_validation_a1_d129_retrieval_only | 100 | 46.0% | 6.0% | 86.0% |
| strict3072_map_validation_a1_d129_retrieval_rerank | 100 | 48.0% | 0.0% | 96.0% |
| strict3072_map_validation_a1_d129_retrieval_init | 100 | 48.0% | 6.0% | 90.0% |
| strict3072_map_validation_a0_d65_retrieval_only | 100 | 15.0% | 0.0% | 30.0% |
| strict3072_map_validation_a0_d65_retrieval_rerank | 100 | 11.0% | 0.0% | 22.0% |
| strict3072_map_validation_a0_d65_retrieval_init | 100 | 14.0% | 0.0% | 28.0% |
| strict3072_map_validation_a0_d129_retrieval_only | 100 | 12.0% | 0.0% | 24.0% |
| strict3072_map_validation_a0_d129_retrieval_rerank | 100 | 14.0% | 0.0% | 28.0% |
| strict3072_map_validation_a0_d129_retrieval_init | 100 | 14.0% | 0.0% | 28.0% |

相对原始几何（竖墙、门49）的同起终点配对差值：

- strict3072_map_train_a1_d81：-12.0个百分点，病例配对95%区间 [-20.0, -4.0]。
- strict3072_map_train_a1_d113：-20.0个百分点，病例配对95%区间 [-28.000000000000004, -12.0]。
- strict3072_map_train_a1_d145：-15.0个百分点，病例配对95%区间 [-25.0, -5.0]。
- strict3072_map_train_a0_d49：-19.0个百分点，病例配对95%区间 [-30.0, -8.0]。
- strict3072_map_train_a0_d81：-20.0个百分点，病例配对95%区间 [-31.0, -9.0]。
- strict3072_map_train_a0_d113：-21.0个百分点，病例配对95%区间 [-32.0, -10.0]。
- strict3072_map_train_a0_d145：-21.0个百分点，病例配对95%区间 [-32.0, -10.0]。
- strict3072_map_validation_a1_d65：-6.0个百分点，病例配对95%区间 [-14.000000000000002, 2.0]。
- strict3072_map_validation_a1_d129：-21.0个百分点，病例配对95%区间 [-30.0, -12.0]。
- strict3072_map_validation_a0_d65：-22.0个百分点，病例配对95%区间 [-33.0, -11.0]。
- strict3072_map_validation_a0_d129：-20.0个百分点，病例配对95%区间 [-31.0, -9.0]。
- strict3072_map_test_a1_d97：-20.0个百分点，病例配对95%区间 [-28.999999999999996, -12.0]。
- strict3072_map_test_a1_d161：-15.0个百分点，病例配对95%区间 [-23.0, -7.000000000000001]。
- strict3072_map_test_a1_d177：-15.0个百分点，病例配对95%区间 [-24.0, -6.0]。
- strict3072_map_test_a0_d97：-19.0个百分点，病例配对95%区间 [-31.0, -7.000000000000001]。
- strict3072_map_test_a0_d161：-18.0个百分点，病例配对95%区间 [-30.0, -6.0]。
- strict3072_map_test_a0_d177：-19.0个百分点，病例配对95%区间 [-31.0, -7.000000000000001]。
- geometry_single4001_map_train_a1_d49：+0.0个百分点，病例配对95%区间 [-11.0, 11.0]。
- geometry_single4001_map_validation_a1_d65：-7.0个百分点，病例配对95%区间 [-17.0, 3.0]。
- geometry_single4001_map_validation_a1_d129：-17.0个百分点，病例配对95%区间 [-28.999999999999996, -5.0]。
- geometry_single4001_map_validation_a0_d65：-19.0个百分点，病例配对95%区间 [-30.0, -8.0]。
- geometry_single4001_map_validation_a0_d129：-16.0个百分点，病例配对95%区间 [-27.0, -5.0]。
- geometry_multi4001_map_train_a1_d49：-21.0个百分点，病例配对95%区间 [-30.0, -12.0]。
- geometry_multi4001_map_validation_a1_d65：-24.0个百分点，病例配对95%区间 [-33.0, -15.0]。
- geometry_multi4001_map_validation_a1_d129：-21.0个百分点，病例配对95%区间 [-31.0, -10.0]。
- geometry_multi4001_map_validation_a0_d65：-19.0个百分点，病例配对95%区间 [-28.000000000000004, -11.0]。
- geometry_multi4001_map_validation_a0_d129：-19.0个百分点，病例配对95%区间 [-30.0, -8.0]。

上述区间条件于单个训练模型和固定地图，不代表跨训练种子或地图总体不确定性；不同布局复用病例，不汇总成独立的千次试验。两房间移动门或旋转墙不是新连通拓扑。当前低成功率可能包含布局外推、目标分布和数据覆盖等因素；不能单独归因于表示、动态、规划器或记忆。

时间评价头的已知布局97%—98%结果尚不能外推到这些布局。其4个验证几何布局已完成，具体差值与解释见transfer_mechanism_results.md。匹配的单布局/多布局采集方案与额外采集先验见geometry_collection_protocol.md，尚未完成相应训练。
