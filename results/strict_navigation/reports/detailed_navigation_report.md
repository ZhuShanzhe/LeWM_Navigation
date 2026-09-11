# 严格导航实验：阶段结果

更新时间：2026-09-11T04:55:51+08:00

当前阶段：summarize_three_seed_baseline。有限队列 626 条记录，已完成 384 条；记录数包括冒烟/训练/报告任务，不等于正式独立实验数量。

结果一致性检查：已核对 336 个正式评估输出，发现 0 项需处理问题。

## 已知布局、保留轨迹

| 模型与设置 | 病例/计划数 | SR (%) | 排除初始成功 SR (%) | 状态 |
|---|---:|---:|---:|---|
| strict3072_validation_short50 | 100/100 | 95.0 | 94.3820224719101 | 完整组 |
| strict3072_validation_short150 | 100/100 | 97.0 | 96.62921348314607 | 完整组 |
| strict3072_validation_long_base | 100/100 | 62.0 | 62.0 | 完整组 |
| strict3072_validation_long_feedback_budget | 100/100 | 62.0 | 62.0 | 完整组 |
| strict3072_validation_long_feedback_extra | 100/100 | 44.0 | 44.0 | 完整组 |
| strict3072_validation_long_h10_budget | 100/100 | 28.0 | 28.000000000000004 | 完整组 |
| strict3072_validation_long_feedback_budget_warm | 100/100 | 64.0 | 64.0 | 完整组 |
| strict3072_validation_long_feedback_extra_warm | 100/100 | 64.0 | 64.0 | 完整组 |
| strict3072_validation_long_h10_budget_warm | 100/100 | 66.0 | 66.0 | 完整组 |
| strict3072_validation_long_base_bounded | 100/100 | 49.0 | 49.0 | 完整组 |
| strict3072_validation_long_feedback_extra_bounded | 100/100 | 47.0 | 47.0 | 完整组 |
| strict3072_validation_long_feedback_extra_warm_bounded | 100/100 | 46.0 | 46.0 | 完整组 |
| strict3072_test_short50 | 300/300 | 94.0 | 93.47826086956522 | 完整组 |
| strict3072_test_short150 | 300/300 | 95.0 | 94.56521739130434 | 完整组 |
| strict3072_test_long_base | 300/300 | 59.3 | 59.06040268456376 | 完整组 |
| strict3072_test_long_feedback_budget | 300/300 | 53.3 | 53.02013422818792 | 完整组 |
| strict3072_test_long_feedback_extra | 300/300 | 46.0 | 45.63758389261745 | 完整组 |
| strict3072_test_long_h10_budget | 300/300 | 31.7 | 31.20805369127517 | 完整组 |
| strict_random_test_long | 300/300 | 4.3 | 4.026845637583892 | 完整组 |
| strict_noop_test_long | 300/300 | 0.7 | 0.0 | 完整组 |
| strict3072_validation_long_base_temporal_replace | 100/100 | 97.0 | 97.0 | 完整组 |
| strict3072_validation_long_base_temporal_hybrid | 100/100 | 98.0 | 98.0 | 完整组 |
| strict3072_validation_short150_temporal_replace | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3072_validation_long_base_shuffled_replace | 100/100 | 6.0 | 6.0 | 完整组 |
| strict3072_validation_long_base_shuffled_hybrid | 100/100 | 76.0 | 76.0 | 完整组 |
| strict3072_validation_short150_shuffled_replace | 100/100 | 65.0 | 60.67415730337079 | 完整组 |
| strict3072_validation_long_base_retrieval_only | 100/100 | 98.0 | 98.0 | 完整组 |
| strict3072_validation_long_base_retrieval_rerank | 100/100 | 97.0 | 97.0 | 完整组 |
| strict3072_validation_long_base_retrieval_init | 100/100 | 97.0 | 97.0 | 完整组 |
| strict3072_validation_long_base_retrieval_gaussian1 | 100/100 | 52.0 | 52.0 | 完整组 |
| strict3072_validation_short150_retrieval_only | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3072_validation_short150_retrieval_rerank | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3072_validation_short150_retrieval_init | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3072_validation_short150_retrieval_gaussian1 | 100/100 | 95.0 | 94.3820224719101 | 完整组 |
| strict3073_validation_short150 | 100/100 | 95.0 | 94.3820224719101 | 完整组 |
| strict3073_validation_long_base | 100/100 | 62.0 | 62.0 | 完整组 |
| strict3073_test_short150 | 300/300 | 94.7 | 94.20289855072464 | 完整组 |
| strict3073_test_long_base | 300/300 | 56.0 | 55.70469798657718 | 完整组 |
| strict3074_validation_short150 | 100/100 | 95.0 | 94.3820224719101 | 完整组 |
| strict3074_validation_long_base | 100/100 | 60.0 | 60.0 | 完整组 |
| strict3074_test_short150 | 300/300 | 95.0 | 94.56521739130434 | 完整组 |
| strict3074_test_long_base | 300/300 | 59.0 | 58.7248322147651 | 完整组 |
| strict3073_validation_long_feedback_extra | 100/100 | 46.0 | 46.0 | 完整组 |
| strict3073_validation_long_feedback_extra_warm | 100/100 | 66.0 | 66.0 | 完整组 |
| strict3073_validation_long_h10_budget | 100/100 | 36.0 | 36.0 | 完整组 |
| strict3073_validation_long_h10_budget_warm | 100/100 | 77.0 | 77.0 | 完整组 |
| strict3074_validation_long_feedback_extra | 100/100 | 46.0 | 46.0 | 完整组 |
| strict3074_validation_long_feedback_extra_warm | 100/100 | 64.0 | 64.0 | 完整组 |
| strict3074_validation_long_h10_budget | 100/100 | 30.0 | 30.0 | 完整组 |
| strict3074_validation_long_h10_budget_warm | 100/100 | 68.0 | 68.0 | 完整组 |
| strict3073_validation_long_base_temporal_replace | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3073_validation_long_base_temporal_hybrid | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3073_validation_long_base_shuffled_replace | 100/100 | 9.0 | 9.0 | 完整组 |
| strict3073_validation_long_base_shuffled_hybrid | 100/100 | 67.0 | 67.0 | 完整组 |
| strict3073_validation_short150_temporal_replace | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3074_validation_long_base_temporal_replace | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3074_validation_long_base_temporal_hybrid | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3074_validation_long_base_shuffled_replace | 100/100 | 4.0 | 4.0 | 完整组 |
| strict3074_validation_long_base_shuffled_hybrid | 100/100 | 58.0 | 57.99999999999999 | 完整组 |
| strict3074_validation_short150_temporal_replace | 100/100 | 100.0 | 100.0 | 完整组 |
| strict3073_validation_long_base_retrieval_only | 100/100 | 98.0 | 98.0 | 完整组 |
| strict3073_validation_long_base_retrieval_rerank | 100/100 | 96.0 | 96.0 | 完整组 |
| strict3074_validation_long_base_retrieval_only | 100/100 | 99.0 | 99.0 | 完整组 |
| strict3074_validation_long_base_retrieval_rerank | 100/100 | 97.0 | 97.0 | 完整组 |

原始无热启动协议与新增热启动/动作边界对照分开列出。后者用于验证计划延续性和预测—执行动作一致性；尚不能将任何下降单独归因于模型表示。

## 配对差值

- strict3072_validation_long_base_bounded 相对 strict3072_validation_long_base，all，n=100：-13.0 个百分点，病例配对 bootstrap 95% 区间 [-20.0, -6.0]。
- strict3072_validation_long_base_bounded 相对 strict3072_validation_long_base，across_wall，n=67：-11.9 个百分点，病例配对 bootstrap 95% 区间 [-20.9, -3.0]。
- strict3072_validation_long_base_bounded 相对 strict3072_validation_long_base，same_side，n=33：-15.2 个百分点，病例配对 bootstrap 95% 区间 [-27.3, -3.0]。
- strict3072_validation_long_base_retrieval_gaussian1 相对 strict3072_validation_long_base，all，n=100：-10.0 个百分点，病例配对 bootstrap 95% 区间 [-18.0, -3.0]。
- strict3072_validation_long_base_retrieval_gaussian1 相对 strict3072_validation_long_base，across_wall，n=67：-17.9 个百分点，病例配对 bootstrap 95% 区间 [-28.4, -9.0]。
- strict3072_validation_long_base_retrieval_gaussian1 相对 strict3072_validation_long_base，same_side，n=33：+6.1 个百分点，病例配对 bootstrap 95% 区间 [+0.0, +15.2]。
- strict3072_validation_long_base_retrieval_init 相对 strict3072_validation_long_base，all，n=100：+35.0 个百分点，病例配对 bootstrap 95% 区间 [+26.0, +45.0]。
- strict3072_validation_long_base_retrieval_init 相对 strict3072_validation_long_base，across_wall，n=67：+38.8 个百分点，病例配对 bootstrap 95% 区间 [+26.9, +50.7]。
- strict3072_validation_long_base_retrieval_init 相对 strict3072_validation_long_base，same_side，n=33：+27.3 个百分点，病例配对 bootstrap 95% 区间 [+12.1, +42.4]。
- strict3072_validation_long_base_retrieval_only 相对 strict3072_validation_long_base，all，n=100：+36.0 个百分点，病例配对 bootstrap 95% 区间 [+26.0, +46.0]。
- strict3072_validation_long_base_retrieval_only 相对 strict3072_validation_long_base，across_wall，n=67：+38.8 个百分点，病例配对 bootstrap 95% 区间 [+26.9, +50.7]。
- strict3072_validation_long_base_retrieval_only 相对 strict3072_validation_long_base，same_side，n=33：+30.3 个百分点，病例配对 bootstrap 95% 区间 [+15.2, +45.5]。
- strict3072_validation_long_base_retrieval_rerank 相对 strict3072_validation_long_base，all，n=100：+35.0 个百分点，病例配对 bootstrap 95% 区间 [+26.0, +45.0]。
- strict3072_validation_long_base_retrieval_rerank 相对 strict3072_validation_long_base，across_wall，n=67：+40.3 个百分点，病例配对 bootstrap 95% 区间 [+28.4, +52.2]。
- strict3072_validation_long_base_retrieval_rerank 相对 strict3072_validation_long_base，same_side，n=33：+24.2 个百分点，病例配对 bootstrap 95% 区间 [+12.1, +39.4]。
- strict3072_validation_long_base_shuffled_hybrid 相对 strict3072_validation_long_base，all，n=100：+14.0 个百分点，病例配对 bootstrap 95% 区间 [+6.0, +22.0]。
- strict3072_validation_long_base_shuffled_hybrid 相对 strict3072_validation_long_base，across_wall，n=67：+10.4 个百分点，病例配对 bootstrap 95% 区间 [+1.5, +19.4]。
- strict3072_validation_long_base_shuffled_hybrid 相对 strict3072_validation_long_base，same_side，n=33：+21.2 个百分点，病例配对 bootstrap 95% 区间 [+9.1, +36.4]。
- strict3072_validation_long_base_shuffled_replace 相对 strict3072_validation_long_base，all，n=100：-56.0 个百分点，病例配对 bootstrap 95% 区间 [-66.0, -46.0]。
- strict3072_validation_long_base_shuffled_replace 相对 strict3072_validation_long_base，across_wall，n=67：-55.2 个百分点，病例配对 bootstrap 95% 区间 [-67.2, -43.3]。
- strict3072_validation_long_base_shuffled_replace 相对 strict3072_validation_long_base，same_side，n=33：-57.6 个百分点，病例配对 bootstrap 95% 区间 [-75.8, -39.4]。
- strict3072_validation_long_base_temporal_hybrid 相对 strict3072_validation_long_base，all，n=100：+36.0 个百分点，病例配对 bootstrap 95% 区间 [+27.0, +45.0]。
- strict3072_validation_long_base_temporal_hybrid 相对 strict3072_validation_long_base，across_wall，n=67：+41.8 个百分点，病例配对 bootstrap 95% 区间 [+29.9, +53.7]。
- strict3072_validation_long_base_temporal_hybrid 相对 strict3072_validation_long_base，same_side，n=33：+24.2 个百分点，病例配对 bootstrap 95% 区间 [+9.1, +39.4]。
- strict3072_validation_long_base_temporal_replace 相对 strict3072_validation_long_base，all，n=100：+35.0 个百分点，病例配对 bootstrap 95% 区间 [+26.0, +44.0]。
- strict3072_validation_long_base_temporal_replace 相对 strict3072_validation_long_base，across_wall，n=67：+38.8 个百分点，病例配对 bootstrap 95% 区间 [+26.9, +50.7]。
- strict3072_validation_long_base_temporal_replace 相对 strict3072_validation_long_base，same_side，n=33：+27.3 个百分点，病例配对 bootstrap 95% 区间 [+12.1, +42.4]。
- strict3072_validation_long_feedback_budget 相对 strict3072_validation_long_base，all，n=100：+0.0 个百分点，病例配对 bootstrap 95% 区间 [-7.0, +7.0]。
- strict3072_validation_long_feedback_budget 相对 strict3072_validation_long_base，across_wall，n=67：-3.0 个百分点，病例配对 bootstrap 95% 区间 [-11.9, +6.0]。
- strict3072_validation_long_feedback_budget 相对 strict3072_validation_long_base，same_side，n=33：+6.1 个百分点，病例配对 bootstrap 95% 区间 [+0.0, +15.2]。
- strict3072_validation_long_feedback_budget_warm 相对 strict3072_validation_long_base，all，n=100：+2.0 个百分点，病例配对 bootstrap 95% 区间 [-3.0, +7.0]。
- strict3072_validation_long_feedback_budget_warm 相对 strict3072_validation_long_base，across_wall，n=67：+1.5 个百分点，病例配对 bootstrap 95% 区间 [-4.5, +7.5]。
- strict3072_validation_long_feedback_budget_warm 相对 strict3072_validation_long_base，same_side，n=33：+3.0 个百分点，病例配对 bootstrap 95% 区间 [+0.0, +9.1]。
- strict3072_validation_long_feedback_extra 相对 strict3072_validation_long_base，all，n=100：-18.0 个百分点，病例配对 bootstrap 95% 区间 [-27.0, -10.0]。
- strict3072_validation_long_feedback_extra 相对 strict3072_validation_long_base，across_wall，n=67：-20.9 个百分点，病例配对 bootstrap 95% 区间 [-32.8, -10.4]。
- strict3072_validation_long_feedback_extra 相对 strict3072_validation_long_base，same_side，n=33：-12.1 个百分点，病例配对 bootstrap 95% 区间 [-24.2, -3.0]。
- strict3072_validation_long_feedback_extra_bounded 相对 strict3072_validation_long_base，all，n=100：-15.0 个百分点，病例配对 bootstrap 95% 区间 [-23.0, -8.0]。
- strict3072_validation_long_feedback_extra_bounded 相对 strict3072_validation_long_base，across_wall，n=67：-14.9 个百分点，病例配对 bootstrap 95% 区间 [-23.9, -6.0]。
- strict3072_validation_long_feedback_extra_bounded 相对 strict3072_validation_long_base，same_side，n=33：-15.2 个百分点，病例配对 bootstrap 95% 区间 [-27.3, -3.0]。
- strict3072_validation_long_feedback_extra_warm 相对 strict3072_validation_long_base，all，n=100：+2.0 个百分点，病例配对 bootstrap 95% 区间 [-5.0, +9.0]。
- strict3072_validation_long_feedback_extra_warm 相对 strict3072_validation_long_base，across_wall，n=67：+6.0 个百分点，病例配对 bootstrap 95% 区间 [-3.0, +14.9]。
- strict3072_validation_long_feedback_extra_warm 相对 strict3072_validation_long_base，same_side，n=33：-6.1 个百分点，病例配对 bootstrap 95% 区间 [-15.2, +0.0]。
- strict3072_validation_long_feedback_extra_warm_bounded 相对 strict3072_validation_long_base，all，n=100：-16.0 个百分点，病例配对 bootstrap 95% 区间 [-24.0, -9.0]。
- strict3072_validation_long_feedback_extra_warm_bounded 相对 strict3072_validation_long_base，across_wall，n=67：-14.9 个百分点，病例配对 bootstrap 95% 区间 [-23.9, -6.0]。
- strict3072_validation_long_feedback_extra_warm_bounded 相对 strict3072_validation_long_base，same_side，n=33：-18.2 个百分点，病例配对 bootstrap 95% 区间 [-33.3, -6.1]。
- strict3072_validation_long_h10_budget 相对 strict3072_validation_long_base，all，n=100：-34.0 个百分点，病例配对 bootstrap 95% 区间 [-45.0, -22.0]。
- strict3072_validation_long_h10_budget 相对 strict3072_validation_long_base，across_wall，n=67：-38.8 个百分点，病例配对 bootstrap 95% 区间 [-52.2, -23.9]。
- strict3072_validation_long_h10_budget 相对 strict3072_validation_long_base，same_side，n=33：-24.2 个百分点，病例配对 bootstrap 95% 区间 [-42.4, -6.1]。
- strict3072_validation_long_h10_budget_warm 相对 strict3072_validation_long_base，all，n=100：+4.0 个百分点，病例配对 bootstrap 95% 区间 [-5.0, +13.0]。
- strict3072_validation_long_h10_budget_warm 相对 strict3072_validation_long_base，across_wall，n=67：+1.5 个百分点，病例配对 bootstrap 95% 区间 [-10.4, +13.4]。
- strict3072_validation_long_h10_budget_warm 相对 strict3072_validation_long_base，same_side，n=33：+9.1 个百分点，病例配对 bootstrap 95% 区间 [-3.0, +21.2]。
- strict3072_test_long_feedback_budget 相对 strict3072_test_long_base，all，n=300：-6.0 个百分点，病例配对 bootstrap 95% 区间 [-10.0, -2.3]。
- strict3072_test_long_feedback_budget 相对 strict3072_test_long_base，across_wall，n=193：-11.4 个百分点，病例配对 bootstrap 95% 区间 [-16.6, -6.2]。
- strict3072_test_long_feedback_budget 相对 strict3072_test_long_base，same_side，n=107：+3.7 个百分点，病例配对 bootstrap 95% 区间 [+0.0, +8.4]。
- strict3072_test_long_feedback_extra 相对 strict3072_test_long_base，all，n=300：-13.3 个百分点，病例配对 bootstrap 95% 区间 [-17.3, -9.7]。
- strict3072_test_long_feedback_extra 相对 strict3072_test_long_base，across_wall，n=193：-18.1 个百分点，病例配对 bootstrap 95% 区间 [-23.3, -13.0]。
- strict3072_test_long_feedback_extra 相对 strict3072_test_long_base，same_side，n=107：-4.7 个百分点，病例配对 bootstrap 95% 区间 [-9.3, -0.9]。
- strict3072_test_long_h10_budget 相对 strict3072_test_long_base，all，n=300：-27.7 个百分点，病例配对 bootstrap 95% 区间 [-34.0, -21.7]。
- strict3072_test_long_h10_budget 相对 strict3072_test_long_base，across_wall，n=193：-40.9 个百分点，病例配对 bootstrap 95% 区间 [-48.7, -33.2]。
- strict3072_test_long_h10_budget 相对 strict3072_test_long_base，same_side，n=107：-3.7 个百分点，病例配对 bootstrap 95% 区间 [-12.1, +4.7]。
- strict3073_validation_long_base_retrieval_only 相对 strict3073_validation_long_base，all，n=100：+36.0 个百分点，病例配对 bootstrap 95% 区间 [+27.0, +45.0]。
- strict3073_validation_long_base_retrieval_only 相对 strict3073_validation_long_base，across_wall，n=67：+38.8 个百分点，病例配对 bootstrap 95% 区间 [+26.9, +50.7]。
- strict3073_validation_long_base_retrieval_only 相对 strict3073_validation_long_base，same_side，n=33：+30.3 个百分点，病例配对 bootstrap 95% 区间 [+15.2, +45.5]。
- strict3073_validation_long_base_retrieval_rerank 相对 strict3073_validation_long_base，all，n=100：+34.0 个百分点，病例配对 bootstrap 95% 区间 [+25.0, +43.0]。
- strict3073_validation_long_base_retrieval_rerank 相对 strict3073_validation_long_base，across_wall，n=67：+35.8 个百分点，病例配对 bootstrap 95% 区间 [+23.9, +47.8]。
- strict3073_validation_long_base_retrieval_rerank 相对 strict3073_validation_long_base，same_side，n=33：+30.3 个百分点，病例配对 bootstrap 95% 区间 [+15.2, +45.5]。
- strict3073_validation_long_base_shuffled_hybrid 相对 strict3073_validation_long_base，all，n=100：+5.0 个百分点，病例配对 bootstrap 95% 区间 [-3.0, +13.0]。
- strict3073_validation_long_base_shuffled_hybrid 相对 strict3073_validation_long_base，across_wall，n=67：+0.0 个百分点，病例配对 bootstrap 95% 区间 [-9.0, +9.0]。
- strict3073_validation_long_base_shuffled_hybrid 相对 strict3073_validation_long_base，same_side，n=33：+15.2 个百分点，病例配对 bootstrap 95% 区间 [+0.0, +30.3]。
- strict3073_validation_long_base_shuffled_replace 相对 strict3073_validation_long_base，all，n=100：-53.0 个百分点，病例配对 bootstrap 95% 区间 [-64.0, -42.0]。
- strict3073_validation_long_base_shuffled_replace 相对 strict3073_validation_long_base，across_wall，n=67：-55.2 个百分点，病例配对 bootstrap 95% 区间 [-67.2, -41.8]。
- strict3073_validation_long_base_shuffled_replace 相对 strict3073_validation_long_base，same_side，n=33：-48.5 个百分点，病例配对 bootstrap 95% 区间 [-69.7, -27.3]。
- strict3073_validation_long_base_temporal_hybrid 相对 strict3073_validation_long_base，all，n=100：+38.0 个百分点，病例配对 bootstrap 95% 区间 [+29.0, +48.0]。
- strict3073_validation_long_base_temporal_hybrid 相对 strict3073_validation_long_base，across_wall，n=67：+41.8 个百分点，病例配对 bootstrap 95% 区间 [+29.9, +53.7]。
- strict3073_validation_long_base_temporal_hybrid 相对 strict3073_validation_long_base，same_side，n=33：+30.3 个百分点，病例配对 bootstrap 95% 区间 [+15.2, +45.5]。
- strict3073_validation_long_base_temporal_replace 相对 strict3073_validation_long_base，all，n=100：+38.0 个百分点，病例配对 bootstrap 95% 区间 [+29.0, +48.0]。
- strict3073_validation_long_base_temporal_replace 相对 strict3073_validation_long_base，across_wall，n=67：+41.8 个百分点，病例配对 bootstrap 95% 区间 [+29.9, +53.7]。
- strict3073_validation_long_base_temporal_replace 相对 strict3073_validation_long_base，same_side，n=33：+30.3 个百分点，病例配对 bootstrap 95% 区间 [+15.2, +45.5]。
- strict3073_validation_long_feedback_extra 相对 strict3073_validation_long_base，all，n=100：-16.0 个百分点，病例配对 bootstrap 95% 区间 [-24.0, -8.0]。
- strict3073_validation_long_feedback_extra 相对 strict3073_validation_long_base，across_wall，n=67：-19.4 个百分点，病例配对 bootstrap 95% 区间 [-29.9, -9.0]。
- strict3073_validation_long_feedback_extra 相对 strict3073_validation_long_base，same_side，n=33：-9.1 个百分点，病例配对 bootstrap 95% 区间 [-21.2, +3.0]。
- strict3073_validation_long_feedback_extra_warm 相对 strict3073_validation_long_base，all，n=100：+4.0 个百分点，病例配对 bootstrap 95% 区间 [-3.0, +11.0]。
- strict3073_validation_long_feedback_extra_warm 相对 strict3073_validation_long_base，across_wall，n=67：+6.0 个百分点，病例配对 bootstrap 95% 区间 [+0.0, +13.4]。
- strict3073_validation_long_feedback_extra_warm 相对 strict3073_validation_long_base，same_side，n=33：+0.0 个百分点，病例配对 bootstrap 95% 区间 [-15.2, +15.2]。
- strict3073_validation_long_h10_budget 相对 strict3073_validation_long_base，all，n=100：-26.0 个百分点，病例配对 bootstrap 95% 区间 [-38.0, -14.0]。
- strict3073_validation_long_h10_budget 相对 strict3073_validation_long_base，across_wall，n=67：-32.8 个百分点，病例配对 bootstrap 95% 区间 [-46.3, -19.4]。
- strict3073_validation_long_h10_budget 相对 strict3073_validation_long_base，same_side，n=33：-12.1 个百分点，病例配对 bootstrap 95% 区间 [-33.3, +9.1]。
- strict3073_validation_long_h10_budget_warm 相对 strict3073_validation_long_base，all，n=100：+15.0 个百分点，病例配对 bootstrap 95% 区间 [+7.0, +24.0]。
- strict3073_validation_long_h10_budget_warm 相对 strict3073_validation_long_base，across_wall，n=67：+9.0 个百分点，病例配对 bootstrap 95% 区间 [-1.5, +19.4]。
- strict3073_validation_long_h10_budget_warm 相对 strict3073_validation_long_base，same_side，n=33：+27.3 个百分点，病例配对 bootstrap 95% 区间 [+12.1, +42.4]。
- strict3074_validation_long_base_retrieval_only 相对 strict3074_validation_long_base，all，n=100：+39.0 个百分点，病例配对 bootstrap 95% 区间 [+30.0, +49.0]。
- strict3074_validation_long_base_retrieval_only 相对 strict3074_validation_long_base，across_wall，n=67：+40.3 个百分点，病例配对 bootstrap 95% 区间 [+28.4, +52.2]。
- strict3074_validation_long_base_retrieval_only 相对 strict3074_validation_long_base，same_side，n=33：+36.4 个百分点，病例配对 bootstrap 95% 区间 [+21.2, +54.5]。
- strict3074_validation_long_base_retrieval_rerank 相对 strict3074_validation_long_base，all，n=100：+37.0 个百分点，病例配对 bootstrap 95% 区间 [+28.0, +47.0]。
- strict3074_validation_long_base_retrieval_rerank 相对 strict3074_validation_long_base，across_wall，n=67：+37.3 个百分点，病例配对 bootstrap 95% 区间 [+25.4, +49.3]。
- strict3074_validation_long_base_retrieval_rerank 相对 strict3074_validation_long_base，same_side，n=33：+36.4 个百分点，病例配对 bootstrap 95% 区间 [+21.2, +51.5]。
- strict3074_validation_long_base_shuffled_hybrid 相对 strict3074_validation_long_base，all，n=100：-2.0 个百分点，病例配对 bootstrap 95% 区间 [-11.0, +7.0]。
- strict3074_validation_long_base_shuffled_hybrid 相对 strict3074_validation_long_base，across_wall，n=67：-6.0 个百分点，病例配对 bootstrap 95% 区间 [-16.4, +4.5]。
- strict3074_validation_long_base_shuffled_hybrid 相对 strict3074_validation_long_base，same_side，n=33：+6.1 个百分点，病例配对 bootstrap 95% 区间 [-9.1, +24.2]。
- strict3074_validation_long_base_shuffled_replace 相对 strict3074_validation_long_base，all，n=100：-56.0 个百分点，病例配对 bootstrap 95% 区间 [-66.0, -46.0]。
- strict3074_validation_long_base_shuffled_replace 相对 strict3074_validation_long_base，across_wall，n=67：-56.7 个百分点，病例配对 bootstrap 95% 区间 [-68.7, -44.8]。
- strict3074_validation_long_base_shuffled_replace 相对 strict3074_validation_long_base，same_side，n=33：-54.5 个百分点，病例配对 bootstrap 95% 区间 [-69.7, -36.4]。
- strict3074_validation_long_base_temporal_hybrid 相对 strict3074_validation_long_base，all，n=100：+40.0 个百分点，病例配对 bootstrap 95% 区间 [+30.0, +50.0]。
- strict3074_validation_long_base_temporal_hybrid 相对 strict3074_validation_long_base，across_wall，n=67：+41.8 个百分点，病例配对 bootstrap 95% 区间 [+29.9, +53.7]。
- strict3074_validation_long_base_temporal_hybrid 相对 strict3074_validation_long_base，same_side，n=33：+36.4 个百分点，病例配对 bootstrap 95% 区间 [+21.2, +54.5]。
- strict3074_validation_long_base_temporal_replace 相对 strict3074_validation_long_base，all，n=100：+40.0 个百分点，病例配对 bootstrap 95% 区间 [+31.0, +49.0]。
- strict3074_validation_long_base_temporal_replace 相对 strict3074_validation_long_base，across_wall，n=67：+41.8 个百分点，病例配对 bootstrap 95% 区间 [+29.9, +53.7]。
- strict3074_validation_long_base_temporal_replace 相对 strict3074_validation_long_base，same_side，n=33：+36.4 个百分点，病例配对 bootstrap 95% 区间 [+21.2, +54.5]。
- strict3074_validation_long_feedback_extra 相对 strict3074_validation_long_base，all，n=100：-14.0 个百分点，病例配对 bootstrap 95% 区间 [-23.0, -6.0]。
- strict3074_validation_long_feedback_extra 相对 strict3074_validation_long_base，across_wall，n=67：-17.9 个百分点，病例配对 bootstrap 95% 区间 [-29.9, -7.5]。
- strict3074_validation_long_feedback_extra 相对 strict3074_validation_long_base，same_side，n=33：-6.1 个百分点，病例配对 bootstrap 95% 区间 [-18.2, +6.1]。
- strict3074_validation_long_feedback_extra_warm 相对 strict3074_validation_long_base，all，n=100：+4.0 个百分点，病例配对 bootstrap 95% 区间 [-3.0, +11.0]。
- strict3074_validation_long_feedback_extra_warm 相对 strict3074_validation_long_base，across_wall，n=67：+7.5 个百分点，病例配对 bootstrap 95% 区间 [-1.5, +16.4]。
- strict3074_validation_long_feedback_extra_warm 相对 strict3074_validation_long_base，same_side，n=33：-3.0 个百分点，病例配对 bootstrap 95% 区间 [-12.1, +6.1]。
- strict3074_validation_long_h10_budget 相对 strict3074_validation_long_base，all，n=100：-30.0 个百分点，病例配对 bootstrap 95% 区间 [-42.0, -18.0]。
- strict3074_validation_long_h10_budget 相对 strict3074_validation_long_base，across_wall，n=67：-35.8 个百分点，病例配对 bootstrap 95% 区间 [-50.7, -20.9]。
- strict3074_validation_long_h10_budget 相对 strict3074_validation_long_base，same_side，n=33：-18.2 个百分点，病例配对 bootstrap 95% 区间 [-36.4, +0.0]。
- strict3074_validation_long_h10_budget_warm 相对 strict3074_validation_long_base，all，n=100：+8.0 个百分点，病例配对 bootstrap 95% 区间 [-2.0, +18.0]。
- strict3074_validation_long_h10_budget_warm 相对 strict3074_validation_long_base，across_wall，n=67：+0.0 个百分点，病例配对 bootstrap 95% 区间 [-13.4, +13.4]。
- strict3074_validation_long_h10_budget_warm 相对 strict3074_validation_long_base，same_side，n=33：+24.2 个百分点，病例配对 bootstrap 95% 区间 [+12.1, +39.4]。

## 几何布局变化

- strict3072_map_train_a1_d49：100/100，SR 30.0%；完整组。
- strict3072_map_train_a1_d81：100/100，SR 18.0%；完整组。
- strict3072_map_train_a1_d113：100/100，SR 10.0%；完整组。
- strict3072_map_train_a1_d145：100/100，SR 15.0%；完整组。
- strict3072_map_train_a0_d49：100/100，SR 11.0%；完整组。
- strict3072_map_train_a0_d81：100/100，SR 10.0%；完整组。
- strict3072_map_train_a0_d113：100/100，SR 9.0%；完整组。
- strict3072_map_train_a0_d145：100/100，SR 9.0%；完整组。
- strict3072_map_validation_a1_d65：100/100，SR 24.0%；完整组。
- strict3072_map_validation_a1_d129：100/100，SR 9.0%；完整组。
- strict3072_map_validation_a0_d65：100/100，SR 8.0%；完整组。
- strict3072_map_validation_a0_d129：100/100，SR 10.0%；完整组。
- strict3072_map_test_a1_d97：100/100，SR 10.0%；完整组。
- strict3072_map_test_a1_d161：100/100，SR 15.0%；完整组。
- strict3072_map_test_a1_d177：100/100，SR 15.0%；完整组。
- strict3072_map_test_a0_d97：100/100，SR 11.0%；完整组。
- strict3072_map_test_a0_d161：100/100，SR 12.0%；完整组。
- strict3072_map_test_a0_d177：100/100，SR 11.0%；完整组。
- strict3072_map_validation_a1_d65_temporal_replace：100/100，SR 88.0%；完整组。
- strict3072_map_validation_a1_d129_temporal_replace：100/100，SR 54.0%；完整组。
- strict3072_map_validation_a0_d65_temporal_replace：100/100，SR 7.0%；完整组。
- strict3072_map_validation_a0_d129_temporal_replace：100/100，SR 13.0%；完整组。
- geometry_single4001_map_train_a1_d49：100/100，SR 30.0%；完整组。
- geometry_single4001_map_validation_a1_d65：100/100，SR 23.0%；完整组。
- geometry_single4001_map_validation_a1_d129：100/100，SR 13.0%；完整组。
- geometry_single4001_map_validation_a0_d65：100/100，SR 11.0%；完整组。
- geometry_single4001_map_validation_a0_d129：100/100，SR 14.0%；完整组。
- geometry_multi4001_map_train_a1_d49：100/100，SR 9.0%；完整组。
- geometry_multi4001_map_validation_a1_d65：100/100，SR 6.0%；完整组。
- geometry_multi4001_map_validation_a1_d129：100/100，SR 9.0%；完整组。
- geometry_multi4001_map_validation_a0_d65：100/100，SR 11.0%；完整组。
- geometry_multi4001_map_validation_a0_d129：100/100，SR 11.0%；完整组。
- strict3072_map_validation_a1_d65_retrieval_only：100/100，SR 83.0%；完整组。
- strict3072_map_validation_a1_d65_retrieval_rerank：100/100，SR 81.0%；完整组。
- strict3072_map_validation_a1_d65_retrieval_init：100/100，SR 70.0%；完整组。
- strict3072_map_validation_a1_d129_retrieval_only：100/100，SR 46.0%；完整组。
- strict3072_map_validation_a1_d129_retrieval_rerank：100/100，SR 48.0%；完整组。
- strict3072_map_validation_a1_d129_retrieval_init：100/100，SR 48.0%；完整组。
- strict3072_map_validation_a0_d65_retrieval_only：100/100，SR 15.0%；完整组。
- strict3072_map_validation_a0_d65_retrieval_rerank：100/100，SR 11.0%；完整组。
- strict3072_map_validation_a0_d65_retrieval_init：100/100，SR 14.0%；完整组。
- strict3072_map_validation_a0_d129_retrieval_only：100/100，SR 12.0%；完整组。
- strict3072_map_validation_a0_d129_retrieval_rerank：100/100，SR 14.0%；完整组。
- strict3072_map_validation_a0_d129_retrieval_init：100/100，SR 14.0%；完整组。

## 解释边界与后续

以上区间仅衡量固定训练模型下的病例不确定性，未包含训练种子不确定性。不同地图复用起终点，不能把所有地图病例视为独立抽样。这里的门位置与朝向变化不构成新连通拓扑；单地图训练的新布局下降也不能单独证明模型架构无法泛化。

剩余工作以报告末尾带日期的阶段审阅和队列为准，勿将历史计划当作当前未完成项；部分观测是否加入由证据决定。整轮研究尚未完成。前沿方法及旧结果参见 frontier_transfer_review.md 和根目录 reports/navigation_capability_merged.md。
