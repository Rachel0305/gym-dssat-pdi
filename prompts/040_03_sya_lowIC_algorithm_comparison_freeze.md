# 040_03 SYA lowIC 自由时序算法对照冻结

## 目的

在不重新训练、不重新评估 DSSAT 的前提下，统一整理 040_00、040_01、040_02 三套已经完成的 SYA lowIC 自由时序结果：

- 040_00：MaskablePPO
- 040_01：SB3 DQN + masked-greedy evaluation
- 040_02：StrictMaskableDQN

本任务只回答一个问题：

> 在相同输入、相同年份划分、相同动作空间、相同安全约束、相同奖励函数下，哪一种算法当前表现更好？

## 数据来源

仅读取以下已有结果：

- `benchmark_results/040_00_sya_lowIC_free_timing_maskableppo/evaluation/040_00_validation_summary_by_station_checkpoint.csv`
- `benchmark_results/040_01_sya_lowIC_free_timing_dqn/evaluation/040_01_validation_summary_by_checkpoint.csv`
- `benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/evaluation/040_02_validation_summary_by_checkpoint.csv`
- 各自对应的 checkpoint-year 明细表。

## 固定比较规则

1. 每个算法只从其预先保存的 checkpoint 中选择验证年均产量最高的 checkpoint。
2. 不挑单个年份、单个事件或单个指标作为算法胜负依据。
3. 不重算 reward，不重跑 DSSAT，不改变 checkpoint。
4. 同时报告资源使用、PFP_N、水分胁迫天数、氮胁迫天数，防止只看产量。
5. 如果某算法后期 checkpoint 相比最佳 checkpoint 明显退化，需要在记录中标注。

## 输出

- `benchmark_results/040_03_sya_lowIC_algorithm_comparison_freeze/tables/040_03_algorithm_checkpoint_summary.csv`
- `benchmark_results/040_03_sya_lowIC_algorithm_comparison_freeze/tables/040_03_algorithm_best_checkpoint_summary.csv`
- `benchmark_results/040_03_sya_lowIC_algorithm_comparison_freeze/tables/040_03_algorithm_best_year_detail.csv`
- `docs/040_03_sya_lowIC_algorithm_comparison_freeze_record.md`

## 预期结论边界

本任务可以说明：

- 在当前 SYA lowIC 自由时序框架下，PPO、SB3-DQN、StrictMaskableDQN 的相对表现。

本任务不能说明：

- 所有站点上 PPO 永远优于 DQN；
- DQN 理论上不可行；
- 当前 PPO 已经达到最终可发表主结果。

