# 014_04 HLA2010 DQN no-op 退化 Q 值诊断记录

## 目的

014_03 中 200 step smoke 有操作，但 5K 确定性评估退化为 no-op。本轮不训练，只读取两个模型在同一环境轨迹上的 Q 值。

## 结果摘要

| case | steps | final_grnwt | irrigation_total | nitrogen_total | max_swfac | max_nstres | action_counts | mean_q_noop | mean_q_irrig | mean_q_fert | mean_q_both | mean_q_gap_best_second |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| free_daily_200steps | 131.000 | 7853.665 | 120.000 | 200.000 | 0.414 | 0.016 | {"0": 71, "1": 52, "2": 7, "3": 1} | 137.201 | 105.478 | -7.165 | 59.334 | 36.037 |
| free_daily_5000steps | 131.000 | 6956.454 | 0.000 | 0.000 | 0.919 | 0.157 | {"0": 131} | 149.343 | 138.426 | 139.249 | 138.247 | 7.364 |

## 初步判读

- free_daily_200steps: 动作计数 {"0": 71, "1": 52, "2": 7, "3": 1}，I=120.0, N=200.0, final GRNWT=7853.7。
- free_daily_5000steps: 动作计数 {"0": 131}，I=0.0, N=0.0, final GRNWT=6956.5。

如果 5K 模型 action_counts 几乎全是 0，且 mean_q_noop 高于其他动作，说明不是动作链路问题，而是 DQN 学到 no-op 价值最高。
如果 Q 值差距很小，则说明模型没有学清楚，可能需要改奖励尺度、探索策略或动作/时间窗口，而不是继续直接加 seed。

## 输出文件

- 日值 Q 表：`DSSAT_auto_validation/HLA_2004/hla2010_dqn_noop_q_diagnosis_014_04/014_04_hla2010_dqn_q_values_daily.csv`
- 汇总表：`DSSAT_auto_validation/HLA_2004/hla2010_dqn_noop_q_diagnosis_014_04/014_04_hla2010_dqn_q_summary.csv`
