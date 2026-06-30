# 014_10 HLA2015 成功策略复现记录

## 目的

把 HLA2010 的成功策略迁移到 HLA2015，验证能否稳定优于 null 并输出有意义的水氮操作。

## 设置

- 9 动作：灌溉 0/15/30 mm，施氮 0/50/100 kg/ha
- baseline-relative reward
- 输入包：`DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/2015/literature_aligned_seed0_5000steps/input`

## 结果

| timesteps | seed | null_baseline_yield | final_grnwt | action_irrigation_total | action_nitrogen_total | max_water_stress | max_nitrogen_stress | eval_reward_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200.000 | 0.000 | 6486.000 | 7635.344 | 120.000 | 300.000 | 0.000 | 0.015 | -470.656 |
| 5000.000 | 0.000 | 6486.000 | 7626.842 | 120.000 | 300.000 | 0.000 | 0.015 | -479.158 |
| 5000.000 | 1.000 | 6486.000 | 7651.901 | 120.000 | 300.000 | 0.000 | 0.015 | -454.099 |

## 文件

- 汇总：`DSSAT_auto_validation/HLA_2004/hla2015_success_strategy_validation_014_10/014_10_hla2015_baseline_relative_reward_summary.csv`
- 输出目录：`DSSAT_auto_validation/HLA_2004/hla2015_success_strategy_validation_014_10`
