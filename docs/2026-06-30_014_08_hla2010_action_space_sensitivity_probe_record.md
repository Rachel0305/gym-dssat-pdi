# 014_08 HLA2010 action-space sensitivity probe 记录

## 目的

在同一个 baseline-relative reward 下，只比较动作粒度：4 动作 vs 9 动作。

## 结果

| action_name | timesteps | seed | final_grnwt | action_irrigation_total | action_nitrogen_total | max_water_stress | max_nitrogen_stress | eval_reward_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action4 | 200.000 | 0.000 | 6970.795 | 0.000 | 200.000 | 0.919 | 0.016 | -985.205 |
| action9 | 200.000 | 0.000 | 7853.665 | 120.000 | 300.000 | 0.436 | 0.016 | -722.335 |

## 文件

- 汇总：`DSSAT_auto_validation/HLA_2004/hla2010_action_space_sensitivity_probe_014_08/014_08_hla2010_action_space_summary.csv`
- 输出目录：`DSSAT_auto_validation/HLA_2004/hla2010_action_space_sensitivity_probe_014_08`
