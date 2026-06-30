# 014_07 HLA2010 baseline-relative reward DQN 记录

## 目的

测试把奖励从绝对产量改成“相对 null 基线的增产”以后，HLA2010 是否还会退化为 no-op。

## 奖励函数

```text
reward = (Yield_policy - Yield_null_baseline) - 1.0 * irrigation - 5.0 * nitrogen
```

## 结果

| timesteps | seed | null_baseline_yield | final_grnwt | action_irrigation_total | action_nitrogen_total | max_water_stress | max_nitrogen_stress | eval_reward_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200.000 | 0.000 | 6956.000 | 6970.795 | 0.000 | 200.000 | 0.919 | 0.016 | -985.205 |
| 5000.000 | 0.000 | 6956.000 | 6956.454 | 0.000 | 0.000 | 0.919 | 0.157 | 0.454 |

## 文件

- 汇总：`DSSAT_auto_validation/HLA_2004/hla2010_baseline_relative_reward_dqn_probe_014_07/014_07_hla2010_baseline_relative_reward_summary.csv`
- 输出目录：`DSSAT_auto_validation/HLA_2004/hla2010_baseline_relative_reward_dqn_probe_014_07`
