# 019_05 FQ2016 leaching-aware reward smoke test 记录

## 结论先行

本轮只做低步数 smoke test，不作为策略优劣结论。目的只是确认 full-state `cleach` 能进入日值表和 reward component。

结论：链路通过。日值表中 `cleach`、`delta_cleach`、`leaching_cost_term` 均已写出；`final_cleach` 与 `SoilNi.OUT` 的 `NLCC` 基本一致。当前可以进入不同 `leaching_cost` 系数的小范围敏感性测试，但还不应该直接长训练或写成策略优劣结论。

## 奖励函数

```text
reward = max(0, final_grnwt - local_null_yield) - water_cost * irrigation - nitrogen_cost * nitrogen - leaching_cost * delta_cleach
```

## 结果

| checkpoint_step | leaching_cost | final_grain_kg_ha | action_irrigation_total | action_fertilizer_total | final_cleach | soilni_final_NLCC | sum_delta_cleach | sum_leaching_cost_term | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000.000 | 20.000 | 7751.000 | 30.000 | 300.000 | 3.193 | 3.190 | 3.193 | 63.866 | -908.981 |
| 2000.000 | 20.000 | 8012.000 | 90.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | -643.578 |
| 3000.000 | 20.000 | 8012.000 | 120.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | -673.578 |
| 4000.000 | 20.000 | 7106.000 | 0.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1459.553 |
| 5000.000 | 20.000 | 7106.000 | 0.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1459.808 |

## 文件

- 运行目录：`DSSAT_auto_validation/fq2016_leaching_aware_reward_smoke_019_05/seed0_5000steps_lc20`
- 日值表：`DSSAT_auto_validation/fq2016_leaching_aware_reward_smoke_019_05/seed0_5000steps_lc20/leaching_aware_eval_daily.csv`
- 汇总表：`DSSAT_auto_validation/fq2016_leaching_aware_reward_smoke_019_05/seed0_5000steps_lc20/leaching_aware_checkpoint_summary.csv`

## 判断

本轮链路通过。下一步建议只做小范围 `leaching_cost` 敏感性，例如 0、20、50、100 的 500-step smoke 或离线回放评分；确认惩罚强度方向后，再考虑 5K 训练。
