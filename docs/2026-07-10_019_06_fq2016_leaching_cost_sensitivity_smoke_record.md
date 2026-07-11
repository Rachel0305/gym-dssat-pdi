# 019_06 FQ2016 leaching-cost sensitivity smoke 记录

## 结论先行

本轮比较 `leaching_cost=0/20/50/100` 的 500-step smoke 结果。注意：500 steps 只能用于检查奖励项方向和链路稳定性，不能作为最终策略优劣结论。

500-step smoke 下，`leaching_cost=20` 是当前最值得进入 5K 短训练的候选；它在本轮中达到最高 `GWAD=8012 kg/ha`，灌溉总量为 `90 mm`，施氮总量仍为 `300 kg/ha`，且 `final_cleach/NLCC=0`。但这不是最终策略结论，因为 500 steps 训练量太小，且所有系数下施氮总量仍为 300 kg/ha。

`leaching_cost=50/100` 在本轮明显把灌溉压到 `30 mm`，并导致产量下降到约 `7700 kg/ha`，不建议直接放大做长训练。淋洗惩罚目前更像是在影响灌溉/时机，而不是直接降低总施氮。

## 结果

| leaching_cost | final_grain_kg_ha | action_irrigation_total | action_fertilizer_total | final_cleach | soilni_final_NLCC | sum_delta_cleach | sum_leaching_cost_term | total_reward | nonzero_irrigation_events | nonzero_fertilizer_events | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 7970.000 | 60.000 | 300.000 | 6.280 | 6.280 | 6.280 | 0.000 | -656.462 | 2.000 | 3.000 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_500steps_lc0 |
| 20.000 | 8012.000 | 90.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | -643.578 | 3.000 | 3.000 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_500steps_lc20 |
| 50.000 | 7703.000 | 30.000 | 300.000 | 3.109 | 3.110 | 3.109 | 155.455 | -1048.541 | 1.000 | 3.000 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_500steps_lc50_20260710_022225 |
| 100.000 | 7701.000 | 30.000 | 300.000 | 3.109 | 3.110 | 3.109 | 310.910 | -1205.575 | 1.000 | 3.000 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_500steps_lc100 |

## 文件

- 汇总表：`DSSAT_auto_validation\fq2016_leaching_cost_sensitivity_019_06\019_06_fq2016_leaching_cost_sensitivity_summary.csv`
- 来源目录：`DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05`

## 下一步建议

1. 先用 `leaching_cost=20` 做 FQ2016 5K seed0 短训练。
2. 保留 `leaching_cost=0` 作为无淋洗惩罚对照，不要只看单一结果。
3. 暂不建议直接上 `leaching_cost=50/100` 的长训练，因为短训练已经显示高系数可能压低灌溉和产量。
4. 如果 5K seed0 能维持高产并降低 `cleach/NLCC`，再做 seed1 稳定性复核。
