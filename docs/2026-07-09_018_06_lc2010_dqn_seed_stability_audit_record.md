# 018_06 LC2010 DQN 跨 seed 稳定性复核记录

## 做了什么

本轮没有训练、没有重跑 DSSAT。只读取 LC2010 现有 seed0/seed1 checkpoint summary，并与 LC2010 DSSAT auto 和 official extension expert 进行对照。

## best checkpoint 汇总

| seed | selection | checkpoint_step | final_grain_kg_ha | final_biomass_kg_ha | irrigation_mm | nitrogen_kg_ha | max_water_stress | max_nitrogen_stress | total_reward | matches_auto_yield | matches_extension_yield | saves_water_vs_extension | saves_nitrogen_vs_extension | not_more_n_than_extension |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | best_reward | 5000 | 8739.000 | 16377.000 | 90.000 | 0.000 | 0.000 | 0.019 | 598.149 | True | True | True | True | True |
| 0 | best_yield_then_resource | 5000 | 8739.000 | 16377.000 | 90.000 | 0.000 | 0.000 | 0.019 | 598.149 | True | True | True | True | True |
| 1 | best_reward | 5000 | 8739.000 | 16374.000 | 120.000 | 300.000 | 0.000 | 0.019 | -932.467 | True | True | True | False | False |
| 1 | best_yield_then_resource | 5000 | 8739.000 | 16374.000 | 120.000 | 300.000 | 0.000 | 0.019 | -932.467 | True | True | True | False | False |

## 判定

- stable_yield = True
- stable_water_saving_vs_extension = True
- stable_not_more_n_than_extension = False
- stable_success = False

LC2010 产量较稳定，但资源效率跨 seed 不稳定；目前应作为 promising candidate，而不是稳定成功案例。

## 解释

seed0 的 best-reward checkpoint 非常理想：产量追平/略高于 auto 和官方推广 expert，同时灌溉、施氮都更少。seed1 的 best-reward checkpoint 产量也追平关键基线，但用水和用氮明显高于 seed0，说明 LC2010 当前更像“产量稳定、资源效率不稳定”。

## 下一步建议

如果要把 LC2010 写成稳定成功案例，需要继续做一个小规模 seed 稳定性补充：要么增加 seed2，要么延长/复核 seed1 的 checkpoint 选择。但在当前阶段，它已经足以作为 DQN 有潜力的代表案例之一。

## 输出文件

- checkpoint table: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_06_lc2010_seed_stability_audit\018_06_lc2010_seed_checkpoint_table.csv`
- best summary: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_06_lc2010_seed_stability_audit\018_06_lc2010_seed_best_summary.csv`
- figure: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_06_lc2010_seed_stability_audit\figures\018_06_lc2010_seed_stability.png`
