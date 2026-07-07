# 017_04 FQ2019 确定性上界扫描记录

## 目的

FQ2016 seed1 checkpoint30000 迁移到 FQ2019 后只达到 7759 kg/ha，低于 recorded 和 DSSAT auto。本实验不训练模型，只用人工调度组合检查当前约束下是否存在更高产的确定性方案。

## 参考值

- null: 7050.0 kg/ha
- recorded: 8041.0 kg/ha
- auto: 8458.0 kg/ha
- dqn_transfer: 7759.0 kg/ha

## 当前 I≤120/N≤300 约束内前 5 名

| case | final_gwad | final_cwad | planned_irrigation_total | planned_fertilizer_total | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- |
| I90_mid_N0 | 8829.000 | 15186.000 | 90.000 | 0.000 | 0.000 | 0.048 |
| I120_even_N0 | 8829.000 | 15186.000 | 120.000 | 0.000 | 0.000 | 0.054 |
| I90_mid_N300_split | 8827.000 | 15179.000 | 90.000 | 300.000 | 0.000 | 0.012 |
| I60_early_N100_early | 8827.000 | 15179.000 | 60.000 | 100.000 | 0.000 | 0.012 |
| I60_early_N200_split | 8827.000 | 15179.000 | 60.000 | 200.000 | 0.000 | 0.012 |

## 全部扫描前 8 名（含 I180 参考）

| case | within_current_budget | final_gwad | final_cwad | planned_irrigation_total | planned_fertilizer_total | max_swfac | max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I180_even_ref_N300_split | False | 8838.000 | 15135.000 | 180.000 | 300.000 | 0.000 | 0.012 |
| I180_even_ref_N200_split | False | 8838.000 | 15135.000 | 180.000 | 200.000 | 0.000 | 0.012 |
| I180_even_ref_N0 | False | 8838.000 | 15132.000 | 180.000 | 0.000 | 0.000 | 0.020 |
| I90_mid_N0 | True | 8829.000 | 15186.000 | 90.000 | 0.000 | 0.000 | 0.048 |
| I120_even_N0 | True | 8829.000 | 15186.000 | 120.000 | 0.000 | 0.000 | 0.054 |
| I120_even_N300_split | True | 8827.000 | 15179.000 | 120.000 | 300.000 | 0.000 | 0.012 |
| I120_even_N200_split | True | 8827.000 | 15179.000 | 120.000 | 200.000 | 0.000 | 0.012 |
| I120_even_N100_early | True | 8827.000 | 15179.000 | 120.000 | 100.000 | 0.000 | 0.012 |

## 初步判读

- 如果约束内最优仍明显低于 DSSAT auto，说明当前 DQN 预算可能限制了追上 auto 的空间。
- 如果约束内最优高于 DQN transfer 很多，说明 FQ2019 并非不能优化，而是 FQ2016 模型迁移没有学到合适时机。
- I180 仅作为水预算诊断参考，不作为当前 DQN 可行策略。

## 输出

- summary: `DSSAT_auto_validation/fq2019_deterministic_upper_bound_scan_017_04/fq2019_deterministic_upper_bound_scan_summary.csv`
- daily: `DSSAT_auto_validation/fq2019_deterministic_upper_bound_scan_017_04/fq2019_deterministic_upper_bound_scan_daily.csv`
- events: `DSSAT_auto_validation/fq2019_deterministic_upper_bound_scan_017_04/fq2019_deterministic_upper_bound_scan_events.csv`
- figures: `DSSAT_auto_validation/fq2019_deterministic_upper_bound_scan_017_04/figures`
