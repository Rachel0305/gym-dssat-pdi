# 017_06 FQ2019 过程图与 seed1 稳定性验证记录

## 已完成：FQ2019 seed0 checkpoint10000 四情景过程图

| year | scenario | final_grain_kg_ha | final_biomass_kg_ha | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | total_reward_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | null_zero | 7049.941 | 13565.004 | 0.000 | 0.000 | 0.707 | 0.012 | 0.000 |
| 2019 | recorded_shifted | 8040.552 | 14539.677 | 75.000 | 144.000 | 0.555 | 0.012 | 195.552 |
| 2019 | dssat_auto | 8458.223 | 14959.358 | 114.500 | 0.000 | 0.000 | 0.012 | 1293.723 |
| 2019 | fq2019_dqn_seed0_ckpt10000 | 8848.849 | 15300.906 | 120.000 | 300.000 | 0.000 | 0.012 | 178.849 |

## 输出

- 日值表：`DSSAT_auto_validation/fq2019_process_plot_and_seed1_stability_017_06/fq2019_seed0_ckpt10000_four_scenario_daily.csv`
- 事件表：`DSSAT_auto_validation/fq2019_process_plot_and_seed1_stability_017_06/fq2019_seed0_ckpt10000_four_scenario_events.csv`
- summary：`DSSAT_auto_validation/fq2019_process_plot_and_seed1_stability_017_06/fq2019_seed0_ckpt10000_four_scenario_summary.csv`
- 图：`DSSAT_auto_validation/fq2019_process_plot_and_seed1_stability_017_06/figures/fq2019_seed0_ckpt10000_four_scenario_process.png`

## 待补充

- FQ2019 seed1 50K checkpoint 稳定性结果。

## ????FQ2019 seed1 50K ?????

seed1 ???? seed0 checkpoint10000 ??????seed1 ?? checkpoint ?? no-op?I=0?N=0?GWAD=7050?reward=0?

### seed0 / seed1 checkpoint ??

| seed | checkpoint_step | final_grain_kg_ha | action_irrigation_total | action_fertilizer_total | total_reward | is_best_reward |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 5000 | 7211.000 | 0.000 | 300.000 | -1339.190 | False |
| 0 | 10000 | 8849.000 | 120.000 | 300.000 | 178.849 | True |
| 0 | 15000 | 7216.000 | 0.000 | 300.000 | -1333.972 | False |
| 0 | 20000 | 7608.000 | 60.000 | 300.000 | -1002.206 | False |
| 0 | 25000 | 7759.000 | 90.000 | 300.000 | -881.318 | False |
| 0 | 30000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 0 | 35000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 0 | 40000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 0 | 45000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 0 | 50000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 5000 | 7050.000 | 0.000 | 0.000 | 0.000 | True |
| 1 | 10000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 15000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 20000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 25000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 30000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 35000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 40000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 45000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |
| 1 | 50000 | 7050.000 | 0.000 | 0.000 | 0.000 | False |

### seed0 checkpoint10000 ?????

- ???DAP 47 / 54 / 61?? 100 kg/ha?? N=300 kg/ha?
- ???DAP 61 / 68 / 75 / 82?? 30 mm?? I=120 mm?
- ?????????????????????????DSSAT auto ??????????? auto ?????????

## ????

1. FQ2019 seed0 checkpoint10000 ?? seed ????????????????
2. seed0 ????????????? DAP 61?82??? 017_04 ? mid-season irrigation ???????
3. seed0 ????????N300 ????? 017_04 ??? N0 ????????????/????????????????
4. seed1 ?? no-op????? DQN ??? seed ?????

## ????

- seed ????`DSSAT_auto_validation/fq2019_process_plot_and_seed1_stability_017_06/fq2019_seed0_seed1_checkpoint_comparison.csv`
- seed1 ?????`DSSAT_auto_validation/fq2019_baseline_relative_dqn_train_transfer_017_05/seed1_50000steps`
