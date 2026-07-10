# 018_07 SY2014 DQN 跨 seed 稳定性复核记录

## 结论

SY2014 目前只有 seed0 的明确 DQN 结果证据，没有发现 seed1 的训练输出或 checkpoint 评估结果。因此当前不能判断 SY2014 是否跨 seed 稳定成功。

## 已有证据盘点

| artifact | path | exists |
| --- | --- | --- |
| seed0_training_run | DSSAT_auto_validation\sy_local_dqn_train_cross_year_transfer_017_08\train_runs\2014\seed0\dqn_train | True |
| seed0_checkpoint_summary | DSSAT_auto_validation\sy_local_dqn_train_cross_year_transfer_017_08\017_08_sy_dqn_checkpoint_summary.csv | True |
| seed0_four_scenario_summary | DSSAT_auto_validation\sy2014_dqn_resource_space_017_09\017_09_sy2014_four_scenario_summary.csv | True |
| seed1_training_run | DSSAT_auto_validation\sy_local_dqn_train_cross_year_transfer_017_08\train_runs\2014\seed1\dqn_train | False |
| seed1_checkpoint_summary | DSSAT_auto_validation\sy_local_dqn_train_cross_year_transfer_017_08\017_08_sy_dqn_checkpoint_summary_seed1.csv | False |
| seed1_eval_run | DSSAT_auto_validation\sy_local_dqn_train_cross_year_transfer_017_08\eval_runs\2014\seed1 | False |

## 当前可用 seed 结果

| seed | source | final_gwad | final_cwad | irrigation_mm | nitrogen_kg_ha | max_water_stress | max_nitrogen_stress | yield_diff_vs_extension | irrigation_diff_vs_extension | nitrogen_diff_vs_extension | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | DSSAT_auto_validation\sy2014_dqn_resource_space_017_09\017_09_sy2014_four_scenario_summary.csv | 11216.000 | 20250.000 | 120.000 | 300.000 | 0.000 | 0.012 | 139.000 | -146.100 | 0.000 | available |
| 1 |  |  |  |  |  |  |  |  |  |  | missing_seed_result |

## 当前判断

1. `seed0` 证据很强：DQN 产量约 11216 kg/ha，高于官方推广 expert 11077 kg/ha；灌溉约 120 mm，低于官方推广 expert 266.1 mm；施氮 300 kg/ha，与官方推广 expert 基本相同。
2. `seed1` 证据缺失：现在不是 seed1 表现不好，而是我们根本还没有对应结果。
3. 因此，SY2014 当前应标记为 `promising but missing seed1 evidence`，不能直接升级为稳定成功案例。

## 如果要最小补充实验

最小补充应是：在 SY2014 现有框架下补跑一个 `seed1`，不必先加长训练，也不必先改奖励函数。先做与 seed0 同长度、同设置的最小复现实验即可。

## 输出文件

- inventory: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_07_sy2014_seed_stability_audit\018_07_sy2014_seed_inventory.csv`
- summary: `DSSAT_auto_validation\extension_expert_baseline_018_03\018_07_sy2014_seed_stability_audit\018_07_sy2014_seed_stability_summary.csv`
