# 014_03 HLA2010 DQN 统一流程复核记录

## 目的

用当前 YC/FQ linked DQN 方法复核 HLA2010，先 smoke test，确认动作链路、奖励、日志和输出正常。

## 方法

- 输入年份：HLA2010
- 初始条件：IC=1，沿用 HLA 新品种参数输入。
- linked 管理：脚本通过 `prepare_case_at` 插入 Jinja 占位符，并切换为 PDI/gym-DSSAT 可接收动作的管理方式。
- DQN 动作：0 不操作；1 灌溉30mm；2 施氮100kg/ha；3 灌溉30mm+施氮100kg/ha。
- 预算：I120/N300，单次 I30/N100，最小操作间隔7天。
- 奖励：`delta_grnwt - 1.0*irrigation - 5.0*nitrogen`。

## 当前结果

| window | seed | timesteps | daily_final_grnwt | daily_final_topwt | action_irrigation_total | action_nitrogen_total | max_water_stress | max_nitrogen_stress | irrigation_total_mgmtevent | fertilizer_total_mgmtevent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agronomic_window | 0.000 | 200.000 | 7853.665 | 20771.658 | 120.000 | 300.000 | 0.414 | 0.016 | 120.000 | 300.000 |
| free_daily | 0.000 | 200.000 | 7853.665 | 20732.576 | 120.000 | 200.000 | 0.414 | 0.016 | 120.000 | 200.000 |
| free_daily | 0.000 | 5000.000 | 6956.454 | 19344.495 | 0.000 | 0.000 | 0.919 | 0.157 | 0.000 | 0.000 |

## 文件

- 汇总：`DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03/014_03_hla2010_dqn_unified_recheck_summary.csv`
- 输出目录：`DSSAT_auto_validation/HLA_2004/hla2010_dqn_unified_recheck_014_03`
