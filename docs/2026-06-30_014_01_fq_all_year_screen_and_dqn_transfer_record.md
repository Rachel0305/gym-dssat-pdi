# 014_01 封丘站全年份筛选与 DQN 方法迁移记录

## 执行目的

筛选封丘站 2000–2023 年中具有水氮优化空间的年份，并将禹城站 linked DQN 离散动作方法迁移到封丘站。

## 输入与口径

- 输入目录：`/workspaces/gym-dssat-pdi/DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ`
- 基础 MZX：`CNFQ0801.MZX`
- 专家迁移模板：FQ2008 treatment 2
- 专家迁移方式：将 FQ2008 管理日期和天气站代码平移到目标年份。
- DQN：I120/N300，单次 I30/N100，最小操作间隔 7 天，5K timesteps，seed0。

## 输出文件

- 筛选汇总：`/workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_all_year_screening_summary.csv`
- 候选年份排序：`/workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_selected_years.csv`
- DQN 汇总：`/workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_dqn_summary.csv`
- 四情景日值：`/workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_selected_four_scenario_daily.csv`
- 四情景事件：`/workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/014_01_fq_selected_four_scenario_events.csv`
- 图目录：`/workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq_all_year_screen_and_dqn_transfer_014_01/figures`

## 筛选结果预览

```text
 year  dssat_auto   null  recorded_shifted  max_water_stress  max_nitrogen_stress  final_dap  best_reference  management_gain  stress_space  usable_for_training_display
 2019      8458.0 7050.0            8041.0          0.706692             0.012191      108.0          8458.0           1408.0      0.706692                         True
 2016      8012.0 7066.0            7933.0          0.656856             0.012191       96.0          8012.0            946.0      0.656856                         True
 2020      8514.0 8514.0            8956.0          0.000000             0.436539      102.0          8956.0            442.0      0.436539                         True
 2013      7429.0 7429.0            7807.0          0.000000             0.386158       95.0          7807.0            378.0      0.386158                         True
 2014      8014.0 8014.0            8273.0          0.000000             0.363553      109.0          8273.0            259.0      0.363553                         True
 2023      8861.0 8868.0            9097.0          0.000000             0.282942       94.0          9097.0            229.0      0.282942                         True
 2001      8461.0    0.0            1576.0          1.000000             0.012191       78.0          8461.0           8461.0      1.000000                        False
 2004      8371.0  966.0            1922.0          1.000000             0.012191       84.0          8371.0           7405.0      1.000000                        False
```

## DQN 结果

```text
 year                    scenario  seed  action_irrigation_total  action_fertilizer_total  final_grain_kg_ha  final_biomass_kg_ha  max_water_stress  max_nitrogen_stress
 2016       dqn_linked_free_daily     1                     90.0                    300.0             8012.0              14093.0               0.0             0.012191
 2016 dqn_linked_agronomic_window     1                     30.0                    300.0             8012.0              14093.0               0.0             0.012191
```

## 已生成四情景图年份

2016
