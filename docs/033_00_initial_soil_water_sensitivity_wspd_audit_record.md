# 033_00 初始土壤水分敏感性与 WSPD 审计记录

## 结论先说

- 状态：完成。
- 本任务未训练 PPO/DQN，只做 DSSAT 前向 no-op/null 回放。
- 原始 `my_data/UFGA8201-*.jinja2` 和 `.SOL` 文件没有修改；所有 SH2O/IC 因子改动只发生在本任务输出目录下的派生模板中。
- 由于当前渲染模板的 treatment `IC` 因子可能为 0，本任务同时记录 `current_ic_factor` 和 `force_ic1` 两条分支。

## 汇总结果

| station | site | year | ic_branch | force_ic1 | fraction | final_grnwt | final_topwt | max_wspd | mean_wspd | wspd_days_gt_0 | wspd_days_gt_005 | max_nstd | mean_nstd | total_irrigation | total_nitrogen | snapshot_path | ic_forced | treatment_line_before | treatment_line_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | HL | 2015 | current_ic_factor | False | 0.55 | 393.4342 | 793.3484 | 0.0 | 0.0 | 0 | 0 | 0.7988 | 0.3622 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/HLA/2015/current_ic_factor/f0p55_null_noop | False |  |  |
| HLA | HL | 2015 | current_ic_factor | False | 0.3 | 393.4342 | 793.3484 | 0.0 | 0.0 | 0 | 0 | 0.7988 | 0.3622 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/HLA/2015/current_ic_factor/f0p3_null_noop | False |  |  |
| HLA | HL | 2015 | current_ic_factor | False | 0.15 | 393.4342 | 793.3484 | 0.0 | 0.0 | 0 | 0 | 0.7988 | 0.3622 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/HLA/2015/current_ic_factor/f0p15_null_noop | False |  |  |
| HLA | HL | 2015 | force_ic1 | True | 0.55 | 5658.1812 | 16038.7744 | 0.9704 | 0.0931 | 21 | 21 | 0.0145 | 0.0003 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/HLA/2015/force_ic1/f0p55_null_noop | True |  1 1 1 0 Sim2015                    1  1  0  0  1  1  1  0  0  0  0  0  1 |  1 1 1 0 Sim2015                    1  1  0  1  1  1  1  0  0  0  0  0  1 |
| HLA | HL | 2015 | force_ic1 | True | 0.3 | 3270.6784 | 10408.009 | 1.0 | 0.1729 | 36 | 36 | 0.0145 | 0.0003 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/HLA/2015/force_ic1/f0p3_null_noop | True |  1 1 1 0 Sim2015                    1  1  0  0  1  1  1  0  0  0  0  0  1 |  1 1 1 0 Sim2015                    1  1  0  1  1  1  1  0  0  0  0  0  1 |
| HLA | HL | 2015 | force_ic1 | True | 0.15 | 2818.3765 | 7802.6904 | 1.0 | 0.2112 | 45 | 43 | 0.0145 | 0.0003 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/HLA/2015/force_ic1/f0p15_null_noop | True |  1 1 1 0 Sim2015                    1  1  0  0  1  1  1  0  0  0  0  0  1 |  1 1 1 0 Sim2015                    1  1  0  1  1  1  1  0  0  0  0  0  1 |
| LCA | LC | 2019 | current_ic_factor | False | 0.55 | 2982.4234 | 5357.5543 | 0.0 | 0.0 | 0 | 0 | 0.5037 | 0.2545 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/LCA/2019/current_ic_factor/f0p55_null_noop | False |  |  |
| LCA | LC | 2019 | current_ic_factor | False | 0.3 | 2982.4234 | 5357.5543 | 0.0 | 0.0 | 0 | 0 | 0.5037 | 0.2545 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/LCA/2019/current_ic_factor/f0p3_null_noop | False |  |  |
| LCA | LC | 2019 | current_ic_factor | False | 0.15 | 2982.4234 | 5357.5543 | 0.0 | 0.0 | 0 | 0 | 0.5037 | 0.2545 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/LCA/2019/current_ic_factor/f0p15_null_noop | False |  |  |
| LCA | LC | 2019 | force_ic1 | True | 0.55 | 9807.0001 | 19505.1636 | 0.0802 | 0.0017 | 2 | 2 | 0.0122 | 0.0002 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/LCA/2019/force_ic1/f0p55_null_noop | True |  1 1 1 0 Sim2019                    1  1  0  0  1  1  1  0  0  0  0  0  1 |  1 1 1 0 Sim2019                    1  1  0  1  1  1  1  0  0  0  0  0  1 |
| LCA | LC | 2019 | force_ic1 | True | 0.3 | 6664.4257 | 13703.418 | 0.9106 | 0.1646 | 27 | 25 | 0.0122 | 0.0002 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/LCA/2019/force_ic1/f0p3_null_noop | True |  1 1 1 0 Sim2019                    1  1  0  0  1  1  1  0  0  0  0  0  1 |  1 1 1 0 Sim2019                    1  1  0  1  1  1  1  0  0  0  0  0  1 |
| LCA | LC | 2019 | force_ic1 | True | 0.15 | 2489.1202 | 3358.7418 | 0.9055 | 0.2801 | 41 | 41 | 0.0122 | 0.0002 | 0.0 | 0.0 | benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/snapshots/LCA/2019/force_ic1/f0p15_null_noop | True |  1 1 1 0 Sim2019                    1  1  0  0  1  1  1  0  0  0  0  0  1 |  1 1 1 0 Sim2019                    1  1  0  1  1  1  1  0  0  0  0  0  1 |

## SH2O 派生设定

| station | soil_id | fraction | ICBL_cm | SLLL | SDUL | old_SH2O | new_SH2O | SNH4 | SNO3 | ic_branch | force_ic1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LCA | LC99001200 | 0.55 | 20 | 0.09 | 0.33 | 0.22 | 0.222 | 4.0 | 5.3 | current_ic_factor | False |
| LCA | LC99001200 | 0.55 | 40 | 0.11 | 0.36 | 0.25 | 0.247 | 4.0 | 4.5 | current_ic_factor | False |
| LCA | LC99001200 | 0.55 | 110 | 0.12 | 0.37 | 0.26 | 0.258 | 4.0 | 6.2 | current_ic_factor | False |
| LCA | LC99001200 | 0.55 | 150 | 0.07 | 0.4 | 0.25 | 0.252 | 4.0 | 14.6 | current_ic_factor | False |
| LCA | LC99001200 | 0.3 | 20 | 0.09 | 0.33 | 0.22 | 0.162 | 4.0 | 5.3 | current_ic_factor | False |
| LCA | LC99001200 | 0.3 | 40 | 0.11 | 0.36 | 0.25 | 0.185 | 4.0 | 4.5 | current_ic_factor | False |
| LCA | LC99001200 | 0.3 | 110 | 0.12 | 0.37 | 0.26 | 0.195 | 4.0 | 6.2 | current_ic_factor | False |
| LCA | LC99001200 | 0.3 | 150 | 0.07 | 0.4 | 0.25 | 0.169 | 4.0 | 14.6 | current_ic_factor | False |
| LCA | LC99001200 | 0.15 | 20 | 0.09 | 0.33 | 0.22 | 0.126 | 4.0 | 5.3 | current_ic_factor | False |
| LCA | LC99001200 | 0.15 | 40 | 0.11 | 0.36 | 0.25 | 0.147 | 4.0 | 4.5 | current_ic_factor | False |
| LCA | LC99001200 | 0.15 | 110 | 0.12 | 0.37 | 0.26 | 0.158 | 4.0 | 6.2 | current_ic_factor | False |
| LCA | LC99001200 | 0.15 | 150 | 0.07 | 0.4 | 0.25 | 0.12 | 4.0 | 14.6 | current_ic_factor | False |
| LCA | LC99001200 | 0.55 | 20 | 0.09 | 0.33 | 0.22 | 0.222 | 4.0 | 5.3 | force_ic1 | True |
| LCA | LC99001200 | 0.55 | 40 | 0.11 | 0.36 | 0.25 | 0.247 | 4.0 | 4.5 | force_ic1 | True |
| LCA | LC99001200 | 0.55 | 110 | 0.12 | 0.37 | 0.26 | 0.258 | 4.0 | 6.2 | force_ic1 | True |
| LCA | LC99001200 | 0.55 | 150 | 0.07 | 0.4 | 0.25 | 0.252 | 4.0 | 14.6 | force_ic1 | True |
| LCA | LC99001200 | 0.3 | 20 | 0.09 | 0.33 | 0.22 | 0.162 | 4.0 | 5.3 | force_ic1 | True |
| LCA | LC99001200 | 0.3 | 40 | 0.11 | 0.36 | 0.25 | 0.185 | 4.0 | 4.5 | force_ic1 | True |
| LCA | LC99001200 | 0.3 | 110 | 0.12 | 0.37 | 0.26 | 0.195 | 4.0 | 6.2 | force_ic1 | True |
| LCA | LC99001200 | 0.3 | 150 | 0.07 | 0.4 | 0.25 | 0.169 | 4.0 | 14.6 | force_ic1 | True |
| LCA | LC99001200 | 0.15 | 20 | 0.09 | 0.33 | 0.22 | 0.126 | 4.0 | 5.3 | force_ic1 | True |
| LCA | LC99001200 | 0.15 | 40 | 0.11 | 0.36 | 0.25 | 0.147 | 4.0 | 4.5 | force_ic1 | True |
| LCA | LC99001200 | 0.15 | 110 | 0.12 | 0.37 | 0.26 | 0.158 | 4.0 | 6.2 | force_ic1 | True |
| LCA | LC99001200 | 0.15 | 150 | 0.07 | 0.4 | 0.25 | 0.12 | 4.0 | 14.6 | force_ic1 | True |
| HLA | HL99001200 | 0.55 | 20 | 0.11 | 0.39 | 0.26 | 0.264 | 22.7 | 10.1 | current_ic_factor | False |
| HLA | HL99001200 | 0.55 | 40 | 0.11 | 0.38 | 0.26 | 0.259 | 10.8 | 11.3 | current_ic_factor | False |
| HLA | HL99001200 | 0.55 | 60 | 0.11 | 0.38 | 0.26 | 0.259 | 15.1 | 12.5 | current_ic_factor | False |
| HLA | HL99001200 | 0.55 | 90 | 0.11 | 0.36 | 0.25 | 0.247 | 8.2 | 8.1 | current_ic_factor | False |
| HLA | HL99001200 | 0.3 | 20 | 0.11 | 0.39 | 0.26 | 0.194 | 22.7 | 10.1 | current_ic_factor | False |
| HLA | HL99001200 | 0.3 | 40 | 0.11 | 0.38 | 0.26 | 0.191 | 10.8 | 11.3 | current_ic_factor | False |
| HLA | HL99001200 | 0.3 | 60 | 0.11 | 0.38 | 0.26 | 0.191 | 15.1 | 12.5 | current_ic_factor | False |
| HLA | HL99001200 | 0.3 | 90 | 0.11 | 0.36 | 0.25 | 0.185 | 8.2 | 8.1 | current_ic_factor | False |
| HLA | HL99001200 | 0.15 | 20 | 0.11 | 0.39 | 0.26 | 0.152 | 22.7 | 10.1 | current_ic_factor | False |
| HLA | HL99001200 | 0.15 | 40 | 0.11 | 0.38 | 0.26 | 0.15 | 10.8 | 11.3 | current_ic_factor | False |
| HLA | HL99001200 | 0.15 | 60 | 0.11 | 0.38 | 0.26 | 0.15 | 15.1 | 12.5 | current_ic_factor | False |
| HLA | HL99001200 | 0.15 | 90 | 0.11 | 0.36 | 0.25 | 0.147 | 8.2 | 8.1 | current_ic_factor | False |
| HLA | HL99001200 | 0.55 | 20 | 0.11 | 0.39 | 0.26 | 0.264 | 22.7 | 10.1 | force_ic1 | True |
| HLA | HL99001200 | 0.55 | 40 | 0.11 | 0.38 | 0.26 | 0.259 | 10.8 | 11.3 | force_ic1 | True |
| HLA | HL99001200 | 0.55 | 60 | 0.11 | 0.38 | 0.26 | 0.259 | 15.1 | 12.5 | force_ic1 | True |
| HLA | HL99001200 | 0.55 | 90 | 0.11 | 0.36 | 0.25 | 0.247 | 8.2 | 8.1 | force_ic1 | True |
| HLA | HL99001200 | 0.3 | 20 | 0.11 | 0.39 | 0.26 | 0.194 | 22.7 | 10.1 | force_ic1 | True |
| HLA | HL99001200 | 0.3 | 40 | 0.11 | 0.38 | 0.26 | 0.191 | 10.8 | 11.3 | force_ic1 | True |
| HLA | HL99001200 | 0.3 | 60 | 0.11 | 0.38 | 0.26 | 0.191 | 15.1 | 12.5 | force_ic1 | True |
| HLA | HL99001200 | 0.3 | 90 | 0.11 | 0.36 | 0.25 | 0.185 | 8.2 | 8.1 | force_ic1 | True |
| HLA | HL99001200 | 0.15 | 20 | 0.11 | 0.39 | 0.26 | 0.152 | 22.7 | 10.1 | force_ic1 | True |
| HLA | HL99001200 | 0.15 | 40 | 0.11 | 0.38 | 0.26 | 0.15 | 10.8 | 11.3 | force_ic1 | True |
| HLA | HL99001200 | 0.15 | 60 | 0.11 | 0.38 | 0.26 | 0.15 | 15.1 | 12.5 | force_ic1 | True |
| HLA | HL99001200 | 0.15 | 90 | 0.11 | 0.36 | 0.25 | 0.147 | 8.2 | 8.1 | force_ic1 | True |

## 失败记录

_空表_

## 输出文件

- 日值表：`benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/tables/033_00_initial_water_sensitivity_daily.csv`
- SH2O 设定表：`benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/tables/033_00_initial_water_sensitivity_sh2o_settings.csv`
- 汇总表：`benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/tables/033_00_initial_water_sensitivity_summary.csv`
- 失败表：`benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/tables/033_00_failures.csv`

## 图件

- `benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/figures/033_00_hla2015_current_ic_factor_initial_water_sensitivity.png`
- `benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/figures/033_00_hla2015_force_ic1_initial_water_sensitivity.png`
- `benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/figures/033_00_lca2019_current_ic_factor_initial_water_sensitivity.png`
- `benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit/figures/033_00_lca2019_force_ic1_initial_water_sensitivity.png`

## 判读边界

- 如果降低初始水分后 WSPD 升高，只能说明初始水分设定对水分胁迫有影响；不能直接说明主实验必须改初始水分。
- 如果 WSPD 仍不升高，说明低 WSPD 不太可能由初始 SH2O 单独解释，需要继续查土壤蓄水能力、天气过程、DSSAT 水分胁迫变量定义或解析链条。
