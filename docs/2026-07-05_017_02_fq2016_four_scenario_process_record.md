# 017_02 FQ2016 四情景过程图整理记录

## 目的

在不新增训练的前提下，把 FQ2016 的 null、记录管理平移、DSSAT auto、DQN seed1 best-reward checkpoint 统一整理为可汇报的过程图和日值表。

## 执行口径

- 非 DQN 三个情景使用同一套 FQ2016 输入重新 forward；
- DQN 情景读取 `017_01` seed1 50K 训练中的 checkpoint 30000，这是 seed1 的 best-reward checkpoint；
- 本轮没有训练，没有修改奖励函数，没有改动作空间；
- 奖励代理值仅用于图中对齐展示：terminal max(0, GWAD-null) - 1×灌溉 - 5×施氮。

## 汇总结果

| site | station | year | scenario | final_grain_kg_ha | final_biomass_kg_ha | final_dap | max_water_stress | max_nitrogen_stress | irrigation_total | fertilizer_total | total_reward | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | Fengqiu | 2016 | null_zero | 7066.000 | 13148.000 | 96.000 | 0.657 | 0.012 | 0.000 | 0.000 | 0.075 | /workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq2016_four_scenario_process_017_02/runs/null |
| FQ | Fengqiu | 2016 | recorded_shifted | 7933.000 | 14008.000 | 96.000 | 0.373 | 0.012 | 75.000 | 144.000 | 866.604 | /workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq2016_four_scenario_process_017_02/runs/recorded_shifted |
| FQ | Fengqiu | 2016 | dssat_auto | 8012.000 | 14095.000 | 96.000 | 0.000 | 0.012 | 59.900 | 0.000 | 946.422 | /workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq2016_four_scenario_process_017_02/runs/dssat_auto |
| FQ | Fengqiu | 2016 | dqn_seed1_best_reward | 7995.000 | 14078.000 | 96.000 | 0.050 | 0.012 | 60.000 | 0.000 | 869.176 | /workspaces/gym-dssat-pdi/DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/pdi_tmp_snapshot_eval_30000 |

## 输出文件

- 日值表：`DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_daily.csv`
- 管理事件：`DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_management_events.csv`
- 汇总表：`DSSAT_auto_validation/fq2016_four_scenario_process_017_02/fq2016_four_scenario_summary.csv`
- 过程图 PNG：`DSSAT_auto_validation/fq2016_four_scenario_process_017_02/figures/fq2016_four_scenario_process.png`
- 过程图 SVG/PDF：同名 `.svg` / `.pdf`

## 初步结论

FQ2016 的 DQN seed1 best-reward 策略为约 I60/N0，符合 017_01 中 seed0/seed1 都倾向节水、零氮、接近高产的判断。该图可作为后续 FQ 跨年份迁移前的站点内代表过程图。
