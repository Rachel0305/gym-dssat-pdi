# 021_01 最终 DQN 策略判定

## 1. 任务状态

状态：`partial`。HLA、YC、FQ、LC 的必要诊断已经完成；LC 显示资源投入的 seed 敏感性，SY 仍因输入来源冲突而阻塞。本任务没有修改 reward、IC、DSSAT 输入值或动作空间，也没有扩大参数扫描。

## 2. 当前统一配置

- 算法：DQN；`n_steps=5`；50K steps；每 5K 保存 checkpoint。
- 动作：灌溉 `[0, 15, 30] mm`；施氮 `[0, 50, 100] kg/ha`。
- 季节预算：`I<=120 mm`，`N<=300 kg/ha`；决策间隔 7 DAP。
- 奖励：相对本地 null 的终端产量增益，减去 `1*I + 5*N`。
- checkpoint 规则：最大 `reward_total`；并列时选择更早 checkpoint。

## 3. 五站点判定

| station_code   | focus_year               | status                      |   recommended_water_cost |   recommended_nitrogen_cost |   recommended_n_steps | evidence                                                                                                                                                         | next_action                                                                    |
|:---------------|:-------------------------|:----------------------------|-------------------------:|----------------------------:|----------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|
| HLA            | 2007,2010,2015,2016,2022 | keep_current_config         |                        1 |                           5 |                     5 | Existing formal 020_11 matrix already covers 5 years x 3 seeds under the frozen n-step setup.                                                                    | Reuse existing HLA formal results; no rerun needed in 021_01.                  |
| YC             | 2014                     | keep_current_config         |                        1 |                           5 |                     5 | nitrogen_cost=2/5/8 formal 50K comparison shows no meaningful policy or outcome separation; selected checkpoints all stay at zero-N behavior with similar yield. | Retain current nitrogen_cost=5 for continuity and cross-station comparability. |
| FQ             | 2016                     | keep_current_config         |                        1 |                           5 |                     5 | water_cost=0.5/1/2 formal 50K comparison shows the same checkpoint family and nearly identical resource-use pattern.                                             | Retain current water_cost=1 for continuity and cross-station comparability.    |
| LC             | 2010                     | resource_use_seed_sensitive |                        1 |                           5 |                     5 | After repairing the LC adapter source, seed0/1/2 formal 50K yields were 8737/8728/8707 kg/ha while irrigation was 120/60/30 mm; nitrogen was 0 for all seeds.    | Retain as diagnostic evidence; do not claim cross-seed resource stability.     |
| SY             | 2014                     | blocked_input_provenance    |                        1 |                           5 |                     5 | 2014 treatment row in prepared MZX is the authoritative on-disk candidate used by current adapter path.                                                          | Freeze formal training until IC/MZX provenance is authoritative.               |

## 4. YC nitrogen_cost 诊断

| station_code   |   year |   nitrogen_cost |   seed |   checkpoint |   yield_kg_ha |   biomass_kg_ha |   irrigation_mm |   nitrogen_kg_ha |   reward_total |   max_water_stress |   max_nitrogen_stress | source                                                                                                              |
|:---------------|-------:|----------------:|-------:|-------------:|--------------:|----------------:|----------------:|-----------------:|---------------:|-------------------:|----------------------:|:--------------------------------------------------------------------------------------------------------------------|
| YC             |   2014 |               2 |      0 |        35000 |          8659 |           18831 |              60 |                0 |        774.153 |                nan |            nan        | new:benchmark_results\021_01\021_01_yc2014_ncost2__yc_2014_seed0\evaluations\season_summary.csv                     |
| YC             |   2014 |               2 |      1 |        50000 |          8676 |           18869 |              90 |                0 |        760.573 |                nan |            nan        | new:benchmark_results\021_01\021_01_yc2014_ncost2__yc_2014_seed1\evaluations\season_summary.csv                     |
| YC             |   2014 |               2 |      2 |        50000 |          8665 |           18842 |              75 |                0 |        764.78  |                nan |            nan        | new:benchmark_results\021_01\021_01_yc2014_ncost2__yc_2014_seed2\evaluations\season_summary.csv                     |
| YC             |   2014 |               5 |      0 |        35000 |          8659 |           18831 |              60 |                0 |        774.153 |                  0 |              0.449023 | reused:DSSAT_auto_validation\frozen_nstep_cross_site_020_12\YC2014\seed0_50000steps\selected_checkpoint_summary.csv |
| YC             |   2014 |               5 |      1 |        50000 |          8676 |           18869 |              90 |                0 |        760.573 |                  0 |              0.447316 | reused:DSSAT_auto_validation\frozen_nstep_cross_site_020_12\YC2014\seed1_50000steps\selected_checkpoint_summary.csv |
| YC             |   2014 |               5 |      2 |        50000 |          8665 |           18842 |              75 |                0 |        764.78  |                nan |            nan        | new:benchmark_results\021_01\021_01_yc2014_ncost5__yc_2014_seed2\evaluations\season_summary.csv                     |
| YC             |   2014 |               8 |      0 |        35000 |          8659 |           18831 |              60 |                0 |        774.153 |                nan |            nan        | new:benchmark_results\021_01\021_01_yc2014_ncost8__yc_2014_seed0\evaluations\season_summary.csv                     |
| YC             |   2014 |               8 |      1 |        50000 |          8676 |           18869 |              90 |                0 |        760.573 |                nan |            nan        | new:benchmark_results\021_01\021_01_yc2014_ncost8__yc_2014_seed1\evaluations\season_summary.csv                     |
| YC             |   2014 |               8 |      2 |        50000 |          8665 |           18842 |              75 |                0 |        764.78  |                nan |            nan        | new:benchmark_results\021_01\021_01_yc2014_ncost8__yc_2014_seed2\evaluations\season_summary.csv                     |

`nitrogen_cost=2/5/8` 没有形成足以支持修改统一系数的稳定分离，因此保留 5。

## 5. FQ water_cost 诊断

| station_code   |   year |   water_cost |   seed |   checkpoint |   yield_kg_ha |   biomass_kg_ha |   irrigation_mm |   nitrogen_kg_ha |   reward_total |   max_water_stress |   max_nitrogen_stress | source                                                                                                              |
|:---------------|-------:|-------------:|-------:|-------------:|--------------:|----------------:|----------------:|-----------------:|---------------:|-------------------:|----------------------:|:--------------------------------------------------------------------------------------------------------------------|
| FQ             |   2016 |          0.5 |      0 |        50000 |          7985 |           14028 |             120 |               50 |        548.564 |                nan |           nan         | new:benchmark_results\021_01\021_01_fq2016_wcost0p5__fq_2016_seed0\evaluations\season_summary.csv                   |
| FQ             |   2016 |          0.5 |      1 |        40000 |          7985 |           13988 |             105 |                0 |        813.536 |                nan |           nan         | new:benchmark_results\021_01\021_01_fq2016_wcost0p5__fq_2016_seed1\evaluations\season_summary.csv                   |
| FQ             |   2016 |          0.5 |      2 |        20000 |          8012 |           14086 |             105 |                0 |        841.422 |                nan |           nan         | new:benchmark_results\021_01\021_01_fq2016_wcost0p5__fq_2016_seed2\evaluations\season_summary.csv                   |
| FQ             |   2016 |          1   |      0 |        50000 |          7985 |           14028 |             120 |               50 |        548.564 |                  0 |             0.0121911 | reused:DSSAT_auto_validation\frozen_nstep_cross_site_020_12\FQ2016\seed0_50000steps\selected_checkpoint_summary.csv |
| FQ             |   2016 |          1   |      1 |        40000 |          7985 |           13988 |             105 |                0 |        813.536 |                  0 |             0.0121911 | reused:DSSAT_auto_validation\frozen_nstep_cross_site_020_12\FQ2016\seed1_50000steps\selected_checkpoint_summary.csv |
| FQ             |   2016 |          1   |      2 |        20000 |          8012 |           14086 |             105 |                0 |        841.422 |                nan |           nan         | new:benchmark_results\021_01\021_01_fq2016_wcost1__fq_2016_seed2\evaluations\season_summary.csv                     |
| FQ             |   2016 |          2   |      0 |        50000 |          7985 |           14028 |             120 |               50 |        548.564 |                nan |           nan         | new:benchmark_results\021_01\021_01_fq2016_wcost2__fq_2016_seed0\evaluations\season_summary.csv                     |
| FQ             |   2016 |          2   |      1 |        40000 |          7985 |           13988 |             105 |                0 |        813.536 |                nan |           nan         | new:benchmark_results\021_01\021_01_fq2016_wcost2__fq_2016_seed1\evaluations\season_summary.csv                     |
| FQ             |   2016 |          2   |      2 |        20000 |          8012 |           14086 |             105 |                0 |        841.422 |                nan |           nan         | new:benchmark_results\021_01\021_01_fq2016_wcost2__fq_2016_seed2\evaluations\season_summary.csv                     |

`water_cost=0.5/1/2` 基本落在同一策略族，没有证据支持修改统一系数，因此保留 1。

## 6. LC2010 正式 50K 跨 seed 复核

| station_code   |   year |   seed | selection                                         |   checkpoint_step |   yield_kg_ha |   biomass_kg_ha |   irrigation_mm |   nitrogen_kg_ha |   total_reward |   irrigation_events |   nitrogen_events |   water_budget_use_ratio |   nitrogen_budget_use_ratio | runtime_audit_passed   | status        | source                                                                                                    |
|:---------------|-------:|-------:|:--------------------------------------------------|------------------:|--------------:|----------------:|----------------:|-----------------:|---------------:|--------------------:|------------------:|-------------------------:|----------------------------:|:-----------------------|:--------------|:----------------------------------------------------------------------------------------------------------|
| LC             |   2010 |      0 | maximum reward_total; earliest checkpoint on ties |             15000 |          8737 |           16366 |             120 |                0 |        566.123 |                   6 |                 0 |                     1    |                           0 | True                   | completed_50k | benchmark_results\021_01\021_01_lc2010_seed_stability_retry__lc_2010_seed0\evaluations\season_summary.csv |
| LC             |   2010 |      1 | maximum reward_total; earliest checkpoint on ties |             40000 |          8728 |           16311 |              60 |                0 |        616.584 |                   4 |                 0 |                     0.5  |                           0 | True                   | completed_50k | benchmark_results\021_01\021_01_lc2010_seed_stability_retry__lc_2010_seed1\evaluations\season_summary.csv |
| LC             |   2010 |      2 | maximum reward_total; earliest checkpoint on ties |             50000 |          8707 |           16199 |              30 |                0 |        626.408 |                   2 |                 0 |                     0.25 |                           0 | True                   | completed_50k | benchmark_results\021_01\021_01_lc2010_seed_stability_retry__lc_2010_seed2\evaluations\season_summary.csv |

三 seed 的最佳 checkpoint 不同（15K、40K、50K）。产量分别为 8737、8728、8707 kg/ha，差异仅 30 kg/ha；灌溉分别为 120、60、30 mm，差异达 90 mm；施氮均为 0。DSSAT auto 为 8738 kg/ha、138.5 mm、0 kg/ha，official expert 为 8739 kg/ha。LC 因而属于“产量近乎持平但资源投入跨 seed 敏感”，不能写成跨 seed 稳定成功，也不是 DQN 显著增产案例。

第一次运行挂起的根因不是训练或端口：Benchmark adapter 错把已经 null 化的 LC 输入作为 DQN 源，形成 `FERTI=L` 但 `MF=0`，DSSAT/PDI 在 `FertType_mod.for` 触发 `fertfile(0)`。修复后先通过 5K smoke，再串行完成三组 50K；全部 runtime audit 通过。

## 7. SY 输入来源审计

| item                               | configured_value   | observed_value    | source                                                                                                                                 | status     | note                                                                                                             |
|:-----------------------------------|:-------------------|:------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:-----------|:-----------------------------------------------------------------------------------------------------------------|
| site_config.initial_condition_mode | 2                  | 2                 | configs\sites\sya.yaml                                                                                                                 | configured | Benchmark site config explicitly points to IC=2 expectation.                                                     |
| prepared_mzx_treatment2_IC         | 2                  | 0                 | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | mismatch   | 2014 treatment row in prepared MZX is the authoritative on-disk candidate used by current adapter path.          |
| prepared_mzx_soil_id               | SY99001200         | SY99001200        | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | matched    | Prepared input package keeps SY99001200 as soil id.                                                              |
| prepared_mzx_weather_2014          | CNSY1401.WTH       | CNSY1401.WTH      | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | matched    | 2014 field row points to CNSY1401.                                                                               |
| prepared_mzx_cultivar              | FY0985             | FY0985            | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | matched    | Cultivar block in prepared MZX is FY0985.                                                                        |
| diagnostic_sy2012_ic1              | nan                | 16789.09423828125 | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=158, terminated=True, description=2012 treatment, keep original IC=1                                       |
| diagnostic_sy2014_ic0              | nan                | 18302.4365234375  | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=160, terminated=True, description=2014 treatment, keep original IC=0                                       |
| diagnostic_sy2014_ic1              | nan                | 13299.51171875    | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=160, terminated=True, description=2014 treatment, patch treatment row back to IC=1                         |
| diagnostic_sy2014_ic1_icdat14100   | nan                | 13299.51171875    | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=160, terminated=True, description=2014 treatment, patch to IC=1 and set ICDAT=14100                        |
| historical_transfer_result_2014    | nan                | 11216.0           | DSSAT_auto_validation\sy_local_dqn_train_cross_year_transfer_017_08\017_08_sy_combined_summary.csv                                     | evidence   | Historical transfer run exists, but current benchmark adapter still points to provenance-ambiguous prepared MZX. |

SY 的 authoritative IC/MZX 尚未统一，因此不能启动正式训练，也不能算作统一配置已完成五站验证。

## 8. 统一参数证据

| station_code      | status                      |   water_cost |   nitrogen_cost |   n_steps | irrigation_action_levels_mm   | nitrogen_action_levels_kg_ha   | season_budgets   | evidence                                                                                                                                                         |
|:------------------|:----------------------------|-------------:|----------------:|----------:|:------------------------------|:-------------------------------|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HLA               | keep_current_config         |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | Existing formal 020_11 matrix already covers 5 years x 3 seeds under the frozen n-step setup.                                                                    |
| YC                | keep_current_config         |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | nitrogen_cost=2/5/8 formal 50K comparison shows no meaningful policy or outcome separation; selected checkpoints all stay at zero-N behavior with similar yield. |
| FQ                | keep_current_config         |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | water_cost=0.5/1/2 formal 50K comparison shows the same checkpoint family and nearly identical resource-use pattern.                                             |
| LC                | resource_use_seed_sensitive |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | Formal LC2010 50K seed0/1/2 had similar yields but irrigation differed by 90 mm.                                                                                 |
| SY                | blocked_input_provenance    |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | 2014 treatment row in prepared MZX is the authoritative on-disk candidate used by current adapter path.                                                          |
| UNIFIED_CANDIDATE | partial_lc_sy               |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | Frozen configuration is supported by HLA/YC/FQ; LC remains resource-use seed-sensitive and SY input provenance is blocked.                                       |

## 9. 最终候选配置

```yaml
meta:
  task: 021_01_final_dqn_strategy_determination
  date: '2026-07-12'
  status: partial_lc_sy
  last_updated: '2026-07-13'
candidate:
  algorithm: DQN
  reward:
    type: null_relative_terminal_gain_minus_resource_cost
    water_cost: 1.0
    nitrogen_cost: 5.0
    note: Same formula kept; diagnosis did not justify changing coefficients.
  actions:
    irrigation_mm:
    - 0
    - 15
    - 30
    nitrogen_kg_ha:
    - 0
    - 50
    - 100
  budgets:
    irrigation_mm: 120
    nitrogen_kg_ha: 300
  training:
    n_steps: 5
    checkpoint_selection: best_reward_existing_protocol
  applicability:
    confirmed_sites:
    - HLA
    - YC
    - FQ
    blocked_sites:
      LC: formal 50K yields are similar across seeds, but irrigation differs by 90
        mm
      SY: input provenance ambiguous (site config expects IC=2, prepared 2014 treatment
        row is IC=0)
```

## 10. 已解决与未解决问题

- 已解决：YC nitrogen_cost、FQ water_cost、LC adapter 和三 seed 50K 正式诊断。
- 未解决：LC 资源投入跨 seed 敏感；SY2014 的 IC/MZX 输入来源冲突。
- 下一步：先确认 SY authoritative input，再按同一冻结配置做 smoke 和正式 seed 复核；不新增敏感性扫描。

## 11. Methods Source

- `benchmark/environment_adapter.py`
- `benchmark/train_runner.py`
- `configs/experiments/021_01_lc2010_seed_stability_retry*.yaml`
- `src/finalize_021_01_after_lc_retry.py`

## 12. 输出与 Git

- LC 图：`benchmark_results/021_01/figures/lc2010_50k_cross_seed_stability.png/.svg`
- 汇总：`benchmark_results/021_01/lc_seed_stability_summary.csv`
- 结果与修复提交：`445acb8`（`Complete LC2010 DQN stability diagnosis`）。
- Git push：2026-07-13 再次尝试失败；SSH `198.18.0.90:22` 主动关闭连接。本地提交完整保留，未伪造远端备份成功。
