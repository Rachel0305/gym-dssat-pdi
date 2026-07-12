# 021_01 Final DQN Strategy Determination

## 1. 任务目标

严格执行 `prompts/021_01_final_dqn_strategy_determination.md`，不继续工程重构，不扩展新的敏感性实验，优先复用已有结果，只对问题站点做必要诊断，并据此确定统一 DQN 正式候选策略。

## 2. 已复用结果与新增诊断

- HLA：完全复用 `020_11` 已有正式五情景/多年份/多 seed 结果。
- YC：复用旧 `nitrogen_cost=5` 的 seed0/1，并补做 `seed2`；新增 `nitrogen_cost=2/8` 正式 50K 结果。
- FQ：复用旧 `water_cost=1` 的 seed0/1，并补做 `seed2`；新增 `water_cost=0.5/2` 正式 50K 结果。
- LC：启动 frozen 正式复核，但 benchmark runtime 出现挂起。
- SY：只做 provenance audit，不启动正式训练。

## 3. 站点诊断结论

| station_code   | focus_year               | status                    |   recommended_water_cost |   recommended_nitrogen_cost |   recommended_n_steps | evidence                                                                                                                                                                    | next_action                                                                    |
|:---------------|:-------------------------|:--------------------------|-------------------------:|----------------------------:|----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|
| HLA            | 2007,2010,2015,2016,2022 | keep_current_config       |                        1 |                           5 |                     5 | Existing formal 020_11 matrix already covers 5 years x 3 seeds under the frozen n-step setup.                                                                               | Reuse existing HLA formal results; no rerun needed in 021_01.                  |
| YC             | 2014                     | keep_current_config       |                        1 |                           5 |                     5 | nitrogen_cost=2/5/8 formal 50K comparison shows no meaningful policy or outcome separation; selected checkpoints all stay at zero-N behavior with similar yield.            | Retain current nitrogen_cost=5 for continuity and cross-station comparability. |
| FQ             | 2016                     | keep_current_config       |                        1 |                           5 |                     5 | water_cost=0.5/1/2 formal 50K comparison shows the same checkpoint family and nearly identical resource-use pattern.                                                        | Retain current water_cost=1 for continuity and cross-station comparability.    |
| LC             | 2010                     | blocked_runtime_poll_wait |                        1 |                           5 |                     5 | null run finished; DQN run created input/runtime shell only; pdi_gym.log ends with Client started; process slept >3h with near-zero CPU and no season_summary/training_log. | Do not launch new sensitivity runs; fix benchmark/runtime handshake first.     |
| SY             | 2014                     | blocked_input_provenance  |                        1 |                           5 |                     5 | 2014 treatment row in prepared MZX is the authoritative on-disk candidate used by current adapter path.                                                                     | Freeze formal training until IC/MZX provenance is authoritative.               |

## 4. HLA 结论

- HLA 已有正式结果矩阵完整，覆盖 2007/2010/2015/2016/2022，且包含多 seed。
- 021_01 不重复训练 HLA，直接保留当前配置结论：`keep_current_config`。

## 5. YC nitrogen_cost 诊断

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

结论：

- `nitrogen_cost=2/5/8` 在正式 50K 结果上没有形成可解释的策略分叉。
- 现有最优 checkpoint 仍是“零追加氮、有限灌溉”的同类策略族。
- 因此不建议仅为了 YC 单站点去改单独 reward，保留 `nitrogen_cost=5`。

## 6. FQ water_cost 诊断

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

结论：

- `water_cost=0.5/1/2` 的正式 50K 结果基本落在同一策略簇。
- 未观察到足以支撑改动统一参数的收益。
- 因此保留 `water_cost=1`。

## 7. LC frozen configuration 复核

|   seed | selection                | checkpoint_step   |   yield_kg_ha |   biomass_kg_ha |   irrigation_mm |   nitrogen_kg_ha |   max_water_stress |   max_nitrogen_stress |   total_reward | status                    | source                                                                                                                               | note                                                                                                                                                                        |
|-------:|:-------------------------|:------------------|--------------:|----------------:|----------------:|-----------------:|-------------------:|----------------------:|---------------:|:--------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      0 | best_reward              | 5000              |          8739 |           16377 |              90 |                0 |                  0 |             0.0191814 |        598.149 | legacy_reference_only     | legacy:DSSAT_auto_validation\extension_expert_baseline_018_03\018_06_lc2010_seed_stability_audit\018_06_lc2010_seed_best_summary.csv | Only 5K legacy evidence exists; not accepted as 021_01 formal frozen 50K seed review.                                                                                       |
|      0 | best_yield_then_resource | 5000              |          8739 |           16377 |              90 |                0 |                  0 |             0.0191814 |        598.149 | legacy_reference_only     | legacy:DSSAT_auto_validation\extension_expert_baseline_018_03\018_06_lc2010_seed_stability_audit\018_06_lc2010_seed_best_summary.csv | Only 5K legacy evidence exists; not accepted as 021_01 formal frozen 50K seed review.                                                                                       |
|      1 | best_reward              | 5000              |          8739 |           16374 |             120 |              300 |                  0 |             0.0191814 |       -932.467 | legacy_reference_only     | legacy:DSSAT_auto_validation\extension_expert_baseline_018_03\018_06_lc2010_seed_stability_audit\018_06_lc2010_seed_best_summary.csv | Only 5K legacy evidence exists; not accepted as 021_01 formal frozen 50K seed review.                                                                                       |
|      1 | best_yield_then_resource | 5000              |          8739 |           16374 |             120 |              300 |                  0 |             0.0191814 |       -932.467 | legacy_reference_only     | legacy:DSSAT_auto_validation\extension_expert_baseline_018_03\018_06_lc2010_seed_stability_audit\018_06_lc2010_seed_best_summary.csv | Only 5K legacy evidence exists; not accepted as 021_01 formal frozen 50K seed review.                                                                                       |
|      0 | 021_01_attempt           | <NA>              |           nan |             nan |             nan |              nan |                nan |           nan         |        nan     | blocked_runtime_poll_wait | new:benchmark_results/021_01/021_01_lc2010_seed_stability__lc_2010_seed0                                                             | null run finished; DQN run created input/runtime shell only; pdi_gym.log ends with Client started; process slept >3h with near-zero CPU and no season_summary/training_log. |

结论：

- 021_01 新框架下，LC2010 的 null 已完成，但 DQN 运行在 `Client started` 后长时间停滞。
- 进程累计运行数小时、CPU 接近 0、未生成 `season_summary.csv` 与训练日志。
- 因此本任务内将 LC 标为 `blocked_runtime_poll_wait`，不继续追加训练。

## 8. SY input provenance audit

| item                               | configured_value   | observed_value    | source                                                                                                                                 | status     | note                                                                                                             |
|:-----------------------------------|:-------------------|:------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:-----------|:-----------------------------------------------------------------------------------------------------------------|
| site_config.initial_condition_mode | 2                  | 2                 | configs\sites\sya.yaml                                                                                                                 | configured | Benchmark site config explicitly points to IC=2 expectation.                                                     |
| prepared_mzx_treatment2_IC         | 2                  | 0                 | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | mismatch   | 2014 treatment row in prepared MZX is the authoritative on-disk candidate used by current adapter path.          |
| prepared_mzx_soil_id               | SY99001200         | SY99001200        | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | matched    | Prepared input package keeps SY99001200 as soil id.                                                              |
| prepared_mzx_weather_2014          | CNSY1401.WTH       | CNSY1401.WTH      | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | matched    | 2014 field row points to CNSY1401.                                                                               |
| prepared_mzx_cultivar              | FY0985             | FY0985            | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\CNSY1201.MZX                                                                | matched    | Cultivar block in prepared MZX is FY0985.                                                                        |
| diagnostic_sy2012_ic1              | n/a                | 16789.09423828125 | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=158, terminated=True, description=2012 treatment, keep original IC=1                                       |
| diagnostic_sy2014_ic0              | n/a                | 18302.4365234375  | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=160, terminated=True, description=2014 treatment, keep original IC=0                                       |
| diagnostic_sy2014_ic1              | n/a                | 13299.51171875    | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=160, terminated=True, description=2014 treatment, patch treatment row back to IC=1                         |
| diagnostic_sy2014_ic1_icdat14100   | n/a                | 13299.51171875    | DSSAT_auto_validation\multisite_new_cultivar_inputs_013\SY\sy_2012_2014_ic_diagnosis_016_01_runs\sy_2012_2014_ic_diagnosis_summary.csv | evidence   | steps=160, terminated=True, description=2014 treatment, patch to IC=1 and set ICDAT=14100                        |
| historical_transfer_result_2014    | n/a                | 11216.0           | DSSAT_auto_validation\sy_local_dqn_train_cross_year_transfer_017_08\017_08_sy_combined_summary.csv                                     | evidence   | Historical transfer run exists, but current benchmark adapter still points to provenance-ambiguous prepared MZX. |

结论：

- `configs/sites/sya.yaml` 期待 `IC=2`。
- 但当前准备输入 `CNSY1201.MZX` 的 2014 treatment 行仍显示 `IC=0`。
- 同时历史诊断已显示 `IC=0` 与 `IC=1`/`IC=2` 会导致明显不同的产量水平。
- 因此本任务内将 SY 标为 `blocked_input_provenance`。

## 9. 统一参数证据表

| station_code      | status                    |   water_cost |   nitrogen_cost |   n_steps | irrigation_action_levels_mm   | nitrogen_action_levels_kg_ha   | season_budgets   | evidence                                                                                                                                                                    |
|:------------------|:--------------------------|-------------:|----------------:|----------:|:------------------------------|:-------------------------------|:-----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HLA               | keep_current_config       |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | Existing formal 020_11 matrix already covers 5 years x 3 seeds under the frozen n-step setup.                                                                               |
| YC                | keep_current_config       |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | nitrogen_cost=2/5/8 formal 50K comparison shows no meaningful policy or outcome separation; selected checkpoints all stay at zero-N behavior with similar yield.            |
| FQ                | keep_current_config       |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | water_cost=0.5/1/2 formal 50K comparison shows the same checkpoint family and nearly identical resource-use pattern.                                                        |
| LC                | blocked_runtime_poll_wait |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | null run finished; DQN run created input/runtime shell only; pdi_gym.log ends with Client started; process slept >3h with near-zero CPU and no season_summary/training_log. |
| SY                | blocked_input_provenance  |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | 2014 treatment row in prepared MZX is the authoritative on-disk candidate used by current adapter path.                                                                     |
| UNIFIED_CANDIDATE | partial                   |            1 |               5 |         5 | [0, 15, 30]                   | [0, 50, 100]                   | I<=120, N<=300   | Keep the frozen HLA/YC/FQ-compatible setup; LC runtime and SY provenance remain blocked items.                                                                              |

## 10. 最终候选策略

```yaml
meta:
  task: 021_01_final_dqn_strategy_determination
  date: '2026-07-12'
  status: partial
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
      LC: runtime poll-wait hang in 021_01 frozen benchmark run
      SY: input provenance ambiguous (config expects IC=2, prepared 2014 treatment
        row is IC=0)

```

## 11. 当前问题与已解决问题

已解决：

- YC `nitrogen_cost` 诊断完成。
- FQ `water_cost` 诊断完成。
- HLA 当前正式配置可以直接复用。

未解决：

- LC benchmark runtime 挂起，未完成 seed0/1/2 冻结正式复核。
- SY 输入 provenance 未统一，仍不能进入正式训练。

## 12. 后续实验计划

1. 先修 LC runtime/blocking 机制，再重新执行 frozen 50K seed 复核。
2. 先明确 SY authoritative IC/MZX，再生成正式训练配置。
3. 在 LC/SY 未解决前，不扩大统一参数搜索，不改 reward 结构。

## 13. Methods Source

- `benchmark/benchmark_runner.py`
- `configs/experiments/021_01_*.yaml`
- `configs/sites/hla.yaml`
- `configs/sites/yca.yaml`
- `configs/sites/fqa.yaml`
- `configs/sites/lca.yaml`
- `configs/sites/sya.yaml`
- `src/run_021_01_final_dqn_strategy_determination.py`

## 14. References

- `prompts/021_01_final_dqn_strategy_determination.md`
- `DSSAT_auto_validation/HLA_2004/hla_five_scenario_nstep_020_11/020_11_hla_five_scenario_summary.csv`
- `DSSAT_auto_validation/frozen_nstep_cross_site_020_12/YC2014/*`
- `DSSAT_auto_validation/frozen_nstep_cross_site_020_12/FQ2016/*`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv`
- `DSSAT_auto_validation/extension_expert_baseline_018_03/018_06_lc2010_seed_stability_audit/018_06_lc2010_seed_best_summary.csv`
- `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX`
- `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/sy_2012_2014_ic_diagnosis_016_01_runs/sy_2012_2014_ic_diagnosis_summary.csv`

## 15. 输出文件

- `benchmark_results/021_01/station_diagnosis.csv`
- `benchmark_results/021_01/yc_nitrogen_cost_summary.csv`
- `benchmark_results/021_01/fq_water_cost_summary.csv`
- `benchmark_results/021_01/lc_seed_stability_summary.csv`
- `benchmark_results/021_01/sy_input_provenance.csv`
- `benchmark_results/021_01/unified_parameter_evidence.csv`
- `benchmark_results/021_01/final_dqn_candidate.yaml`
- `configs/final_dqn_candidate.yaml`
- `docs/2026-07-12_021_01_final_dqn_strategy_determination.md`
- `docs/2026-07-12_021_01_final_dqn_strategy_determination.pptx`

## 16. Git commit 信息

- Git commit: `b3a1ebf` (`Diagnose and determine unified DQN strategy`)
- Git push: failed (`Connection closed by 198.18.0.90 port 22`)
