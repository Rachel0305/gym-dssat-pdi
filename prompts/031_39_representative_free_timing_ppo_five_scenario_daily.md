# 031_39 representative free-timing PPO five-scenario daily plot

## Purpose

为当前 031 系列自由时序 MaskablePPO 阶段性结果选择一个代表站点年份，输出与历史 `027_05` 样式一致的五情景日过程图，用于导师汇报。

## Representative case

- Site-year: LCA2010
- Reason:
  - LCA 在 031_38 中 19/19 年至少一项严格超过四情景 envelope。
  - LCA2010 是 031_33 的 LCA 训练年之一，模型来源清楚。
  - 031_38 中 LCA2010 的展示候选为 seed1 checkpoint75000。

## Boundary

- 不训练。
- 不重新选择 checkpoint。
- 不运行新的 DSSAT 模拟。
- 只读取已有 snapshot 和日值证据，复用 `027_05` 的绘图样式。

## Inputs

- PPO candidate snapshot:
  - `benchmark_results/031_34_four_site_all_year_frozen_maskableppo_transfer/runs/LCA/2010/seed1_ckpt75000/snapshot`
- Baseline snapshots:
  - `benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/LC/readiness/baseline_runs/{null,recorded_farmer,dssat_auto,official_extension_expert}/pdi_tmp_snapshot_eval`
- Selected candidate table:
  - `benchmark_results/031_38_five_station_ppo_water_n_saving_summary/tables/031_38_selected_station_year_ppo_deltas_vs_official.csv`

## Outputs

- Five-scenario daily process PNG/SVG.
- Five-scenario endpoint PNG/SVG.
- Daily CSV, summary CSV, manifest, and record.

