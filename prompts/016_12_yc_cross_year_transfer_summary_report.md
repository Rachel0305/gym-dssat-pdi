# 016_12 YC 跨年份迁移总对比汇总图表

## 目标

把 `016_11` 已选出的 4 个代表年份（2006/2009/2015/2018）进一步整理成：

1. 一张可直接汇报的跨年份总表；
2. 一张跨年份总对比汇总图；
3. 保证图、表、年份选择和 `016_11` 保持完全一致。

## 输入

- `DSSAT_auto_validation/yc2014_cross_year_transfer_success_plots_016_11/summary_selected_years.csv`
- `DSSAT_auto_validation/yc2014_station_level3_true_model_transfer_016_04/yc2014_true_model_transfer_summary.csv`

## 输出

- `DSSAT_auto_validation/yc2014_cross_year_transfer_summary_report_016_12/`
  - `yc_cross_year_transfer_summary_table.csv`
  - `figures/yc_cross_year_transfer_summary_report.png`
- `docs/2026-07-05_016_12_yc_cross_year_transfer_summary_report.md`

## 说明

- 不新增模拟；
- 不新增训练；
- 仅做结果整理与展示；
- 汇总图重点突出 DQN transfer 相对 auto / expert / null 的位置关系。
