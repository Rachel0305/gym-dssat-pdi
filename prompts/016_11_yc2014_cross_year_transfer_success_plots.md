# 016_11 YC2014 跨年份迁移成功案例四情景图

## 目标

在不新增训练、不重跑多年迁移模拟的前提下，直接利用 `016_04` 已有跨年份迁移日值结果：

1. 筛出 YC2014 迁移到其他年份后表现最有代表性的年份；
2. 为这些年份生成正式四情景对照图；
3. 同时导出逐年日值表，保证“图—表—摘要”一一对应，方便后续汇报与追溯。

## 数据来源

- `DSSAT_auto_validation/yc2014_station_level3_true_model_transfer_016_04/yc2014_true_model_transfer_daily.csv`
- `DSSAT_auto_validation/yc2014_station_level3_true_model_transfer_016_04/yc2014_true_model_transfer_summary.csv`
- `DSSAT_auto_validation/yc2014_station_level3_true_model_transfer_016_04/yc2014_transfer_success_by_year.csv`

## 选择逻辑

优先使用 `success_flag=True` 的年份，并保留实际迁移成功的代表年份：

- 2006
- 2009
- 2015
- 2018

迁移情景使用每年表现最好的那条 transfer 记录，不再重新搜索。

## 输出

- `DSSAT_auto_validation/yc2014_cross_year_transfer_success_plots_016_11/`
  - `summary_selected_years.csv`
  - `daily_tables/`
  - `figures/`
- `docs/2026-07-05_016_11_yc2014_cross_year_transfer_success_plots_record.md`

## 执行

```bash
python src/plot_yc2014_cross_year_transfer_success_plots_016_11.py
```
