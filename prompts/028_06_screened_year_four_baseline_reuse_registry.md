# 028_06 已筛选年份四基线复用与缺口登记

## 目标

在不启动 DSSAT、不训练模型的前提下，逐项核对 028_03 候选池中尚未完成当前汇总的 10 个站点—年份的四个比较情景：`null`、`recorded`、`dssat_auto`、`official_extension_expert`。

本任务只回答两件事：已有快照能否复用；真正还缺哪些情景。不得为了目录完整而重复运行已有情景。

## 范围

- HLA：2007、2015、2016、2022。
- YC：2008。
- FQ：2013、2014、2019、2020、2023。
- 共 40 个“站点—年份—情景”单元。

## 强制审计

每个已有快照必须同时具有：

- `Summary.OUT`
- `PlantGro.OUT`
- `Weather.OUT`
- `SoilWat.OUT`
- `SoilNi.OUT`
- `MgmtEvent.OUT`
- `fileX.MZX`

并保存关键文件 SHA256。目录存在但文件不完整时不得标记为可复用。

## Provenance 边界

- HLA 2007/2015/2016/2022 使用 `020_11` prepared-adapter 输入链路；不得冒充未经转换的原始 treatment。
- FQ 2013/2014/2019/2020/2023 使用 `014_01` weather-year 派生链路，recorded 情景为 `recorded_shifted`；必须原样标注。
- YC2008 使用 `013_01` forward-screening 输入链路。
- official expert 缺失不得用 recorded 或其他年份 expert 替代。

## 输出

- `benchmark_results/028_06_screened_year_four_baseline_reuse_registry/028_06_baseline_scenario_registry.csv`
- `benchmark_results/028_06_screened_year_four_baseline_reuse_registry/028_06_summary.json`
- `docs/2026-07-18_028_06_screened_year_four_baseline_reuse_registry.md`

## 停止条件

若审计确认只有 official expert 缺失，则下一任务只能补这些缺失情景，不得重跑其余 34 个情景。

