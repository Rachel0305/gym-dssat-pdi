# 019_10 WUE/NUE 文献定义与 DSSAT 原生指标计算

## 目的

谨慎定义本项目的水分利用效率与氮素利用效率，并将定义落实到现有 DSSAT 4.8.0 `Summary.OUT` 的实际变量上，避免以后因术语混用返工。

## 原则

1. 不笼统使用含义不清的 `WUE` 或 `NUE`，每个指标必须写全名称、公式、单位和适用条件；
2. 优先使用 DSSAT 原生输出，并用原始变量复算核对；
3. `IRCM=0` 或 `NICM=0` 时，相应投入效率记为 `NA`，不得填 0 或无穷大；
4. 不用同时改变水氮的 null 情景计算氮肥农学效率或灌溉边际效率；
5. 不训练、不调用 DSSAT，只读取已有 `Summary.OUT`；
6. 正式比较使用 null、DSSAT auto、官方推广 expert、DQN 四个情景；农民记录另作补充。

## 推荐指标

- `WP_ET = HWAM / ETCP`：基于生育季实际蒸散的籽粒水分生产率；DSSAT 对应 `YPEM`。
- `IWP_gross = HWAM / irrigation`：单位灌溉水籽粒生产率；DSSAT 对应 `YPIM`，仅在灌溉量大于 0 时定义。
- `PFP_N = HWAM / NICM`：施氮偏生产力；DSSAT 对应 `YPNAM`，仅在施氮量大于 0 时定义。
- `NUtE = HWAM / NUCM`：植株吸收氮的内部利用效率；DSSAT 对应 `YPNUM`。
- `PNB_N = NUCM / NICM`：植株吸氮量与施氮量之比，不等同于肥料回收率。
- `N_leaching = NLCM`：季节氮淋洗量。

## 输出

- `DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_native_wue_nue_metrics.csv`
- `DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_dqn_vs_baseline_efficiency_comparison.csv`
- `DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_source_match_audit.csv`
- `DSSAT_auto_validation/five_site_wue_nue_019_10/019_10_metric_dictionary.csv`
- `docs/2026-07-10_019_10_wue_nue_definition_and_dssat_calculation.md`
