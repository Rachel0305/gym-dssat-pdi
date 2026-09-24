# YC 2004–2013 天气定稿与 train-only `.CLI` 准备（003_05）

**最终 Gate：** `BLOCKED_EXTERNAL_GAPFILL`
**分析范围：** YC/YCA；训练期 2004–2013；未启动 WGEN、DSSAT 或 PPO。

## 1. 数据源层级

- **2004**：ChinaFLUX 半小时“近地面空气温度”按完整 48 条记录计算 TMAX/TMIN；SRAD 按 `sum(W m-2 × 1800 s) / 1e6` 积分；RAIN 仅对完整 48 条日求和。温度按 2005–2006 重叠期一致性选择，不宣称已经确认 2 m 高度。
- **2005–2013**：禹城站官方 QC 日产品直接提供 TMAX、TMIN、RAIN、SRAD。官方发布值被视为 QC 产品值，不等同于未经处理的原始观测；发布的 0 mm 雨量接受为正式值。
- 外部源只填缺口，不替换完整主源记录；每个候选值保留 `source` 和 `qc`。
- 上一轮项目内来源盘点没有找到覆盖 2004 五个雨量缺口的可信逐日记录；1998–2006 产品只有月尺度，未拆分成日值。
- NASA POWER 官方说明：[Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)；[参数字典](https://power.larc.nasa.gov/docs/tutorials/parameters/)。

## 2. 缺口清单与外部数据源

原始缺口数（负值、缺测、TMAX<TMIN 和 2004 不完整聚合均计入）：

| 年份 | 变量 | 缺口日数 |
|---:|---|---:|
| 2004 | RAIN | 5 |
| 2005 | SRAD | 40 |
| 2005 | TMAX | 8 |
| 2005 | TMIN | 8 |
| 2006 | SRAD | 19 |
| 2006 | TMAX | 17 |
| 2006 | TMIN | 17 |
| 2007 | SRAD | 1 |
| 2008 | SRAD | 30 |
| 2008 | TMAX | 26 |
| 2008 | TMIN | 26 |
| 2009 | SRAD | 17 |
| 2009 | TMAX | 14 |
| 2009 | TMIN | 14 |
| 2010 | SRAD | 3 |
| 2010 | TMAX | 3 |
| 2010 | TMIN | 3 |
| 2011 | SRAD | 38 |
| 2011 | TMAX | 38 |
| 2011 | TMIN | 38 |
| 2012 | SRAD | 1 |
| 2012 | TMAX | 1 |
| 2012 | TMIN | 1 |
| 2013 | SRAD | 2 |

逐日缺口日期（连续日期已合并成首末范围；逐行值与原因见 `gaps_before_external_fill.csv`）：

| 年份 | 变量 | 日期/范围 |
|---:|---|---|
| 2004 | RAIN | 2004-10-16 至 2004-10-20 |
| 2005 | SRAD | 2005-02-04 至 2005-02-15、2005-03-14 至 2005-03-17、2005-03-19 至 2005-03-23、2005-03-26 至 2005-03-28、2005-03-31 至 2005-04-06、2005-06-10 至 2005-06-18 |
| 2005 | TMAX | 2005-06-11 至 2005-06-18 |
| 2005 | TMIN | 2005-06-11 至 2005-06-18 |
| 2006 | SRAD | 2006-01-05 至 2006-01-16、2006-02-16、2006-12-02 至 2006-12-07 |
| 2006 | TMAX | 2006-01-05 至 2006-01-16、2006-12-03 至 2006-12-07 |
| 2006 | TMIN | 2006-01-05 至 2006-01-16、2006-12-03 至 2006-12-07 |
| 2007 | SRAD | 2007-11-06 |
| 2008 | SRAD | 2008-06-28 至 2008-07-02、2008-08-01 至 2008-08-08、2008-08-11、2008-10-01 至 2008-10-03、2008-10-25 至 2008-10-28、2008-10-31 至 2008-11-01、2008-11-06 至 2008-11-12 |
| 2008 | TMAX | 2008-06-28 至 2008-07-02、2008-08-01 至 2008-08-08、2008-10-01 至 2008-10-03、2008-10-26 至 2008-10-28、2008-11-01、2008-11-07 至 2008-11-12 |
| 2008 | TMIN | 2008-06-28 至 2008-07-02、2008-08-01 至 2008-08-08、2008-10-01 至 2008-10-03、2008-10-26 至 2008-10-28、2008-11-01、2008-11-07 至 2008-11-12 |
| 2009 | SRAD | 2009-02-03、2009-03-11、2009-03-24 至 2009-04-07 |
| 2009 | TMAX | 2009-03-25 至 2009-04-07 |
| 2009 | TMIN | 2009-03-25 至 2009-04-07 |
| 2010 | SRAD | 2010-03-18、2010-08-10 至 2010-08-11 |
| 2010 | TMAX | 2010-03-18、2010-08-10 至 2010-08-11 |
| 2010 | TMIN | 2010-03-18、2010-08-10 至 2010-08-11 |
| 2011 | SRAD | 2011-03-09、2011-04-20、2011-05-18 至 2011-06-03、2011-08-17 至 2011-09-04 |
| 2011 | TMAX | 2011-03-09、2011-05-18 至 2011-06-03、2011-07-27、2011-08-17 至 2011-09-04 |
| 2011 | TMIN | 2011-03-09、2011-05-18 至 2011-06-03、2011-07-27、2011-08-17 至 2011-09-04 |
| 2012 | SRAD | 2012-06-03 |
| 2012 | TMAX | 2012-06-03 |
| 2012 | TMIN | 2012-06-03 |
| 2013 | SRAD | 2013-01-02、2013-10-06 |

外部源为 NASA POWER Daily API（AG community，LST，点位 36.830°N、116.570°E）。它是格点/模型和卫星衍生产品，仅作为缺口补值候选。原始响应缓存于 `data/external/yc_weather_finalize_and_cli/nasa_power/`，请求 URL、API 版本、单位、来源和 SHA256 见 `run_manifest.json`。

### 重叠期验证

只用 2005–2013 官方日值非缺测且物理有效日期；偏差为外部原值减官方值。加性校正仅用于 TMAX/TMIN/SRAD，参数为 `mean(official - external)`；RAIN 不做均值平移。

| 变量 | 配对日 | 原始 bias | 校正后 MAE | 校正后 RMSE | Pearson r |
|---|---:|---:|---:|---:|---:|
| TMAX | 3180 | 0.564 | 1.721 | 2.316 | 0.9808 |
| TMIN | 3180 | -0.435 | 1.630 | 2.120 | 0.9814 |
| SRAD | 3136 | 0.077 | 1.562 | 2.482 | 0.9363 |

降雨事件一致率：71.98%；官方湿日事件量 MAE：7.16 mm/d；月总量比中位数：1.134；年总量比中位数：1.082。所有 overlap 指标是同站点数据产品对照，不是独立观测验证。门槛和逐年/逐月比值见 `external_gapfill_validation.json`。

外部 overlap Gate：`False`；未通过项：`RAIN_event_agreement`。本轮缺口提案数 370，接受填补数 0。门槛是本轮预先写入脚本的工程筛选规则，不是 NASA 或 DSSAT 发布的标准。

## 3. 候选数据、统计与 QC

候选文件：`yc_wgen_fitting_weather_2004_2013.csv`；存在：`False`。
候选 QC：`False`；行数 None；日期覆盖 2004-01-01 至 2013-12-31。
2004 年最终降水量：未计算（candidate Gate 未通过）。

| 年份 | 变量 | 外部 gap-fill 日数 | 当年比例 |
|---:|---|---:|---:|
| 无 | - | 0 | - |

年/月气候统计见 `candidate_annual_summary.csv` 和 `candidate_monthly_summary.csv`。缺口填补比例不是“实测率”；外部补值仍按来源单独识别。2011–2013 各变量填补量在上述表和 `gap_filled_days_by_year_variable.csv` 中列出。

若 candidate 未形成，年/月气候统计文件会显式标为未运行；`external_gapfill_values.csv` 中的数值只是外部提案，`qc_status=REJECTED_EXTERNAL_OVERLAP_GATE` 时不得读入 candidate。

### 2005–2010 雨量源对照

下表分别汇总 ChinaFLUX 完整 48 半小时日和官方日产品有效日；不同源有效天数可能不同，因此以完整 365/366 天候选总量为准，paired-day 对照数值另见 `rain_source_comparison_2005_2010.csv`。

| 年份 | ChinaFLUX 完整日 | ChinaFLUX 总量 mm | 官方有效日 | 官方有效值总量 mm | 官方−ChinaFLUX mm |
|---:|---:|---:|---:|---:|---:|
| 2005 | 365 | 627.2 | 365 | 678.4 | 51.2 |
| 2006 | 365 | 380.2 | 365 | 403.6 | 23.4 |
| 2007 | 362 | 535.7 | 365 | 571.7 | 36.0 |
| 2008 | 363 | 477.9 | 366 | 528.7 | 50.8 |
| 2009 | 359 | 716.2 | 365 | 818.6 | 102.4 |
| 2010 | 339 | 151.8 | 365 | 739.6 | 587.8 |

WeatherMan 操作依据：[DSSAT User's Guide Volume 3](https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf)。

## 4. Leakage audit

`leakage_audit.json` 记录：官方表只解析到 2013 年气象变量列；为停止顺序扫描，仅读取下一行的年份标记 `2014`，该行其余气象单元格未解码/使用；源文件 SHA256 是字节级完整性校验，不解析单元格。偏差参数只由 2005–2013 计算；NASA 请求止于 2013-12-31；validation 天气数据仍独立。代码断言 `max(source_weather_years_used_for_values_or_calibration) <= 2013`。

## 5. `.CLI` 状态与下一步

官方 WeatherMan 自动生成：`False`。状态：`NOT_RUN`。
候选天气 QC 尚未通过，CLI 准备未启动。

天气 candidate 通过后仍不运行 WGEN。本轮只准备 train-only `.CLI`；下一任务才进入受限 seed pilot 和 DSSAT 单季 smoke，之后再独立审核能否进入 PPO。

## 6. 可复现文件

- 主脚本：`scripts/finalize_yc_weather_and_prepare_cli.py`
- 原始缺口：`gaps_before_external_fill.csv`
- 外部 overlap / 补值：`external_gapfill_validation.json`、`external_gapfill_values.csv`
- 泄漏审计：`leakage_audit.json`
- run source hash：`run_manifest.json`
- 年/月 summary 与 gap 比例：`candidate_annual_summary.csv`、`candidate_monthly_summary.csv`、`gap_filled_days_by_year_variable.csv`
- 中文 PPT：`docs/yc_weather_finalize_and_cli.pptx`；结构检查：`pptx_validation_summary.json`
- 实验记录：`results/yc_weather_finalize_and_cli/experiment_log.md`
