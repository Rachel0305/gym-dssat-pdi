# 004_18A YC 随机天气汇报统计

## 范围与来源
使用 `004_17/weather_archive_manifest_verified.csv` 所指向的100套 canonical daily CSV；Observed 使用 004_05 seed5 正式 evaluation 的原始 `CNYCxxxx.WTH`，其 FileX treatment-1 PDATE 定义 DAP0，step trace 的最大 DAP 界定季末。004_05 step trace 天气四列全空，未以缺失值代替天气。历史 exact：training 78/80、held-out 20/20；seed1003、1036 暂定。未运行 WGEN、DSSAT 或 PPO。

## 对齐与质量控制
WGEN runtime DATE 为统一参考日期，不等同于 crop year；crop year/context 取 manifest。每套 WGEN 序列保留最后一条 DAP=0 及其后记录，排除前置 DAP=0；每套裁剪数见 `season_alignment_qc.csv`。该裁剪是描述性可比口径，不更改归档或历史 hash。Observed 的 WTH 按真实日期、FileX PDATE、trace 末日对齐。80/20/10 季已通过 DAP 唯一、连续和气象值有效性检查。各组季节长度中位数分别为 99/98/98 天；季长差异会影响整季累计量。

## FULL80 vs EXACT78
`SENSITIVITY_CHANGES_IDENTIFIED`。预设判据：percentile 差至少10个百分点或 P5/P95 tail、min/max support 标记翻转。触发条目 6/850；详情见 `observed_training_percentiles.csv`，所有分布量见 `full80_vs_exact78_sensitivity.csv`。这只是统计敏感性，不等于对 provisional 两套完成历史验证。

## 描述性结果
整季汇报表见 `weather_summary_for_presentation.csv`，逐年见 `observed_weather_2014_2023.csv`；阶段数据见 `weather_stage_summary.csv`。Observed 逐年在 FULL80 的位置见 `observed_training_percentiles.csv` 和 Figure 8。Figure 9 对照 seed5 的2014/2019和已有五情景报告中产量相对较好的2017/2023；只描述天气，不推断产量原因。

## 证据边界
Descriptive climate-position analysis; formal coverage verdict pending provenance closure / sensitivity review. 004_17 两条 provisional 历史 hash 尚未闭合；正式 climate coverage verdict 留待004_18B/004_19。

## 图表
- Figure 1: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure1_dataset_overview.png`
- Figure 2: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure2_seasonal_rainfall.png`
- Figure 3: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure3_rainfall_extremes.png`
- Figure 4: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure4_temperature_exposure.png`
- Figure 5: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure5_srad_distribution.png`
- Figure 6: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure6_stage_rainfall.png`
- Figure 7: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure7_stage_temperature.png`
- Figure 8: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure8_observed_percentile_heatmap.png`
- Figure 9: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/figure9_seed5_key_year_weather.png`

### 敏感项
| scope   | stage    | metric         |   year |   percentile_delta_pp |   full80_tail_p5_p95 |   exact78_tail_p5_p95 | full80_outside_support   | exact78_outside_support   |
|:--------|:---------|:---------------|-------:|----------------------:|---------------------:|----------------------:|:-------------------------|:--------------------------|
| season  | ALL      | rainfall_total |   2014 |               1.25    |                    1 |                     1 | False                    | True                      |
| season  | ALL      | rainfall_total |   2016 |               1.25    |                    1 |                     1 | False                    | True                      |
| season  | ALL      | rainfall_total |   2019 |               1.25    |                    1 |                     1 | False                    | True                      |
| season  | ALL      | rx5day         |   2014 |               1.15385 |                    0 |                     1 | False                    | False                     |
| stage   | DAP31_60 | rx1day         |   2014 |               1.12179 |                    0 |                     1 | False                    | False                     |
| stage   | DAP61_90 | srad_max       |   2017 |               1.12179 |                    0 |                     1 | False                    | False                     |

尤其整季降雨：2014、2016、2019 在 FULL80 中仅靠 provisional 低降雨样本保持在 min/max 内，EXACT78 则落在 min 以下。故不能称两条 provisional 对 tail identification 无影响。
