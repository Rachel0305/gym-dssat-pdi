# Phase 2c 实测生育期降雨诊断

## 1. 任务背景
Phase 2b 已完成 QC 后天气与代表年份初选，但由于 jinja2 模板缺少明确收获期，当时只能使用 `planting + 150 days` 作为临时生育期窗口。本阶段改用实测 `planting_date -> harvest_date`，只诊断已有真实试验年份，不训练 PPO，不修改 reward，也不生成新的情景天气。

## 2. 输入数据
- 实测物候输入：`data/observed_phenology_dates.csv`
- QC 后天气目录：`weather_clean_qc`
- Phase 2b 对比文件：`Leave_One_experiments/year_classification/selected_years_qc.csv`

## 3. 日期 QC 结果
所有日期已标准化为 `YYYY-MM-DD`，并生成 DSSAT 常用 `YYDDD` 字段。检查规则包括播种-吐丝-成熟-收获顺序，以及几个阶段持续天数阈值。

| station_code   |   year | planting_date   | mature_date   | harvest_date   |   harvest_window_days | phenology_qc_flag   |
|:---------------|-------:|:----------------|:--------------|:---------------|----------------------:|:--------------------|
| FQA            |   2008 | 2008-06-10      | 2008-09-20    | 2008-09-20     |                   103 | pass                |
| FQA            |   2010 | 2010-06-10      | 2010-09-12    | 2010-09-26     |                   109 | pass                |
| HLA            |   2007 | 2007-05-05      | 2007-09-24    | 2007-10-03     |                   152 | pass                |
| HLA            |   2011 | 2011-05-05      | 2011-09-20    | 2011-09-23     |                   142 | pass                |
| HLA            |   2009 | 2009-05-02      | 2009-09-28    | 2009-10-02     |                   154 | pass                |
| LCA            |   2010 | 2010-06-06      | 2010-09-25    | 2010-09-27     |                   114 | pass                |
| LCA            |   2011 | 2011-06-15      | 2011-09-26    | 2011-09-27     |                   105 | pass                |
| LCA            |   2008 | 2008-06-19      | 2008-09-29    | 2008-10-02     |                   106 | pass                |
| LCA            |   2009 | 2009-06-14      | 2009-09-23    | 2009-09-23     |                   102 | pass                |
| SYA            |   2014 | 2014-04-21      | 2014-09-09    | 2014-09-15     |                   148 | pass                |
| SYA            |   2015 | 2015-04-18      | 2015-09-16    | 2015-09-22     |                   158 | pass                |
| SYA            |   2012 | 2012-04-28      | 2012-09-06    | 2012-09-06     |                   132 | pass                |
| YCA            |   2014 | 2014-06-17      | 2014-09-28    | 2014-10-02     |                   108 | pass                |
| YCA            |   2008 | 2008-06-18      | 2008-09-24    | 2008-09-26     |                   101 | pass                |

异常日期记录：
无。


## 4. 天气数据读取和 RAIN 字段识别
- FQA: date=`date`, RAIN=`RAIN`
- HLA: date=`date`, RAIN=`RAIN`
- LCA: date=`date`, RAIN=`RAIN`
- SYA: date=`date`, RAIN=`RAIN`
- YCA: date=`date`, RAIN=`RAIN`

RAIN QC 检查包括窗口内天气日期完整性、RAIN 缺失、负值、极端值和全 0 情况。

异常 RAIN 记录：
无。


## 5. 每个站点-年份的完整生育期降雨
| station_code   |   year |   harvest_window_days |   harvest_window_rain_mm |   mature_window_rain_mm |   post_mature_window_rain_mm |   rain_difference_ratio |
|:---------------|-------:|----------------------:|-------------------------:|------------------------:|-----------------------------:|------------------------:|
| FQA            |   2008 |                   103 |                    221.8 |                   221.8 |                          0   |                  0      |
| FQA            |   2010 |                   109 |                    362.3 |                   353.4 |                          8.9 |                  0.0246 |
| HLA            |   2007 |                   152 |                    345.6 |                   339.4 |                          6.2 |                  0.0179 |
| HLA            |   2011 |                   142 |                    479.2 |                   479.2 |                          0   |                  0      |
| HLA            |   2009 |                   154 |                    489.5 |                   486.9 |                          2.6 |                  0.0053 |
| LCA            |   2010 |                   114 |                    288.5 |                   288.5 |                          0   |                  0      |
| LCA            |   2011 |                   105 |                    300.9 |                   300.9 |                          0   |                  0      |
| LCA            |   2008 |                   106 |                    357.1 |                   357.1 |                          0   |                  0      |
| LCA            |   2009 |                   102 |                    417.1 |                   417.1 |                          0   |                  0      |
| SYA            |   2014 |                   148 |                    331.8 |                   329.8 |                          2   |                  0.006  |
| SYA            |   2015 |                   158 |                    424.1 |                   420.1 |                          4   |                  0.0094 |
| SYA            |   2012 |                   132 |                    649.4 |                   649.4 |                          0   |                  0      |
| YCA            |   2014 |                   108 |                    211.7 |                   210.7 |                          1   |                  0.0047 |
| YCA            |   2008 |                   101 |                    346.9 |                   345   |                          1.9 |                  0.0055 |

## 6. 成熟到收获期间降雨影响
成熟窗口和收获窗口的差值即成熟后到收获期的降雨。若差值很小，使用成熟期或收获期作为窗口终点对降雨排序影响有限；若差值较大，后续分类应优先使用收获窗口。

## 7. 真实年份降雨排序
| station_code   |   year |   harvest_window_rain_mm | observed_rain_rank   | observed_rain_label               | can_use_relative_low_mid_high   | recommended_next_step                                                          |
|:---------------|-------:|-------------------------:|:---------------------|:----------------------------------|:--------------------------------|:-------------------------------------------------------------------------------|
| FQA            |   2008 |                    221.8 | rank_1_lower         | observed_lower_rain_year          | limited                         | historical_weather_substitution_scenario;rainfall_scaling_sensitivity_scenario |
| FQA            |   2010 |                    362.3 | rank_2_higher        | observed_higher_rain_year         | limited                         | historical_weather_substitution_scenario;rainfall_scaling_sensitivity_scenario |
| HLA            |   2007 |                    345.6 | rank_1               | observed_low_rain_year            | yes                             | observed_year_smoke_test                                                       |
| HLA            |   2011 |                    479.2 | rank_2               | observed_mid_rain_year            | yes                             | observed_year_smoke_test                                                       |
| HLA            |   2009 |                    489.5 | rank_3               | observed_high_rain_year           | yes                             | observed_year_smoke_test                                                       |
| LCA            |   2010 |                    288.5 | rank_1               | observed_low_rain_year            | yes                             | observed_year_smoke_test                                                       |
| LCA            |   2011 |                    300.9 | rank_2               | observed_mid_rain_year            | yes                             | observed_year_smoke_test                                                       |
| LCA            |   2008 |                    357.1 | rank_3               | observed_intermediate_rain_year_3 | yes                             | observed_year_smoke_test                                                       |
| LCA            |   2009 |                    417.1 | rank_4               | observed_high_rain_year           | yes                             | observed_year_smoke_test                                                       |
| SYA            |   2014 |                    331.8 | rank_1               | observed_low_rain_year            | yes                             | observed_year_smoke_test                                                       |
| SYA            |   2015 |                    424.1 | rank_2               | observed_mid_rain_year            | yes                             | observed_year_smoke_test                                                       |
| SYA            |   2012 |                    649.4 | rank_3               | observed_high_rain_year           | yes                             | observed_year_smoke_test                                                       |
| YCA            |   2014 |                    211.7 | rank_1_lower         | observed_lower_rain_year          | limited                         | historical_weather_substitution_scenario;rainfall_scaling_sensitivity_scenario |
| YCA            |   2008 |                    346.9 | rank_2_higher        | observed_higher_rain_year         | limited                         | historical_weather_substitution_scenario;rainfall_scaling_sensitivity_scenario |

## 8. 是否可以称为相对低雨/中雨/高雨
本阶段默认不使用 dry / normal / wet 命名。只有至少 3 个实测年份且站内降雨跨度较明显时，才建议作为 observed experimental years 内部的相对低雨/中雨/高雨候选，不代表 2000-2022 长期气候分位数。

- FQA: limited
- HLA: yes
- LCA: yes
- SYA: yes
- YCA: limited

## 9. 与 Phase 2b 代表年份对比
Phase 2b 使用 QC 后天气和 `planting + 150 days` fallback 选择长期历史代表年；Phase 2c 只使用已有实测物候年份。因此 Phase 2c 不直接继承 Phase 2b 代表年，也不把 2-4 年真实试验记录强行扩展到 2000-2022 全部历史年份。

- HLA: Phase 2b=[2009, 2021, 2022], Phase 2c observed=[2007, 2009, 2011], overlap=[2009]
- SYA: Phase 2b=[2006, 2008, 2016], Phase 2c observed=[2012, 2014, 2015], overlap=[]
- LCA: Phase 2b=[2005, 2013, 2017], Phase 2c observed=[2008, 2009, 2010, 2011], overlap=[]
- YCA: Phase 2b=[2005, 2007, 2016], Phase 2c observed=[2008, 2014], overlap=[]
- FQA: Phase 2b=[2005, 2015, 2020], Phase 2c observed=[2008, 2010], overlap=[]

## 10. 推荐下一步 smoke test 年份
建议先对通过 QC 的实测年份做 NullAgent、fixed_low_N、fixed_medium_N、fixed_high_N smoke test，不直接进入 PPO。

| station_code   |   year | observed_rain_label               |   harvest_window_rain_mm | recommended_for_smoke_test   |
|:---------------|-------:|:----------------------------------|-------------------------:|:-----------------------------|
| FQA            |   2008 | observed_lower_rain_year          |                    221.8 | yes                          |
| FQA            |   2010 | observed_higher_rain_year         |                    362.3 | yes                          |
| HLA            |   2007 | observed_low_rain_year            |                    345.6 | yes                          |
| HLA            |   2011 | observed_mid_rain_year            |                    479.2 | yes                          |
| HLA            |   2009 | observed_high_rain_year           |                    489.5 | yes                          |
| LCA            |   2010 | observed_low_rain_year            |                    288.5 | yes                          |
| LCA            |   2011 | observed_mid_rain_year            |                    300.9 | yes                          |
| LCA            |   2008 | observed_intermediate_rain_year_3 |                    357.1 | yes                          |
| LCA            |   2009 | observed_high_rain_year           |                    417.1 | yes                          |
| SYA            |   2014 | observed_low_rain_year            |                    331.8 | yes                          |
| SYA            |   2015 | observed_mid_rain_year            |                    424.1 | yes                          |
| SYA            |   2012 | observed_high_rain_year           |                    649.4 | yes                          |
| YCA            |   2014 | observed_lower_rain_year          |                    211.7 | yes                          |
| YCA            |   2008 | observed_higher_rain_year         |                    346.9 | yes                          |

## 11. 后续情景方案建议
- 对真实年份梯度不足或只有 2 年记录的站点，建议设计 `historical_weather_substitution_scenario`：固定真实试验年的品种、土壤初始条件和管理方案，替换为同站点历史低雨/中雨/高雨年份天气，并明确标记为虚拟天气替换情景。
- 如需机制敏感性分析，建议设计 `rainfall_scaling_sensitivity_scenario`：以真实试验年份天气为 baseline，构造 RAIN x 0.6、0.8、1.0、1.2、1.4，只作为降雨敏感性分析。

## 12. 本阶段生成的文件
- `data/observed_phenology_dates.csv`
- `data/observed_phenology_dates_standardized.csv`
- `Leave_One_experiments/year_classification/observed_phenology_rainfall_diagnosis.csv`
- `Leave_One_experiments/year_classification/observed_phenology_rainfall_rank_by_station.csv`
- `docs/phase2c_observed_phenology_rainfall_diagnosis.md`
- `docs/phase2c_observed_phenology_rainfall_diagnosis.pptx`
- `Leave_One_experiments/figures/observed_phenology_rainfall_diagnosis/phase2c_harvest_window_rain_by_station_year.png`
- `Leave_One_experiments/figures/observed_phenology_rainfall_diagnosis/phase2c_mature_vs_harvest_window_rain.png`
- `Leave_One_experiments/figures/observed_phenology_rainfall_diagnosis/phase2c_planting_to_harvest_days.png`
- `Leave_One_experiments/figures/observed_phenology_rainfall_diagnosis/phase2b_fallback_representative_years.png`

## 13. 尚未解决的问题和风险
- 部分站点只有 2 年真实试验记录，不能构成完整低雨/中雨/高雨梯度。
- 实测年份排序是站内 observed years 的相对排序，不等同长期气候 dry/normal/wet 分类。
- 后续 smoke test 应先使用固定策略验证环境、天气、管理文件是否能稳定跑通，再进入 PPO。
