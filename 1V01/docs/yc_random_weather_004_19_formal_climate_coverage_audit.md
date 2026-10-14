# YC正式气候覆盖审计：EXACT78主分析与FULL80敏感性

## 结论摘要

- 主网格共850格，其中24格标记为`INSUFFICIENT`（EXACT78原因：OBSERVED_METRIC_UNAVAILABLE：24格）。其中Observed无日记录的阶段为无，对应0格；其余缺失来自指标在短窗口内不可定义。有效格中，EXACT78有**69格（8.4%）**超出min/max；FULL80有**66格（8.0%）**超出。两条口径的总体覆盖等级均为**PARTIAL**，稳健总体结论为**PARTIAL**。
- EXACT78与FULL80共有6格分类改变。2014、2016、2019整季降雨在EXACT78中低于最小值；加入provisional seed后FULL80的下界被拉低，三年都进入min/max范围，但仍处于P5以下。因此这是对极端min/max的敏感性变化，不是已验证的历史训练域覆盖。
- 另有不依赖1003/1036的重复低SRAD最小值缺口：整季 `srad_min` 在EXACT78有10/10年、FULL80有10/10年超出min/max；held-out整季最低SRAD范围为4.60–5.81 MJ m⁻² d⁻¹。 training两条口径和held-out都没有覆盖Observed低值，是当前最明确的稳健生成分布问题。
- 决策：`robust_tail_gap_type = SYSTEMATIC_GENERATOR_GAP`；`next_weather_action = REVISE_WGEN_GENERATOR`。本任务只建议后续先审查WGEN低SRAD尾部表示，不生成天气、不训练PPO。

## 机器结论

```text
exact78_observed_coverage = PARTIAL
full80_candidate_coverage = PARTIAL
robust_coverage_conclusion = PARTIAL
robust_tail_gap_type = SYSTEMATIC_GENERATOR_GAP
next_weather_action = REVISE_WGEN_GENERATOR
```

EXACT78是正式历史训练域主分析。FULL80仅为候选敏感性口径，不能称为80/80 exact historical reconstruction。seed1003、1036仍为provisional；held-out 1081–1100继续隔离，不回流training。

## 主要结果

### 2014、2016、2019整季降雨

| 年份 | Observed (mm) | EXACT78 min–max | FULL80 min–max | EXACT78 P5–P95 | FULL80 P5–P95 | FULL80下界seed | EXACT78分类 | FULL80分类 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2014 | 210.7 | 249.4–1052.5 | 188.9–1052.5 | 277.8–771.8 | 275.4–768.3 | 1036 | OUTSIDE_SUPPORT | TAIL_BUT_INSIDE_SUPPORT |
| 2016 | 203.4 | 249.4–1052.5 | 188.9–1052.5 | 277.8–771.8 | 275.4–768.3 | 1036 | OUTSIDE_SUPPORT | TAIL_BUT_INSIDE_SUPPORT |
| 2019 | 198.7 | 249.4–1052.5 | 188.9–1052.5 | 277.8–771.8 | 275.4–768.3 | 1036 | OUTSIDE_SUPPORT | TAIL_BUT_INSIDE_SUPPORT |

在provisional样本中，seed **1036**决定了FULL80最低降雨下界。2014/2016/2019的EXACT78分类为`OUTSIDE_SUPPORT`，FULL80分类为`TAIL_BUT_INSIDE_SUPPORT`；FULL80虽扩展min/max支持，但三年仍低于两套口径的P5。004_18A观察到的经验百分位变化约1.25个百分点。故不得把FULL80的边界纳入当成历史已覆盖的证据。

全部分类变化与候选seed归因：

| 年份 | 范围/阶段 | 指标 | EXACT78 | FULL80 | 单独足以改变分类的候选seed |
|---:|---|---|---|---|---|
| 2014 | 整季 | rainfall_total | OUTSIDE_SUPPORT | TAIL_BUT_INSIDE_SUPPORT | seed1036 |
| 2014 | 整季 | rx5day | TAIL_BUT_INSIDE_SUPPORT | WELL_COVERED | seed1003 |
| 2016 | 整季 | rainfall_total | OUTSIDE_SUPPORT | TAIL_BUT_INSIDE_SUPPORT | seed1036 |
| 2019 | 整季 | rainfall_total | OUTSIDE_SUPPORT | TAIL_BUT_INSIDE_SUPPORT | seed1036 |
| 2014 | DAP31_60 | rx1day | TAIL_BUT_INSIDE_SUPPORT | WELL_COVERED | seed1036 |
| 2017 | DAP61_90 | srad_max | TAIL_BUT_INSIDE_SUPPORT | WELL_COVERED | seed1036 |

其中2014整季Rx5day的P5分类改变由seed1003单独即可触发；DAP31–60的Rx1day与DAP61–90的SRAD max两项变化由seed1036单独即可触发。seed1036单独扩展季节总降雨的min/max支持；以上FULL80影响均为敏感性，不升级为已核验的历史覆盖。

### 三个重点年份的分阶段降雨

| 年份 | DAP阶段 | Observed降雨(mm) | EXACT78范围 | FULL80范围 | EXACT78分类 | FULL80分类 | held-out是否包含 |
|---:|---|---:|---:|---:|---|---|---|
| 2014 | DAP0_30 | 96.3 | 1.4–324.1 | 1.4–324.1 | WELL_COVERED | WELL_COVERED | True |
| 2014 | DAP31_60 | 45.9 | 33.3–539.4 | 33.3–539.4 | TAIL_BUT_INSIDE_SUPPORT | TAIL_BUT_INSIDE_SUPPORT | False |
| 2014 | DAP61_90 | 29.3 | 0.0–497.4 | 0.0–497.4 | WELL_COVERED | WELL_COVERED | True |
| 2014 | DAP_gt90 | 39.2 | 0.0–75.9 | 0.0–75.9 | WELL_COVERED | WELL_COVERED | True |
| 2016 | DAP0_30 | 78.4 | 1.4–324.1 | 1.4–324.1 | WELL_COVERED | WELL_COVERED | True |
| 2016 | DAP31_60 | 96.0 | 33.3–539.4 | 33.3–539.4 | WELL_COVERED | WELL_COVERED | True |
| 2016 | DAP61_90 | 29.0 | 0.0–497.4 | 0.0–497.4 | WELL_COVERED | WELL_COVERED | True |
| 2016 | DAP_gt90 | 0.0 | 0.0–75.9 | 0.0–75.9 | WELL_COVERED | WELL_COVERED | True |
| 2019 | DAP0_30 | 36.5 | 1.4–324.1 | 1.4–324.1 | WELL_COVERED | WELL_COVERED | True |
| 2019 | DAP31_60 | 140.4 | 33.3–539.4 | 33.3–539.4 | WELL_COVERED | WELL_COVERED | True |
| 2019 | DAP61_90 | 21.8 | 0.0–497.4 | 0.0–497.4 | WELL_COVERED | WELL_COVERED | False |
| 2019 | DAP_gt90 | 0.0 | 0.0–75.9 | 0.0–75.9 | WELL_COVERED | WELL_COVERED | True |

该表用于辨别整季低雨量对应单一阶段还是多个阶段的降雨差异。DAP按冻结的season alignment定义，不把WGEN的参考DATE误当作作物日历日期。

### 稳健缺口与held-out检查

预先固定的重复缺口规则：EXACT78与FULL80至少8/10年同方向超出，且held-out也至少8年未覆盖。达到规则的指标：**season ALL srad_min：10年超出，held-out同样不覆盖10年；stage DAP0_30 srad_min：8年超出，held-out同样不覆盖8年；stage DAP31_60 srad_min：10年超出，held-out同样不覆盖10年；stage DAP61_90 srad_min：8年超出，held-out同样不覆盖8年**

held-out季节降雨范围为306.6–881.4 mm（中位数515.2）；2014、2016、2019均低于held-out最小值。10个Observed年份中共有4年超出held-out季节降雨范围。KS检验仅作探索，不能证明分布等价。

Observed低SRAD最小值超出EXACT78、FULL80及held-out支持。对该项而言，证据更符合WGEN当前生成分布没有到达低值区域，而非仅仅80个seed随机漏抽；但不能据此推断所有天气变量都存在生成器问题。held-out比较仅为描述分析，KS不显著不代表分布相同。

代表性稳健越界格：

| 范围 | 阶段 | 指标 | 年份 | Observed | EXACT78范围 | FULL80范围 | held-out包含？ |
|---|---|---|---:|---:|---:|---:|---|
| season | ALL | hot_days_gt30 | 2023 | 77 | 45–73 | 45–73 | False |
| season | ALL | hot_days_gt35 | 2015 | 3 | 4–31 | 4–31 | False |
| season | ALL | hot_days_gt35 | 2020 | 2 | 4–31 | 4–31 | False |
| season | ALL | hot_dry_days_gt32_rain_le0 | 2018 | 51 | 23–48 | 23–48 | False |
| season | ALL | hotspell_dry_overlap_days_ge3 | 2018 | 50 | 17–46 | 17–46 | False |
| season | ALL | longest_hot_spell | 2018 | 31 | 4–22 | 4–22 | False |
| season | ALL | lowrain7_hot_days_le10mm | 2019 | 35 | 0–29 | 0–29 | False |
| season | ALL | rainfall_days | 2016 | 11 | 13–40 | 13–40 | False |
| season | ALL | rainfall_days | 2019 | 12 | 13–40 | 13–40 | False |
| season | ALL | srad_mean | 2018 | 19.5 | 14.9–18.6 | 14.9–18.6 | False |
| season | ALL | srad_mean | 2023 | 20.7 | 14.9–18.6 | 14.9–18.6 | False |
| season | ALL | srad_min | 2014 | 2.4 | 4.44–6.36 | 4.44–6.36 | False |
| season | ALL | srad_min | 2015 | 2.9 | 4.44–6.36 | 4.44–6.36 | False |
| season | ALL | srad_min | 2016 | 2.5 | 4.44–6.36 | 4.44–6.36 | False |

### 复合高温—干旱条件

复合指标阈值已在计算前固定：同日高温干燥（`Tmax>32°C`且`RAIN<=0`）；高温日与至少连续3天无雨段重叠；以及当日及前6天累计雨量不超过10 mm的高温日数。DAP阶段内滚动窗口重新起算。全部年份和两条training口径见`compound_extreme_support.csv`；这些补充指标不替代850格核心网格。

## 逐项回答

1. Observed 2014–2023并非全部处于历史训练天气支持内；两种口径按预设规则均为`PARTIAL`。
2. 2014/2016/2019整季低降雨分类依赖provisional seed 1036 扩展min/max；P5/P95仍在尾部，因此不属于稳健的历史覆盖证据。分阶段降雨见上表，并需与温度、辐射、复合指标一起判断。
3. 排除1003/1036后仍有稳健缺口，尤其整季最低SRAD在Observed十年均超出两套training边界，held-out也无法覆盖。其他stage、温度、降雨和复合指标的逐项结果见CSV。
4. held-out与training逐项分布比较见`heldout_distribution_check.csv`。有限样本可以表明现有WGEN realization是否实际达到Observed尾部，但不能证明总体分布等价。
5. 当前问题不只是“80个抽样是否太少”：低SRAD最小值呈training与held-out共同的稳健缺口，更支持先检查WGEN生成分布/辐射尾部，而不是直接随机扩到160。此结论是天气分布诊断建议，不是对作物表现的因果结论。

## 方法与证据边界

1. 100个canonical文件均按`weather_archive_manifest_verified.csv`校验；计算前核实78 exact、2 provisional、20 held-out的划分。SHA和逐文件记录见`input_integrity.csv`。
2. season对齐与17项核心指标直接调用冻结的004_18A runner；Observed使用相同官方WTH、FileX treatment-1 PDATE和seed5 episode最大DAP流程。
3. 经验百分位为`100 × mean(training metric <= observed)`；P5/P95采用与004_18A相同的pandas线性分位数。越界距离为Observed到最近min/max边界的绝对距离，范围内记0。
4. 核心网格为17项指标×整季和4个DAP阶段×10年，共850格。Observed某阶段无日记录、该指标在Observed阶段不可定义，或training指标无有效样本时记`INSUFFICIENT`，不填零；排除于coverage比例分母。EXACT78原因分布：OBSERVED_METRIC_UNAVAILABLE：24格。FULL80原因分布：OBSERVED_METRIC_UNAVAILABLE：24格。两条口径各有826格可计算。GOOD=0个min/max越界；PARTIAL=1–84；POOR=85格及以上。复合指标单独报告。
5. held-out双样本KS p值为探索性结果，不作等价性检验，也未作多重检验校正。本任务未运行WGEN、DSSAT或PPO。

## 图表

- Figure 1: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure1_seasonal_rainfall_support.png`
- Figure 2: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure2_dryspell_rx1_rx5_support.png`
- Figure 3: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure3_dap_stage_rainfall_support.png`
- Figure 4: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure4_temperature_hotday_support.png`
- Figure 5: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure5_srad_support.png`
- Figure 6: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure6_observed_year_metric_support_heatmap.png`
- Figure 7: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure7_exact78_full80_classification_changes.png`
- Figure 8: `results/yc_random_weather_ppo/004_19_yc_formal_climate_coverage_audit/figures/figure8_key_year_weather_profiles.png`

## 机器可读文件

`weather_metrics_exact78.csv`、`weather_metrics_full80.csv`、`weather_metrics_heldout20.csv`、`observed_metrics.csv`、`observed_percentiles_exact78.csv`、`observed_percentiles_full80.csv`、`observed_support_comparison.csv`、`stage_support_comparison.csv`、`compound_extreme_support.csv`、`heldout_distribution_check.csv`、`robust_gap_inventory.csv`、`insufficient_data_cells.csv`和`decision.json`。
