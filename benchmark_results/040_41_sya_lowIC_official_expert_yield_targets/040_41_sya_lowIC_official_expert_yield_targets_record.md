# 040_21 SYA lowIC 四情景基线重建记录

## 结论边界

- 本任务只重建四情景基线，不训练 PPO/DQN。
- 目的：给 040_20 lowIC PPO 结果提供同源、同条件的 null / recorded farmer template / official expert / DSSAT auto 对照。
- 如果 rendered template 不是 LOWIC，本任务结果不得用于 lowIC 比较。

## 运行配置

- mode：`full`
- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 源 MZX：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/SY/CNSY1201.MZX`
- 源 MZX 家族：`LOWIC`
- 源检查状态：`pass`
- 耗时：110.2 秒

## 年份范围

| station_code | site | year | split |
| --- | --- | --- | --- |
| SYA | SY | 2005 | train |
| SYA | SY | 2006 | train |
| SYA | SY | 2007 | train |
| SYA | SY | 2008 | train |
| SYA | SY | 2009 | train |
| SYA | SY | 2010 | train |
| SYA | SY | 2011 | train |
| SYA | SY | 2012 | train |
| SYA | SY | 2013 | train |
| SYA | SY | 2014 | validation |
| SYA | SY | 2015 | validation |
| SYA | SY | 2016 | validation |
| SYA | SY | 2017 | validation |
| SYA | SY | 2018 | validation |
| SYA | SY | 2019 | validation |
| SYA | SY | 2020 | validation |
| SYA | SY | 2021 | validation |
| SYA | SY | 2022 | validation |
| SYA | SY | 2023 | validation |

## 覆盖状态

| scenario | status | n |
| --- | --- | --- |
| official_extension_expert | ok | 19 |

## rendered template IC 家族

| template_family_checked | n |
| --- | --- |
| LOWIC | 19 |

## 情景均值预览

| scenario | n | mean_yield | mean_irrigation | mean_nitrogen | mean_wp_et | mean_pfp_n | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_extension_expert | 19 | 10100.8953 | 266.0 | 297.0 | 2.1468 | 34.0158 | 0.0 | 0.017 |

## 非 ok 记录

无记录。

## 管理事件审计非 ok 记录

无记录。

## 输出文件

- summary：`benchmark_results/040_41_sya_lowIC_official_expert_yield_targets/evaluation/040_21_baseline_summary.csv`
- daily：`benchmark_results/040_41_sya_lowIC_official_expert_yield_targets/evaluation/040_21_baseline_daily.csv`
- management audit：`benchmark_results/040_41_sya_lowIC_official_expert_yield_targets/evaluation/040_21_management_event_audit.csv`
- manifest：`benchmark_results/040_41_sya_lowIC_official_expert_yield_targets/evaluation/040_21_coverage_manifest.csv`
- render check：`benchmark_results/040_41_sya_lowIC_official_expert_yield_targets/evaluation/040_21_render_check.csv`
