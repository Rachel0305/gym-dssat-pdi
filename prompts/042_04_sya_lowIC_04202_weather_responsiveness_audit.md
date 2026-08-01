# 042_04 SYA lowIC 042_02 天气/胁迫响应性审计

## 目的

`042_02` 的 75K checkpoint 指标表现较好，但逐年措施看起来仍然像固定模板。本任务不训练、不跑 DSSAT，只审计：

1. 动作是否在不同年份之间有多样性；
2. 灌溉/施氮动作是否与天气、土壤水分、SWFAC/NSTRES 胁迫相关；
3. 75K 的好结果是否可以被解释为“天气响应策略”，还是更接近“固定模板策略”。

## 输入

- `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/daily_outputs/SYA/*_daily.csv`
- `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/evaluation/041_03_checkpoint_validation_summary.csv`
- `042_02` 的配置和 SYA lowIC weather 表。

## 审计指标

### 一、动作多样性

- 每个 checkpoint、每个年份的非零灌溉/施氮日期和剂量；
- 不同年份之间 action sequence 的 Hamming distance；
- 每个 checkpoint 的 unique action sequence 数量；
- 灌溉/施氮 event DAP 的跨年标准差。

如果 unique sequence 很少、event DAP 标准差接近 0，则判定为模板化。

### 二、天气/胁迫关联

对每个 checkpoint 比较：

- 灌溉日 vs 非灌溉日：
  - past7 rain；
  - future7 rain；
  - SWFAC；
  - NSTRES；
  - DAP。
- 施氮日 vs 非施氮日：
  - NSTRES；
  - SWFAC；
  - DAP；
  - past7/future7 rain。

如果动作主要由 DAP 区分，而不是由 weather/stress 区分，则判定为阶段模板策略。

## 边界

- 本任务只读已有结果；
- 不训练；
- 不运行 DSSAT；
- 不修改 checkpoint；
- 不把 75K 直接指定为正式结果。

