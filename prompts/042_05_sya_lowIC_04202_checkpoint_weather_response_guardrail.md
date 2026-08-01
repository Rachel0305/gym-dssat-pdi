# 042_05 SYA lowIC 042_02 checkpoint 天气响应性 guardrail 审计

## 背景

042_02 在 SYA lowIC 输入下加入了：

- 全部观测归一化；
- rain/tmin、过去 7 天降雨、未来 7 天降雨/温度等完美天气预报特征；
- 氮胁迫过程惩罚；
- 041_04 balanced teacher warm-start。

042_02 rerun100k 的 75K checkpoint 在终点指标上较好：10 个验证年份均至少有一项指标超过四情景最高值，且部分年份三项全超。但 042_04 审计显示，75K 的非零水氮管理动作在 2014–2023 年间几乎是同一套固定模板。这说明它的好指标不能直接解释为“模型根据不同年份天气自适应决策”。

因此 042_05 不进入新训练，只做一次纯离线 checkpoint weather-response guardrail 审计。

## 任务性质

- 只读取已有 042_02 rerun100k daily/evaluation CSV。
- 不重新训练 PPO。
- 不运行 DSSAT。
- 不修改 checkpoint。
- 不挑选新的最终 checkpoint。
- 目的不是证明 75K 不可用，而是把“指标表现”和“天气响应性”分开量化，作为下一轮训练或 checkpoint 选择规则的依据。

## 输入

- `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/evaluation/041_03_checkpoint_validation_summary.csv`
- `benchmark_results/042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k/daily_outputs/SYA/*_daily.csv`

## 审计指标

### 1. 终点指标表现

按 checkpoint 汇总：

- `any_metric_win_years`
- `all3_win_years`
- mean yield
- mean WP_ET
- mean PFP_N
- mean/maximum water stress
- mean/maximum nitrogen stress

### 2. 动作多样性

按 checkpoint 统计：

- 10 个验证年份中非零动作 signature 的唯一数量；
- 总灌溉量、总施氮量的跨年标准差；
- 灌溉事件数、施肥事件数的跨年标准差；
- 非零灌溉事件 DAP 的跨年变异；
- 非零施肥事件 DAP 的跨年变异。

### 3. 天气/胁迫响应性

对每个 checkpoint 统计：

- 灌溉日与非灌溉日相比，过去 7 天降雨、未来 7 天降雨、SWFAC 是否有明显差异；
- 施肥日与非施肥日相比，NSTRES、作物生长量、DAP 是否有明显差异；
- 使用标准化均值差 `standardized_difference`，只作为相关性审计，不作为因果证据。

### 4. 候选 checkpoint guardrail

本任务预注册一个“是否值得作为下一轮候选”的审计门槛：

- 终点指标门槛：`any_metric_win_years == 10` 且 `all3_win_years >= 5`；
- 非模板化门槛：`unique_nonzero_action_signatures >= 3`；
- 灌溉时机响应门槛：`mean_irrigation_event_dap_sd >= 3` 或 `irrigation_future7_rain_abs_std_diff >= 0.2`；
- 施肥时机响应门槛：`mean_nitrogen_event_dap_sd >= 3` 或 `nitrogen_nstres_abs_std_diff >= 0.2`。

说明：

- 这些门槛不是最终论文判据，只是下一步训练/选点前的工程审计标准。
- 若 75K 指标好但响应性不达标，应记录为“高指标但模板化候选”，不能直接包装成天气自适应策略。
- 若没有任何 checkpoint 同时满足终点指标与响应性门槛，下一步应修改训练目标或 checkpoint selection guardrail，而不是事后硬挑 75K。

## 输出

- `benchmark_results/042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail/tables/042_05_checkpoint_response_guardrail_summary.csv`
- `benchmark_results/042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail/tables/042_05_action_sequence_by_year_checkpoint.csv`
- `benchmark_results/042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail/tables/042_05_event_position_variability.csv`
- `benchmark_results/042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail/tables/042_05_action_feature_association.csv`
- `benchmark_results/042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail/figures/042_05_checkpoint_metric_response_guardrail.png`
- `docs/042_05_sya_lowIC_04202_checkpoint_weather_response_guardrail_record.md`

## 停止条件

- 如果输入 daily/evaluation 文件缺失，停止，不补跑训练。
- 如果天气重建失败，停止，不根据不完整结果判断。
- 如果没有 checkpoint 通过响应性 guardrail，记录阴性结果，不现场修改门槛。
