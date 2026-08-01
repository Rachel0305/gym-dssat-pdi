# 042_13 SYA lowIC binary-timing PPO 25K 天气/胁迫响应性审计

## 背景

042_12 显示 042_11 的 25K binary-timing PPO checkpoint 在指标上表现较好：

- 10/10 验证年份至少一项指标超过四基线最高值；
- 7/10 产量超过四基线最高值；
- 9/10 WP_ET 超过四基线最高值；
- 10/10 PFP_N 超过四基线最高值；
- 10 年验证集存在 7 种动作签名。

但导师关心的不只是终值指标，还包括 PPO 的决策是否能解释：它是不是根据天气、土壤水分/氮胁迫状态改变灌溉和施肥时机。

## 本任务目标

本任务不训练、不调参、不换 checkpoint，只读取 042_11/042_12 已有结果，审计 25K checkpoint 的动作响应性：

1. 统计 2014–2023 每一年灌溉/施氮事件的 DAP；
2. 检查动作序列是否跨年变化；
3. 统计每个灌溉事件附近的天气和水分胁迫背景：
   - 事件前 7 天降雨；
   - 事件后 7 天降雨；
   - 事件前 7 天最大 WSPD；
   - 事件后 7 天最大 WSPD；
4. 统计每个施氮事件附近的氮胁迫背景：
   - 事件前 7 天最大 NSTD；
   - 事件后 7 天最大 NSTD；
5. 输出事件 DAP heatmap/scatter 图，判断是否存在固定模板成分。

## 预注册判定

- A：强响应  
  动作序列跨年明显变化，且早期固定动作不主导；灌溉/施氮时点与雨量/胁迫背景有明确对应。

- B：部分响应、仍有模板成分  
  动作序列跨年有变化，但存在固定早期或固定后期动作；指标可以继续作为候选，但需要在报告中诚实说明“当前 PPO 更像是在有限几套时序模板中切换”。

- C：模板化失败  
  动作序列几乎完全固定，或事件时点与天气/胁迫背景没有任何可解释关联。

## 输出

- `docs/042_13_sya_lowIC_binary_timing_25k_weather_response_audit_record.md`
- `benchmark_results/042_13_sya_lowIC_binary_timing_25k_weather_response_audit/tables/`
- `benchmark_results/042_13_sya_lowIC_binary_timing_25k_weather_response_audit/figures/`

