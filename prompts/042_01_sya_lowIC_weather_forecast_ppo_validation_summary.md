# 042_01 SYA lowIC 天气/预报 observation PPO 验证结果后处理

## 目的

对已经完成的 `042_00_sya_lowIC_weather_forecast_observation_maskableppo` 结果做纯后处理汇总：

1. 不重新训练；
2. 不重新运行 DSSAT；
3. 只读取 `042_00` 的验证结果和 `040_21` 的可信 lowIC 四基线；
4. 逐年计算 PPO 相对四基线最高值的差距；
5. 汇总每个 checkpoint 在 2014--2023 验证年上的胜出数量。

## 输入

- PPO 结果：
  - `benchmark_results/042_00_sya_lowIC_weather_forecast_observation_maskableppo/evaluation/042_00_checkpoint_validation_summary.csv`
- 四基线：
  - `benchmark_results/040_21_sya_lowIC_four_baseline_rebuild/evaluation/040_21_baseline_summary.csv`

## 指标

四基线包括：

- null
- recorded farmer
- DSSAT auto
- official extension expert

对每个验证年，分别计算：

- `gap_yield_vs_four_max = PPO_final_grnwt - max(four_baseline_yield)`
- `gap_wp_et_vs_four_max = PPO_WP_ET - max(four_baseline_WP_ET)`
- `gap_pfp_n_vs_four_max = PPO_PFP_N - max(four_baseline_PFP_N)`

并给出：

- `yield_win_four`
- `wp_et_win_four`
- `pfp_n_win_four`
- `any_metric_win_four`
- `all3_metric_win_four`

## 判读边界

这是后处理汇总，不改变 `042_00` 科学结果。若天气/预报 observation 分支表现不佳，结论只限于：

> 在当前 100K、无 teacher warm-start、原 040_40 约束和 reward 保持不变的配置下，直接加入 rain/tmin/短期天气窗口没有带来更好的验证表现。

不得据此宣称“天气预报无用”。
