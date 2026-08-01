# 042_01 SYA lowIC 天气/预报 observation PPO 验证结果后处理记录

## 任务性质

- 只做后处理汇总；没有重新训练；没有重新运行 DSSAT。
- 输入 PPO 结果来自 `042_00`。
- 四基线来自 `040_21` lowIC 可信基线重建。
- 注意：`042_00` 的验证摘要没有保存 ET/ETCP，因此本次不能重新计算 PPO 的 WP_ET；WP_ET 胜出列标记为不可用。

## observation 预检

- observation smoke audit 通过：True
- 原 observation 维度：25
- 增强后 observation 维度：30
- 新增变量：rain_today_mm, tmin_today_c, rain_past7_mm, rain_future7_mm, tmean_future7_c

## checkpoint 汇总

| station_code   | site   |   checkpoint_step |   validation_years |   mean_final_grnwt |   mean_total_irrigation |   mean_total_n |   mean_pfp_n |   yield_win_four_count |   pfp_n_win_four_count |   any_metric_win_four_partial_count |   all_available_metric_win_four_count |   mean_gap_yield_vs_four_max |   mean_gap_pfp_n_vs_four_max |
|:---------------|:-------|------------------:|-------------------:|-------------------:|------------------------:|---------------:|-------------:|-----------------------:|-----------------------:|------------------------------------:|--------------------------------------:|-----------------------------:|-----------------------------:|
| SYA            | SY     |             25000 |                 10 |            8755.75 |                   177   |            240 |      36.4823 |                      4 |                      8 |                                   8 |                                     4 |                     -1317.71 |                     2.56229  |
| SYA            | SY     |             50000 |                 10 |            5649.63 |                   124.5 |             92 |      67.4758 |                      0 |                      8 |                                   8 |                                     0 |                     -4423.84 |                    33.5558   |
| SYA            | SY     |             75000 |                 10 |            7741.21 |                   150   |            228 |      34.3134 |                      2 |                      6 |                                   6 |                                     2 |                     -2332.26 |                     0.393352 |
| SYA            | SY     |            100000 |                 10 |            6838.43 |                   123   |            200 |      34.1922 |                      2 |                      5 |                                   5 |                                     2 |                     -3235.03 |                     0.272168 |

## 初步判读

- 平均产量最高 checkpoint：25000，平均产量 8755.75 kg/ha。
- 可用指标中至少一项胜出最多 checkpoint：25000，8/10。
- 但 WP_ET 在当前 042_00 摘要中不可直接判定；若要完整三指标比较，需要补一次只读 daily 输出的 ET 汇总后处理。
- 现有数字已足以说明：直接加入天气/预报变量，在无 teacher warm-start 的 100K PPO 下没有明显优于 040_40/041_04 主线。

## 输出

- enriched: `benchmark_results\042_01_sya_lowIC_weather_forecast_ppo_validation_summary\tables\042_01_ppo_validation_enriched_vs_four_baselines.csv`
- by checkpoint: `benchmark_results\042_01_sya_lowIC_weather_forecast_ppo_validation_summary\tables\042_01_validation_summary_by_checkpoint_vs_four_baselines.csv`
