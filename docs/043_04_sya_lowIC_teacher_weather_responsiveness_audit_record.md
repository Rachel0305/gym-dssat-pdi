# 043_04 SYA lowIC teacher 天气响应性审计记录

## 一句话结论

- 分支：`A_teacher_has_more_weather_responsive_signal_than_04302_ppo`
- 本任务不训练 PPO、不运行 DSSAT，只读取已有 041_02 lowIC teacher 与 043_02 PPO 输出。
- 目的：判断已有 lowIC teacher 是否比 043_02 PPO 模板更适合作为“天气/胁迫敏感性”的训练信号。

## 核心对比指标

| metric | value | interpretation |
| --- | --- | --- |
| teacher_unique_action_signatures | 9.0 | teacher逐年完整动作序列唯一数；越高越非模板化 |
| teacher_unique_irrigation_signatures | 7.0 | teacher逐年灌溉序列唯一数 |
| teacher_unique_nitrogen_signatures | 4.0 | teacher逐年施氮序列唯一数 |
| teacher_total_irrigation_sd | 34.7275 | teacher逐年总灌溉量标准差 |
| teacher_total_n_sd | 26.533 | teacher逐年总施氮量标准差 |
| teacher_first_irrigation_dap_sd | 18.2551 | teacher首次灌溉DAP标准差 |
| teacher_first_nitrogen_dap_sd | 16.8 | teacher首次施氮DAP标准差 |
| corr_total_irrigation_vs_max_swfac | 0.1223 | 总灌溉与水分胁迫强度相关性；正值说明胁迫重年份用水更多 |
| corr_total_n_vs_max_nstres | -0.3751 | 总施氮与氮胁迫强度相关性 |
| corr_total_irrigation_vs_season_rain | -0.5512 | 总灌溉与季节降雨相关性；负值通常更符合补水直觉 |
| ppo_04302_50k_unique_action_signatures | 1.0 | 043_02 50K PPO验证年动作序列唯一数 |
| ppo_04302_75k_unique_action_signatures | 1.0 | 043_02 75K PPO验证年动作序列唯一数 |

## Teacher 逐年措施摘要

| year | teacher_tier | candidate_id | total_irrigation | total_n | grain_yield_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | max_swfac | max_nstres | irrigation_signature | nitrogen_signature |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | strong_all3 | W240_mid_late__N240_three80 | 240.0 | 240.0 | 12824.0 | 2.54 | 53.4 | 0.0 | 0.1784 | DAP30:45; DAP45:45; DAP60:45; DAP75:30; DAP95:45; DAP110:30 | DAP1:80; DAP43:80; DAP61:80 |
| 2015 | near_miss | W210_drop_dap8__N240_three80 | 210.0 | 240.0 | 11081.0 | 2.27 | 46.2 | 0.0 | 0.0322 | DAP1:45; DAP31:45; DAP38:30; DAP61:45; DAP91:45 | DAP1:80; DAP43:80; DAP61:80 |
| 2016 | strong_all3 | W150_late_saving__N160_two80 | 150.0 | 160.0 | 8052.0 | 1.75 | 50.3 | 0.5226 | 0.1507 | DAP45:30; DAP75:45; DAP95:45; DAP115:30 | DAP1:80; DAP43:80 |
| 2017 | near_miss | W240_mid_late__N240_three80 | 240.0 | 240.0 | 10718.0 | 2.5 | 44.7 | 0.8959 | 0.0167 | DAP30:45; DAP45:45; DAP60:45; DAP75:30; DAP95:45; DAP110:30 | DAP1:80; DAP43:80; DAP61:80 |
| 2018 | strong_all3 | W240_late_balanced__N200_mid_late | 240.0 | 200.0 | 8334.0 | 1.85 | 41.7 | 0.0 | 0.017 | DAP1:30; DAP30:45; DAP60:45; DAP90:45; DAP105:45; DAP120:30 | DAP43:80; DAP61:120 |
| 2019 | near_miss | W225_ppo_like__N240_three80 | 225.0 | 240.0 | 10415.0 | 2.05 | 43.4 | 0.0 | 0.013 | DAP1:45; DAP8:30; DAP31:45; DAP38:30; DAP61:45; DAP91:30 | DAP1:80; DAP43:80; DAP61:80 |
| 2020 | strong_all3 | W240_mid_late__N200_mid_late | 240.0 | 200.0 | 9961.0 | 2.17 | 49.8 | 0.9016 | 0.0162 | DAP30:45; DAP45:45; DAP60:45; DAP75:30; DAP95:45; DAP110:30 | DAP43:80; DAP61:120 |
| 2021 | strong_all3 | W150_late_saving__N240_three80 | 150.0 | 240.0 | 10500.0 | 2.3 | 43.8 | 0.0 | 0.0136 | DAP45:30; DAP75:45; DAP95:45; DAP115:30 | DAP1:80; DAP43:80; DAP61:80 |
| 2022 | near_miss | W195_no_dap8_late_save__N240_two120 | 195.0 | 240.0 | 11305.0 | 2.45 | 47.1 | 0.0 | 0.0122 | DAP1:45; DAP31:45; DAP61:45; DAP91:30; DAP110:30 | DAP1:120; DAP43:120 |
| 2023 | near_miss | W240_ppo_plus_late__N240_three80 | 240.0 | 240.0 | 11005.0 | 2.24 | 45.9 | 0.0 | 0.016 | DAP1:45; DAP8:30; DAP31:45; DAP38:30; DAP61:45; DAP91:45 | DAP1:80; DAP43:80; DAP61:80 |

## Teacher 年份天气/胁迫背景

| year | planting_date | season_rain_0_140_mm | early_rain_0_60_mm | mid_rain_61_100_mm | late_rain_101_140_mm | mean_tmax_0_140_c | mean_tmin_0_140_c | teacher_total_irrigation | teacher_total_n | teacher_first_irrigation_dap | teacher_first_nitrogen_dap | teacher_max_swfac | teacher_max_nstres |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 2014-04-21 | 320.5 | 122.4 | 135.0 | 63.1 | 27.8844 | 16.1064 | 240.0 | 240.0 | 30.0 | 1.0 | 0.0 | 0.1784 |
| 2015 | 2015-04-21 | 420.1 | 193.5 | 135.3 | 91.3 | 27.9064 | 16.4404 | 210.0 | 240.0 | 1.0 | 1.0 | 0.0 | 0.0322 |
| 2016 | 2016-04-20 | 696.9 | 216.6 | 297.1 | 183.2 | 27.366 | 16.5043 | 150.0 | 160.0 | 45.0 | 1.0 | 0.5226 | 0.1507 |
| 2017 | 2017-04-21 | 301.7 | 43.8 | 104.4 | 153.5 | 28.473 | 16.5184 | 240.0 | 240.0 | 30.0 | 1.0 | 0.8959 | 0.0167 |
| 2018 | 2018-04-21 | 515.0 | 186.6 | 134.7 | 193.7 | 27.9135 | 17.5128 | 240.0 | 200.0 | 1.0 | 43.0 | 0.0 | 0.017 |
| 2019 | 2019-04-21 | 598.4 | 129.4 | 133.2 | 335.8 | 27.5433 | 16.7496 | 225.0 | 240.0 | 1.0 | 1.0 | 0.0 | 0.013 |
| 2020 | 2020-04-20 | 560.9 | 127.8 | 70.0 | 363.1 | 27.0255 | 16.7035 | 240.0 | 200.0 | 30.0 | 43.0 | 0.9016 | 0.0162 |
| 2021 | 2021-04-21 | 523.4 | 204.0 | 134.4 | 185.0 | 26.8574 | 16.4426 | 150.0 | 240.0 | 45.0 | 1.0 | 0.0 | 0.0136 |
| 2022 | 2022-04-21 | 670.1 | 145.0 | 384.4 | 140.7 | 26.8532 | 16.3766 | 195.0 | 240.0 | 1.0 | 1.0 | 0.0 | 0.0122 |
| 2023 | 2023-04-21 | 481.9 | 89.2 | 197.1 | 195.6 | 27.7255 | 16.8887 | 240.0 | 240.0 | 1.0 | 1.0 | 0.0 | 0.016 |

## 非零动作日天气样例

| year | dap_before_action | safe_irrigation_mm_action | safe_nitrogen_kg_ha_action | rain_today_mm | rain_past7_mm | rain_future7_mm | swfac | nstres | teacher_tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 1 | 0.0 | 80.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | strong_all3 |
| 2014 | 30 | 45.0 | 0.0 | 19.0 | 34.6 | 30.4 | 0.0 | 0.0 | strong_all3 |
| 2014 | 43 | 0.0 | 80.0 | 0.0 | 6.8 | 10.9 | 0.0 | 0.0 | strong_all3 |
| 2014 | 45 | 45.0 | 0.0 | 0.0 | 6.8 | 10.9 | 0.0 | 0.0 | strong_all3 |
| 2014 | 60 | 45.0 | 0.0 | 2.6 | 17.7 | 40.9 | 0.0 | 0.0 | strong_all3 |
| 2014 | 61 | 0.0 | 80.0 | 0.0 | 17.7 | 41.7 | 0.0 | 0.0 | strong_all3 |
| 2014 | 75 | 30.0 | 0.0 | 0.0 | 0.0 | 2.5 | 0.0 | 0.0 | strong_all3 |
| 2014 | 95 | 45.0 | 0.0 | 0.0 | 35.8 | 0.0 | 0.0 | 0.0 | strong_all3 |
| 2014 | 110 | 30.0 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 0.0 | strong_all3 |
| 2015 | 1 | 45.0 | 80.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | near_miss |
| 2015 | 31 | 45.0 | 0.0 | 0.0 | 6.9 | 0.0 | 0.0 | 0.0 | near_miss |
| 2015 | 38 | 30.0 | 0.0 | 0.4 | 0.4 | 2.4 | 0.0 | 0.0 | near_miss |
| 2015 | 43 | 0.0 | 80.0 | 0.0 | 2.4 | 60.4 | 0.0 | 0.0 | near_miss |
| 2015 | 61 | 45.0 | 80.0 | 0.0 | 30.3 | 2.5 | 0.0 | 0.0 | near_miss |
| 2015 | 91 | 45.0 | 0.0 | 0.0 | 0.0 | 33.8 | 0.0 | 0.0 | near_miss |
| 2016 | 1 | 0.0 | 80.0 | 0.0 | 9.0 | 0.0 | 0.0 | 0.0 | strong_all3 |
| 2016 | 43 | 0.0 | 80.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | strong_all3 |
| 2016 | 45 | 30.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | strong_all3 |
| 2016 | 75 | 45.0 | 0.0 | 0.0 | 49.5 | 0.0 | 0.0 | 0.0 | strong_all3 |
| 2016 | 95 | 45.0 | 0.0 | 0.0 | 94.5 | 133.0 | 0.0 | 0.0 | strong_all3 |
| 2016 | 115 | 30.0 | 0.0 | 4.0 | 15.5 | 18.1 | 0.0 | 0.0 | strong_all3 |
| 2017 | 1 | 0.0 | 80.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | near_miss |
| 2017 | 30 | 45.0 | 0.0 | 0.0 | 0.0 | 4.5 | 0.8959 | 0.0 | near_miss |
| 2017 | 43 | 0.0 | 80.0 | 0.0 | 2.3 | 0.0 | 0.0 | 0.0 | near_miss |
| 2017 | 45 | 45.0 | 0.0 | 0.0 | 2.3 | 0.0 | 0.0 | 0.0 | near_miss |
| 2017 | 60 | 45.0 | 0.0 | 17.0 | 19.0 | 30.7 | 0.0 | 0.0 | near_miss |
| 2017 | 61 | 0.0 | 80.0 | 0.0 | 19.0 | 13.7 | 0.0 | 0.0 | near_miss |
| 2017 | 75 | 30.0 | 0.0 | 0.0 | 0.0 | 46.7 | 0.0 | 0.0 | near_miss |
| 2017 | 95 | 45.0 | 0.0 | 0.0 | 34.7 | 0.0 | 0.0 | 0.0 | near_miss |
| 2017 | 110 | 30.0 | 0.0 | 2.7 | 18.5 | 5.9 | 0.0 | 0.0 | near_miss |

## 输出文件

- teacher_year_summary: `benchmark_results/043_04_sya_lowIC_teacher_weather_responsiveness_audit/tables/043_04_teacher_year_summary.csv`
- teacher_year_weather_context: `benchmark_results/043_04_sya_lowIC_teacher_weather_responsiveness_audit/tables/043_04_teacher_year_weather_context.csv`
- teacher_event_weather: `benchmark_results/043_04_sya_lowIC_teacher_weather_responsiveness_audit/tables/043_04_teacher_event_weather.csv`
- teacher_vs_04302_ppo_summary: `benchmark_results/043_04_sya_lowIC_teacher_weather_responsiveness_audit/tables/043_04_teacher_vs_04302_ppo_summary.csv`

## 对下一步的含义

- 如果分支 A：已有 lowIC teacher 至少比 043_02 PPO 模板更非模板化，可进入 warm-start 训练设计，但还要检查 teacher 是否真的与天气/胁迫存在可解释关系。
- 如果分支 B：teacher 比 PPO 非模板化，但天气关联弱；可以作为 warm-start 候选，但需要额外天气响应 reward 或分层采样。
- 如果分支 C：teacher 也不能提供足够天气响应信号，不应继续 teacher warm-start，应改做天气响应 reward/采样机制。