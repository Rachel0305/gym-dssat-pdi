# YC 004_11：seed 5 mask sensitivity 评估

## 执行与边界

- 固定 seed 5 checkpoint 100K，B/C/D 按各自 mask 从 2014–2023 每个 episode 起点完整 evaluation-only 重放；A 复用已核验的 canonical 结果。没有训练，没有写入配置、模型、wrapper 源码或 DSSAT 输入。
- B 仅取消 DAP≤30、75 mm cap；C 仅取消 DAP≤60、150 mm cap；D 同时取消两者。I/N 7 日间隔、season I240/N250、每日剂量、DAP 窗口、DAP≤90/I195 late reserve 与其他规则均保持。
- 重放年份数：30 / 30；RSS 峰值 452.6 MiB；阈值 2048 MiB。
- 004_07_02 的 Summary.OUT / ETCP helper 用于 B/C/D。每季要求 action dose、MgmtEvent.OUT、Summary.OUT 闭合；A 使用 004_07_02 已通过 finalize 的摘要。

## 逐版平均差值

| Variant | Yield Δ kg/ha | I Δ mm | N Δ kg/ha | WP_ET Δ | PFP_N Δ | Common reward Δ | Canonical reward Δ | Yield 提升年份 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B no-early | -11.4 | 25.5 | -40.0 | -0.074 | 6.730 | 33.34 | 0.0435 | 1/10 |
| C no-mid | -44.8 | 0.0 | 0.0 | -0.006 | -0.200 | -7.08 | -0.0157 | 0/10 |
| D no-early-mid | -43.5 | 25.5 | -40.0 | -0.080 | 6.560 | 28.28 | 0.0321 | 1/10 |

## 按年指标

| 年份 | Variant | Yield | I | N | ETCP | WP_ET | PFP_N | Common reward | Canonical reward | 管理事件 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2014 | A canonical | 9253.2 | 165.0 | 240.0 | 362.5 | 2.550 | 38.60 | 901.30 | 0.9296 | 9 |
| 2014 | B no-early | 8991.6 | 180.0 | 200.0 | 378.4 | 2.380 | 45.00 | 906.67 | 0.9117 | 10 |
| 2014 | C no-mid | 9034.6 | 165.0 | 240.0 | 361.9 | 2.500 | 37.60 | 866.77 | 0.8829 | 9 |
| 2014 | D no-early-mid | 8928.4 | 180.0 | 200.0 | 378.1 | 2.360 | 44.60 | 896.69 | 0.8841 | 10 |
| 2015 | A canonical | 9088.7 | 165.0 | 240.0 | 348.8 | 2.610 | 37.90 | 875.31 | 0.9441 | 9 |
| 2015 | B no-early | 9086.0 | 180.0 | 200.0 | 364.4 | 2.490 | 45.40 | 921.58 | 0.9903 | 10 |
| 2015 | C no-mid | 9088.7 | 165.0 | 240.0 | 348.7 | 2.610 | 37.90 | 875.31 | 0.9441 | 9 |
| 2015 | D no-early-mid | 9079.6 | 180.0 | 200.0 | 364.8 | 2.490 | 45.40 | 920.58 | 0.9893 | 10 |
| 2016 | A canonical | 6496.6 | 165.0 | 240.0 | 334.1 | 1.940 | 27.10 | 465.77 | 0.5341 | 9 |
| 2016 | B no-early | 6480.9 | 180.0 | 200.0 | 341.7 | 1.900 | 32.40 | 509.99 | 0.5783 | 10 |
| 2016 | C no-mid | 6496.6 | 165.0 | 240.0 | 332.5 | 1.950 | 27.10 | 465.77 | 0.5341 | 9 |
| 2016 | D no-early-mid | 6459.6 | 180.0 | 200.0 | 338.9 | 1.910 | 32.30 | 506.61 | 0.5749 | 10 |
| 2017 | A canonical | 7827.1 | 165.0 | 240.0 | 338.4 | 2.310 | 32.60 | 675.99 | 0.7449 | 9 |
| 2017 | B no-early | 7816.0 | 195.0 | 200.0 | 354.7 | 2.200 | 39.10 | 704.42 | 0.7733 | 10 |
| 2017 | C no-mid | 7820.5 | 165.0 | 240.0 | 338.3 | 2.310 | 32.60 | 674.93 | 0.7438 | 9 |
| 2017 | D no-early-mid | 7796.6 | 195.0 | 200.0 | 355.0 | 2.200 | 39.00 | 701.36 | 0.7703 | 10 |
| 2018 | A canonical | 8323.5 | 165.0 | 240.0 | 357.5 | 2.330 | 34.70 | 754.41 | 0.8222 | 9 |
| 2018 | B no-early | 8264.5 | 195.0 | 200.0 | 370.8 | 2.230 | 41.30 | 775.29 | 0.8431 | 10 |
| 2018 | C no-mid | 8323.0 | 165.0 | 240.0 | 357.5 | 2.330 | 34.70 | 754.33 | 0.8222 | 9 |
| 2018 | D no-early-mid | 8227.2 | 195.0 | 200.0 | 371.0 | 2.220 | 41.10 | 769.40 | 0.8372 | 10 |
| 2019 | A canonical | 7225.0 | 165.0 | 240.0 | 344.8 | 2.100 | 30.10 | 580.85 | 0.4510 | 9 |
| 2019 | B no-early | 7524.8 | 195.0 | 200.0 | 368.8 | 2.040 | 37.60 | 658.41 | 0.6537 | 10 |
| 2019 | C no-mid | 7037.5 | 165.0 | 240.0 | 338.0 | 2.080 | 29.30 | 551.22 | 0.3476 | 9 |
| 2019 | D no-early-mid | 7417.3 | 195.0 | 200.0 | 366.5 | 2.020 | 37.10 | 641.43 | 0.5902 | 10 |
| 2020 | A canonical | 6516.7 | 165.0 | 240.0 | 379.1 | 1.720 | 27.20 | 468.94 | 0.5378 | 9 |
| 2020 | B no-early | 6500.9 | 195.0 | 200.0 | 380.2 | 1.710 | 32.50 | 496.64 | 0.5655 | 10 |
| 2020 | C no-mid | 6513.6 | 165.0 | 240.0 | 379.1 | 1.720 | 27.10 | 468.44 | 0.5373 | 9 |
| 2020 | D no-early-mid | 6484.4 | 195.0 | 200.0 | 380.8 | 1.700 | 32.40 | 494.03 | 0.5629 | 10 |
| 2021 | A canonical | 8625.5 | 165.0 | 240.0 | 364.9 | 2.360 | 35.90 | 802.13 | 0.8698 | 9 |
| 2021 | B no-early | 8593.9 | 195.0 | 200.0 | 376.2 | 2.280 | 43.00 | 827.34 | 0.8950 | 10 |
| 2021 | C no-mid | 8614.9 | 165.0 | 240.0 | 365.1 | 2.360 | 35.90 | 800.45 | 0.8681 | 9 |
| 2021 | D no-early-mid | 8575.1 | 195.0 | 200.0 | 376.6 | 2.280 | 42.90 | 824.36 | 0.8920 | 10 |
| 2022 | A canonical | 8543.3 | 165.0 | 240.0 | 386.4 | 2.210 | 35.60 | 789.15 | 0.8566 | 9 |
| 2022 | B no-early | 8532.7 | 195.0 | 200.0 | 388.7 | 2.200 | 42.70 | 817.66 | 0.8852 | 10 |
| 2022 | C no-mid | 8524.4 | 165.0 | 240.0 | 386.3 | 2.210 | 35.50 | 786.16 | 0.8537 | 9 |
| 2022 | D no-early-mid | 8506.0 | 195.0 | 200.0 | 388.8 | 2.190 | 42.50 | 813.46 | 0.8810 | 10 |
| 2023 | A canonical | 9587.2 | 165.0 | 240.0 | 406.2 | 2.360 | 39.90 | 954.07 | 1.0217 | 9 |
| 2023 | B no-early | 9581.3 | 195.0 | 200.0 | 412.2 | 2.320 | 47.90 | 983.35 | 1.0510 | 10 |
| 2023 | C no-mid | 9585.1 | 165.0 | 240.0 | 406.2 | 2.360 | 39.90 | 953.75 | 1.0214 | 9 |
| 2023 | D no-early-mid | 9577.8 | 195.0 | 200.0 | 412.3 | 2.320 | 47.90 | 982.79 | 1.0504 | 10 |

## 直接结论

- B no-early：产量提高 1/10 年；灌溉增加 10/10 年；N 有变化 10/10 年。平均产量变化 -11.4 kg/ha，I 25.5 mm，N -40.0 kg/ha，WP_ET -0.074，PFP_N 6.730。
- C no-mid：产量提高 0/10 年；灌溉增加 0/10 年；N 有变化 0/10 年。平均产量变化 -44.8 kg/ha，I 0.0 mm，N 0.0 kg/ha，WP_ET -0.006，PFP_N -0.200。
- D no-early-mid：产量提高 1/10 年；灌溉增加 10/10 年；N 有变化 10/10 年。平均产量变化 -43.5 kg/ha，I 25.5 mm，N -40.0 kg/ha，WP_ET -0.080，PFP_N 6.560。

- 移除 cap 后是否改变灌溉：YES；是否改变 N：YES。N 的变化来自整季策略重跑后观测到的结果，不能仅由单个 DAP15 动作推断。
- 联合动作的逐年证据：A 在 DAP15 的 raw argmax `I30+N40` 被早期 cap 拦截，执行 `I0+N80`（004_10/004_09）；B 和 D 在 2014–2023 每年 DAP15 的 raw argmax `I30+N40` 均变为合法并被执行，且两版十年 N 都从 240 降到 200 kg/ha。这说明 early cap 在固定 seed 5 checkpoint 的闭环反事实中确实连带塑造了 N 总量；不是对新 mask 训练效果的结论。
- 各季节策略从起点闭环演化；年度差值是固定 actor 在不同 mask 下的反事实结果。它不预测在新 mask 下重新训练会得到什么策略，也不构成修改 canonical mask 的单独依据。

```text
early_mid_caps_change_irrigation = MIXED (B/D early-cap release changes I in 10/10 years; C mid-only changes annual I in 0/10 years)
early_mid_caps_change_nitrogen = MIXED (B/D early-cap release changes N in 10/10 years; C mid-only changes N in 0/10 years)
additional_irrigation_yield_return = B +25.5 mm / -11.4 kg/ha; D +25.5 mm / -43.5 kg/ha, yield rose in 1/10 years
efficiency_response = B WP_ET -0.074, PFP_N +6.730; D WP_ET -0.080, PFP_N +6.560
seed5_mask_counterfactual_verdict = mixed; early-cap release changes the joint I×N path and helps 2019, while mean yield and WP_ET decline
next_step = targeted mask redesign study; do not promote this fixed-checkpoint result to a retrained-policy claim
```

## Raw argmax 禁用规则

| Variant | 首个禁用规则 | 次数 |
|---|---|---:|
| B no-early | `I_LATE_RESERVE` | 81 |
| B no-early | `I_MID_CAP` | 320 |
| B no-early | `I_MIN_INTERVAL` | 350 |
| C no-mid | `I_EARLY_CAP` | 100 |
| C no-mid | `I_MIN_INTERVAL` | 290 |
| C no-mid | `N_MIN_INTERVAL` | 60 |
| C no-mid | `N_SEASON_BUDGET` | 311 |
| D no-early-mid | `I_LATE_RESERVE` | 400 |
| D no-early-mid | `I_MIN_INTERVAL` | 350 |

## 动作日程变化

- 与 A 的年度 action signature 不同：B 10/10 年，C 10/10 年，D 10/10 年。
- 十年内不同 signature 数：A 1，B 2，C 1，D 2。C 的日程虽然和 A 不同，但十年仍复用一个相同日程；B/D 分别出现两个日程模板，说明放开早期 cap 后有少量跨年分化，仍不是广泛逐年适应。

## 图表

- `results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/figures/figure_yield_by_year.png`
- `results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/figures/figure_irrigation_by_year.png`
- `results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/figures/figure_nitrogen_by_year.png`
- `results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/figures/figure_efficiency_by_year.png`
- `results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/figures/figure_common_reward_by_year.png`

## 数据来源

- Baseline A：`results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay/replay_episode_summary.csv` 与 `results/yc_random_weather_ppo/004_09_observation_mask_action_audit/decision_trace_seed345.csv`。
- B/C/D episode traces：`results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/decision_trace_BCD.csv`；逐规则归因：`results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/decision_trace_BCD_rule_attributed.csv`；summary：`results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/annual_metrics_A_B_C_D.csv`。
- 逐年差值：`results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02/variant_differences_vs_A_refined.csv`；provenance、资源峰值、checkpoint/config/input hashes 与逐季闭合状态见结果目录 JSON manifests。

未提交、未推送。
