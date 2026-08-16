# SYA E2 5K 与 no-forecast 的 WP_ET 补全比较

## 口径

- 范围：SYA originIC，验证年 2014–2023，固定 5000-step checkpoint。
- 本轮没有训练；对 6 个冻结策略逐年串行回放 DSSAT，共 60 次。
- WP_ET = YPEM × 0.1；若 YPEM 不可用，则用 HWAM / (ETCP × 10)。ETCP 单位为 mm。
- 回放结果与原 daily CSV 的产量、灌溉、施氮、动作序列逐行闭合后，才纳入比较。

## 6 个冻结策略汇总

| policy   |   years |   mean_yield |   mean_etcp_mm |   mean_WP_ET |   weighted_WP_ET |   mean_PFP_N |   weighted_PFP_N |
|:---------|--------:|-------------:|---------------:|-------------:|-----------------:|-------------:|-----------------:|
| e2_s0    |      10 |     10070.2  |         476.07 |        2.115 |           2.1153 |        41.95 |          41.9591 |
| e2_s1    |      10 |     10039.3  |         500.28 |        2.005 |           2.0067 |        41.83 |          41.8305 |
| e2_s2    |      10 |      9988.66 |         496.26 |        2.013 |           2.0128 |        41.62 |          41.6194 |
| nof_s0   |      10 |     10009.4  |         476.89 |        2.102 |           2.0989 |        41.71 |          41.7057 |
| nof_s1   |      10 |     10015.6  |         483.62 |        2.074 |           2.071  |        41.73 |          41.7316 |
| nof_s2   |      10 |      9965.57 |         487.64 |        2.044 |           2.0436 |        43.12 |          42.955  |

## 同 seed E2 - no-forecast

|   seed |   years |   mean_yield_delta |   mean_WP_ET_delta |   mean_PFP_N_delta |   WP_ET_win_years |   yield_win_years |   weighted_forecast_WP_ET |   weighted_noforecast_WP_ET |
|-------:|--------:|-------------------:|-------------------:|-------------------:|------------------:|------------------:|--------------------------:|----------------------------:|
|      0 |      10 |            60.8138 |              0.013 |               0.24 |                 5 |                 4 |                    2.1153 |                      2.0989 |
|      1 |      10 |            23.7306 |             -0.069 |               0.1  |                 1 |                 6 |                    2.0067 |                      2.071  |
|      2 |      10 |            23.0854 |             -0.031 |              -1.5  |                 1 |                 6 |                    2.0128 |                      2.0436 |

## 与四个基线的 WP_ET 均值

| scenario                  |   years |   mean_WP_ET |   E2_three_seed_mean_WP_ET |   E2_minus_baseline_mean_WP_ET |
|:--------------------------|--------:|-------------:|---------------------------:|-------------------------------:|
| dssat_auto                |      10 |        1.482 |                     2.0443 |                         0.5623 |
| null                      |      10 |        1.38  |                     2.0443 |                         0.6643 |
| official_extension_expert |      10 |        2.112 |                     2.0443 |                        -0.0677 |
| recorded_farmer_template  |      10 |        1.887 |                     2.0443 |                         0.1573 |

## E2 对四基线逐年最高值

| policy   |   years |   strict_WP_ET_wins_vs_four_baselines |   mean_gap_vs_four_baseline_max |
|:---------|--------:|--------------------------------------:|--------------------------------:|
| e2_s0    |      10 |                                     3 |                          -0.064 |
| e2_s1    |      10 |                                     0 |                          -0.174 |
| e2_s2    |      10 |                                     0 |                          -0.166 |

## 结论

- E2 的 WP_ET 并未稳定超过 no-forecast：seed0 略高，seed1/seed2 较低；三 seed 年度均值 pooled 差为负。
- E2 三 seed 的 WP_ET 均值高于 null、dssat_auto、recorded_farmer，但低于 official_extension_expert；不能写成“超过四个情景”。
- 由于 E2 同时更换了天气输入表达和 dual-branch 特征提取器，这仍是 E2 package 对 no-forecast 的比较，不是只改变天气编码的纯因果对照。

- 年度明细：`benchmark_results/151E2N_wp_et_5k_replay/151_wp_et_replay_by_policy_year.csv`
- 配对明细：`benchmark_results/151E2N_wp_et_5k_replay/151_wp_et_paired_forecast_noforecast_by_year.csv`
- 可复现性核验：`benchmark_results/151E2N_wp_et_5k_replay/151_wp_et_replay_reproducibility_check.csv`
