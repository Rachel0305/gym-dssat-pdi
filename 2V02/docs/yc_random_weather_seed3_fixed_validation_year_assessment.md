# YC 固定 PPO seed 3：2014–2023 验证年五情景审计

## 目的与边界

本次把同一个已训练的 Random-weather PPO seed 3 固定用于 2014–2023 十个验证年，逐年对照 Null、DSSAT auto + external N、Official expert、Recorded farmer 四个基准。没有逐年换模型或拼接年度冠军，也没有重训、重跑 DSSAT 或修改任何实验设置。

seed 3 是此前按 observed 与 heldout WGEN canonical reward 等权平均事后选出的候选（均值 0.763），选择使用了评估结果，因此存在选择偏差。本报告是描述性复核，不是独立确认性结论。

## 指标口径

- 五情景共同 reward 按 `0.158 * yield - 1.1 * irrigation - 1.58 * nitrogen` 计算，所有情景同口径。
- 基准 common reward 使用 055_03 season summary 的正式 `unified_reward`；用已舍入的 yield/水氮字段重算，与正式值最大差异为 **0.440**。PPO common reward 使用同一公式和 episode 精度。
- PPO canonical reward 单独列示，不与基准 `common_reward_05503` 混排。
- `PFP_N = yield / N`，仅在 N > 0 时定义；N=0 为 N/A。
- PPO 没有匹配的 ETCP replay，`WP_ET` 保留缺失。因此不能给 PPO 做 WP_ET 排名，也不能据此宣称完整农业效率 Pareto 最优。
- 下文的“部分 Pareto 前沿”仅使用五情景都有值的四维指标：yield 最大、灌溉最小、N 最小、common reward 最大。common reward 本身由前三项公式构成，非独立的新证据。

## 数据质量与管理闭合

- seed 3 observed episode：10 行，覆盖 2014–2023，每年唯一一行。
- 四基准：40 行，十年每年恰好四个情景；站点限定为 YC。
- PPO 管理事件与正式 episode 的总灌溉、施氮总量逐年一致：是。
- 源文件：`results\yc_random_weather_ppo\004_05\all_evaluation_episode_level_0_7.csv`、`benchmark_results\055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000\tables\055_03_yca_five_scenario_season_summary.csv`、`results\yc_random_weather_ppo\004_05_review_figures\055_03_five_scenario\best_seed_seed3\seed3_observed_management_events.csv`、`results\yc_random_weather_ppo\004_05_review_figures\055_03_five_scenario\best_seed_seed3\seed3_diagnostic_manifest.json`。

## 逐年表现

| 年份 | PPO yield | 当年最高产量基准 | PPO-最高基准 | 灌溉 mm | N kg/ha | PFP_N | common reward | PPO在部分前沿 | 当年部分前沿情景 |
|---:|---:|---|---:|---:|---:|---:|---:|:---:|---|
| 2014 | 4203 | Official expert (9518) | -5315 | 75 | 120 | 35.0 | 391.9 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2015 | 7992 | Official expert (9089) | -1097 | 75 | 120 | 66.6 | 990.6 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2016 | 5608 | Recorded farmer (6508) | -900 | 75 | 120 | 46.7 | 614.0 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2017 | 7828 | Official expert (7820) | +8 | 75 | 80 | 97.9 | 1027.9 | 是 | DSSAT auto + external N, Null, Random-weather PPO seed 3 |
| 2018 | 8323 | Recorded farmer (8328) | -5 | 75 | 120 | 69.4 | 1043.0 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2019 | 4101 | Official expert (7454) | -3353 | 75 | 80 | 51.3 | 439.1 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2020 | 6517 | Official expert (6518) | -1 | 75 | 120 | 54.3 | 757.6 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2021 | 8316 | Recorded farmer (8644) | -328 | 75 | 80 | 104.0 | 1105.0 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2022 | 8379 | Official expert (8564) | -185 | 75 | 80 | 104.7 | 1115.0 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |
| 2023 | 7751 | Official expert (9568) | -1817 | 75 | 80 | 96.9 | 1015.8 | 是 | DSSAT auto + external N, Null, Official expert, Recorded farmer, Random-weather PPO seed 3 |

## 汇总判断

- 同一个 seed 3 完成十年评估。平均产量 **6901.9 kg/ha**，平均灌溉 **75.0 mm**，平均施氮 **100.0 kg/ha**，平均 `PFP_N` **72.7 kg grain/kg N**。
- 逐年排名第一次数（可比范围）：yield **1/10**，灌溉最低 **0/10**，N最低 **0/10**，`PFP_N` **9/10**，common reward **8/10**。`WP_ET` 对 PPO 缺数，**0/10 可排名**。
- PPO 在四指标部分 Pareto 前沿中的年份数：**10/10**。该结果不能替代包含 WP_ET 的完整农业效率判定。
- PPO 超过当年四基准最高产量的年份数：**1/10**；其余年份需以表中产量差阅读，不能把“接近最高基准”说成胜出。
- 相对官方专家，seed 3 每年灌溉节省约 **136.0 mm**、施氮节省约 **145.0 kg/ha**；相对记录农户模板，分别节省 **45.0 mm** 和 **274.0 kg/ha**。节省投入并不自动等于综合改进，需与产量和效率并列判断。
- 管理事件表显示 PPO 多年重复使用相同灌溉总量或少数氮剂量档位；逐年措施数量和汇总值见 CSV。该行为适应性有限，不能仅由 episode return 较高推断策略已找到逐年精细最优措施。

seed 3 的管理画像：2014–2023 年灌溉总量均为 **75 mm**；施氮在 **80 或 120 kg/ha** 两档，正动作日为 **3–4 天/季**（十年合计 35 个动作日）。这说明当前固定策略虽给出一致的低灌溉方案，但年度动作响应较粗，2014/2019 的低产尤其需要逐日状态及 ETCP replay 复核。

## 结论与下一步

seed 3 是一个已训练、固定并实际用于十个验证年的候选，不是逐年挑选的不同策略。它呈现低于官方专家/农户的水氮投入和较高 `PFP_N` 的权衡，但产量并未稳定达到当年基准最高值；`WP_ET` 缺失又使综合效率结论不完整。因此当前证据支持“存在值得核验的资源效率权衡”，不支持“已经在每个验证年综合胜出”或“已获得完整 Pareto 最优策略”。

低算力下一步：优先为这个固定 seed 3 补做**evaluation-only、精确 ETCP/WP_ET replay**并核验同年基准口径；若 replay 通过，再按逐年完整指标重算前沿。完成前不扩大训练、不增加 seed、不按年挑选模型。

可复现命令：`python src/run_004_07_yc_fixed_seed3_validation_pareto_audit.py`

输出 CSV：`results\yc_random_weather_ppo\004_05_review_figures\055_03_five_scenario\best_seed_seed3\seed3_fixed_validation_year_comparison.csv`。未 commit、未 push。
