# YC PPO Seed Variability and Policy Behavior Diagnosis

## 1. Why diagnosis before more training
004_03 的 pooled 指标隐藏了 seed 间方向冲突，因此先检查策略行为，再决定是否值得扩展 PPO seeds；本轮不训练、不调参。

## 2. Existing pilot result
004_03 的 held-out pooled reward 为 historical 0.830541、random-weather 0.907329（+9.2456%），但逐 seed 方向并不一致。本轮独立复核策略轨迹与同状态响应。

## 3. Six models
H0/H1/H2 与 W0/W1/W2 分别为两种训练域的 seed 0/1/2。checkpoint 清单及 SHA256 见 results/yc_random_weather_ppo/004_04/model_manifest.csv；仅使用 004_03/training_verified。

## 4. Evaluation replay design
确定性回放每模型 20 个 held-out WGEN seeds (1081-1100) 与 observed 2014-2023，共 30 集。W0/1081 smoke 的 runtime weather hash、reward、yield、N、I、天数与 004_03 对齐后再进行全量回放。环境、reward 与 mask 复用 004_03 canonical runner；进程树 RSS 守卫 4000 MB。逐步文件和 manifest 位于 004_04/trajectories。

## 5. Action occupancy
| model_id | steps | noop_fraction | irrigation_probability | fertilizer_probability | mean_irrigation_dose | mean_fertilizer_dose |
| --- | --- | --- | --- | --- | --- | --- |
| H0 | 3075 | 0.829 | 0.122 | 0.059 | 2.122 | 2.341 |
| H1 | 3061 | 0.990 | 0.010 | 0.010 | 0.441 | 0.392 |
| H2 | 3075 | 0.951 | 0.049 | 0.010 | 1.024 | 0.780 |
| W0 | 3061 | 0.990 | 0.010 | 0.010 | 0.441 | 0.392 |
| W1 | 3075 | 0.990 | 0.010 | 0.010 | 0.439 | 1.171 |
| W2 | 3061 | 0.972 | 0.020 | 0.028 | 0.735 | 1.895 |

16 动作和 marginal dose occupancy 按总体、held-out WGEN、observed weather 的分组见 results/yc_random_weather_ppo/004_04/action_occupancy_by_model.csv。W0 的 no-op、边际剂量概率和累计季节投入需联合阅读；动作事件数与单次事件剂量见第 6、7 节，不能仅凭逐日平均动作量解释节水/减氮。

## 5.1 Model by evaluation weather
下表补充每种模型在两类天气下的回报、产量、季节总投入、事件数、首次管理时间与 no-op 频率；可直接横向比较 W0 与 H0/H1/W1/W2。

| model_id | evaluation_weather_type | episodes | mean_return | mean_yield | mean_season_irrigation | mean_season_fertilizer | mean_irrigation_events | mean_fertilizer_events | mean_first_irrigation_dap | mean_first_fertilizer_dap | noop_fraction | p_irrigation_action | p_fertilizer_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | heldout_wgen | 20 | 0.586 | 7210.487 | 219.000 | 240.000 | 12.600 | 6.000 | 0.000 | 0.000 | 0.829 | 0.122 | 0.058 |
| H0 | observed_weather | 10 | 0.748 | 8202.040 | 214.500 | 240.000 | 12.300 | 6.000 | 0.000 | 0.000 | 0.830 | 0.121 | 0.059 |
| H1 | heldout_wgen | 20 | 0.977 | 6626.578 | 45.000 | 40.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.010 |
| H1 | observed_weather | 10 | 0.331 | 5348.288 | 45.000 | 40.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.010 |
| H2 | heldout_wgen | 20 | 0.928 | 7119.284 | 106.500 | 80.000 | 5.100 | 1.000 | 0.000 | 0.000 | 0.950 | 0.050 | 0.010 |
| H2 | observed_weather | 10 | 0.685 | 6871.321 | 102.000 | 80.000 | 4.800 | 1.000 | 0.000 | 0.000 | 0.953 | 0.047 | 0.010 |
| W0 | heldout_wgen | 20 | 0.977 | 6626.578 | 45.000 | 40.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.010 |
| W0 | observed_weather | 10 | 0.331 | 5348.288 | 45.000 | 40.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.010 |
| W1 | heldout_wgen | 20 | 0.939 | 7186.100 | 45.000 | 120.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.010 |
| W1 | observed_weather | 10 | 0.351 | 6303.987 | 45.000 | 120.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.990 | 0.010 | 0.010 |
| W2 | heldout_wgen | 20 | 0.806 | 7206.341 | 75.000 | 196.000 | 2.000 | 2.900 | 0.000 | 0.000 | 0.972 | 0.019 | 0.028 |
| W2 | observed_weather | 10 | 0.423 | 6821.757 | 75.000 | 188.000 | 2.000 | 2.700 | 0.000 | 0.000 | 0.973 | 0.020 | 0.027 |

## 6. Resource use
每集资源 totals、episode return 和 yield 见 evaluation_episode_summary.csv；DAP 累积资源 median/P25/P75 见 resource_trajectories/cumulative_resource_by_dap.csv。

## 7. Management timing
management_event_summary.csv 包含事件数、事件均量、首末 DAP；management_timing_summary.csv 给出按 DAP 的 action probability 与累计资源分位数。
本 runner 的 `DAP=0` 对应播种前 5 天（播种日−5 至 −1），`DAP=1` 才是播种日；因此表中首次管理 DAP=0 表示播前首个决策窗口，不应误读为播种当天。

## 8. Stress response
SWFAC/NSTRES 分箱为 <=0.05、(0.05,0.25]、(0.25,0.50]、>0.50。按仓库后处理语义，0 近似无胁迫、数值越大胁迫越强。下表合并显示 <=0.05 与 >0.05 条件下的目标动作概率和样本步数；完整四档见 stress_response_summary.csv。

| model_id | evaluation_weather_type | n_irrigation_stress_le_0p05 | p_irrigation_at_stress_le_0p05 | n_irrigation_stress_gt_0p05 | p_irrigation_at_stress_gt_0p05 | n_fertilizer_stress_le_0p05 | p_fertilizer_at_stress_le_0p05 | n_fertilizer_stress_gt_0p05 | p_fertilizer_at_stress_gt_0p05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | heldout_wgen | 2038 | 0.114 | 21 | 0.952 | 2059 | 0.058 | 0 |  |
| H0 | observed_weather | 1005 | 0.112 | 11 | 0.909 | 1016 | 0.059 | 0 |  |
| H1 | heldout_wgen | 2017 | 0.000 | 42 | 0.476 | 1253 | 0.016 | 806 | 0.000 |
| H1 | observed_weather | 865 | 0.000 | 137 | 0.073 | 539 | 0.019 | 463 | 0.000 |
| H2 | heldout_wgen | 2023 | 0.041 | 36 | 0.556 | 1840 | 0.011 | 219 | 0.000 |
| H2 | observed_weather | 933 | 0.032 | 83 | 0.217 | 820 | 0.012 | 196 | 0.000 |
| W0 | heldout_wgen | 2017 | 0.000 | 42 | 0.476 | 1253 | 0.016 | 806 | 0.000 |
| W0 | observed_weather | 865 | 0.000 | 137 | 0.073 | 539 | 0.019 | 463 | 0.000 |
| W1 | heldout_wgen | 2018 | 0.000 | 41 | 0.488 | 2036 | 0.010 | 23 | 0.000 |
| W1 | observed_weather | 873 | 0.000 | 143 | 0.070 | 980 | 0.010 | 36 | 0.000 |
| W2 | heldout_wgen | 2030 | 0.010 | 29 | 0.690 | 2059 | 0.028 | 0 |  |
| W2 | observed_weather | 897 | 0.011 | 105 | 0.095 | 1002 | 0.027 | 0 |  |

这是沿各自策略轨迹的描述性条件关联，不能作因果解释。H1/W0 的条件动作分布相同；其管理集中在 DAP=0 的单次输入，不能据此声称 random-weather 训练增强了持续的 stress-feedback。较高压力分箱尤其 WGEN 端样本较少，证据不足以比较实时 cue 依赖强弱。

## 9. Weather response
前 3/7 天雨量与滞后 3 日 TMAX/SRAD 只作为 POST_HOC_DIAGNOSTIC_ONLY，不作为 PPO 输入。天气分箱结果见 weather_response_summary.csv。
所有 120 个 WGEN episode 的保存 runtime weather hash 均复核一致；但实际天气逐步值仅覆盖每模型 1850/2059 = 89.85% 的 held-out 步数。缺失的末段天气字段保持缺失，未用基础 rendered .WTH 补齐，因此 WGEN weather-response 结论为 PARTIAL；observed weather 步级覆盖为 100%。

## 10. Same-seed paired behavior
同 evaluation realization 按共同 timestep 的轨迹级比较如下。不同动作可能已令后续 DSSAT state 分叉，不能把后续 mismatch 当作同状态策略差异。

| historical_model | random_weather_model | ppo_seed | evaluation_weather_type | paired_episodes | common_trajectory_steps | trajectory_action_agreement_rate | irrigation_action_agreement_rate | fertilizer_action_agreement_rate | mean_abs_irrigation_dose_difference | mean_abs_fertilizer_dose_difference | mean_first_irrigation_dap_difference | mean_first_fertilizer_dap_difference | mean_irrigation_event_count_difference_w_minus_h | mean_fertilizer_event_count_difference_w_minus_h | mean_irrigation_per_event_difference_w_minus_h | mean_fertilizer_per_event_difference_w_minus_h | mean_reward_difference_w_minus_h | mean_yield_difference_w_minus_h | mean_irrigation_difference_w_minus_h | mean_fertilizer_difference_w_minus_h | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | W0 | 0 | heldout_wgen | 20 | 2059 | 0.839 | 0.887 | 0.951 | 1.690 | 1.943 | 0.000 | 0.000 | -11.600 | -5.000 | 27.613 | 0.000 | 0.391 | -583.909 | -174.000 | -200.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H0 | W0 | 0 | observed_weather | 10 | 1002 | 0.839 | 0.889 | 0.950 | 1.662 | 1.996 | 0.000 | 0.000 | -11.300 | -5.000 | 27.554 | 0.000 | -0.417 | -2853.752 | -169.500 | -200.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H1 | W1 | 1 | heldout_wgen | 20 | 2059 | 0.990 | 1.000 | 0.990 | 0.000 | 0.777 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 80.000 | -0.038 | 559.522 | 0.000 | 80.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H1 | W1 | 1 | observed_weather | 10 | 1002 | 0.990 | 1.000 | 0.990 | 0.000 | 0.798 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 80.000 | 0.020 | 955.699 | 0.000 | 80.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H2 | W2 | 2 | heldout_wgen | 20 | 2059 | 0.932 | 0.950 | 0.972 | 0.889 | 1.127 | 0.000 | 0.000 | -3.100 | 1.900 | 16.550 | -12.000 | -0.122 | 87.057 | -31.500 | 116.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H2 | W2 | 2 | observed_weather | 10 | 1002 | 0.937 | 0.954 | 0.973 | 0.838 | 1.078 | 0.000 | 0.000 | -2.800 | 1.700 | 16.200 | -9.333 | -0.262 | -49.563 | -27.000 | 108.000 | TRAJECTORY_LEVEL; states may diverge after the first action |

## 11. Same-state policy probe
从六模型轨迹 union 按天气类型、season phase、SWFAC、NSTRES、累计资源分层，以固定 seed 404006 抽取 600 个 probe state。先保存 state 和 action mask，再由六个模型在完全相同输入上预测；不 step DSSAT。概率/JSD 跳过，未改 SB3。矩阵见 policy_similarity/policy_similarity_matrix.csv。

## 12. H1 vs W0
同状态 action agreement=1.000，rollout trajectory action agreement=1.000，配对 episode return/yield/N/I 一致=True，分类 **IDENTICAL_POLICY_BEHAVIOR**。因此这里不是仅凭 pooled aggregate 得出相似，而是本评估 support 上的确定性动作序列也完全一致。此结果说明 W0 的低投入型行为已在历史天气 PPO 的 H1 seed 中出现，不构成 random-weather augmentation 独有行为的证据。

## 13. W0 mechanism
W0 与 H0/H1/W1/W2 在两类天气下的平均产量、投入、事件频率及 no-op 见第 5.1 节。与 H0 相比，canonical runtime reward 的 signed component effect 为：

- held-out WGEN：yield=-0.0923; water-cost savings=+0.1914; N-cost savings=+0.3160; stress-relief=-0.0003; SWFAC-penalty effect=-0.0241; component sum=+0.3908, runtime return delta=+0.3908, residual=-5.55e-17
- observed weather：yield=-0.4509; water-cost savings=+0.1865; N-cost savings=+0.3160; stress-relief=-0.0002; SWFAC-penalty effect=-0.4683; component sum=-0.4170, runtime return delta=-0.4170, residual=-5.55e-17

逐 episode 分项与重建误差见 reward_component_summary.csv。WGEN 评价期内 W0 的资源节省足以抵消产量项损失；observed 期回报下降则由产量项和 SWFAC guardrail penalty 主导，投入成本节省仅部分抵消。以上是精确 reward 分项对账，不把 yield/N/I 单项替代总回报。

## 14. Held-out vs observed case studies
按 W0-H0 episode return 差自动选择 WGEN 改善最大 2 例、优势最小 2 例及 observed 下降最大 2 例；20 个 WGEN seed 均未出现 W0 低于 H0；按规则报告 W0 优势最小的两例，不将其误称为下降案例。首次动作分歧与 episode 环境/胁迫摘要如下：

| case_group | episode_key | w0_minus_h0_reward | first_action_divergence_timestep | first_action_divergence_dap | first_divergent_H0_action | first_divergent_W0_action |
| --- | --- | --- | --- | --- | --- | --- |
| WGEN_W0_largest_improvement | heldout_wgen:1085 | 0.500 | 13 | 8 | 4 | 0 |
| WGEN_W0_largest_improvement | heldout_wgen:1092 | 0.494 | 13 | 8 | 4 | 0 |
| WGEN_W0_smallest_improvement_no_decline | heldout_wgen:1082 | 0.128 | 13 | 8 | 4 | 0 |
| WGEN_W0_smallest_improvement_no_decline | heldout_wgen:1099 | 0.229 | 13 | 8 | 4 | 0 |
| OBSERVED_W0_largest_decline | observed_weather:2014 | -1.959 | 13 | 8 | 4 | 0 |
| OBSERVED_W0_largest_decline | observed_weather:2019 | -1.271 | 13 | 8 | 4 | 0 |

| case_group | episode_key | model_id | episode_return | yield | total_irrigation | total_fertilizer | total_rain | mean_TMAX | mean_SRAD | mean_SWFAC | mean_NSTRES | irrigation_event_count | fertilizer_event_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WGEN_W0_largest_improvement | heldout_wgen:1085 | H0 | 0.511 | 6661.489 | 210.000 | 240.000 | 627.369 | 31.333 | 14.191 | 0.002 | 0.000 | 12 | 6 |
| WGEN_W0_largest_improvement | heldout_wgen:1085 | W0 | 1.011 | 6686.457 | 45.000 | 40.000 | 627.369 | 31.333 | 14.191 | 0.002 | 0.027 | 1 | 1 |
| WGEN_W0_largest_improvement | heldout_wgen:1092 | H0 | 0.550 | 7016.917 | 225.000 | 240.000 | 395.886 | 31.563 | 15.283 | 0.001 | 0.000 | 13 | 6 |
| WGEN_W0_largest_improvement | heldout_wgen:1092 | W0 | 1.044 | 6892.736 | 45.000 | 40.000 | 395.886 | 31.563 | 15.283 | 0.001 | 0.055 | 1 | 1 |
| WGEN_W0_smallest_improvement_no_decline | heldout_wgen:1082 | H0 | 0.505 | 6628.010 | 210.000 | 240.000 | 361.702 | 31.764 | 16.238 | 0.002 | 0.000 | 12 | 6 |
| WGEN_W0_smallest_improvement_no_decline | heldout_wgen:1082 | W0 | 0.633 | 5912.953 | 45.000 | 40.000 | 361.702 | 31.764 | 16.238 | 0.059 | 0.047 | 1 | 1 |
| WGEN_W0_smallest_improvement_no_decline | heldout_wgen:1099 | H0 | 0.795 | 8570.650 | 225.000 | 240.000 | 273.394 | 30.772 | 14.643 | 0.001 | 0.000 | 13 | 6 |
| WGEN_W0_smallest_improvement_no_decline | heldout_wgen:1099 | W0 | 1.024 | 7226.467 | 45.000 | 40.000 | 273.394 | 30.772 | 14.643 | 0.012 | 0.087 | 1 | 1 |
| OBSERVED_W0_largest_decline | observed_weather:2014 | H0 | 0.927 | 9402.930 | 225.000 | 240.000 | 205.300 | 30.212 | 16.776 | 0.001 | 0.000 | 13 | 6 |
| OBSERVED_W0_largest_decline | observed_weather:2014 | W0 | -1.032 | 1833.745 | 45.000 | 40.000 | 157.000 | 31.054 | 17.701 | 0.294 | 0.031 | 1 | 1 |
| OBSERVED_W0_largest_decline | observed_weather:2019 | H0 | 0.656 | 7602.567 | 210.000 | 240.000 | 198.700 | 31.978 | 18.791 | 0.002 | 0.000 | 12 | 6 |
| OBSERVED_W0_largest_decline | observed_weather:2019 | W0 | -0.614 | 3700.841 | 45.000 | 40.000 | 198.700 | 31.978 | 18.791 | 0.238 | 0.020 | 1 | 1 |

逐日文件保留天气、SWFAC/NSTRES、动作、累计 N/I 与 reward trajectory；规则为排序，未人工挑 seed/year。

## 15. Reward decomposition
只使用 canonical runtime reward components；旧式 0.06 * final_grnwt - 0.04 * cumfert 为 SUPERSEDED / NOT_APPLICABLE。缺失的精确 wrapper component 不猜；reward_component_summary.csv 给出重建和对账状态。

## 16. Policy archetypes
依据逐步 no-op 频率与 episode 季节总投入、事件数识别行为 profile（而非把低日均剂量误作低季节投入）：

| model_id | labels | mean_season_irrigation | mean_season_fertilizer | noop_fraction | mean_irrigation_events | mean_fertilizer_events |
| --- | --- | --- | --- | --- | --- | --- |
| H0 | 高灌溉高氮 | 217.500 | 240.000 | 0.829 | 12.500 | 6.000 |
| H1 | 低季节投入, 高no-op频率 | 45.000 | 40.000 | 0.990 | 1.000 | 1.000 |
| H2 | 中等/混合投入, 高no-op频率 | 105.000 | 80.000 | 0.951 | 5.000 | 1.000 |
| W0 | 低季节投入, 高no-op频率 | 45.000 | 40.000 | 0.990 | 1.000 | 1.000 |
| W1 | 低灌溉高氮, 高no-op频率 | 45.000 | 120.000 | 0.990 | 1.000 | 1.000 |
| W2 | 低灌溉高氮, 高no-op频率 | 75.000 | 193.333 | 0.972 | 2.000 | 2.833 |

命名是对这批轨迹的描述，不是稳定类别或 seed 价值排序。

## 17. What weather augmentation changed
random_weather_changes_policy_behavior=MIXED。同 seed rollout divergence 与同状态 probe 分开解释，不能将轨迹级差异误读为纯策略映射差异。

## 18. What does this diagnosis imply about weather augmentation?
4. Evidence remains insufficient (D): same-seed behavior changes are mixed, and W0 matches the existing H1 policy rather than demonstrating a systematic shift.

## 19. Recommended next experiment
本轮不扩增 seeds；若后续正式立项，预注册可区分 favorable-archetype discovery 与 weather-specific response 的 seed-level 假设，并冻结 site/reward/action/runtime，再开展预定规模的多 seed 诊断。 暂不挑选最佳 seed；若扩展 seed，应预注册行为机制与 reward component 假设，同时冻结 site/reward/action/runtime。

## 20. Limitations
- 每种训练域只有 3 个 PPO seeds，不能估计稳定的有利 archetype 概率。
- Probe 状态来自策略自身诱导的有限 state support；trajectory agreement 混合环境状态分叉。
- Stress 与 post-hoc 天气分箱是描述性关联；weather windows 未输入 policy。
- WGEN 天气分箱使用 004_03 同一 runtime state capture/hash 口径；基础 rendered .WTH 不代表 WTHER=W 的实际生成序列。
- WGEN 末段 runtime weather state capture 不完整（逐步天气字段覆盖 89.85%）；未以其他来源插补，限制天气响应分析。
- 未提取动作概率分布，因此没有报告 JSD。
- observed 2014-2023 不足以断言普遍泛化失败。

## 21. Files / tests / git
机器结果位于 results/yc_random_weather_ppo/004_04/。QC 覆盖六个 verified model、固定评估集合、smoke 复现、逐步资源闭合、episode reward 对账和 probe 覆盖。训练/reward/runtime/CLI 修改均为 NO。commit subject: analysis: diagnose YC PPO seed policy behavior；不 push。

## Summary
- Models H0,H1,H2,W0,W1,W2; no training; 6 x 30 evaluation episodes.
- Random-weather policy behavior: MIXED.
- H1/W0 agreement/class: 1.000 / IDENTICAL_POLICY_BEHAVIOR.
- W0 heldout mechanism: YES; observed drop: YES.
- Interpretation: 4. Evidence remains insufficient (D): same-seed behavior changes are mixed, and W0 matches the existing H1 policy rather than demonstrating a systematic shift.
