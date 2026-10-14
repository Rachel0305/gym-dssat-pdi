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

16 动作和 marginal dose occupancy 按天气类型见 action_occupancy_by_model.csv。

## 6. Resource use
每集资源 totals、episode return 和 yield 见 evaluation_episode_summary.csv；DAP 累积资源 median/P25/P75 见 resource_trajectories/cumulative_resource_by_dap.csv。

## 7. Management timing
management_event_summary.csv 包含事件数、事件均量、首末 DAP；management_timing_summary.csv 给出按 DAP 的 action probability 与累计资源分位数。

## 8. Stress response
SWFAC/NSTRES 分箱为 <=0.05、(0.05,0.25]、(0.25,0.50]、>0.50。按仓库现有记录，高值按更强 stress exposure 解释。结果是描述性分层而非因果效应，见 stress_response_summary.csv。

## 9. Weather response
前 3/7 天雨量与滞后 3 日 TMAX/SRAD 只作为 POST_HOC_DIAGNOSTIC_ONLY，不作为 PPO 输入。天气分箱结果见 weather_response_summary.csv。

## 10. Same-seed paired behavior
同 evaluation realization 按共同 timestep 的轨迹级比较如下。不同动作可能已令后续 DSSAT state 分叉，不能把后续 mismatch 当作同状态策略差异。

| historical_model | random_weather_model | ppo_seed | evaluation_weather_type | paired_episodes | common_trajectory_steps | trajectory_action_agreement_rate | irrigation_action_agreement_rate | fertilizer_action_agreement_rate | mean_abs_irrigation_dose_difference | mean_abs_fertilizer_dose_difference | mean_first_irrigation_dap_difference | mean_first_fertilizer_dap_difference | mean_reward_difference_w_minus_h | mean_yield_difference_w_minus_h | mean_irrigation_difference_w_minus_h | mean_fertilizer_difference_w_minus_h | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | W0 | 0 | heldout_wgen | 20 | 2059 | 0.839 | 0.887 | 0.951 | 1.690 | 1.943 | 0.000 | 0.000 | 0.391 | -583.909 | -174.000 | -200.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H0 | W0 | 0 | observed_weather | 10 | 1002 | 0.839 | 0.889 | 0.950 | 1.662 | 1.996 | 0.000 | 0.000 | -0.417 | -2853.752 | -169.500 | -200.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H1 | W1 | 1 | heldout_wgen | 20 | 2059 | 0.990 | 1.000 | 0.990 | 0.000 | 0.777 | 0.000 | 0.000 | -0.038 | 559.522 | 0.000 | 80.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H1 | W1 | 1 | observed_weather | 10 | 1002 | 0.990 | 1.000 | 0.990 | 0.000 | 0.798 | 0.000 | 0.000 | 0.020 | 955.699 | 0.000 | 80.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H2 | W2 | 2 | heldout_wgen | 20 | 2059 | 0.932 | 0.950 | 0.972 | 0.889 | 1.127 | 0.000 | 0.000 | -0.122 | 87.057 | -31.500 | 116.000 | TRAJECTORY_LEVEL; states may diverge after the first action |
| H2 | W2 | 2 | observed_weather | 10 | 1002 | 0.937 | 0.954 | 0.973 | 0.838 | 1.078 | 0.000 | 0.000 | -0.262 | -49.563 | -27.000 | 108.000 | TRAJECTORY_LEVEL; states may diverge after the first action |

## 11. Same-state policy probe
从六模型轨迹 union 按天气类型、season phase、SWFAC、NSTRES、累计资源分层，以固定 seed 404006 抽取 600 个 probe state。先保存 state 和 action mask，再由六个模型在完全相同输入上预测；不 step DSSAT。概率/JSD 跳过，未改 SB3。矩阵见 policy_similarity/policy_similarity_matrix.csv。

## 12. H1 vs W0
同状态 action agreement=1.000，分类 IDENTICAL_POLICY_BEHAVIOR。相同 aggregate metrics 本身不足以说明策略相同。

## 13. W0 mechanism
W0 与其他模型的逐天气产量、投入和运行时 reward components 见 W0_diagnostic_profile.json、reward_component_summary.csv。held-out WGEN 与 observed 的 H0 配对差及分项留存在 diagnostic_decision.json 和逐 episode 文件。

## 14. Held-out vs observed case studies
按 W0-H0 episode return 差自动选择 WGEN 改善最大 2 例、下降最大 2 例及 observed 下降最大 2 例；见 case_study_manifest.csv、case_studies/selected_case_step_trajectories.csv。规则为排序，未人工挑 seed/year。

## 15. Reward decomposition
只使用 canonical runtime reward components；旧式 0.06 * final_grnwt - 0.04 * cumfert 为 SUPERSEDED / NOT_APPLICABLE。缺失的精确 wrapper component 不猜；reward_component_summary.csv 给出重建和对账状态。

## 16. Policy archetypes
依据动作占用、no-op、事件频率/剂量和 timing 识别低投入、灌溉活跃、施氮活跃等 profile；命名是对这批轨迹的描述，不是稳定类别或 seed 价值排序。

## 17. What weather augmentation changed
random_weather_changes_policy_behavior=MIXED。同 seed rollout divergence 与同状态 probe 分开解释，不能将轨迹级差异误读为纯策略映射差异。

## 18. What this means for the hypothesis
4. Evidence remains insufficient.

## 19. Recommended next experiment
先复核 W0 的同状态响应及选定 observed 年份 reward 分项；若继续，预注册 seed-level 行为机制假设并冻结 reward/action/runtime。 暂不挑选最佳 seed；若扩展 seed，应预注册行为机制与 reward component 假设，同时冻结 site/reward/action/runtime。

## 20. Limitations
- 每种训练域只有 3 个 PPO seeds，不能估计稳定的有利 archetype 概率。
- Probe 状态来自策略自身诱导的有限 state support；trajectory agreement 混合环境状态分叉。
- Stress 与 post-hoc 天气分箱是描述性关联；weather windows 未输入 policy。
- WGEN 天气分箱使用 004_03 同一 runtime state capture/hash 口径；基础 rendered .WTH 不代表 WTHER=W 的实际生成序列。
- 未提取动作概率分布，因此没有报告 JSD。
- observed 2014-2023 不足以断言普遍泛化失败。

## 21. Files / tests / git
机器结果位于 results/yc_random_weather_ppo/004_04/。QC 覆盖六个 verified model、固定评估集合、smoke 复现、逐步资源闭合、episode reward 对账和 probe 覆盖。训练/reward/runtime/CLI 修改均为 NO。commit subject: analysis: diagnose YC PPO seed policy behavior；不 push。

## Summary
- Models H0,H1,H2,W0,W1,W2; no training; 6 x 30 evaluation episodes.
- Random-weather policy behavior: MIXED.
- H1/W0 agreement/class: 1.000 / IDENTICAL_POLICY_BEHAVIOR.
- W0 heldout mechanism: YES; observed drop: YES.
- Interpretation: 4. Evidence remains insufficient.
