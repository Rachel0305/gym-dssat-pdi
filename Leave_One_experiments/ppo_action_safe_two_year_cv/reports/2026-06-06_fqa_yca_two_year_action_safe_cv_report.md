# FQA/YCA two-year action-safe PPO cross validation report

Generated at: 2026-06-06

## Goal

This stage trains only action-safe PPO models for FQA and YCA. Each site has two observed years, so the result is explicitly marked as `limited_two_year_cross_validation`, not full dry/normal/wet stability validation.

## Constraints followed

- No PPO without action safety was trained.
- Reward functions and my_data originals were not modified.
- No multi-seed training was performed.
- HLA/SYA/LCA previous results were not overwritten.
- Timesteps per model: 5000

## Action safety parameters

- `enabled`: True
- `daily_irrigation_max`: 40.0
- `daily_n_max`: 80.0
- `season_irrigation_soft_limit`: 200.0
- `season_n_soft_limit`: 300.0
- `min_days_between_irrigation`: 7
- `min_days_between_fertilization`: 10
- `fertilization_allowed_dap_range`: [1, 90]
- `irrigation_allowed_dap_range`: [1, 120]

## Pretrain smoke checks

| station | train_year | policy | status | completed |
|---|---:|---|---|---|
| FQA | 2008 | null_zero | ok | True |
| FQA | 2008 | fixed_low_input | ok | True |
| FQA | 2010 | null_zero | ok | True |
| FQA | 2010 | fixed_low_input | ok | True |
| YCA | 2014 | null_zero | ok | True |
| YCA | 2014 | fixed_low_input | ok | True |
| YCA | 2008 | null_zero | ok | True |
| YCA | 2008 | fixed_low_input | ok | True |

## Evaluation summary

| station | policy | train_year | eval_year | yield | irrigation | nitrogen | mean_reward | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FQA | FQA_train2008_seed0_action_safe | 2008 | 2008 | 7066.73 | 200.00 | 300.00 | 90.67 | True |
| FQA | FQA_train2008_seed0_action_safe | 2008 | 2010 | 5834.45 | 200.00 | 300.00 | 89.58 | True |
| FQA | FQA_train2010_seed0_action_safe | 2010 | 2010 | 5834.70 | 200.00 | 300.00 | 89.59 | True |
| FQA | FQA_train2010_seed0_action_safe | 2010 | 2008 | 7063.59 | 200.00 | 300.00 | 90.60 | True |
| YCA | YCA_train2014_seed0_action_safe | 2014 | 2014 | 9365.60 | 200.00 | 300.00 | 159.20 | True |
| YCA | YCA_train2014_seed0_action_safe | 2014 | 2008 | 8133.53 | 200.00 | 300.00 | 138.65 | True |
| YCA | YCA_train2008_seed0_action_safe | 2008 | 2008 | 8132.17 | 200.00 | 300.00 | 138.48 | True |
| YCA | YCA_train2008_seed0_action_safe | 2008 | 2014 | 9364.83 | 200.00 | 300.00 | 159.05 | True |

## Action safety saturation

- All FQA/YCA evaluations at 200 mm / 300 kg/ha cap: True
| station | policy | eval_year | irrigation | nitrogen | trigger_days | dominant_rule |
|---|---|---:|---:|---:|---:|---|
| FQA | FQA_train2008_seed0_action_safe | 2008 | 200.00 | 300.00 | 105 | amir_min_interval |
| FQA | FQA_train2008_seed0_action_safe | 2010 | 200.00 | 300.00 | 100 | amir_min_interval |
| FQA | FQA_train2010_seed0_action_safe | 2010 | 200.00 | 300.00 | 100 | amir_min_interval |
| FQA | FQA_train2010_seed0_action_safe | 2008 | 200.00 | 300.00 | 105 | amir_min_interval |
| YCA | YCA_train2014_seed0_action_safe | 2014 | 200.00 | 300.00 | 109 | amir_min_interval |
| YCA | YCA_train2014_seed0_action_safe | 2008 | 200.00 | 300.00 | 111 | amir_min_interval |
| YCA | YCA_train2008_seed0_action_safe | 2008 | 200.00 | 300.00 | 111 | anfer_season_limit |
| YCA | YCA_train2008_seed0_action_safe | 2014 | 200.00 | 300.00 | 109 | anfer_season_limit |

当前 action-safe PPO 的策略比较更多反映了固定 season cap 下的时序分配差异，而不是自由水氮优化；下一阶段需要考虑 reward 成本项或 season cap 敏感性分析。

## FQA/YCA best policy

| station | best_policy | train_year | validation_years | mean_yield | mean_irrigation | mean_n | stability_score |
|---|---|---:|---:|---:|---:|---:|---:|
| FQA | FQA_train2010_seed0_action_safe | 2010 | 2008 | 7063.59 | 200.00 | 300.00 | 0.500 |
| YCA | YCA_train2008_seed0_action_safe | 2008 | 2014 | 9364.83 | 200.00 | 300.00 | 0.500 |

## Five-site best policy summary

- Path: `Leave_One_experiments\ppo_action_safe_summary\all_site_best_policy_summary.csv`
| station | best_policy | train_year | cv_type | mean_yield | irrigation | nitrogen | limited_data |
|---|---|---:|---|---:|---:|---:|---|
| FQA | FQA_train2010_seed0_action_safe | 2010 | limited_two_year_cross_validation | 7063.59 | 200.00 | 300.00 | True |
| HLA | HLA_train2011_seed0_action_safe | 2011 | observed_year_leave_one | 7321.97 | 200.00 | 300.00 | False |
| LCA | LCA_train2010_seed0_action_safe | 2010 | observed_year_leave_one | 9040.48 | 200.00 | 300.00 | False |
| SYA | SYA_train2012_seed0_action_safe | 2012 | observed_year_leave_one | 10781.32 | 200.00 | 300.00 | False |
| YCA | YCA_train2008_seed0_action_safe | 2008 | limited_two_year_cross_validation | 9364.83 | 200.00 | 300.00 | True |

## Next step recommendation

Because FQA/YCA are also expected to be checked for cap saturation, the next robust step is season cap sensitivity analysis and/or adding explicit water/nitrogen cost terms before treating these policies as final management recommendations. Multi-seed stability analysis should come after the action/cost design is less dominated by fixed caps.