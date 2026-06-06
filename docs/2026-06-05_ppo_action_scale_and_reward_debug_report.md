# PPO action scale and reward debug report

## 1. 为什么不能直接批量 PPO

上一阶段 HLA 2007 debug PPO 能跑通，但 total_irrigation 超过 3000 mm，total_n_fertilizer 超过 14000 kg/ha，远高于 fixed_high_input 的 120 mm / 250 kg/ha，因此不能直接批量训练。

## 2. 当前 PPO 每日动作诊断

| eval_year | daily_csv_path | days | nonzero_irrigation_days | nonzero_fertilization_days | max_daily_irrigation | max_daily_fertilization | mean_daily_irrigation | mean_daily_fertilization | sum_irrigation | sum_fertilization | median_normalized_action_amir | median_normalized_action_anfer | max_normalized_action_amir | max_normalized_action_anfer | min_normalized_action_amir | min_normalized_action_anfer | totir_last | tofer_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007 | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2007_seed0_daily.csv | 149 | 149 | 149 | 25.6 | 102.1 | 20.59 | 97.34 | 3068 | 1.45e+04 | -0.1861 | -0.02769 | 0.02385 | 0.02113 | -0.1959 | -0.03464 | 3047 | 1.45e+04 |
| 2009 | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2009_seed0_daily.csv | 159 | 159 | 159 | 25.27 | 102.2 | 20.62 | 97.39 | 3279 | 1.549e+04 | -0.186 | -0.02768 | 0.01067 | 0.02245 | -0.1959 | -0.03464 | 3257 | 1.549e+04 |
| 2011 | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2011_seed0_daily.csv | 169 | 169 | 169 | 25.48 | 102.1 | 20.63 | 97.38 | 3486 | 1.646e+04 | -0.186 | -0.02769 | 0.01935 | 0.02129 | -0.1959 | -0.03464 | 3465 | 1.646e+04 |

## 3. action_space 和动作转换

| station | year | action_name | action_space_low | action_space_high | normalization_formula | denormalization_formula | example_normalized_minus1 | example_normalized_0 | example_normalized_plus1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | 2007 | amir | 0 | 50 | normalized = 2 * ((real - low) / (high - low)) - 1 | real = low + 0.5 * (normalized + 1) * (high - low) | 0 | 25 | 50 | PPO outputs normalized action in [-1, 1]. |
| HLA | 2007 | anfer | 0 | 200 | normalized = 2 * ((real - low) / (high - low)) - 1 | real = low + 0.5 * (normalized + 1) * (high - low) | 0 | 100 | 200 | PPO outputs normalized action in [-1, 1]. |

## 4. total input 重算

| eval_year | manual_sum_irrigation | summary_total_irrigation | irrigation_match | manual_sum_fertilization | summary_total_n_fertilizer | fertilization_match | totir_last | tofer_last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2007 | 3068 | 3068 | True | 1.45e+04 | 1.45e+04 | True | 3047 | 1.45e+04 |
| 2009 | 3279 | 3279 | True | 1.549e+04 | 1.549e+04 | True | 3257 | 1.549e+04 |
| 2011 | 3486 | 3486 | True | 1.646e+04 | 1.646e+04 | True | 3465 | 1.646e+04 |

## 5. reward 诊断

- 当前 reward review: `Leave_One_experiments/ppo_action_debug/reward_diagnostics/current_reward_function_review.md`
- reward candidates: `Leave_One_experiments/ppo_action_debug/reward_diagnostics/reward_revision_candidates.md`
- 诊断结论：当前 reward/成本不足以在 debug PPO 中阻止高频水氮动作；本阶段先不改 reward，只隔离测试 action safety。

## 6. action safety 方案

参数来自 `experiments/ppo_observed_years/config_ppo_action_safe_debug.yaml`：daily irrigation <= 40 mm，daily N <= 80 kg/ha，season irrigation <= 200 mm，season N <= 300 kg/ha，并限制最小间隔和 DAP 范围。

## 7. action-safe debug PPO 新旧对比

| model_type | eval_year | final_grnwt | total_irrigation | total_n_fertilizer | mean_reward | sum_reward | run_status | daily_csv_path | figure_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_ppo | 2007 | 4463 | 3068 | 1.45e+04 | -301 | -4.486e+04 | ok | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2007_seed0_daily.csv | Leave_One_experiments/ppo_observed_years/figures/HLA/HLA_train2007_seed0_debug/eval_2007 |
| current_ppo | 2011 | 4444 | 3486 | 1.646e+04 | -315.6 | -5.333e+04 | ok | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2011_seed0_daily.csv | Leave_One_experiments/ppo_observed_years/figures/HLA/HLA_train2007_seed0_debug/eval_2011 |
| current_ppo | 2009 | 4614 | 3279 | 1.549e+04 | -309.1 | -4.914e+04 | ok | Leave_One_experiments/ppo_observed_years/daily_outputs/HLA/HLA_train2007_eval2009_seed0_daily.csv | Leave_One_experiments/ppo_observed_years/figures/HLA/HLA_train2007_seed0_debug/eval_2009 |
| action_safe_ppo | 2007 | 7236 | 200 | 300 | 108.3 | 1.614e+04 | ok | Leave_One_experiments/ppo_action_debug/daily_outputs/HLA/HLA_train2007_eval2007_seed0_daily.csv | Leave_One_experiments/ppo_action_debug/figures/HLA/HLA_train2007_seed0_action_safe_debug/eval_2007 |
| action_safe_ppo | 2011 | 6739 | 200 | 300 | 73.09 | 1.235e+04 | ok | Leave_One_experiments/ppo_action_debug/daily_outputs/HLA/HLA_train2007_eval2011_seed0_daily.csv | Leave_One_experiments/ppo_action_debug/figures/HLA/HLA_train2007_seed0_action_safe_debug/eval_2011 |
| action_safe_ppo | 2009 | 7403 | 200 | 300 | 73.71 | 1.172e+04 | ok | Leave_One_experiments/ppo_action_debug/daily_outputs/HLA/HLA_train2007_eval2009_seed0_daily.csv | Leave_One_experiments/ppo_action_debug/figures/HLA/HLA_train2007_seed0_action_safe_debug/eval_2009 |

## 8. 是否可以进入批量训练

可以进入启用 action safety 的逐模型小批量训练准备，但不能使用无 safety 的旧 PPO 直接批量训练；仍建议先跑 HLA/SYA/LCA 每个模型的 pretrain smoke check。
