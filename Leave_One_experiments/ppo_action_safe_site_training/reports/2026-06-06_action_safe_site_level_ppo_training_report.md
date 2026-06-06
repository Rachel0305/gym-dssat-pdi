# Action-safe site-level PPO training report

Generated at: 2026-06-06

## Scope

- Only action-safe PPO was trained.
- Only HLA, SYA, and LCA were included.
- FQA/YCA, multi-seed training, reward modification, and my_data modification were not performed.
- Timesteps per model: 5000
- Output root: `Leave_One_experiments\ppo_action_safe_site_training`

## Quality gates

- total_irrigation <= 300.0 mm
- total_n_fertilizer <= 400.0 kg/ha
- run_status = ok; episode completed; daily CSV and required figures exist

## Generated files

- `Leave_One_experiments\ppo_action_safe_site_training\configs\action_safe_site_training_plan.csv`
- `Leave_One_experiments\ppo_action_safe_site_training\evaluation\action_safe_site_ppo_evaluation_summary.csv`
- `Leave_One_experiments\ppo_action_safe_site_training\strategy_selection\best_policy_by_site.csv`
- `Leave_One_experiments\ppo_action_safe_site_training\strategy_selection\policy_ranking_by_site.csv`

## Model-level status

| station | policy | evaluations | passed | mean_yield | mean_irrigation | mean_n | mean_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| HLA | HLA_train2007_seed0_action_safe | 3 | 3 | 7125.01 | 200.00 | 300.00 | 85.30 |
| HLA | HLA_train2009_seed0_action_safe | 3 | 3 | 7132.83 | 200.00 | 300.00 | 85.06 |
| HLA | HLA_train2011_seed0_action_safe | 3 | 3 | 7122.66 | 200.00 | 300.00 | 84.60 |
| LCA | LCA_train2008_seed0_action_safe | 4 | 4 | 8760.11 | 200.00 | 300.00 | 135.50 |
| LCA | LCA_train2009_seed0_action_safe | 4 | 4 | 8812.40 | 200.00 | 300.00 | 137.03 |
| LCA | LCA_train2010_seed0_action_safe | 4 | 4 | 8790.05 | 200.00 | 300.00 | 136.05 |
| LCA | LCA_train2011_seed0_action_safe | 4 | 4 | 8815.43 | 200.00 | 300.00 | 137.20 |
| SYA | SYA_train2012_seed0_action_safe | 3 | 3 | 10252.54 | 200.00 | 300.00 | 102.55 |
| SYA | SYA_train2014_seed0_action_safe | 3 | 3 | 10270.50 | 200.00 | 300.00 | 102.98 |
| SYA | SYA_train2015_seed0_action_safe | 3 | 3 | 10275.28 | 200.00 | 300.00 | 103.13 |

## Best policy by site

| station | best_policy | validation_years | mean_yield | mean_irrigation | mean_n | stability_score |
|---|---|---:|---:|---:|---:|---:|
| HLA | HLA_train2011_seed0_action_safe | 2007,2009 | 7321.97 | 200.00 | 300.00 | 1.000 |
| LCA | LCA_train2010_seed0_action_safe | 2008,2009,2011 | 9040.48 | 200.00 | 300.00 | 0.000 |
| SYA | SYA_train2012_seed0_action_safe | 2014,2015 | 10781.32 | 200.00 | 300.00 | 1.000 |

## Interpretation

This stage is a controlled small-batch verification of the observed-year leave-one workflow. The action safety wrapper is a training constraint, not a final paper reward parameter. A policy should only move to longer training, FQA/YCA two-year validation, or multi-seed stability analysis after all quality gates are satisfied.