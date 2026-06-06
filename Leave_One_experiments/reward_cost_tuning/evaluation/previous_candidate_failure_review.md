# Previous candidate failure review

Generated at: 2026-06-06

The 006_03 HLA pilot tested current_reward_baseline, A1-A3, B1-B3, and C1. All completed HLA 2011/2007/2009 evaluations but all candidates still used 300 mm irrigation and 450 kg ha-1 N. Therefore no candidate passed the unsaturated filter and SYA/LCA extension was skipped.

## Previous comparison

| reward_version | reward_family | mean_yield | mean_reward | mean_irrigation | mean_n | mean_irrigation_saturation_ratio | mean_n_saturation_ratio | recommendation_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_A1 | candidate_A_linear_cost | 6843.46923828125 | 66.72177745699577 | 300.0 | 450.0 | 1.0 | 1.0 | False |
| candidate_A2 | candidate_A_linear_cost | 6839.4482421875 | 66.45663154599248 | 300.0 | 450.0 | 1.0 | 1.0 | False |
| candidate_A3 | candidate_A_linear_cost | 6843.167521158855 | 66.12151309558004 | 300.0 | 450.0 | 1.0 | 1.0 | False |
| candidate_B1 | candidate_B_linear_plus_excess_penalty | 6878.356323242188 | 12.07133100773791 | 300.0 | 450.0 | 1.0 | 1.0 | False |
| candidate_B2 | candidate_B_linear_plus_excess_penalty | 7015.935262044271 | -235.3189890858692 | 300.0 | 450.0 | 1.0 | 1.0 | False |
| candidate_B3 | candidate_B_linear_plus_excess_penalty | 6930.003255208333 | -1027.3119886675404 | 300.0 | 450.0 | 1.0 | 1.0 | False |
| candidate_C1 | candidate_C_incremental_yield_minus_total_cost | 6868.2375081380205 | 1.956939893612208 | 300.0 | 450.0 | 1.0 | 1.0 | False |
| current_reward_baseline | baseline | 6843.57401529948 | 66.84092951279358 | 300.0 | 450.0 | 1.0 | 1.0 | False |

## Failure diagnosis

- Linear A costs were too small relative to growth/recovery reward and did not change the action allocation.
- B quadratic excess penalties changed reward scale strongly, but with only 5000 timesteps PPO still evaluated as cap-saturated.
- C1 lacked a true terminal done flag and used incremental grain-yield proxy, so it did not provide a clean season-level objective.
- The reward callback is step-wise, while the desired behavior is season-level water and nitrogen budgeting. This scale mismatch can make coefficient tuning brittle.
- This round strengthens costs and tests cumulative penalties that are visible before the cap is reached.
