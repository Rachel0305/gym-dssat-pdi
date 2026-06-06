# Reward candidate design

Generated at: 2026-06-06

This stage does not overwrite the original gym-DSSAT reward file. Candidate rewards are defined in `src/reward_candidates.py` and are injected at runtime by patching the imported `gym_dssat_pdi.envs.configs.rewards.all_reward` function before environment creation.

## Candidate families

- `current_reward_baseline`: original `all_reward`.
- `candidate_A_linear_cost`: original reward minus extra linear daily irrigation and nitrogen costs.
- `candidate_B_linear_plus_excess_penalty`: original reward minus linear costs and a quadratic penalty after cumulative irrigation exceeds 200 mm or cumulative nitrogen exceeds 300 kg/ha.
- `candidate_C_incremental_yield_minus_total_cost`: incremental grain-yield proxy minus water and nitrogen action costs. The reward callback has no explicit `done` flag, so terminal yield is approximated by positive daily `grnwt` increments.

## Candidate table

| reward_version | reward_family | description | irrigation_cost_coef | nitrogen_cost_coef | target_irrigation | target_n | excess_irrigation_coef | excess_n_coef | yield_value_coef | use_incremental_yield_proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_reward_baseline | baseline | Original gym-DSSAT all_reward used as the baseline. | 0.0 | 0.0 | 200.0 | 300.0 | 0.0 | 0.0 | 0.0 | False |
| candidate_A1 | candidate_A_linear_cost | Original all_reward plus weak linear irrigation and nitrogen action costs. | 0.05 | 0.02 | 200.0 | 300.0 | 0.0 | 0.0 | 0.0 | False |
| candidate_A2 | candidate_A_linear_cost | Original all_reward plus medium linear irrigation and nitrogen action costs. | 0.1 | 0.05 | 200.0 | 300.0 | 0.0 | 0.0 | 0.0 | False |
| candidate_A3 | candidate_A_linear_cost | Original all_reward plus stronger linear irrigation and nitrogen action costs. | 0.2 | 0.1 | 200.0 | 300.0 | 0.0 | 0.0 | 0.0 | False |
| candidate_B1 | candidate_B_linear_plus_excess_penalty | Linear costs plus weak quadratic penalty above 200 mm irrigation and 300 kg/ha N. | 0.1 | 0.05 | 200.0 | 300.0 | 0.005 | 0.002 | 0.0 | False |
| candidate_B2 | candidate_B_linear_plus_excess_penalty | Linear costs plus medium quadratic penalty above 200 mm irrigation and 300 kg/ha N. | 0.2 | 0.1 | 200.0 | 300.0 | 0.03 | 0.012 | 0.0 | False |
| candidate_B3 | candidate_B_linear_plus_excess_penalty | Linear costs plus strong quadratic penalty above 200 mm irrigation and 300 kg/ha N. | 0.5 | 0.25 | 200.0 | 300.0 | 0.1 | 0.04 | 0.0 | False |
| candidate_C1 | candidate_C_incremental_yield_minus_total_cost | Incremental grain-yield proxy minus daily irrigation and nitrogen costs. The gym reward callback has no explicit done flag, so final-yield reward is approximated by positive daily grain-yield increments. | 0.5 | 0.2 | 200.0 | 300.0 | 0.0 | 0.0 | 0.08 | True |