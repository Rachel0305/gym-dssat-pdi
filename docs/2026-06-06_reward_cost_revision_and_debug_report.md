# Reward cost revision and debug report

Generated at: 2026-06-06

## Why revise reward

The 006_02 season-cap sensitivity analysis showed that action-safe PPO filled every tested cap from 100/150 to 300/450. This means the safety wrapper prevents unlimited actions, but the reward itself still encourages using all allowed water and nitrogen.

## Current reward issue

- Current `all_reward` combines nitrogen recovery reward and biomass-growth irrigation reward.
- It includes daily action costs, but no default nonlinear seasonal penalty and no explicit terminal grain-yield economic return.
- PPO therefore treats the season cap as the effective budget and learns to exhaust it.

## Candidate design

- A candidates add linear irrigation and nitrogen costs.
- B candidates add linear costs plus quadratic penalties after 200 mm irrigation and 300 kg/ha nitrogen.
- C1 approximates an economic reward by using positive daily grain-yield increments minus input costs.

## HLA pilot setup

- Station: HLA
- Train year: 2011
- Eval years: 2011, 2007, 2009
- Action safety cap: 300 mm irrigation / 450 kg ha-1 N
- Timesteps: 5000
- Seed: 0

## HLA reward candidate comparison

| reward_version | reward_family | eval_count | ok_count | mean_yield | std_yield | mean_reward | mean_irrigation | mean_n | mean_irrigation_saturation_ratio | mean_n_saturation_ratio | yield_loss_vs_current_reward | input_reduction_vs_current_reward | yield_per_100mm_irrigation | yield_per_100kg_n | recommendation_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_A1 | candidate_A_linear_cost | 3 | 3 | 6843.4692 | 273.1103 | 66.7218 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0 | 2281.1564 | 1520.7709 | False |
| candidate_A2 | candidate_A_linear_cost | 3 | 3 | 6839.4482 | 275.2457 | 66.4566 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0006 | 0.0 | 2279.8161 | 1519.8774 | False |
| candidate_A3 | candidate_A_linear_cost | 3 | 3 | 6843.1675 | 273.8958 | 66.1215 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0001 | 0.0 | 2281.0558 | 1520.7039 | False |
| candidate_B1 | candidate_B_linear_plus_excess_penalty | 3 | 3 | 6878.3563 | 264.4608 | 12.0713 | 300.0 | 450.0 | 1.0 | 1.0 | -0.0051 | 0.0 | 2292.7854 | 1528.5236 | False |
| candidate_B2 | candidate_B_linear_plus_excess_penalty | 3 | 3 | 7015.9353 | 270.5381 | -235.319 | 300.0 | 450.0 | 1.0 | 1.0 | -0.0252 | 0.0 | 2338.6451 | 1559.0967 | False |
| candidate_B3 | candidate_B_linear_plus_excess_penalty | 3 | 3 | 6930.0033 | 264.3231 | -1027.312 | 300.0 | 450.0 | 1.0 | 1.0 | -0.0126 | 0.0 | 2310.0011 | 1540.0007 | False |
| candidate_C1 | candidate_C_incremental_yield_minus_total_cost | 3 | 3 | 6868.2375 | 262.6394 | 1.9569 | 300.0 | 450.0 | 1.0 | 1.0 | -0.0036 | 0.0 | 2289.4125 | 1526.275 | False |
| current_reward_baseline | baseline | 3 | 3 | 6843.574 | 273.1875 | 66.8409 | 300.0 | 450.0 | 1.0 | 1.0 | 0.0 | 0.0 | 2281.1913 | 1520.7942 | False |

## Candidates passing initial filter

No candidate passed the initial filter.

Initial filter:

- mean irrigation saturation ratio < 0.95
- mean nitrogen saturation ratio < 0.95
- yield loss vs current reward <= 10%
- all HLA eval episodes passed quality gate

## Recommended candidate

Recommended reward for next debug stage: `None`.

## SYA/LCA extension

| selected_reward | run_status | notes |
| --- | --- | --- |
|  | skipped | No HLA candidate satisfied saturation and yield-loss criteria. |

## Interpretation

Action safety cap and reward cost are different mechanisms. The cap is a safety valve. Reward cost is the signal that should make PPO internalize water and nitrogen scarcity. If a candidate no longer fills 300/450 while keeping yield loss below 10%, it is a better debug reward for later multi-seed tests. This is still not the final paper reward; the next version should use interpretable water, nitrogen, and grain price parameters.

## Next step

Proceed to multi-seed only if the selected candidate remains below cap on HLA and the SYA/LCA extension is acceptable. If the candidate still saturates or yield collapses, tune the cost coefficients before running rainfall-scaling budget scenarios.
