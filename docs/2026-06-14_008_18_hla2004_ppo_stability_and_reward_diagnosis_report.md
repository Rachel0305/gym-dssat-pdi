# 008_18 HLA 2004 PPO Stability And Reward Diagnosis Report

## Purpose

This report consolidates the HLA 2004 forecast/stress-aware irrigation PPO experiment chain from 008_11 to 008_17.
No new training or DSSAT simulation was performed.

## Core Experiment Chain

| experiment | seed | design | total_irrigation | total_n | final_grnwt | grnwt_fraction_vs_fixed_I120 | swfac_days_gt_0p05 | ppo_autonomous_contribution | raw_action_pattern | strategy_structure | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 008_11_pure_forecast_rule | none | forecast gate pure rule, 30 mm when triggered | 120.0 | 150.0 | 6940.0507 | 0.9827 | 19 | 0.0 | not applicable | rule triggered S2-S5 irrigation | Agronomically interpretable forecast rule, but no PPO decision. |
| 008_12_hard_min_gate_ppo | 0 | PPO under hard-minimum forecast gate | 120.0 | 150.0 | 6940.0507 | 0.9827 | 19 | 0.0 | PPO extra above gate minimum = 0 | rule-dominated | Good yield but PPO contribution was zero. |
| 008_15_soft_stress_ppo_10k | 0 | no hard minimum, soft SWFAC penalty 1.0/0.25 | 80.0 | 150.0 | 6777.9266 | 0.9597 | 22 | 80.0 | S1:raw=-1.00,I=0.0; S2:raw=-1.00,I=0.0; S3:raw=-1.00,I=0.0; S4:raw=1.00,I=40.0; S5:raw=1.00,I=40.0 | S4/S5 full late irrigation | PPO made autonomous nonzero decisions and reached 96% of fixed I120 yield with 80 mm irrigation. |
| 008_16_soft_stress_seed1_5k | 1 | no hard minimum, soft SWFAC penalty 1.0/0.25 | 0.0 | 150.0 | 761.9897 | 0.1079 | 20 | 0.0 | S1-S5 raw=-1.00 | zero-irrigation degeneration | Original soft penalty did not remove the bad zero-irrigation attractor for seed1. |
| 008_17_stronger_penalty_seed1_5k | 1 | no hard minimum, stronger SWFAC penalty 3.0/1.0 | 53.6272 | 150.0 | 5579.4836 | 0.79 | 31 | 53.6272 | S1-S3 raw=-1.00; S4/S5 raw=0.34 | S4/S5 partial late irrigation | Stronger penalty recovered seed1 from zero irrigation but did not reach seed0 yield level. |

## Proven Result

The forecast/stress-aware stage decision framework can produce agronomically interpretable irrigation behavior in HLA 2004.
The pure forecast rule responds to dry forecast windows, and the no-hard-minimum PPO design demonstrates that PPO can make autonomous irrigation decisions when the rule only allows or blocks irrigation.

The strongest positive case is 008_15 seed0: PPO applied 80 mm in S4/S5, reached 6777.93 kg/ha GRNWT, and achieved 95.97% of the fixed I120_N150 reference yield without hard-minimum rule support.

## Unresolved Limitation

The PPO policy is seed-sensitive under the current reward design.
Seed0 found a strong S4/S5 late-irrigation strategy, but seed1 degenerated to zero irrigation under the original soft penalty.
A stronger SWFAC penalty recovered seed1 to a partial S4/S5 irrigation strategy, but yield reached only 79.0% of fixed I120_N150.

This means the framework is promising, but the current PPO configuration should not yet be presented as a production-level stable policy.

## Interpretation For The Paper

Two layers should be separated in the manuscript:

1. Proven: a weather-forecast and stress-diagnosis assisted stage decision framework can make irrigation decisions agronomically interpretable and allows PPO to contribute when hard minimum gate support is removed.
2. Not yet solved: PPO water amount is sensitive to random initialization, so reward scaling and training stability require further study.

## Recommended Next Step

Do not continue blind penalty scanning on HLA 2004.
Use this diagnosis as a staged HLA 2004 result, then run a minimal FQA 2016 second-year validation using the 008_17 stronger penalty configuration without tuning.

## Files

- Core summary CSV: `Leave_One_experiments/hla2004_ppo_stability_reward_diagnosis_008_18/evaluation/008_18_hla2004_core_experiment_summary.csv`