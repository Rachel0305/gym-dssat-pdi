# 008_18 HLA 2004 PPO Stability And Reward Diagnosis Report

## Purpose

Consolidate the HLA 2004 PPO/forecast-gate experiment chain from 008_11 to 008_17 into a clean report.

This is a reporting task only. Do not train PPO.

## Strict Constraints

- Do not train PPO.
- Do not modify `my_data/`.
- Do not modify reward files.
- Do not overwrite 008_11 to 008_17 results.
- Do not run new DSSAT episodes.

## Inputs

Use existing outputs:

- 008_11 pure forecast rule replay;
- 008_12 hard-minimum forecast-gate PPO contribution test;
- 008_15 soft-stress PPO seed0 10k;
- 008_16 seed stability seed1 degeneration;
- 008_17 stronger soft-stress penalty seed1 partial recovery.

## Required Summary Table

Create a core table with:

- experiment;
- seed;
- reward/gate design;
- total irrigation;
- total nitrogen;
- final GRNWT;
- GRNWT / fixed_I120;
- SWFAC stress days;
- `ppo_autonomous_contribution`;
- raw action pattern;
- strategy structure;
- interpretation.

The `ppo_autonomous_contribution` column is essential:

- pure rule: 0 mm;
- hard-minimum PPO: 0 mm;
- soft PPO: total irrigation after gate when hard minimum is removed.

## Required Conclusions

Separate the conclusions into two levels.

### Proven

The forecast/stress-aware stage decision framework can produce agronomically interpretable irrigation behavior in HLA 2004.

Under no-hard-minimum soft-stress PPO, PPO can make autonomous irrigation decisions and learn late-stage supplemental irrigation.

### Not Yet Solved

The PPO policy is seed-sensitive. Seed0 learned a strong S4/S5 irrigation strategy, while seed1 degenerated under the original soft penalty and only partially recovered under stronger penalty.

This means the framework is promising but not yet a production-level stable PPO strategy.

## Required Next Step

Recommend:

1. Do not continue blind penalty scanning.
2. Use the current HLA 2004 diagnosis as a staged result.
3. Run a minimal FQA 2016 second-year validation using the 008_17 stronger penalty configuration, without tuning.

## Required Outputs

Save:

- `Leave_One_experiments/hla2004_ppo_stability_reward_diagnosis_008_18/evaluation/008_18_hla2004_core_experiment_summary.csv`
- `docs/2026-06-14_008_18_hla2004_ppo_stability_and_reward_diagnosis_report.md`

