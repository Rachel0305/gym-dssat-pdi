# 031_30 SY PPO candidate versus completed generated baselines prompt

## Purpose

Merge 031_27 frozen SYA MaskablePPO transfer results with 031_29 completed generated baselines, then quantify whether each PPO seed-year candidate exceeds the baseline envelope.

This is a comparison and reporting task only:

- no PPO/DQN training;
- no DSSAT reruns;
- no change to candidate or baseline metrics.

## Inputs

- PPO candidates: `benchmark_results/031_27_sy_all_year_frozen_maskableppo_transfer/evaluation/031_27_sy_all_year_candidate_summary.csv`
- Generated baselines: `benchmark_results/031_29_sy_auto_and_recorded_template_completion/evaluation/031_29_sy_baseline_summary.csv`

## Baseline envelope

Use all 031_29 scenarios:

- `null`
- `dssat_auto`
- `official_extension_expert`
- `recorded_farmer_template_2012`
- `recorded_farmer_template_2014`
- `recorded_farmer_template_2015`

Do not call the template-transfer scenarios real recorded-farmer observations.

## Metrics

For each PPO seed-year candidate:

- yield winner: candidate `final_grain_kg_ha` > maximum baseline `grain_yield_kg_ha`;
- WP_ET winner: candidate `wp_et_kg_m3` > maximum baseline `WP_ET_kg_m3`;
- PFP_N winner: candidate `pfp_n_kg_kg` > maximum finite baseline `PFP_N_kg_kg`;
- any-metric winner: at least one of the above is true;
- all-three winner: all three are true.

Also report:

- water saving versus official expert;
- N saving versus official expert;
- water saving versus the best-yield recorded template;
- N saving versus the best-yield recorded template.

## Required outputs

Under `benchmark_results/031_30_sy_ppo_vs_completed_baselines/`:

- `evaluation/031_30_candidate_vs_baseline_envelope.csv`
- `evaluation/031_30_baseline_envelope_by_year.csv`
- `evaluation/031_30_seed_level_summary.csv`
- `evaluation/031_30_year_level_best_candidate.csv`
- `docs/031_30_sy_ppo_vs_completed_baselines_record.md`

## Success condition

All 57 candidate rows from 031_27 and all 114 baseline rows from 031_29 must be read and accounted for. If the row counts differ, stop and record failure.
