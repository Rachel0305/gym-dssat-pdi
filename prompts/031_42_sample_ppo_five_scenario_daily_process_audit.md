# 031_42 Sample free-timing PPO five-scenario daily process audit

## Purpose

Create a small set of five-scenario daily process figures for already frozen free-timing MaskablePPO candidates, so that PPO decisions can be visually checked against weather, stress indices, management events, crop growth, and cumulative reward/proxy trajectories.

This task is a plotting/audit task only.

- No PPO/DQN training.
- No DSSAT rerun.
- No checkpoint reselection.
- No reward or parameter change.

## Source data

Use the currently frozen 031_38 selected PPO candidates:

```text
benchmark_results/031_38_five_station_ppo_water_n_saving_summary/tables/031_38_selected_station_year_ppo_deltas_vs_official.csv
```

Use four-scenario daily baselines generated for the 031_34/031_38 evaluation chain:

```text
benchmark_results/031_35_missing_four_baseline_completion_for_03134/evaluation/031_35_full_generated_baseline_daily.csv
benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_generated_dssat_auto_daily.csv
```

The 031_35/031_36 daily CSV files do not include a complete soil-water storage column. Therefore, this audit figure must not fabricate SWTD. Replace the previous soil-water panel with cumulative irrigation amount, and record this limitation explicitly.

## Fixed sample list

Use six samples that cover different stations and management patterns:

| station_code | year | reason |
|---|---:|---|
| HLA | 2008 | HLA candidate with complete five-scenario daily sources |
| HLA | 2023 | HLA later-year candidate with complete five-scenario daily sources |
| LCA | 2012 | LC candidate with moderate water and N |
| LCA | 2023 | LC candidate with lower N |
| YCA | 2015 | YC candidate with relatively low irrigation and complete sources |
| YCA | 2023 | YC later-year candidate with complete sources |

Do not add or remove samples after seeing plots.

## Output

Write outputs under:

```text
benchmark_results/031_42_sample_ppo_five_scenario_daily_process_audit/
```

Required files:

- `figures/*.png` and `figures/*.svg`
- `tables/031_42_sample_manifest.csv`
- `tables/031_42_sample_five_scenario_daily.csv`
- `tables/031_42_sample_five_scenario_summary.csv`
- `tables/031_42_source_coverage_checks.csv`
- `docs/031_42_sample_ppo_five_scenario_daily_process_audit_record.md`

## Figure panels

Use one 4x2 figure per station-year:

1. Weather: rain, Tmax, Tmin
2. Cumulative irrigation amount, because SWTD is unavailable in this daily CSV source
3. Water stress index
4. Nitrogen stress index
5. Irrigation events
6. Nitrogen application events
7. Grain and biomass trajectories
8. Cumulative reward/proxy

## Interpretation constraints

- These figures are visual diagnostics of already frozen candidates.
- A visually plausible action sequence is not a causal proof that every action was necessary.
- If a management action appears questionable, follow-up evidence should come from a separate same-prefix counterfactual DSSAT audit.
