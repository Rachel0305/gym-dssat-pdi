# 008_05 Reselect Water-Stress Showcase Years from Existing Results

Date: 2026-06-13

## Background

After 008_04, the supervisor clarified two rules:

1. if the reference scenario shows no water stress, PPO should not irrigate because it wastes cost and labor;
2. the next PPO showcase should use station-years with real SWFAC water stress and clear irrigation-response evidence.

Previous work already screened observed-year and all-year station-years:

- 006_14 water x nitrogen factorial diagnosis;
- 006_15 all-year weather/stress scenario pool;
- 007_02 fixed irrigation response diagnosis.

Therefore, do not rerun all DSSAT simulations. This task is a secondary selection and reinterpretation of existing results under the supervisor's updated process-oriented standard.

## Constraints

- Do not train PPO.
- Do not run new bulk DSSAT simulations.
- Do not do rainfall scaling.
- Do not modify reward.
- Do not modify `my_data/` original files.
- Do not overwrite 006/007/008 prior outputs.
- Use existing CSV/report outputs as the primary evidence.
- Minimize compute and avoid OOM.

## Main Question

Which station-years are best suited to demonstrate water-nitrogen optimization, where irrigation can be explained as a response to SWFAC water stress rather than as fixed early irrigation?

## Important Method Note

The strict `T0_null_zero` scenario may show no SWFAC water stress because the crop is severely nitrogen-limited and water demand is low. Therefore, report two reference standards:

1. `strict_null_zero_water_stress`: water stress under no water and no nitrogen.
2. `agronomic_rainfed_n_water_stress`: water stress under rainfed but adequate nitrogen, usually `T1_N_only_medium` or fixed `I0_N150`.

The supervisor's standard should be reported honestly, but the agronomic limitation of strict null must also be documented.

## Input Files

Use existing outputs:

```text
Leave_One_experiments/all_year_weather_calibration_validation/scenario_pool/all_year_weather_scenario_pool.csv
Leave_One_experiments/all_year_weather_calibration_validation/stress_diagnostics/all_year_fixed_management_stress_summary.csv
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/irrigation_response_screening.csv
Leave_One_experiments/water_nitrogen_factorial_diagnosis/evaluation/water_nitrogen_factorial_summary.csv
Leave_One_experiments/fixed_irrigation_response_diagnosis/evaluation/fixed_irrigation_response_by_station_year.csv
Leave_One_experiments/fixed_irrigation_response_diagnosis/evaluation/fixed_irrigation_response_summary.csv
```

If a file is missing, use the closest available report and record the missing file. Do not rerun bulk simulations.

## Selection Criteria

For each station-year, compute a candidate score based on:

- `N_only` or `I0_N150` SWFAC stress days;
- maximum SWFAC under rainfed adequate-N condition;
- yield gain from irrigation at the same N level;
- whether irrigation reduces SWFAC;
- whether low-water-cost profit gain is positive;
- whether the year is already present in daily outputs for plotting;
- whether it is an observed year or an all-year weather scenario.

Flag:

- `strict_null_pass`: strict `T0_null_zero` has SWFAC stress.
- `agronomic_water_stress_pass`: rainfed adequate-N scenario has SWFAC stress.
- `biophysical_irrigation_response_pass`: fixed-N irrigation increases yield.
- `economic_low_cost_pass`: low-water-cost profit is positive.
- `showcase_priority`: high / medium / low / reject.

## Required Outputs

Use:

```text
Leave_One_experiments/water_stress_showcase_selection_008_05/
```

Generate:

```text
evaluation/008_05_water_stress_showcase_candidate_ranking.csv
evaluation/008_05_selection_source_audit.csv
figures/008_05_candidate_score_top20.png
figures/008_05_water_stress_vs_yield_gain.png
docs/2026-06-13_008_05_water_stress_showcase_selection_report.md
docs/008_008_05_water_stress_showcase_selection.pptx
```

## Interpretation Rules

- Do not hide that strict `T0_null_zero` may show no water stress.
- Do not automatically reject all years just because strict null has no SWFAC; explain that `N_only` is the agronomically meaningful rainfed-water-stress reference.
- Do not use FQA 2008 as the primary water-optimization showcase if strict process logic cannot convince the supervisor.
- Recommend 1-3 station-years for the next stress-aware PPO training.

## Decision Criteria

After this task, choose:

1. primary water-stress showcase year;
2. backup showcase year;
3. whether the next step should run stress-aware PPO on a single chosen year or first produce four-scenario comparison plots for top candidates.

