# 032_17 LC 75k PPO all-year five-scenario daily package

## Purpose

Freeze the current best LC multiyear 75k free-timing MaskablePPO candidate and build advisor-facing five-scenario daily process figures for all available LC2005-LC2020 years.

## Scope

- Station: LC / LCA.
- Frozen model evidence:
  - LC2005-LC2010: 032_11 train-year deterministic evaluations at checkpoint 75k.
  - LC2011-LC2020: 032_12 frozen transfer evaluations at checkpoint 75k.
- Four baselines:
  - null
  - recorded_farmer
  - dssat_auto
  - official_extension_expert
- No training.
- No DSSAT rerun.
- No model reselection.

## Figure panels

For each year, draw the same eight-panel package:

1. rainfall + Tmax/Tmin;
2. water stress index;
3. nitrogen stress index;
4. cumulative common reward;
5. irrigation events;
6. nitrogen application events;
7. grain and biomass trajectories;
8. endpoint summary table.

Do not plot unavailable SWTD/soil-water storage as if it existed.

## Common reward for plotting

Do not use source `reward` columns for cross-scenario comparison. Recompute one common scaled reward for all scenarios:

```text
common_reward_step =
  0.001 * (
      0.158 * grnwt
      - 1.1 * irrigation_mm
      - 1.58 * nitrogen_kg_ha
      + 10.0 * irrigation_mm * max(previous_wspd - current_wspd, 0)
      + 5.0 * nitrogen_kg_ha * max(previous_nstd - current_nstd, 0)
  )
```

Then plot cumulative sum by DAP.

## Outputs

Write under:

`benchmark_results/032_17_lc_75k_ppo_all_year_five_scenario_daily_package/`

Required:

- one daily CSV and one summary CSV per year;
- one PNG/SVG figure per year;
- combined daily and summary CSV;
- manifest CSV;
- record MD in `docs/`.

## Interpretation boundary

- This is a packaging/reporting task for the frozen 75k LC model.
- Do not claim new training success.
- If figures show suspicious agronomic behavior in a year, record it as an interpretation target rather than silently editing the policy.
