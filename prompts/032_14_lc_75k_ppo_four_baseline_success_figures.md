# 032_14 LC 75k PPO four-baseline success figures

## Purpose

Generate two advisor-facing LC figures for the current frozen 75k no-forecast free-timing MaskablePPO result:

1. Year-by-year bar/delta chart comparing the PPO candidate with the best of the four baseline scenarios.
2. Cross-year mean and standard deviation summary for yield, nitrogen productivity, and water-related metrics.

## Scope

- Station: LC / LCA.
- Model: frozen 032_11 75k checkpoint.
- Years:
  - train-year deterministic evaluation: LC2005-LC2010;
  - frozen future-year transfer: LC2011-LC2020.
- No training.
- No DSSAT rerun.
- No model reselection.

## Four baseline scenarios

Use exactly:

- `null`
- `recorded_farmer`
- `dssat_auto`
- `official_extension_expert`

Do not include `recorded_farmer_template_*` rows as extra baselines.

## Metrics

Compute:

- grain yield: kg/ha;
- PFP_N: grain yield / fertilizer N, kg grain per kg N;
- irrigation total: mm, reported as saving vs official expert;
- nitrogen total: kg/ha, reported as saving vs official expert.

WP_ET should only be computed if the RL result has a defensible ET denominator. If the current RL output has no ET column, mark WP_ET as not available instead of inferring it.

## Output

Write under:

`benchmark_results/032_14_lc_75k_ppo_four_baseline_success_figures/`

Required:

- `tables/032_14_lc_75k_ppo_vs_four_baseline.csv`
- `figures/032_14_lc_75k_ppo_yearly_vs_four_baseline.png`
- `figures/032_14_lc_75k_ppo_crossyear_std_summary.png`
- `docs/032_14_lc_75k_ppo_four_baseline_success_figures_record.md`

## Interpretation boundary

- “Training success” can be claimed only in the relaxed advisor sense: at least one target metric exceeds the best of the four baselines.
- If success is driven mainly by PFP_N rather than yield, state that clearly.
- Do not claim WP_ET success if ET is unavailable for the PPO daily output.
