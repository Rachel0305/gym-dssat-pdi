# 032_19 LC 75k daily stress plot data-quality audit

## Goal

Audit why the LC2005-LC2020 032_17 daily figures show apparently inconsistent water-stress patterns, including:

- dry years with no null water stress;
- PPO-only water-stress spikes after irrigation/rain;
- PPO daily rows exceeding baseline row counts.

## Scope

- No training.
- No DSSAT reruns.
- Do not change 032_17 figures in this task.
- Read existing 032_17 daily package and available baseline DSSAT snapshots.

## Checks

1. Scenario completeness and row-count differences by year.
2. Duplicate `(scenario, date, dap)` rows, especially for `rl_candidate`.
3. Rainfall totals by scenario within each year; weather should normally be identical across scenarios.
4. For baseline scenarios with `PlantGro.OUT` snapshots, compare 032_17 plotted stress columns against true PlantGro `WSPD/NSTD`.
5. Generate focused windows around suspicious PPO water-stress peaks for LC2008, LC2013, LC2018.

## Output

- CSV tables under `benchmark_results/032_19_lc_75k_daily_stress_plot_data_quality_audit/tables/`
- Markdown record under `docs/032_19_lc_75k_daily_stress_plot_data_quality_audit_record.md`

## Interpretation

If 032_17 uses environment `swfac/nstres` where true PlantGro `WSPD/NSTD` is required, or if PPO rows duplicate DAP/date values, the 032_17 stress panels should be treated as unsafe for advisor-facing decision-rationale interpretation until rebuilt from consistent DSSAT daily outputs.
