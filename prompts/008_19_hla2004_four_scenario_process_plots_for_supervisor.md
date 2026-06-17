# 008_19 HLA 2004 Four-Scenario Process Plots For Supervisor

## Purpose

Generate supervisor-facing HLA 2004 process plots for four management scenarios:

1. `null_zero`
2. `expert_reference_recorded`
3. `dssat_auto_attempt`
4. `ppo_soft_stress_seed0_00815`

This is a plotting/reporting task only. Do not train PPO and do not rerun DSSAT.

## Strict Constraints

- Do not train PPO.
- Do not rerun DSSAT.
- Do not modify `my_data/`.
- Do not overwrite 008_06 or 008_15 outputs.
- Use existing daily CSVs.
- Clearly label `dssat_auto_attempt` as an attempt through gym-DSSAT, not a confirmed full native DSSAT automatic-management baseline.

## Required Input Data

Use:

- `Leave_One_experiments/representative_management_comparison_008_06/daily_outputs/HLA/2004_null_zero_daily.csv`
- `Leave_One_experiments/representative_management_comparison_008_06/daily_outputs/HLA/2004_expert_reference_recorded_daily.csv`
- `Leave_One_experiments/representative_management_comparison_008_06/daily_outputs/HLA/2004_dssat_auto_attempt_daily.csv`
- `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_15_10k/daily_outputs/HLA/HLA_2004_seed0_stress_aware_stage_daily.csv`
- `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_15_10k/daily_outputs/HLA/HLA_2004_seed0_stress_aware_stage_steps.csv`

## Required Outputs

For each scenario, generate four process plots:

1. Rainfall/weather plot;
2. SWFAC/NSTRES stress plot;
3. Irrigation/fertilization management action plot;
4. TOPWT/GRNWT crop growth plot.

Also generate:

- a combined comparison figure;
- a scenario summary CSV;
- a Markdown report.

Output root:

- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19`

Markdown report:

- `docs/2026-06-14_008_19_hla2004_four_scenario_process_plots_for_supervisor_report.md`

## Interpretation Rules

State clearly:

- HLA 2004 is the current main water-limited demonstration case.
- PPO should be represented by 008_15 soft-stress PPO, not by the old FQA transfer replay in 008_06.
- `dssat_auto_attempt` behaved like null in the gym-DSSAT pathway and should be treated as diagnostic rather than confirmed native DSSAT automatic management.
- PPO demonstrates autonomous contribution under no-hard-minimum gate, but seed stability remains an unresolved limitation.

