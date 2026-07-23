# 031_37 PPO water/nitrogen saving summary after completed four-baseline comparison

## Goal

Build station-level and overall water/nitrogen saving figures for the 031_34 frozen free-timing MaskablePPO candidates after 031_35/031_36 completed the four-baseline comparison table.

## Inputs

- Candidate comparison table: `benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_ppo_vs_completed_template_aware_four_baselines.csv`
- Completed template-aware baseline table: `benchmark_results/031_36_missing_dssat_auto_completion_for_03134/evaluation/031_36_full_completed_template_aware_unified_baseline_summary.csv`

## Comparison logic

- Main water/nitrogen saving baseline: `official_extension_expert`.
- Reason: `null` and some true `dssat_auto` rows can use zero water or zero nitrogen, so "saving relative to all four baselines" is not agronomically meaningful for input-use totals.
- Four-baseline outcome success is still read from `advisor_any_metric_strict_winner`, which means the PPO candidate beats the four-baseline maximum in at least one of yield, WP_ET, or PFP_N.

## Candidate selection per station-year

The comparison table has up to three seed candidates per station-year. For the station-year display, select one candidate by a fixed rule:

1. higher number of strict metric wins among yield/WP_ET/PFP_N;
2. `advisor_any_metric_strict_winner=True` before false;
3. higher `profit_simple`;
4. higher final grain yield;
5. lower irrigation;
6. lower nitrogen;
7. lower seed id.

The all-seed table is retained separately.

## Outputs

- One station figure per station (`FQA`, `HLA`, `LCA`, `YCA`): yearly irrigation and nitrogen deltas vs official expert.
- One overall figure: station-level mean irrigation and nitrogen saving vs official expert with min-max ranges.
- CSVs for selected station-year candidates, all seed deltas, and station summaries.
- Markdown record.

## Interpretation boundary

The figures summarize water/nitrogen savings relative to official expert, not proof of agronomic causality for each action. Decision reasonableness still requires separate daily-process or counterfactual audits.
