# 2026-08-07 Recent experiment synthesis and GitHub backup record

## Scope

This record summarizes the recent DSSAT/gym-DSSAT experiments prepared for GitHub backup. It covers the SYA originIC line, external stress-triggered nitrogen baselines, and the lowIC/originIC site-transfer experiments for LCA, HLA, YCA, and FQA.

The purpose is archival and manuscript-readiness triage. It does not change any training script, reward function, baseline, or result file.

## Experiment families

### SYA originIC method development

- `046_02`: SYA originIC binary-timing PPO baseline.
- `046_07`: train-stat normalized PPO variant.
- `046_09`: raw-weather observation PPO variant.
- `046_10`: expanded-action MaskablePPO, used as the main SYA PPO candidate for subsequent reporting.
- `046_11`: external automatic nitrogen rule implementation, using daily `NSTRES`-based trigger logic through the gym-DSSAT/PDI interaction path.

Important interpretation:

- The daily interface itself is a gym-DSSAT capability and should not be claimed as the core novelty.
- The defensible engineering contribution is the reproducible diagnosis of non-operational DSSAT-native auto-N pathways in the current workflow, plus an explicit external controller with season-level closure checks.
- `recorded_farmer_template` remains frozen unless explicitly reopened; it should not be retuned to make PPO look better.

### External automatic N threshold baselines

Current threshold naming must be kept exact:

- `nstd005` means `nitrogen_stress_threshold = 0.05`.
- `nstd030` means `nitrogen_stress_threshold = 0.30`.
- `nstd050` means `nitrogen_stress_threshold = 0.50`.
- The implemented trigger is `NSTRES >= threshold`, with a fixed dose of 25 kg/ha N per triggered day in the minimal controller.

Manuscript label:

- Use `DSSAT automatic irrigation + external stress-triggered N`.
- Do not call it `DSSAT native automatic fertilization`.

### Cross-site transfer experiments

Recent site-transfer experiments reused the SYA expanded-action MaskablePPO pattern and compared it with four baselines:

- null management
- recorded farmer template
- official extension expert
- DSSAT automatic irrigation + external stress-triggered N
- PPO candidate

Key result directories:

- SYA accepted formal baseline: `benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_nstd050_minimal_auto_nstd050_minimal_five_scenario_daily_ckpt100000`
- SYA sensitivity baseline: `benchmark_results/046_05_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_nstd005_minimal_auto_nstd005_minimal_reporting_ckpt100000`
- LCA accepted result: `benchmark_results/053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd030_minimal_five_scenario_figures_ckpt100000_auto_nstd030_minimal_run_auto_nstd030_minimal`
- HLA weak result: `benchmark_results/054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000`
- YCA mixed result: `benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000`
- FQA weak result: `benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000`

## Current quantitative triage

Values below are means across validation years in the available summary CSVs.

| Site/result | Scenario | Yield kg/ha | WP_ET kg/m3 | PFP_N kg/kg | Irrigation mm | N kg/ha | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| SYA nstd005 | PPO | 10072.6 | 2.110 | 42.66 | 240.0 | 236.0 | Nearly ties auto yield, but uses more water and N. |
| SYA nstd005 | external auto N | 10049.2 | 2.162 | 65.22 | 111.5 | 157.5 | Strong Pareto baseline; not a weak comparator. |
| LCA nstd030 | PPO | 9140.7 | 2.723 | 38.09 | 208.5 | 240.0 | Accepted; PPO leads yield and WP_ET, auto leads N efficiency. |
| LCA nstd030 | external auto N | 6983.9 | 2.041 | 110.18 | 181.2 | 47.5 | High PFP_N due to low N use; lower yield. |
| HLA nstd050 | PPO | 6244.5 | 1.447 | 50.71 | 78.0 | 124.0 | Weak-to-mixed; above auto/farmer yield but below expert. |
| HLA nstd050 | official expert | 6797.0 | 1.475 | 22.89 | 266.0 | 297.0 | Best mean yield and WP_ET in current HLA result. |
| YCA nstd050 | PPO | 6072.6 | 1.879 | 37.95 | 45.0 | 160.0 | Mixed; PFP_N strong, yield/WP_ET not competitive. |
| YCA nstd050 | official expert | 8200.6 | 2.293 | 33.48 | 211.0 | 245.0 | Better yield, lower PFP_N than PPO. |
| FQA nstd050 | PPO | 7315.3 | 1.996 | 33.87 | 219.0 | 240.0 | Not publication-ready as a success case. |
| FQA nstd050 | recorded farmer | 7201.7 | 2.145 | 55.58 | 75.0 | 144.0 | Strong baseline for water and N efficiency. |

## Current manuscript implication

The current results do not support the story `RL universally beats traditional management`.

A stronger and more defensible story is:

> Across five site-specific DSSAT/gym-DSSAT water-nitrogen experiments, free-timing PPO showed site-dependent advantages rather than universal dominance. In some environments it improved yield or water productivity, whereas a simple stress-triggered nitrogen controller formed a strong Pareto baseline for nitrogen efficiency. This benchmark therefore identifies when RL adds value and when interpretable threshold control already approaches the local management frontier.

This framing is better aligned with the evidence and with reviewer expectations for strong baselines.

## Station-level status

- SYA: usable. Keep `nstd050` as the formal auto-N baseline and `nstd005` as a sensitivity/strong-controller result.
- LCA: usable. This is the cleanest site for PPO yield/WP_ET advantage with a clear N-efficiency tradeoff.
- YCA: mixed. PPO has PFP_N value but insufficient yield/WP_ET robustness; use as boundary or sensitivity evidence.
- HLA: weak. PPO does not dominate expert management; use cautiously.
- FQA: weak. Current PPO is not a success case and should be diagnosed separately if the paper needs all five sites to show RL value.

## FQA diagnosis note

The current FQA PPO run used the SYA-derived expanded-action setup:

- observation contract: raw observation, no normalization, no weather forecast
- irrigation levels: 0, 15, 30, 45 mm
- nitrogen levels: 0, 40, 80, 120 kg/ha
- main 100k non-zero action pairs: `I0/N80`, `I15/N0`, `I45/N80`

This suggests that the FQA issue is not a transmission failure. It is more likely a station-specific objective/action-structure mismatch, especially because PPO used relatively high irrigation and N while failing to improve WP_ET or PFP_N.

## Recommended next steps

1. Back up this evidence state to GitHub with lightweight configs, scripts, docs, summary tables, and figures.
2. Avoid committing large model files, repeated DSSAT runtime folders, snapshots, tensorboard logs, and generated cache files.
3. Build a five-site manuscript matrix from existing CSVs: yield, WP_ET, PFP_N, irrigation, N, and per-year win counts.
4. For FQA, run only a small diagnostic before any full retraining: inspect checkpoint-level action diversity, reward decomposition, and water/N event timing.
5. For manuscript figures, prefer Pareto plots and rank heatmaps over a single `PPO wins` bar chart.

## Backup intent

This backup is intended to preserve the current experimental state before further FQA-specific debugging or manuscript restructuring. The backup should include enough files to reproduce the evidence trail without pushing large runtime artifacts.
