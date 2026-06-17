# Workspace cleanup plan for the current paper mainline

## Why this document exists

The workspace contains many historical experiments, failed attempts, generated model outputs, rendered DSSAT inputs, logs, archives, and current paper-mainline artifacts. Direct deletion is risky because some old outputs are still used as references in reports and comparison tables.

This cleanup plan separates files into:

1. keep for the current paper mainline;
2. keep as source data or reproducibility infrastructure;
3. archive candidates;
4. do-not-touch items.

No files are deleted by this document.

## Current paper mainline

The current paper direction is:

> Weather-forecast and stress-diagnosis assisted stage-level PPO for maize water-nitrogen management, with representative drought-year validation and explicit diagnosis of PPO contribution.

The current experimental chain starts mainly from 008 and 009:

- 008_05: water-stress showcase year selection;
- 008_06: HLA 2004 representative management scenario comparison;
- 008_11 and 008_12: rule gate versus PPO contribution diagnosis;
- 008_14 to 008_18: HLA 2004 soft-stress PPO and stability diagnosis;
- 008_19: HLA 2004 four-scenario supervisor figures and daily values;
- 008_20: FQA 2016 second water-stress year validation;
- 008_21: HLA 2004 + FQA 2016 two-case framework summary;
- 009_01 to 009_04: water-nitrogen joint PPO design and HLA 2004 probes.

## Keep: prompts

Keep these prompt files:

- `prompts/008_05_reselect_water_stress_showcase_years_from_existing_results.md`
- `prompts/008_06_representative_hla2004_water_stress_management_comparison.md`
- `prompts/008_11_forecast_gate_rule_replay_validation.md`
- `prompts/008_12_ppo_contribution_under_forecast_gate.md`
- `prompts/008_13_official_maize_vs_project_data_ppo_gap_analysis.md`
- `prompts/008_14_hla2004_irrigation_only_ppo_soft_stress_reward.md`
- `prompts/008_15_hla2004_irrigation_only_soft_stress_ppo_10k_validation.md`
- `prompts/008_16_hla2004_irrigation_only_soft_stress_ppo_seed_stability.md`
- `prompts/008_17_hla2004_stronger_soft_stress_penalty_seed_recovery.md`
- `prompts/008_18_hla2004_ppo_stability_and_reward_diagnosis_report.md`
- `prompts/008_19_hla2004_four_scenario_process_plots_for_supervisor.md`
- `prompts/008_20_fqa2016_soft_stress_ppo_second_water_stress_year_validation.md`
- `prompts/009_01_design_stage_level_water_nitrogen_joint_ppo.md`
- `prompts/009_01_design_stage_level_water_nitrogen_joint_ppo_zh.md`
- `prompts/009_02_hla2004_stage_level_water_nitrogen_joint_ppo_smoke.md`
- `prompts/009_03_water_nitrogen_reward_balance_probe.md`
- `prompts/009_04_hla2004_joint_ppo_n_cap_tightening_probe.md`

## Keep: source code

Keep these active scripts and wrappers:

- `src/soft_stress_gate_wrapper_008_14.py`
- `src/joint_soft_stress_gate_wrapper_009.py`
- `src/run_hla2004_representative_comparison_008_06.py`
- `src/run_forecast_gate_rule_replay_008_11.py`
- `src/run_ppo_contribution_forecast_gate_008_12.py`
- `src/run_hla2004_irrigation_only_ppo_soft_stress_reward_008_14.py`
- `src/run_hla2004_irrigation_only_soft_stress_seed_stability_008_16.py`
- `src/run_hla2004_stronger_soft_stress_penalty_seed_recovery_008_17.py`
- `src/generate_hla2004_ppo_stability_and_reward_diagnosis_008_18.py`
- `src/generate_hla2004_four_scenario_process_plots_008_19.py`
- `src/run_fqa2016_irrigation_only_soft_stress_ppo_008_20.py`
- `src/generate_two_case_framework_summary_008_21.py`
- `src/run_hla2004_stage_level_water_nitrogen_joint_ppo_009_02.py`
- `src/run_hla2004_joint_ppo_reward_balance_probe_009_03.py`
- `src/run_hla2004_joint_ppo_n_cap_tightening_probe_009_04.py`

Also keep shared infrastructure:

- `src/stage_action_wrapper.py`
- `src/stress_aware_stage_action_wrapper.py`
- `src/run_all_year_direct_action_safe_ppo.py`
- `src/ppo_safe_rendering.py`

## Keep: configs

Keep these experiment configs:

- `experiments/ppo_observed_years/config_008_07_hla2004_stress_aware_stage_ppo_smoke.yaml`
- `experiments/ppo_observed_years/config_008_09_hla2004_forecast_stress_gate_stage_ppo_smoke.yaml`
- `experiments/ppo_observed_years/config_008_14_hla2004_irrigation_only_ppo_soft_stress_reward.yaml`
- `experiments/ppo_observed_years/config_008_15_hla2004_irrigation_only_soft_stress_ppo_10k_validation.yaml`
- `experiments/ppo_observed_years/config_008_16_hla2004_irrigation_only_soft_stress_ppo_seed_stability.yaml`
- `experiments/ppo_observed_years/config_008_17_hla2004_stronger_soft_stress_penalty_seed_recovery.yaml`
- `experiments/ppo_observed_years/config_008_20_fqa2016_irrigation_only_soft_stress_ppo_validation.yaml`
- `experiments/ppo_observed_years/config_009_02_hla2004_stage_level_water_nitrogen_joint_ppo_smoke.yaml`

009_03 and 009_04 generate scenario configs under their own output folders and should be kept with those output folders.

## Keep: docs and reports

Keep current thesis/paper-mainline reports:

- `docs/2026-06-14_008_13_official_maize_vs_project_data_ppo_gap_analysis.md`
- `docs/2026-06-14_008_18_hla2004_ppo_stability_and_reward_diagnosis_report.md`
- `docs/2026-06-14_008_19_hla2004_four_scenario_process_plots_for_supervisor_report.md`
- `docs/2026-06-14_paper_framework_technical_roadmap.md`
- `docs/2026-06-15_008_20_fqa2016_irrigation_only_soft_stress_ppo_validation_report.md`
- `docs/2026-06-16_008_21_hla2004_fqa2016_two_case_framework_summary.md`
- `docs/2026-06-16_weather_stress_assisted_ppo_framework_summary.md`
- `docs/2026-06-16_weather_stress_assisted_ppo_framework_summary_zh.md`
- `docs/2026-06-17_009_01_stage_level_water_nitrogen_joint_ppo_design.md`
- `docs/2026-06-17_009_02_hla2004_stage_level_water_nitrogen_joint_ppo_smoke_report.md`
- `docs/2026-06-17_009_03_hla2004_joint_ppo_reward_balance_probe_report.md`
- `docs/2026-06-17_009_04_hla2004_joint_ppo_n_cap_tightening_probe_report.md`
- `docs/2026-06-17_workspace_cleanup_mainline_plan.md`

Keep PPT files that are used for supervisor meetings, but do not upload large PPT/model/cache files unless needed.

## Keep: mainline outputs

Keep these output folders:

- `Leave_One_experiments/water_stress_showcase_selection_008_05/`
- `Leave_One_experiments/representative_management_comparison_008_06/`
- `Leave_One_experiments/forecast_gate_rule_replay_008_11/`
- `Leave_One_experiments/ppo_contribution_forecast_gate_008_12/`
- `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_14/`
- `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_15_10k/`
- `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_16_seed_stability/`
- `Leave_One_experiments/hla2004_stronger_soft_stress_penalty_ppo_008_17/`
- `Leave_One_experiments/hla2004_four_scenario_process_plots_008_19/`
- `Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20/`
- `Leave_One_experiments/two_case_framework_summary_008_21/`
- `Leave_One_experiments/hla2004_stage_level_water_nitrogen_joint_ppo_009_02/`
- `Leave_One_experiments/hla2004_joint_ppo_reward_balance_probe_009_03/`
- `Leave_One_experiments/hla2004_joint_ppo_n_cap_tightening_probe_009_04/`

## Do not touch

Do not delete, move, or overwrite:

- `my_data/`
- original `.xlsx`, `.WTH`, `.SOL`, `.jinja2`, `.CUL` files;
- `weather_clean/`
- `weather_clean_qc/`
- `backups/`
- existing scripts that are not yet classified;
- existing reward/wrapper files;
- `AGENTS.md`
- `README.md`
- `.git/`

## Archive candidates

The following are likely archive candidates because they belong to old reward sweeps, early failed PPO attempts, duplicated figures, old output folders, old RAR archives, logs, or rendered inputs:

- `output_hl/`
- `output_fq/`
- `output_sy/`
- `output_lc/`
- `output_yc/`
- `output_*.rar`
- `logs_hl/`, `logs_fq/`, `logs_sy/`, `logs_lc/`, `logs_yc/`
- old top-level `figures_*` folders not referenced by 008/009 reports;
- old `Leave_One_experiments/*` folders outside the 008/009 mainline list above;
- `tensorboard/`
- rendered input subfolders from old unrelated experiments.

Archive candidates should be moved into an archive folder only after review, not deleted directly.

Suggested archive folder:

```text
archive_non_mainline_2026_06_17/
```

## Safe delete candidates

The following are generated caches and do not need to be archived:

- `__pycache__/`
- nested `__pycache__/` folders under `src/`, `scripts/`, or other code folders;
- `.pyc` files.

These can be deleted after confirming they are inside the project directory. They should not be committed to Git.

## Recommended safe cleanup sequence

1. Confirm the keep list above.
2. Create a Git commit containing only the mainline prompts, docs, configs, scripts, and small CSV summaries.
3. Do not commit model `.zip`, rendered DSSAT temporary files, large logs, or large old sweeps.
4. Move archive candidates into `archive_non_mainline_2026_06_17/` instead of deleting them.
5. After the project has been backed up and the archive has been reviewed, delete only explicitly approved archive contents.

## Current 009_04 note

009_04 completed without OOM. It reduced total N to 150 kg/ha and kept GRNWT near the high-N 009_03 runs, but it still reached the tightened N cap:

- total irrigation: about 105.7 mm;
- total N: 150.0 kg/ha;
- PPO extra N: 50.0 kg/ha;
- final GRNWT: about 6969.0 kg/ha;
- N cap saturated: true.

This means the 150 kg/ha cap is agronomically reasonable, but the policy still wants all available supplemental N under the current reward/action design.
