# 008_13 Official Maize Example vs Project Data PPO Gap Analysis

## Purpose

Compare the official gym-DSSAT maize example with the current project data and explain why PPO shows clear improvement in the official baseline but contributes little or behaves poorly in the current real-station water-nitrogen experiments.

This task is diagnostic only. It should help decide whether the next step should be:

1. Continue direct PPO training on current real station-years;
2. Build an official-style benchmark inside this project to verify PPO capability;
3. Redesign the project task so PPO has a measurable but agronomically defensible role.

## Strict Constraints

- Do not train PPO.
- Do not modify `my_data/`.
- Do not modify reward functions.
- Do not modify wrapper files.
- Do not overwrite previous 006/007/008 results.
- Do not delete or move existing data, reports, scripts, or figures.
- Do not claim that real project data should be changed or falsified to match the official example.

## Required Inputs

Use the following sources:

- Official gym-DSSAT maize baseline documentation:
  - `https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/Baselines/maize_baseline.html`
- Installed official maize config in the Docker environment:
  - `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/maize/env_config.yml`
  - `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/maize/UFGA8201.jinja2`
  - `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/envs/configs/rewards.py`
- Existing project reports:
  - `docs/2026-06-13_008_05_water_stress_showcase_selection_report.md`
  - `docs/2026-06-13_008_06_hla2004_representative_management_comparison_report.md`
  - `docs/2026-06-14_008_11_forecast_gate_rule_replay_validation_report.md`
  - `docs/2026-06-14_008_12_ppo_contribution_under_forecast_gate_report.md`

## Required Analysis

### 1. Official Example Summary

Summarize the official maize example:

- station/site and experiment type;
- cultivar;
- soil and initial condition structure;
- action space;
- observation space;
- management/treatment design;
- reward functions;
- why PPO has a clearer learning signal.

### 2. Project Data Summary

Summarize the current project data and experiments:

- five-station, multi-year, observed-weather setting;
- water-nitrogen joint management target;
- observed-year and all-year stress screening;
- HLA 2004 as a water-limited representative case;
- FQA 2008 as an example where irrigation decision logic was not convincing;
- 008_11 and 008_12 finding that forecast gate currently explains the result and PPO contribution is zero under that design.

### 3. Difference Matrix

Create a table comparing:

- task objective;
- data type;
- treatment/control structure;
- water-stress signal clarity;
- nitrogen-stress signal clarity;
- action frequency;
- observation consistency;
- reward delay;
- agronomic interpretability requirement;
- expected PPO difficulty.

### 4. Can We Modify Project Data to Become Like the Official Example?

Answer carefully:

- Do not falsify real observations.
- It is acceptable to create a separate official-style diagnostic benchmark.
- The benchmark should be clearly labeled as a controlled/synthetic diagnostic task, not the final real-station experiment.

### 5. Recommended Official-Style Benchmark

Design a minimal next benchmark:

- one station-year or small scenario pool;
- fixed station-invariant observation adapter;
- use official-like action scale and reward first;
- start with single-task fertilization or irrigation;
- then add water-nitrogen joint action;
- compare PPO, null, expert/fixed schedule, and pure rule baseline;
- explicitly measure PPO's contribution above rule/fixed management.

### 6. Decision Recommendation

State whether the project is drifting away from PPO and how to bring PPO back:

- Keep the real-data agronomic diagnostics;
- Add the official-style PPO benchmark as an algorithm sanity check;
- Then return to water-nitrogen joint optimization with a clearer separation between rule constraints and PPO-controlled decisions.

## Required Outputs

Save a Markdown report:

- `docs/2026-06-14_008_13_official_maize_vs_project_data_ppo_gap_analysis.md`

The report must be suitable for discussion with the supervisor or Claude.

