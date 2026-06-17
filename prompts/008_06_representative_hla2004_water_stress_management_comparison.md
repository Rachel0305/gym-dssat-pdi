# 008_06 Representative HLA 2004 Water-Stress Management Comparison

Date: 2026-06-13

## Background

008_04 showed that FQA 2008 is not a convincing water-optimization showcase because strict null management has no SWFAC water stress. 008_05 reselected water-stress showcase years from existing results and identified HLA 2004 as the strongest candidate:

- rainfed adequate-N SWFAC stress days: 18;
- maximum SWFAC under N-only: 1.0;
- same-N irrigation yield gain: about 6383 kg/ha;
- low-water-cost profit gain: positive.

The supervisor requested process-based evidence showing how management changes water/nitrogen stress, actions, and yield. This task builds a lightweight representative scenario comparison for HLA 2004 before any new PPO training.

## Constraints

- Do not train PPO.
- Do not run bulk DSSAT simulations.
- Do not modify reward.
- Do not modify `my_data/` original files.
- Do not overwrite 006/007/008 previous outputs.
- Reuse existing all-year fixed-management daily CSVs where possible.
- Only run missing HLA 2004 single-episode scenarios if needed.
- Minimize compute and avoid OOM.

## Scenario List

For HLA 2004, compare:

1. `null_zero`
   - reuse all-year `T0_null_zero`;
   - no nitrogen, no irrigation.

2. `n_only_medium`
   - reuse all-year `T1_N_only_medium`;
   - N=150 kg/ha, no irrigation;
   - agronomic rainfed water-stress reference.

3. `fixed_I60_N150`
   - reuse all-year `T2_N_medium_I_low`;
   - N=150 kg/ha, irrigation=60 mm;
   - fixed irrigation response benchmark.

4. `fixed_I120_N150`
   - reuse all-year `T3_N_medium_I_mid`;
   - N=150 kg/ha, irrigation=120 mm;
   - adequate irrigation response benchmark.

5. `expert_reference_recorded`
   - run one HLA 2004 episode using recorded site management:
     - DAP 1: N=165 kg/ha;
     - DAP 49, 70, 95: irrigation=10 mm each;
   - source: site observed management record from a single year;
   - limitation: not year-specific optimum and may underperform in extreme drought years.

6. `ppo_replay_transfer`
   - run one HLA 2004 episode using FQA 007_04J stage schedule transfer;
   - diagnostic only, not a trained HLA PPO and not model-weight transfer.

7. `dssat_auto_attempt`
   - run one HLA 2004 episode attempting DSSAT automatic management through the gym-DSSAT wrapper;
   - label diagnostic and do not claim full native DSSAT automatic management unless logs confirm it.

## Required Outputs

Use:

```text
Leave_One_experiments/representative_management_comparison_008_06/
```

Generate:

```text
daily_outputs/HLA/2004_<scenario>_daily.csv
event_outputs/HLA_2004_<scenario>_events.csv
evaluation/HLA_2004_representative_scenario_summary.csv
evaluation/HLA_2004_process_diagnosis.csv
figures/HLA_2004_rain_stress_actions_growth.png
figures/HLA_2004_management_events.png
figures/HLA_2004_yield_water_n_summary.png
docs/2026-06-13_008_06_hla2004_representative_management_comparison_report.md
docs/009_008_06_hla2004_representative_management_comparison.pptx
```

## Interpretation Rules

- Treat `n_only_medium`, `fixed_I60_N150`, and `fixed_I120_N150` as fixed irrigation response controls, not expert strategies.
- Treat `expert_reference_recorded` as a single-year observed management reference, not an optimum.
- Treat `ppo_replay_transfer` as schedule-transfer diagnosis, not HLA-trained PPO.
- Treat `dssat_auto_attempt` as diagnostic unless native automatic totals are clearly parsed from DSSAT output.
- The goal is to show whether HLA 2004 is a valid water-stress showcase and what the next stress-aware PPO should improve.

## Decision Criteria

After this task, decide whether to:

1. train a stress-aware PPO on HLA 2004 first;
2. generate the same scenario comparison for FQA 2016 and SYA 2017;
3. revise irrigation action gating before PPO training.

