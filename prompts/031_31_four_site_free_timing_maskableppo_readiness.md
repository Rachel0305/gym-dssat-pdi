# 031_31 Four-site free-timing MaskablePPO expansion readiness prompt

## Purpose

Prepare the next formal expansion after the SYA 031_27-031_30 result:

- freeze the current free-timing discrete MaskablePPO configuration;
- do not tune SYA-specific parameters;
- audit whether HLA, FQA, LCA, and YCA are ready for the same train/checkpoint-select/frozen-transfer workflow used for SYA.

This task is an audit and planning task only. No PPO/DQN training and no DSSAT rerun.

## Frozen method to carry forward

Use the same algorithmic frame as the current successful SYA branch:

- free timing, not expert-DAP fixed;
- discrete MaskablePPO;
- irrigation levels: `[0, 6, 12, 18, 24]` mm per action;
- nitrogen levels: `[0, 40, 80, 120, 160]` kg/ha per action;
- 7-day minimum interval for irrigation and fertilization;
- seasonal soft caps: irrigation 160 mm, nitrogen 250 kg/ha;
- fertilization allowed through DAP90;
- reward: literature-style scaled terminal yield minus water and nitrogen costs, as in 031_18/031_20/031_26.

## Stations and candidate train years

Use the five-station smoke choices as default candidate train years:

- HLA: 2015
- FQA: 2016
- LCA: 2010
- YCA: 2014

SYA is already completed and is included only as a reference.

## Audit questions

For each target station, report:

1. available all-year weather years from the scenario pool;
2. whether the 031_20 smoke for the proposed train year ran successfully;
3. whether all required files for the all-year rendered environment exist;
4. proposed next task type:
   - `ready_for_checkpoint_selection`;
   - `needs_baseline_completion_first`;
   - `blocked_missing_inputs`.

## Required outputs

Under `benchmark_results/031_31_four_site_free_timing_maskableppo_readiness/`:

- `evaluation/031_31_station_year_inventory.csv`
- `evaluation/031_31_train_year_readiness.csv`
- `evaluation/031_31_next_task_plan.csv`
- `docs/031_31_four_site_free_timing_maskableppo_readiness_record.md`

## Stop condition

If any proposed train year is missing weather or failed the 031_20 smoke, do not propose formal training for that station until the missing issue is resolved.
