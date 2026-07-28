# 032_09 LC historical/future split and DSSAT-auto gap audit

## Motivation

The next formal free-timing PPO line should use a literature-aligned historical-weather training and future-year validation protocol instead of ad hoc diagnostic years. Before training, audit which LC years from 2000 onward have weather/scenario support and which four-scenario baselines are complete.

## Proposed split

Primary protocol:

- Training/development years: 2000-2010.
- Future test years: 2011-2020.
- External/extension test years: 2021-2023, if available.

This follows the idea used in crop-management RL literature: train on historical years and evaluate on unseen future years.

## Scope

- Station: LCA/LC only.
- No training.
- No DSSAT execution in this audit.
- Read existing scenario pool, weather inventory, and completed baseline CSVs.
- Identify missing DSSAT-auto years that need a separate completion task.

## Required checks

1. List all LCA years >= 2000 available in the scenario pool.
2. List all LCA years >= 2000 available in weather inventory / cleaned weather sources if readable.
3. For each available LCA year, mark whether current daily baselines contain:
   - null
   - official_extension_expert
   - recorded_farmer_template_02705 or recorded_farmer
   - dssat_auto
4. Assign each year to one of:
   - train_2000_2010
   - test_2011_2020
   - extension_2021_2023
   - outside_scope
5. Record whether the year can support:
   - RL training/evaluation,
   - complete five-scenario plotting,
   - future-year metric comparison against all four baselines.

## Interpretation

This audit only decides the data split and baseline gap list. It does not claim model success or failure.

If DSSAT-auto is missing for some otherwise usable years, write a follow-up recommendation to run a separate auto-completion task before producing final five-scenario figures for those years.
