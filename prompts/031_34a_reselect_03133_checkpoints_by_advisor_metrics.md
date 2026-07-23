# 031_34a Reselect 031_33 checkpoints by advisor metrics

## Purpose

031_33 selected checkpoints by training-year `literature_reward_sum`. Initial comparison showed that this may not align with the advisor-facing criterion:

> At least one of yield / WP_ET / PFP_N should exceed the best of the four baseline scenarios.

This task re-screens all already-trained 031_33 checkpoints without retraining.

## Scope

- No new training.
- No DSSAT rerun.
- Input:
  - all 72 checkpoint evaluations from 031_33.
  - four-baseline summaries for HLA2015, FQ2016, LC2010, YC2014.
- Output:
  - all checkpoint gaps against the four-baseline envelope;
  - best checkpoint per station-seed under advisor known metrics;
  - station-level pass counts.

## Important limitation

031_33 checkpoint summaries do not contain ETCP / WP_ET. Therefore this task can only evaluate:

- yield;
- PFP_N;
- irrigation and nitrogen savings relative to expert.

WP_ET must be calculated in 031_34 full frozen-transfer evaluation, where snapshots / ETCP are saved.

## Selection rule for this audit

For each station-seed, rank checkpoints by:

1. `known_any_metric_win = yield_win or pfp_n_win`;
2. highest normalized score:
   - `gap_yield / baseline_max_yield`
   - `gap_pfp_n / baseline_max_pfp_n`
   - take max of available terms;
3. higher yield gap;
4. lower nitrogen total;
5. lower irrigation total;
6. earlier checkpoint.

This is an audit/reselection rule, not a new training result.

## Pass/fail interpretation

- If a station has candidate checkpoints that pass this audit, then 031_33 training produced usable candidates and the original reward-based selection was too narrow.
- If a station has no passing checkpoint across all seeds/checkpoints, then current frozen PPO configuration did not produce advisor-metric winners on that station's training year.

