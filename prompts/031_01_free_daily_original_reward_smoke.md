# 031_01 Free-daily PPO original-reward smoke

## Purpose

Test the advisor-requested fully free timing setup before any all-year run.

This is a smoke test, not a final experiment.

## Key change from 031_00

Use fully free daily decision timing:

- no expert DAP windows;
- no 7-day minimum interval;
- minimum interval is set to 1 day, meaning the agent may act again on the next day;
- only safety caps and biological/management bounds remain.

This directly answers the advisor concern that RL should not be forced to follow expert timing.

## Reward

Use the simple original growth-cost reward:

```text
reward_t = ΔGRNWT_t - water_cost * irrigation_t - nitrogen_cost * nitrogen_t
```

First smoke-test coefficients:

```text
water_cost = 1
nitrogen_cost = 5
```

No terminal feasibility bonus.
No division by 1000.
No TOPWT term.
No expert imitation reward.

## Safety constraints retained

These are retained because they prevent physically unrealistic management, not because they copy expert:

- daily irrigation cap;
- daily nitrogen cap;
- seasonal irrigation cap;
- seasonal nitrogen cap;
- no fertilization after DAP 90.

No minimum operation interval beyond one day.

## Scope

Use 031_00 station-year scope for the final line, but this task only runs one smoke case.

Recommended first smoke:

- station: SY
- year: 2014
- seed: 0
- short training budget only

If the implementation cannot safely run training, stop after config/unit checks and record the blocker.

## Required outputs

Write outputs under:

`benchmark_results/031_01_free_daily_original_reward_smoke/`

Required:

- copied/resolved config;
- smoke training/evaluation log if training runs;
- daily CSV and five-scenario plot if evaluation runs;
- `031_01_result.json`;
- `031_01_free_daily_original_reward_smoke_record.md`.

## Pass / stop rules

Pass smoke if:

- daily free decisions are actually enabled;
- no expert DAP window is used;
- min interval is 1 day;
- reward calculation exactly matches `ΔGRNWT - I - 5N`;
- outputs are written without changing original weather/data files.

Stop if:

- the script still uses expert DAP windows;
- the reward still includes TOPWT, terminal bonus, or `/1000`;
- DSSAT/input path resolution is unclear.

