# 015_10 YC2014 baseline-relative reward recheck prompt

## Purpose

Recheck YC2014 with the same formal cross-site reward definition used by the successful HLA baseline-relative line:

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

This keeps one formula across sites while using each site-year's own null yield.

## Why

The previous YC unified DQN line used an incremental yield reward:

```text
reward_t = max(0, delta_grnwt_t) - 1.0 * I_t - 5.0 * N_t
```

That is not the same reward as the HLA success line. To make the cross-site story defensible, YC must be tested under the baseline-relative reward too.

## Fixed settings

- Site-year: YC2014
- Algorithm: DQN
- Action space: 9 actions
- Irrigation options: 0 / 15 / 30 mm
- Nitrogen options: 0 / 50 / 100 kg/ha
- Budget: I <= 120 mm, N <= 300 kg/ha
- Single-event cap: I <= 30 mm, N <= 100 kg/ha
- Minimum operation interval: 7 days
- Management mode: IRRIG=L, FERTI=L
- Baseline: YC2014 null yield from the same input package

## Execution

First run a low-cost 5K seed0 recheck. If the result is coherent, later run checkpoint-based 50K / seed1.

## Outputs

```text
DSSAT_auto_validation/yc2014_baseline_relative_dqn_015_10/
docs/2026-07-01_015_10_yc2014_baseline_relative_dqn_record.md
```

Required files:

- daily CSV
- summary CSV
- event summary JSON
- Chinese MD record

## Decision rule

- If YC2014 still reaches expert-level yield with less N under baseline-relative reward, it can be kept as a cross-site success case.
- If it degrades, then previous YC success was partly tied to the delta-grnwt reward and must be reported separately.
