# 031_05 SY2014 free-timing rule sensitivity

## Purpose

Before designing a new free-timing RL reward, test whether SY2014 is actually sensitive to operation timing under the same I/N budget caps.

031_03/031_04 showed that random and PPO policies can reach similar outcomes under the original free-daily reward. This task runs fixed, deterministic rule policies with the same budget caps but different timing patterns. If timing barely matters in SY2014, it is a weak testbed for free-timing RL.

## Scope

- Site-year: SYA2014 only.
- No RL training.
- No model inference.
- No expert DAP restriction.
- Same free-daily action safety as 031:
  - irrigation cap <= 160 mm
  - nitrogen cap <= 250 kg/ha
  - daily irrigation <= 40 mm
  - daily nitrogen <= 80 kg/ha
  - fertilization disabled after DAP90
  - minimum interval = 1 day
- Reward:

```text
reward_t = delta_GRNWT_t - 1.0 * irrigation_t - 5.0 * nitrogen_t
```

## Fixed policies

All rules are fixed before execution.

1. `early_dump`
   - Apply water and nitrogen as early as possible until caps are reached.
2. `uniform_spread`
   - Spread I160 and N250 across the season using fixed DAP points:
   - irrigation DAP 1/30/50/65/85/110 = 30/30/30/30/20/20
   - nitrogen DAP 1/30/50/65/85 = 50/50/50/50/50
3. `expert_window_budget`
   - Use the same expert-style DAP points as `uniform_spread`; this is the expert-window timing reference under the same I160/N250 budget, not the original official expert dose.
4. `stress_triggered`
   - If current SWFAC > 0.05, request 40 mm irrigation until I160 is exhausted.
   - If current NSTRES > 0.05 and DAP <= 90, request 80 kg/ha N until N250 is exhausted.
5. `delayed_late`
   - Deliberately late policy:
   - nitrogen DAP 83/84/85/86/87/88/89 = 40/40/40/40/40/40/10
   - irrigation DAP 102/103/104/105/106/107/108/109 = 20 each

## Outputs

- daily CSV for each rule.
- summary CSV.
- docs record.

## Interpretation

- Large yield/profit spread across rules means SY2014 has timing sensitivity and is a useful free-timing testbed.
- Small spread means SY2014 cannot strongly distinguish learned timing from random timing; pick a more timing-sensitive year before reward engineering.
