# HL scale0.6 seasonal-budget all-mode scan findings

Date: 2026-06-02

Purpose:
- Add seasonal fertilizer and irrigation budgets on top of daily safe action caps.
- Force PPO to learn timing under a finite water-nitrogen budget instead of applying small amounts every day.
- Scenario: Hailun 2007 maize, rainfall scaled to 60%.

Implementation:
- `BudgetedSafeActionGymDssatWrapper` maps PPO actions into daily safe caps and then clips them by the remaining seasonal budget.
- Training and PPO diagnosis use the same wrapper.
- The original DSSAT environment and original source files are not modified.

Baseline:
- Expert: yield = 6276 kg/ha, fertilizer = 165 kg/ha, irrigation = 30 mm.
- Null: yield = 469 kg/ha, fertilizer = 0, irrigation = 0.

50k scan results:
- `budget180_60_daily20_5_pen20`: yield = 6165 kg/ha, fertilizer = 166 kg/ha, irrigation = 51 mm.
- `budget220_100_daily20_5_pen20`: yield = 6729 kg/ha, fertilizer = 206 kg/ha, irrigation = 91 mm.
- `budget300_150_daily30_10_pen15`: yield = 6695 kg/ha, fertilizer = 278 kg/ha, irrigation = 133 mm.

Interpretation:
- The seasonal-budget wrapper is the first setup that gives a plausible PPO water-nitrogen policy.
- `budget180_60` is conservative: it nearly matches expert fertilizer use and only adds about 21 mm irrigation, but yield is slightly lower than expert.
- `budget220_100` is the best current tradeoff: it improves yield by about 453 kg/ha over expert while using about 41 kg/ha more fertilizer and 61 mm more irrigation.
- `budget300_150` spends substantially more water and fertilizer but does not improve yield compared with `budget220_100`.

Recommended next step:
- Use `budget220_100_daily20_5_pen20` as the current best debugging configuration.
- Run a 100k confirmation and then inspect the daily action trace to see whether water and nitrogen are applied during plausible crop stages and stress periods.
