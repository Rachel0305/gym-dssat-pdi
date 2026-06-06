# Episode-level profit reward design

This stage does not overwrite the original gym-DSSAT reward file. The new reward is injected by `EpisodeProfitRewardWrapper`.

For P1-P6:

```text
daily_reward = - daily_water_cost * daily_irrigation - daily_n_cost * daily_n
terminal_reward = grain_value_coef * final_grnwt
                  - season_water_cost * total_irrigation
                  - season_n_cost * total_n
```

P0 is a pass-through baseline using the current environment reward under the same action-safety cap.

The coefficients are normalized scores, not real RMB prices.

## Candidate table

| name | family | grain_value_coef | season_water_cost | season_n_cost | daily_water_cost | daily_n_cost | baseline_passthrough |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0_current_reward_baseline | P0_baseline | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| P1_terminal_profit_weak_cost | P_terminal_profit | 0.01 | 0.1 | 0.05 | 0.0 | 0.0 | False |
| P2_terminal_profit_medium_cost | P_terminal_profit | 0.01 | 0.5 | 0.25 | 0.0 | 0.0 | False |
| P3_terminal_profit_strong_cost | P_terminal_profit | 0.01 | 1.0 | 0.5 | 0.0 | 0.0 | False |
| P4_terminal_profit_with_daily_cost | P_terminal_plus_daily_cost | 0.01 | 0.5 | 0.25 | 0.05 | 0.02 | False |
| P5_lower_grain_value_medium_cost | P_terminal_plus_daily_cost | 0.005 | 0.5 | 0.25 | 0.05 | 0.02 | False |
| P6_high_economic_pressure | P_terminal_plus_daily_cost | 0.005 | 1.0 | 0.5 | 0.1 | 0.05 | False |