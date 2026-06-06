# Budget/action design plan

| design | type | reward_version | description |
| --- | --- | --- | --- |
| baseline_daily_action_current | baseline | P6_high_economic_pressure | daily continuous action with action safety |
| D_seasonal_budget_fixed_split | seasonal_budget | P0_current_reward_baseline | PPO selects season budget, wrapper releases fixed split at DAP 25/50/75 and 1/30/60 |
| E_scheduled_events_default | scheduled_event | P0_current_reward_baseline | PPO selects amounts only at fixed event DAPs |
| F_stage_budget_default | stage_budget | P0_current_reward_baseline | PPO selects stage budget at stage starts |
| D_plus_profit_reward | seasonal_budget | P6_high_economic_pressure | PPO selects season budget, wrapper releases fixed split at DAP 25/50/75 and 1/30/60 |
| E_plus_profit_reward | scheduled_event | P6_high_economic_pressure | PPO selects amounts only at fixed event DAPs |