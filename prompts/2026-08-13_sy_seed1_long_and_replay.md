# SY seed1 100K continuation and five-scenario replay

## Fixed contract

Continue only the passed `069_00` SY seed1 2K policy to nominal 100K steps.
The source 2K model is loaded; it is not replaced by a new seed0 model and no
hyperparameter, reward, safety, input, observation, weather-forecast, action-grid,
train/validation split, or baseline scenario is changed.  The grid remains
I `[0,15,30,45]` x N `[0,40,80,120]`; raw observation and no forecast remain.

## Safety and acceptance

Run in `nifty_taussig`, `/opt/gym_dssat_pdi/bin/python`, single process only.
Write only to new `070_*` output roots.  Checkpoint labels are 25K/50K/75K/100K.
For every checkpoint save daily action audits for all ten validation years.
Stop on model/input provenance error, missing daily output, or nonzero action
transmission mismatch.  Do not infer `WP_ET` from PPO training summaries.

## Replay

For each saved checkpoint replay all ten validation years and all five frozen
scenarios.  Set both the engine input root and
`ppo_safe_rendering.MULTISITE_INPUT_ROOT` to the originIC root before PPO replay.
Require each replay endpoint to agree with the corresponding saved validation
endpoint within the established 0.5 kg/ha rounding tolerance.  Report yield,
WP_ET, PFP_N, irrigation, N, yearly wins, and action behaviour only from the
resulting five-scenario season summaries.
