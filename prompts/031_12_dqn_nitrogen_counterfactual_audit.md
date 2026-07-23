# 031_12 DQN nitrogen counterfactual audit

## Purpose

031_10/031_11 showed that the literature-aligned DQN with a 7-day minimum interval is the strongest current free-timing result on SY2014, but all three seeds still apply nitrogen very early and reach the N250 seasonal cap.

This task asks:

> Is early N250 necessary, or is it an algorithmic shortcut?

## Scope

- No training.
- No PPO/DQN updates.
- No reward tuning.
- Site-year: SYA2014.
- Seeds audited: 0, 1, 2 from 031_10/031_11.
- Keep each seed's learned irrigation sequence unchanged.
- Only replace the nitrogen schedule.
- Keep current execution constraints:
  - N <= 250 kg/ha per season.
  - DAP > 90 nitrogen disabled.
  - same-resource interval >= 7 days.

## Counterfactual nitrogen variants

For each seed:

1. `original_replay`
   - Replays the seed's original DQN irrigation and nitrogen schedule.

2. `stage_spread_N250`
   - Keeps seed-specific irrigation.
   - Replaces nitrogen with N50 at DAP1/30/50/65/85.

3. `stage_spread_N200`
   - Keeps seed-specific irrigation.
   - Replaces nitrogen with N40 at DAP1/30/50/65/85.

4. `stage_spread_N160`
   - Keeps seed-specific irrigation.
   - Replaces nitrogen with N40 at DAP1/30/50/65.

5. `early_scaled_N200`
   - Keeps the original DQN nitrogen event dates but scales total nitrogen down to 200.

6. `early_scaled_N160`
   - Keeps the original DQN nitrogen event dates but scales total nitrogen down to 160.

## Metrics

- final grain yield;
- total irrigation;
- total nitrogen;
- simple profit `Y - I - 5N`;
- PFP_N = `Y / N`; if N=0, PFP_N is undefined and must be reported as blank/NaN;
- water stress days (`swfac > 0.05`);
- nitrogen stress days (`nstres > 0.05`);
- nonzero action sequence.

## Interpretation

- If lower-N variants keep yield close while improving PFP_N/profit, DQN N250 is likely excessive.
- If stage-spread N250 beats early N250, DQN's early-heavy timing is likely suboptimal.
- If original early N250 remains best, the early-heavy DQN choice has DSSAT counterfactual support for SY2014, even if it still needs agronomic discussion.

