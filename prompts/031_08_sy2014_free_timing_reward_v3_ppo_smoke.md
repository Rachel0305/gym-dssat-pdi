# 031_08 SY2014 free-timing reward v3 PPO smoke

## Purpose

031_06 showed that reward v2 did not solve free-timing RL:

- PPO still saturated I160/N250 by DAP7.
- DQN improved over delayed-late failure but still saturated caps and underperformed fixed timing references.

031_07 then showed, using DSSAT counterfactual branches, that early cap saturation is not supported by marginal value under a high-performing background:

- DAP1/30/50/65 best simple-profit action was I0/N0;
- DAP85 water was useful but extra nitrogen was not;
- DAP100 extra action was ineffective or clipped.

031_08 tests whether a cleaner reward v3 can move PPO away from early cap saturation.

## Scope

- Site-year: SYA2014 only.
- Algorithm: PPO only.
- Seed: 0 only.
- Training budget: 5,000 timesteps.
- Daily free timing remains enabled.
- No expert DAP windows.
- No minimum interval hard constraint beyond one day.
- Do not expand to DQN, more seeds, or more site-years before reading this result.

## Hard constraints retained

- season irrigation <= 160 mm;
- season nitrogen <= 250 kg/ha;
- daily irrigation <= 40 mm;
- daily nitrogen <= 80 kg/ha;
- fertilization disabled after DAP90.

## Reward v3

Remove the TOPWT term from reward v2:

```text
base_t = delta_GRNWT_t - 1.0 * irrigation_t - 5.0 * nitrogen_t
```

Add soft timing penalties:

```text
repeat_penalty =
    2.00 * irrigation_t * max(0, 7 - days_since_last_irrigation) / 7
  + 2.00 * nitrogen_t   * max(0, 7 - days_since_last_fertilization) / 7

early_excess_penalty =
    2.00 * irrigation_t if DAP <= 10 and cumulative_irrigation_before >= 40
  + 2.00 * nitrogen_t   if DAP <= 10 and cumulative_n_before >= 80

no_stress_early_penalty =
    0.50 * irrigation_t if DAP <= 65 and SWFAC_before <= 0.01
  + 2.00 * nitrogen_t   if DAP <= 65 and NSTRES_before <= 0.01

late_penalty =
    0.50 * irrigation_t if DAP >= 100
  + 2.00 * nitrogen_t   if DAP >= 80

reward_v3 = base_t - all_penalties
```

Interpretation:

- This does not force expert operation dates.
- It still lets PPO act every day.
- It discourages the failure mode directly observed in 031_06: consecutive early water/N dumping before stress.
- It removes the TOPWT reward that may have encouraged early vegetative biomass rather than final grain.

## Success criteria for this smoke

This is not a final success claim. It asks only whether reward v3 is directionally better than 031_06.

Pass signals:

1. PPO does not reach both I160 and N250 within DAP1-DAP10.
2. Final grain yield is at least 9761 kg/ha, the 031_05 `stress_triggered` reference.
3. Preferably approaches the 031_05 `uniform_spread` reference, 10908 kg/ha.
4. Non-zero actions become less front-loaded than 031_06 PPO.
5. Compare against 031_03 random, 031_04 PPO/DQN, 031_05 rules, and 031_06 PPO/DQN.

If reward v3 still causes early cap saturation or severe yield loss, stop and record it; do not tune coefficients in-place.
