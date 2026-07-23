# 031_06 free-timing reward v2 PPO/DQN smoke

## Purpose

Start a true free-timing RL attempt after 031_05 confirmed that SY2014 is sensitive to operation timing.

031_04 showed that simply increasing training under the original reward did not work:

- PPO stayed close to random early cap saturation.
- DQN learned a delayed low-yield strategy.

031_06 keeps daily free decisions, but adds a light timing-aware reward shaping term to help the algorithms distinguish reasonable timing from early dump and delayed late.

## Scope

- Site-year: SYA2014 only.
- Seed: 0 only.
- Algorithms:
  - continuous-action PPO
  - 3x3 discrete-action DQN
- Training budget: 5,000 timesteps each.
- No expert DAP windows.
- No minimum interval hard constraint beyond 1 day.
- Hard constraints retained:
  - irrigation cap <= 160 mm
  - nitrogen cap <= 250 kg/ha
  - daily irrigation <= 40 mm
  - daily nitrogen <= 80 kg/ha
  - fertilization disabled after DAP90

## Reward v2

Base reward:

```text
base_t = delta_GRNWT_t + 0.10 * delta_TOPWT_t - 1.0 * irrigation_t - 5.0 * nitrogen_t
```

Timing shaping:

```text
repeat_penalty =
    0.50 * irrigation_t * max(0, 7 - days_since_last_irrigation) / 7
  + 0.50 * nitrogen_t   * max(0, 7 - days_since_last_fertilization) / 7

early_excess_penalty =
    0.50 * irrigation_t if DAP <= 10 and cumulative_irrigation_before >= 40
  + 0.50 * nitrogen_t   if DAP <= 10 and cumulative_n_before >= 80

late_penalty =
    0.50 * irrigation_t if DAP >= 100
  + 2.00 * nitrogen_t   if DAP >= 80

reward_v2 = base_t - repeat_penalty - early_excess_penalty - late_penalty
```

Interpretation:

- This still allows daily free decisions.
- It does not force expert DAPs.
- It softly discourages the two observed bad modes:
  - early dump;
  - delayed late.

## Success criteria for this smoke

This is not yet a final site-year claim. It asks whether reward v2 moves either algorithm away from the 031_04 bad modes.

Report:

1. final GRNWT;
2. total irrigation and nitrogen;
3. first operation DAP;
4. event counts;
5. whether caps still saturate;
6. comparison to:
   - 031_04 PPO/DQN 5k;
   - 031_05 fixed timing rules.

Do not expand to more seeds/sites before reading this result.
