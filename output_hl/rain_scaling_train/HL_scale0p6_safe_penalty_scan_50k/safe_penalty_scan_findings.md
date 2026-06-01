# HL scale0.6 safe-action penalty scan findings

Date: 2026-06-01

Purpose:
- Add safe daily action caps without changing the original DSSAT environment.
- Keep the total fertilizer / irrigation excess penalties and no-water-stress irrigation penalty.
- Test whether PPO begins to learn a more realistic water-nitrogen strategy in the rainfall-scaled Hailun drought scenario.

Implementation:
- `SafeActionGymDssatWrapper` maps PPO actions from `[-1, 1]` directly into `[0, safe_cap]`.
- This is different from clipping after the original action scaling; direct safe scaling gives PPO a useful action resolution within the agronomic range.
- Training and PPO diagnosis both use the same safe wrapper.

50k scan:
- `safe60_20`: total_anfer = 4066 kg/ha, total_amir = 1403 mm, yield = 5230 kg/ha.
- `safe40_15`: total_anfer = 2720 kg/ha, total_amir = 1055 mm, yield = 5615 kg/ha.
- `safe30_10`: total_anfer = 2039 kg/ha, total_amir = 705 mm, yield = 6977 kg/ha.
- `safe20_5`: total_anfer = 1355 kg/ha, total_amir = 352 mm, yield = 7456 kg/ha.

100k follow-up on best 50k combo:
- `safe20_5` 100k: total_anfer = 1350 kg/ha, total_amir = 351 mm, yield = 7457 kg/ha.
- Extending from 50k to 100k barely changed the policy.

Interpretation:
- The safe action wrapper is effective: compared with the previous no-safe-cap scan, PPO water and nitrogen use dropped by roughly one order of magnitude.
- The best current policy has high yield and no water/nitrogen stress, but it still uses much more water and fertilizer than the expert baseline.
- The policy is still not a final management policy. It is a useful debugging milestone showing that action-space control is necessary.

Next recommendation:
- Add a seasonal budget or cumulative-action wrapper, not only daily caps.
- Suggested next limits for Hailun scale0.6 debugging:
  - seasonal `anfer` budget near 180-300 kg/ha
  - seasonal `amir` budget near 60-150 mm
  - once the seasonal budget is exhausted, force the corresponding action to zero
- This should make the policy choose timing rather than simply applying small amounts almost every day.
