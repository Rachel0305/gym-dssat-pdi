# HL scale0.6 all-mode penalty scan findings

Date: 2026-06-01

Purpose:
- Test whether stronger total fertilizer / irrigation excess penalties and an extra no-water-stress irrigation penalty can make PPO irrigate only when water stress appears.
- Scenario: Hailun 2007 maize with rainfall scaled to 60%.
- Training length: 50,000 timesteps per combination.
- Execution pattern: each combination is trained and diagnosed in a fresh Python subprocess to reduce memory leakage risk.

Implemented reward controls:
- `GYM_DSSAT_ALL_ANFER_EXCESS_LIMIT`
- `GYM_DSSAT_ALL_ANFER_EXCESS_COST`
- `GYM_DSSAT_ALL_AMIR_EXCESS_LIMIT`
- `GYM_DSSAT_ALL_AMIR_EXCESS_COST`
- `GYM_DSSAT_ALL_AMIR_NO_STRESS_COST`
- `GYM_DSSAT_ALL_WATER_STRESS_THRESHOLD`

Main result:
- All three PPO models still over-applied fertilizer and irrigation.
- PPO total irrigation remained around 3,500 mm, far above the expert strategy's 30 mm.
- PPO total fertilizer remained around 13,600 kg/ha, far above the expert strategy's 165 kg/ha.
- PPO yield stayed around 4,866-4,868 kg/ha, below the expert strategy's 6,276 kg/ha.
- The learned PPO policy removed water and nitrogen stress numerically, but by unrealistic over-application rather than adaptive scheduling.

Best of this small scan:
- `pen15_icost15_ns20_nlim300_nex5_ilim120_iex10`
- This was only marginally better than the stricter settings and is not acceptable as a final management policy.

Interpretation:
- Reward penalties alone are not enough for short from-scratch PPO training in this action space.
- The action range is too permissive: random or early PPO policies can apply very large daily water and fertilizer amounts.
- The next safer step is to add a separate capped-action training wrapper, for example `anfer <= 60 kg/ha/day` and `amir <= 20 mm/day`, while keeping the penalty terms. This preserves the original DSSAT environment but prevents physically unrealistic exploration.

Files:
- `penalty_scan_summary.csv`: combined null/expert/PPO diagnostic summary.
- Per-combination folders: training logs, best/final PPO models, and `diagnose_ppo/all_water_stress_summary.csv`.
