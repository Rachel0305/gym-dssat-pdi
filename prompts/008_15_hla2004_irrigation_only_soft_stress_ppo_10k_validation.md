# 008_15 HLA 2004 Irrigation-Only Soft-Stress PPO 10k Validation

## Purpose

Validate whether the promising 008_14 PPO result remains stable or improves when training is increased from 2000 to 10000 timesteps.

This is still a small single-station validation, not a final batch experiment.

## Strict Constraints

- Do not train all stations.
- Do not run multiple seeds.
- Do not train unrestricted daily PPO.
- Do not modify `my_data/`.
- Do not modify site-packages reward files.
- Do not overwrite 008_14 outputs.
- Do not use hard minimum irrigation gate.
- Stop after HLA 2004 seed0.

## Design

Inherit 008_14 exactly except:

- `total_timesteps: 10000`
- output root:
  - `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_15_10k`
- report:
  - `docs/2026-06-14_008_15_hla2004_irrigation_only_soft_stress_ppo_10k_validation_report.md`

## Required Checks

Compare 008_15 with:

- 008_14 2k soft-stress PPO;
- 008_11 pure forecast rule;
- 008_12 hard-minimum gate PPO;
- fixed_I60_N150;
- fixed_I120_N150;
- expert_reference_recorded.

Report:

- total irrigation;
- final GRNWT;
- GRNWT / fixed_I120_N150;
- first irrigation DAP;
- whether S2/S3 begin to irrigate;
- whether PPO remains unsaturated;
- whether PPO collapses to zero irrigation;
- whether PPO reverts to cap saturation;
- whether PPO has autonomous contribution.

## Required Stage Diagnostics

The stage decision table must include:

- `raw_stage_action_irrigation`;
- `stage_action_amir`;
- `irrigation_before_gate`;
- `irrigation_after_gate`;
- `gate_future_rain`;
- `gate_forecast_trigger`;
- `gate_swfac_at_decision`;
- `growth_reward`;
- `terminal_reward`;
- `soft_swfac_penalty`;
- `reward`.

If total irrigation is 0 or cap saturated, inspect whether terminal reward dominates total reward.

Record:

```text
terminal_reward_share =
sum(terminal_reward) / sum(abs(stage reward components))
```

Use this only as a diagnostic, not as a final reward tuning conclusion.

