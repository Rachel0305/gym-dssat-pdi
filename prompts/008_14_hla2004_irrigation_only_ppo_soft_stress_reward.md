# 008_14 HLA 2004 Irrigation-Only PPO With Soft Stress Reward

## Purpose

Run a small official-style PPO sanity benchmark on HLA 2004.

The goal is to test whether PPO can learn irrigation decisions in a clearly water-limited year when:

- nitrogen is fixed at an adequate level;
- irrigation is the only meaningful decision;
- the forecast gate only blocks clearly unreasonable irrigation;
- there is no hard minimum irrigation such as `max(PPO_action, min_irrigation)`;
- a soft water-stress penalty is added to the reward so zero-irrigation behavior is discouraged by the environment response, not by a rule-imposed irrigation amount.

## Strict Constraints

- Do not train unrestricted daily PPO.
- Do not train all stations.
- Do not train multiple seeds in this first smoke test.
- Do not modify `my_data/`.
- Do not modify site-packages reward files.
- Do not overwrite 006/007/008 previous results.
- Do not use hard minimum forecast gate irrigation.
- Keep this as a small smoke test to avoid OOM.

## Scenario

- Station: HLA
- Year: 2004
- Mode: all
- Nitrogen: fixed stage total N150
  - S1: 50 kg/ha
  - S2: 50 kg/ha
  - S3: 50 kg/ha
  - S4/S5: 0 kg/ha
- Irrigation: PPO-controlled stage irrigation
  - S1: blocked
  - S2: 0-40 mm
  - S3: 0-40 mm
  - S4: 0-40 mm
  - S5: 0-40 mm

## Forecast Gate Logic

The gate controls only whether irrigation is allowed.

It must not force a minimum irrigation amount.

- S1: irrigation always blocked.
- S2/S3: irrigation allowed if current SWFAC > 0.05 or future 7-day rain < 10 mm.
- S4/S5: irrigation allowed if current SWFAC > 0.05 or future 7-day rain < 20 mm.

If the gate blocks irrigation, PPO irrigation is set to 0.

If the gate allows irrigation, PPO decides the amount between 0 and the stage cap.

## Reward

Use a stage-level reward:

```text
reward =
  topwt_delta_coef * delta_TOPWT
  + grnwt_delta_coef * delta_GRNWT
  + terminal_grnwt_coef * final_GRNWT
  - water_cost * stage_irrigation
  - soft_swfac_cost * sum(max(0, SWFAC - swfac_threshold)) over daily records in the stage
  - soft_swfac_day_cost * number of SWFAC stress days in the stage
```

No nitrogen cost is needed because nitrogen is fixed at N150.

## Initial Smoke-Test Settings

- Seed: 0
- Timesteps: 2000
- PPO parameters: reuse previous small-stage PPO settings.
- Output root:
  - `Leave_One_experiments/hla2004_irrigation_only_soft_stress_ppo_008_14`

## Required Outputs

Save:

- model zip;
- daily CSV;
- stage CSV;
- training summary CSV;
- evaluation summary CSV;
- process figure;
- Markdown report:
  - `docs/2026-06-14_008_14_hla2004_irrigation_only_ppo_soft_stress_reward_report.md`

## Required Interpretation

Compare the 008_14 result against existing references:

- `n_only_medium`
- `fixed_I60_N150`
- `fixed_I120_N150`
- `008_11` pure forecast rule
- `008_12` PPO under hard-minimum forecast gate

Report:

- whether PPO irrigates without a hard minimum;
- total irrigation;
- final GRNWT;
- SWFAC stress days;
- whether PPO uses allowed stages differently from the rule;
- whether PPO has measurable contribution;
- whether this supports moving to 5k/10k or redesigning the action/reward again.

