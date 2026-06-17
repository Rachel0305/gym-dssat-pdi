# 008_20 FQA 2016 Soft-Stress PPO Second Water-Stress Year Validation

## Goal

Validate whether the HLA 2004 soft-stress stage PPO framework can produce interpretable, nonzero, non-saturated irrigation behavior in a second water-stress station-year.

## Scope

Run only a lightweight FQA 2016 validation:

- station: FQA
- year: 2016
- seed: 0
- timesteps: 5000
- fixed nitrogen: 150 kg ha-1 by stage prior
- PPO controls irrigation only
- use the same no-hard-minimum soft-stress framework as 008_14/008_15
- use stronger soft-stress penalty from 008_17:
  - swfac_excess_cost = 3.0
  - swfac_day_cost = 1.0

## Do Not

- Do not train unrestricted daily PPO.
- Do not run multiple seeds.
- Do not modify `my_data/`.
- Do not modify original reward files in site-packages.
- Do not overwrite HLA 2004 results.
- Do not tune parameters after seeing the first result.
- Do not claim final generalization from one FQA run.

## Required Outputs

Save all outputs under:

```text
Leave_One_experiments/fqa2016_irrigation_only_soft_stress_ppo_008_20
```

Required files:

1. PPO training summary CSV.
2. PPO evaluation summary CSV.
3. Daily CSV.
4. Stage decision CSV.
5. Diagnostic plots.
6. Markdown report under `docs/`.

## Required Diagnosis

Report:

- total irrigation
- total nitrogen
- final GRNWT
- SWFAC stress days
- NSTRES stress days
- first irrigation DAP
- whether irrigation cap is saturated
- whether early irrigation occurred before allowed window
- raw irrigation action by stage
- effective irrigation by stage
- comparison with 008_11 FQA 2016 pure forecast-gate rule replay if available
- whether PPO learned a nonzero autonomous irrigation strategy

## Interpretation Rules

This is a second-year validation, not a final model.

Positive signal:

- irrigation is nonzero
- irrigation is not saturated
- first irrigation is not S1
- PPO decisions are concentrated around water-risk stages
- final GRNWT is reasonably close to the 008_11 rule replay or clearly better than no-irrigation water-stress diagnosis

Negative signal:

- zero irrigation
- cap saturation
- early S1 irrigation
- very low GRNWT
- raw PPO actions collapse to all negative values

