# New Experiment Route After Group Meeting

## Stage A: cultivar calibration / validation

Use 2020-2023 weather years with fixed management. Do not introduce PPO action effects into cultivar calibration.

## Stage B: all-year weather scenario stress pool

Use all available QC weather years and classify them by rainfall, swfac, nstres, and irrigation response.

## Stage C: expert prior / offline schedule search

Run this only if the scenario pool includes enough water-stress or irrigation-responsive years.

## Stage D: PPO

Run only after scenario pool and calibration route are established. Continue action safety and do not use unrestricted PPO.

## Recommended next prompt

`006_16_all_year_offline_schedule_search_and_imitation_prior`

Evidence: water_stress_years=20, irrigation_responsive_years=6.