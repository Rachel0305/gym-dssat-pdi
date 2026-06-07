# Current Site Status Review

This stage does not continue PPO and does not enter rainfall-scaling. It first asks whether irrigation has real observed-year value.

## Why not continue PPO

- 006_13 showed no 300/450 regression, but fine-tuned PPO did not improve over the augmented RF prior.
- MS0 augmented RF prior replay kept water at 0 and N near 83 kg/ha with best mean profit.
- MS2 increased yield by using the strict 100/200 guardrail and produced negative profit, so it is not a low-input optimum.

## Why not directly rainfall-scaling

- Rainfall-scaling is a stress-test scenario, not observed-year evidence.
- The augmented imitation dataset has very few non-zero irrigation rows, so a learned policy can easily become a no-irrigation policy.
- Existing all-mode diagnostics show weak or missing swfac evidence in observed years.

## Site status summary

| station | observed_years | PRCP | ETCP | PRCP_minus_ETCP | all_mode_swfac_available | swfac_stress_days_gt_0p05 | max_swfac | mean_swfac | nstres_days_gt_0p05 | max_nstres | mean_nstres | augmented_rf_mean_yield | augmented_rf_mean_irrigation | augmented_rf_mean_n | augmented_rf_mean_profit | diagnosis_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | 2007,2011,2009 | 345.6 | 496.0 | -150.4 | True | 0.0 | 0.0 | 0.0 | 114.0 | 0.7988 | 0.3971 | 6503.2767 | 0.0 | 81.8291 | 44.5755 | all_mode_available |
| SYA | 2014,2015,2012 | 659.3 | 428.0 | 231.3 | True | 0.0 | 0.0 | 0.0 | 89.0 | 0.3814 | 0.1847 | 6503.2767 | 0.0 | 81.8291 | 44.5755 | all_mode_available |
| LCA | 2010,2011,2008,2009 |  |  |  | False |  |  |  |  |  |  | 6503.2767 | 0.0 | 81.8291 | 44.5755 | all_mode_missing_or_stalled |
| FQA | 2008,2010 | 413.6 | 344.0 | 69.6 | True | 0.0 | 0.0 | 0.0 | 64.0 | 0.4164 | 0.1869 | 6503.2767 | 0.0 | 81.8291 | 44.5755 | all_mode_available |
| YCA | 2014,2008 | 357.1 | 350.0 | 7.1 | True | 0.0 | 0.0 | 0.0 | 88.0 | 0.5364 | 0.3025 | 6503.2767 | 0.0 | 81.8291 | 44.5755 | all_mode_available |