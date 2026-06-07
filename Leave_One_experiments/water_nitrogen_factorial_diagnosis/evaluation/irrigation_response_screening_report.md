# Irrigation Response Screening Report

## Screening table

| station | year | has_water_stress | t2_swfac_stress_days_gt_0p05 | t2_max_swfac | irrigation_reduces_swfac | irrigation_increases_yield_at_same_N | max_yield_gain_from_irrigation_at_same_N | best_irrigation_treatment | irrigation_increases_profit_default | max_profit_gain_default | irrigation_increases_profit_low_water_cost | max_profit_gain_low_water_cost | irrigation_candidate | dominant_yield_driver | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2008 | True | 20.0 | 0.6031 | True | True | 708.8977 | T5_N_medium_I_mid | False | -23.751 | True | 0.249 | True | nitrogen |  |
| FQA | 2010 | False | 0.0 | 0.0 | False | False | 65.9583 | T5_N_medium_I_mid | False | -29.6063 | False | -5.6063 | False | nitrogen |  |
| HLA | 2007 | False | 0.0 | 0.0 | False | False | -48.1677 | T4_N_medium_I_low | False | -30.4817 | False | -6.4817 | False | nitrogen |  |
| HLA | 2009 | False | 0.0 | 0.0 | False | False | -362.027 | T5_N_medium_I_mid | False | -36.4644 | False | -12.4644 | False | nitrogen |  |
| HLA | 2011 | False | 0.0 | 0.0 | False | False | -83.9349 | T4_N_medium_I_low | False | -30.8393 | False | -6.8393 | False | nitrogen |  |
| LCA | 2008 | False | 0.0 | 0.0 | False | False | 19.0576 | T4_N_medium_I_low | False | -29.8094 | False | -5.8094 | False | nitrogen |  |
| LCA | 2009 | False | 0.0 | 0.0 | False | False | -13.0078 | T4_N_medium_I_low | False | -30.1301 | False | -6.1301 | False | nitrogen |  |
| LCA | 2010 | False | 0.0 | 0.0358 | True | False | 93.1885 | T5_N_medium_I_mid | False | -30.0 | False | -6.0 | False | nitrogen |  |
| LCA | 2011 | False | 0.0 | 0.0 | False | False | 191.3782 | T5_N_medium_I_mid | False | -29.2205 | False | -5.2205 | False | nitrogen |  |
| SYA | 2012 | False | 0.0 | 0.0 | False | False | -348.7482 | T4_N_medium_I_low | False | -33.4875 | False | -9.4875 | False | nitrogen |  |
| SYA | 2014 | True | 12.0 | 0.8005 | True | True | 922.9803 | T5_N_medium_I_mid | False | -25.2761 | False | -1.2761 | False | nitrogen |  |
| SYA | 2015 | False | 0.0 | 0.0 | False | False | 100.6763 | T5_N_medium_I_mid | False | -30.1519 | False | -6.1519 | False | nitrogen |  |
| YCA | 2008 | False | 0.0 | 0.0 | False | False | -4.7186 | T4_N_medium_I_low | False | -30.0472 | False | -6.0472 | False | nitrogen |  |
| YCA | 2014 | True | 7.0 | 0.6809 | True | True | 609.9957 | T5_N_medium_I_mid | False | -25.3255 | False | -1.3255 | False | nitrogen |  |

## Findings

- Station-years with water stress: 3.
- Station-years where irrigation reduced swfac: 4.
- Station-years where irrigation increased yield at same N: 3.
- Station-years where irrigation improved default-cost profit: 0.
- Station-years where irrigation improved low-water-cost profit: 1.
- Irrigation candidates: 1.

## HLA and FQA notes

- HLA yield gap is interpreted as nitrogen/action-prior related unless HLA rows pass irrigation_candidate screening.
- FQA 2008 is treated as a possible observed-year water candidate only if it passes stress, yield, and profit screening.