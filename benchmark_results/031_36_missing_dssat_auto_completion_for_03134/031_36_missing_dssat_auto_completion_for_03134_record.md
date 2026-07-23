# 031_36 Missing true DSSAT-auto completion for 031_34/031_35 record

Mode: `full`

## Scope

- No PPO/DQN training.
- Only true `dssat_auto` gaps from 031_35 template-aware incomplete rows were generated.
- Original MZX/WTH/SOL/CUL files were not modified; only rendered per-run FileX templates under 031_36 were edited.
- `recorded_farmer_template_02705` remains a comparison-only transferred recorded-farmer surrogate where true yearly recorded farmer is unavailable.

## Target missing auto station-years

| station_code | site | year |
| --- | --- | --- |
| HLA | HLA | 2004 |
| HLA | HLA | 2005 |
| HLA | HLA | 2006 |
| HLA | HLA | 2008 |
| HLA | HLA | 2009 |
| HLA | HLA | 2011 |
| HLA | HLA | 2012 |
| HLA | HLA | 2013 |
| HLA | HLA | 2014 |
| HLA | HLA | 2017 |
| HLA | HLA | 2018 |
| HLA | HLA | 2019 |
| HLA | HLA | 2020 |
| HLA | HLA | 2021 |
| HLA | HLA | 2023 |
| LCA | LC | 2005 |
| LCA | LC | 2006 |
| LCA | LC | 2007 |
| LCA | LC | 2012 |
| LCA | LC | 2013 |
| LCA | LC | 2014 |
| LCA | LC | 2015 |
| LCA | LC | 2016 |
| LCA | LC | 2017 |
| LCA | LC | 2018 |
| LCA | LC | 2019 |
| LCA | LC | 2020 |
| LCA | LC | 2021 |
| LCA | LC | 2022 |
| LCA | LC | 2023 |
| YCA | YC | 2004 |
| YCA | YC | 2005 |
| YCA | YC | 2006 |
| YCA | YC | 2007 |
| YCA | YC | 2009 |
| YCA | YC | 2010 |
| YCA | YC | 2011 |
| YCA | YC | 2012 |
| YCA | YC | 2013 |
| YCA | YC | 2015 |
| YCA | YC | 2016 |
| YCA | YC | 2017 |
| YCA | YC | 2018 |
| YCA | YC | 2019 |
| YCA | YC | 2020 |
| YCA | YC | 2021 |
| YCA | YC | 2022 |
| YCA | YC | 2023 |

## DSSAT-auto coverage manifest

| station_code | status | n |
| --- | --- | --- |
| HLA | generated_031_36_true_dssat_auto | 15 |
| LCA | generated_031_36_true_dssat_auto | 15 |
| YCA | generated_031_36_true_dssat_auto | 18 |

## Generated auto metric preview

| station_code | site | year | grain_yield_kg_ha | actual_irrigation_mm | actual_nitrogen_kg_ha | WP_ET_kg_m3 | PFP_N_kg_kg | max_water_stress | max_nitrogen_stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | HLA | 2004 | 418.5661 | 0.0 | 0.0 | 0.43 |  | 0.0 | 0.7988 |
| HLA | HLA | 2005 | 354.0693 | 0.0 | 0.0 | 0.12 |  | 0.0 | 0.7988 |
| HLA | HLA | 2006 | 223.7189 | 0.0 | 0.0 | 0.09 |  | 0.0 | 0.7988 |
| HLA | HLA | 2008 | 263.3796 | 0.0 | 0.0 | 0.1 |  | 0.0 | 0.7988 |
| HLA | HLA | 2009 | 235.4001 | 0.0 | 0.0 | 0.09 |  | 0.0 | 0.7988 |
| HLA | HLA | 2011 | 197.7147 | 0.0 | 0.0 | 0.06 |  | 0.0 | 0.7988 |
| HLA | HLA | 2012 | 0.0 | 0.0 | 0.0 | 0.0 |  | 0.0 | 0.7988 |
| HLA | HLA | 2013 | 211.3286 | 0.0 | 0.0 | 0.07 |  | 0.0 | 0.7988 |
| HLA | HLA | 2014 | 207.5701 | 0.0 | 0.0 | 0.06 |  | 0.0 | 0.7988 |
| HLA | HLA | 2017 | 402.5588 | 0.0 | 0.0 | 0.16 |  | 0.0 | 0.7988 |
| HLA | HLA | 2018 | 246.159 | 0.0 | 0.0 | 0.09 |  | 0.0 | 0.7988 |
| HLA | HLA | 2019 | 175.9166 | 0.0 | 0.0 | 0.06 |  | 0.0 | 0.7988 |
| HLA | HLA | 2020 | 240.0147 | 0.0 | 0.0 | 0.08 |  | 0.0 | 0.7988 |
| HLA | HLA | 2021 | 196.5611 | 0.0 | 0.0 | 0.07 |  | 0.0 | 0.7988 |
| HLA | HLA | 2023 | 331.6223 | 0.0 | 0.0 | 0.11 |  | 0.0 | 0.7988 |
| LCA | LC | 2005 | 3192.8918 | 0.0 | 0.0 | 1.21 |  | 0.0 | 0.4902 |
| LCA | LC | 2006 | 2981.5372 | 0.0 | 0.0 | 1.19 |  | 0.0 | 0.5036 |
| LCA | LC | 2007 | 2804.122 | 0.0 | 0.0 | 1.62 |  | 0.0 | 0.4554 |
| LCA | LC | 2012 | 2935.7709 | 0.0 | 0.0 | 1.23 |  | 0.0 | 0.4727 |
| LCA | LC | 2013 | 2971.0599 | 0.0 | 0.0 | 1.08 |  | 0.0 | 0.4944 |
| LCA | LC | 2014 | 3103.2428 | 42.0 | 0.0 | 1.24 |  | 0.0 | 0.4864 |
| LCA | LC | 2015 | 3063.4293 | 0.0 | 0.0 | 1.31 |  | 0.0 | 0.4854 |
| LCA | LC | 2016 | 2874.9451 | 0.0 | 0.0 | 1.17 |  | 0.0 | 0.4611 |
| LCA | LC | 2017 | 2915.7724 | 42.0 | 0.0 | 1.19 |  | 0.0 | 0.4881 |
| LCA | LC | 2018 | 2713.999 | 42.0 | 0.0 | 1.16 |  | 0.0 | 0.4772 |
| LCA | LC | 2019 | 2982.4234 | 42.0 | 0.0 | 1.23 |  | 0.0 | 0.5037 |
| LCA | LC | 2020 | 2475.9288 | 0.0 | 0.0 | 1.05 |  | 0.0 | 0.4901 |
| LCA | LC | 2021 | 3216.6733 | 0.0 | 0.0 | 1.27 |  | 0.0 | 0.4998 |
| LCA | LC | 2022 | 2967.6666 | 0.0 | 0.0 | 1.09 |  | 0.0 | 0.5012 |
| LCA | LC | 2023 | 2473.3853 | 43.0 | 0.0 | 0.82 |  | 0.0 | 0.503 |
| YCA | YC | 2004 | 1703.9523 | 126.0 | 0.0 | 0.61 |  | 0.0 | 0.5098 |
| YCA | YC | 2005 | 1820.4608 | 0.0 | 0.0 | 0.51 |  | 0.0 | 0.532 |
| YCA | YC | 2006 | 1733.618 | 43.0 | 0.0 | 0.5 |  | 0.0 | 0.551 |
| YCA | YC | 2007 | 1595.2084 | 0.0 | 0.0 | 0.44 |  | 0.0 | 0.5411 |
| YCA | YC | 2009 | 1950.7413 | 0.0 | 0.0 | 0.49 |  | 0.0 | 0.5197 |
| YCA | YC | 2010 | 1391.0892 | 0.0 | 0.0 | 0.39 |  | 0.0 | 0.5523 |
| YCA | YC | 2011 | 1866.741 | 0.0 | 0.0 | 0.46 |  | 0.0 | 0.5364 |
| YCA | YC | 2012 | 1752.5841 | 0.0 | 0.0 | 0.52 |  | 0.0 | 0.5398 |
| YCA | YC | 2013 | 1430.3484 | 0.0 | 0.0 | 0.39 |  | 0.0 | 0.5696 |
| YCA | YC | 2015 | 1783.9453 | 42.0 | 0.0 | 0.51 |  | 0.0 | 0.5452 |
| YCA | YC | 2016 | 1387.1745 | 43.0 | 0.0 | 0.4 |  | 0.0 | 0.5473 |
| YCA | YC | 2017 | 1611.7094 | 0.0 | 0.0 | 0.45 |  | 0.0 | 0.5205 |
| YCA | YC | 2018 | 1395.4608 | 42.0 | 0.0 | 0.37 |  | 0.0 | 0.5543 |
| YCA | YC | 2019 | 1479.8129 | 43.0 | 0.0 | 0.5 |  | 0.0 | 0.5388 |
| YCA | YC | 2020 | 1365.6067 | 0.0 | 0.0 | 0.37 |  | 0.0 | 0.5443 |
| YCA | YC | 2021 | 1633.2933 | 0.0 | 0.0 | 0.39 |  | 0.0 | 0.5567 |
| YCA | YC | 2022 | 1453.7663 | 0.0 | 0.0 | 0.37 |  | 0.0 | 0.544 |
| YCA | YC | 2023 | 1349.9446 | 0.0 | 0.0 | 0.34 |  | 0.0 | 0.569 |

## Rebuilt four-baseline comparison status

| station_code | baseline_comparison_status | n |
| --- | --- | --- |
| FQA | ok | 57 |
| HLA | ok | 60 |
| LCA | ok | 57 |
| YCA | ok | 60 |

## Wins among comparable rows

| station_code | compared_ok | candidate_years | any_metric_wins | yield_wins | wp_wins | pfp_wins |
| --- | --- | --- | --- | --- | --- | --- |
| FQA | 57 | 19 | 6 | 2 | 5 | 3 |
| HLA | 60 | 20 | 10 | 9 | 1 | 0 |
| LCA | 57 | 19 | 55 | 24 | 13 | 55 |
| YCA | 60 | 20 | 54 | 9 | 1 | 54 |
