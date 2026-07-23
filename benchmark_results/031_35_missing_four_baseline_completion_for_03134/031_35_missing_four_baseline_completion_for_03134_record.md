# 031_35 Missing four-baseline completion for 031_34 record

Mode: `full`

## Scope

- No PPO/DQN training.
- Fixed-policy baselines were generated only for missing station-years from 031_34.
- Existing true DSSAT auto rows are reused where available; missing true auto rows are not faked.
- `recorded_farmer_template_02705` is a transferred template, not true yearly observed farmer management.
- Generated fixed-policy actions are executed through the gym action interface; very large transferred recorded-template single events may be clipped by the environment action bounds.
- A second `template_aware` comparison table treats transferred recorded templates as comparison-only recorded farmer surrogates when true recorded-farmer rows are absent.

## Missing station-years processed

| station_code | year |
| --- | --- |
| FQA | 2005 |
| FQA | 2006 |
| FQA | 2007 |
| FQA | 2008 |
| FQA | 2009 |
| FQA | 2010 |
| FQA | 2011 |
| FQA | 2012 |
| FQA | 2015 |
| FQA | 2017 |
| FQA | 2018 |
| FQA | 2021 |
| FQA | 2022 |
| HLA | 2004 |
| HLA | 2005 |
| HLA | 2006 |
| HLA | 2008 |
| HLA | 2009 |
| HLA | 2011 |
| HLA | 2012 |
| HLA | 2013 |
| HLA | 2014 |
| HLA | 2017 |
| HLA | 2018 |
| HLA | 2019 |
| HLA | 2020 |
| HLA | 2021 |
| HLA | 2023 |
| LCA | 2005 |
| LCA | 2006 |
| LCA | 2007 |
| LCA | 2008 |
| LCA | 2009 |
| LCA | 2011 |
| LCA | 2012 |
| LCA | 2013 |
| LCA | 2014 |
| LCA | 2015 |
| LCA | 2016 |
| LCA | 2017 |
| LCA | 2018 |
| LCA | 2019 |
| LCA | 2020 |
| LCA | 2021 |
| LCA | 2022 |
| LCA | 2023 |
| YCA | 2004 |
| YCA | 2005 |
| YCA | 2006 |
| YCA | 2007 |
| YCA | 2009 |
| YCA | 2010 |
| YCA | 2011 |
| YCA | 2012 |
| YCA | 2013 |
| YCA | 2015 |
| YCA | 2016 |
| YCA | 2017 |
| YCA | 2018 |
| YCA | 2019 |
| YCA | 2020 |
| YCA | 2021 |
| YCA | 2022 |
| YCA | 2023 |

## Generated/reused coverage status

| scenario | status | n |
| --- | --- | --- |
| null | generated_031_35_fixed_policy | 64 |
| official_extension_expert | generated_031_35_fixed_policy | 64 |
| recorded_farmer_template_02705 | generated_031_35_fixed_policy | 64 |

## Strict expanded PPO comparison status

| station_code | baseline_comparison_status | n |
| --- | --- | --- |
| FQA | baseline_incomplete_3rows | 30 |
| FQA | ok | 27 |
| HLA | baseline_incomplete_2rows | 45 |
| HLA | ok | 15 |
| LCA | baseline_incomplete_2rows | 45 |
| LCA | ok | 12 |
| YCA | baseline_incomplete_2rows | 54 |
| YCA | ok | 6 |

## Strict wins among comparable rows

| station_code | compared_ok | any_metric_wins |
| --- | --- | --- |
| FQA | 27 | 3 |
| HLA | 15 | 0 |
| LCA | 12 | 10 |
| YCA | 6 | 4 |

## Template-aware PPO comparison status

| station_code | baseline_comparison_status | n |
| --- | --- | --- |
| FQA | ok | 57 |
| HLA | baseline_incomplete_3rows | 45 |
| HLA | ok | 15 |
| LCA | baseline_incomplete_3rows | 45 |
| LCA | ok | 12 |
| YCA | baseline_incomplete_3rows | 54 |
| YCA | ok | 6 |

## Template-aware wins among comparable rows

| station_code | compared_ok | any_metric_wins |
| --- | --- | --- |
| FQA | 57 | 6 |
| HLA | 15 | 0 |
| LCA | 12 | 10 |
| YCA | 6 | 4 |
