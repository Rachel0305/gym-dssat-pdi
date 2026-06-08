# All-Year Weather Calibration, Validation, and PPO Scenario Pool Report

## Executive Summary

- This stage did not train PPO, did not run rainfall-scaling, and did not modify original my_data files.
- Weather inventory found 119 station-years with QC weather/WTH data.
- 2020-2023 weather is available for all five stations, subject to the caveat that cultivar calibration still needs observed yield/phenology/fixed management records.
- All-year fixed management diagnostics found 20 water-stress years and 6 irrigation-responsive years, compared with observed-year-only water-stress=3 and irrigation-candidate=1.
- Recommended next stage: `006_16_all_year_offline_schedule_search_and_imitation_prior`.

## Available Weather Years

| station_code | n_years | years |
| --- | --- | --- |
| FQA | 24 | 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 |
| HLA | 24 | 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 |
| LCA | 24 | 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 |
| SYA | 23 | 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 |
| YCA | 24 | 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 |

## 2020-2023 Availability

| station_code | 2020 | 2021 | 2022 | 2023 |
| --- | --- | --- | --- | --- |
| FQA | ok | ok | ok | ok |
| HLA | ok | ok | ok | ok |
| LCA | ok | ok | ok | ok |
| SYA | ok | ok | ok | ok |
| YCA | ok | ok | ok | ok |

## Scenario Pool Summary

| station_code | water_stress_years | nitrogen_stress_years | irrigation_responsive_years | ppo_train_years | ppo_eval_years |
| --- | --- | --- | --- | --- | --- |
| FQA | 11 | 19 | 3 | 19 | 7 |
| HLA | 1 | 20 | 1 | 20 | 7 |
| LCA | 1 | 19 | 0 | 19 | 7 |
| SYA | 4 | 19 | 1 | 19 | 7 |
| YCA | 3 | 20 | 1 | 20 | 8 |

## Water Stress / Irrigation Responsive Years

| station_code | year | scenario_type | growing_season_rain | swfac_stress_days_gt_0p05 | max_swfac | nstres_days_gt_0p05 | yield_gain_from_irrigation_at_same_N | profit_gain_from_irrigation_low_water_cost | recommended_for_ppo_train | recommended_for_ppo_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2005 | wet_year;water_stress_year;nitrogen_stress_year | 535.1 | 6.0 | 0.3782 | 64.0 | 436.6431 | -1.6336 | True | True |
| FQA | 2006 | wet_year;nitrogen_stress_year;low_response_year | 481.7 | 0.0 | 0.0 | 64.0 | -34.0204 | -6.3402 | True | True |
| FQA | 2007 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 325.8 | 13.0 | 0.4622 | 66.0 | 763.7604 | 1.6376 | True | False |
| FQA | 2008 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 221.8 | 20.0 | 0.6031 | 64.0 | 708.8977 | 0.249 | True | False |
| FQA | 2009 | normal_year;nitrogen_stress_year;low_response_year | 292.3 | 0.0 | 0.0 | 75.0 | -33.4875 | -6.3349 | True | False |
| FQA | 2010 | normal_year;nitrogen_stress_year;low_response_year | 355.6 | 0.0 | 0.0 | 65.0 | 54.0936 | -5.9066 | True | False |
| FQA | 2011 | wet_year;water_stress_year;nitrogen_stress_year | 1051.2 | 5.0 | 0.4627 | 71.0 | 252.3895 | -3.921 | True | True |
| FQA | 2012 | wet_year;nitrogen_stress_year;low_response_year | 1078.1 | 0.0 | 0.0 | 57.0 | -27.1582 | -6.2716 | True | True |
| FQA | 2013 | wet_year;nitrogen_stress_year;low_response_year | 601.0 | 0.0 | 0.0 | 61.0 | -173.4369 | -7.8778 | True | True |
| FQA | 2014 | wet_year;nitrogen_stress_year;low_response_year | 1022.2 | 0.0 | 0.0 | 71.0 | 178.4448 | -4.2156 | True | True |
| FQA | 2015 | normal_year;water_stress_year;nitrogen_stress_year | 299.6 | 10.0 | 0.5991 | 60.0 | 222.6593 | -3.7734 | True | False |
| FQA | 2016 | normal_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 262.8 | 22.0 | 0.5806 | 61.0 | 2081.2744 | 8.8127 | True | False |
| FQA | 2017 | normal_year;water_stress_year;nitrogen_stress_year | 270.5 | 6.0 | 0.3714 | 65.0 | 472.9688 | -1.2703 | True | False |
| FQA | 2018 | normal_year;water_stress_year;nitrogen_stress_year | 348.2 | 10.0 | 0.5569 | 43.0 | 0.0 | -6.0 | True | False |
| FQA | 2019 | dry_year;water_stress_year;nitrogen_stress_year | 201.3 | 14.0 | 0.5568 | 64.0 | 425.672 | -7.4511 | True | True |
| FQA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | 411.0 | 7.0 | 0.2008 | 63.0 | 461.557 | -1.3844 | True | False |
| FQA | 2021 | wet_year;nitrogen_stress_year;low_response_year | 856.4 | 0.0 | 0.0 | 68.0 | 82.403 | -6.3813 | True | False |
| FQA | 2022 | normal_year;nitrogen_stress_year;low_response_year | 428.7 | 0.0 | 0.0 | 64.0 | -103.4875 | -7.0349 | True | False |
| FQA | 2023 | normal_year;water_stress_year;nitrogen_stress_year | 389.9 | 5.0 | 0.3482 | 61.0 | 270.614 | -3.6891 | True | False |
| HLA | 2004 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 51.5 | 18.0 | 1.0 | 122.0 | 6383.5418 | 51.8354 | True | True |
| HLA | 2005 | normal_year;nitrogen_stress_year;low_response_year | 478.0 | 0.0 | 0.0 | 126.0 | -108.6548 | -7.0865 | True | False |
| HLA | 2006 | normal_year;nitrogen_stress_year;low_response_year | 539.0 | 0.0 | 0.0 | 112.0 | -117.6752 | -7.1768 | True | False |
| HLA | 2007 | normal_year;nitrogen_stress_year;low_response_year | 345.6 | 0.0 | 0.0 | 114.0 | -48.1677 | -6.4817 | True | False |
| HLA | 2008 | normal_year;nitrogen_stress_year;low_response_year | 476.4 | 0.0 | 0.0 | 115.0 | 2.2766 | -5.9772 | True | False |
| HLA | 2009 | normal_year;nitrogen_stress_year;low_response_year | 489.5 | 0.0 | 0.0 | 103.0 | -408.6078 | -10.0861 | True | False |
| HLA | 2010 | normal_year;nitrogen_stress_year;low_response_year | 353.9 | 0.0 | 0.0 | 110.0 | 5.105 | -5.949 | True | False |
| HLA | 2011 | normal_year;nitrogen_stress_year;low_response_year | 509.1 | 0.0 | 0.0 | 118.0 | -83.9349 | -6.8393 | True | False |
| HLA | 2012 | wet_year;nitrogen_stress_year;low_response_year | 546.5 | 0.0 | 0.0 | 59.0 | 0.0 | -6.0 | True | True |
| HLA | 2013 | wet_year;nitrogen_stress_year;low_response_year | 825.1 | 0.0 | 0.0 | 108.0 | -25.2539 | -6.2525 | True | True |
| HLA | 2014 | normal_year;nitrogen_stress_year;low_response_year | 543.5 | 0.0 | 0.0 | 109.0 | 89.8169 | -6.6762 | True | False |
| HLA | 2015 | dry_year;nitrogen_stress_year;low_response_year | 342.1 | 0.0 | 0.0 | 120.0 | -165.6995 | -7.657 | True | True |
| HLA | 2016 | normal_year;nitrogen_stress_year;low_response_year | 454.2 | 0.0 | 0.0 | 106.0 | -221.5149 | -8.2151 | True | False |
| HLA | 2017 | normal_year;nitrogen_stress_year;low_response_year | 395.2 | 0.0 | 0.0 | 122.0 | -14.4788 | -6.1448 | True | False |
| HLA | 2018 | wet_year;nitrogen_stress_year;low_response_year | 824.4 | 0.0 | 0.0 | 111.0 | -84.2102 | -7.1595 | True | True |
| HLA | 2019 | wet_year;nitrogen_stress_year;low_response_year | 612.4 | 0.0 | 0.0 | 122.0 | -123.772 | -7.2377 | True | True |
| HLA | 2020 | wet_year;nitrogen_stress_year;low_response_year | 782.8 | 0.0 | 0.0 | 117.0 | -153.7537 | -7.5375 | True | False |
| HLA | 2021 | wet_year;nitrogen_stress_year;low_response_year | 646.4 | 0.0 | 0.0 | 112.0 | -7.4921 | -6.0749 | True | False |
| HLA | 2022 | normal_year;nitrogen_stress_year;low_response_year | 368.1 | 0.0 | 0.0 | 110.0 | 9.9438 | -5.9006 | True | False |
| HLA | 2023 | wet_year;nitrogen_stress_year;low_response_year | 740.8 | 0.0 | 0.0 | 110.0 | -15.6085 | -6.1561 | True | True |
| LCA | 2005 | normal_year;nitrogen_stress_year;low_response_year | 312.5 | 0.0 | 0.0 | 82.0 | 203.8525 | -4.0944 | True | False |
| LCA | 2006 | wet_year;nitrogen_stress_year;low_response_year | 347.2 | 0.0 | 0.0 | 80.0 | -11.3263 | -6.1133 | True | True |
| LCA | 2007 | normal_year;nitrogen_stress_year;low_response_year | 186.3 | 0.0 | 0.0 | 78.0 | 102.0557 | -4.9794 | True | False |
| LCA | 2008 | wet_year;nitrogen_stress_year;low_response_year | 393.8 | 0.0 | 0.0 | 84.0 | 6.0883 | -5.9391 | True | True |
| LCA | 2009 | wet_year;nitrogen_stress_year;low_response_year | 417.1 | 0.0 | 0.0 | 84.0 | -16.4856 | -6.1649 | True | True |
| LCA | 2010 | normal_year;nitrogen_stress_year;low_response_year | 288.5 | 0.0 | 0.0 | 77.0 | 23.1171 | -5.7688 | True | False |
| LCA | 2011 | normal_year;nitrogen_stress_year;low_response_year | 300.9 | 0.0 | 0.0 | 84.0 | 191.3782 | -5.2205 | True | False |
| LCA | 2012 | normal_year;nitrogen_stress_year;low_response_year | 262.0 | 0.0 | 0.0 | 85.0 | 307.1844 | -2.9282 | True | False |
| LCA | 2013 | wet_year;nitrogen_stress_year;low_response_year | 417.6 | 0.0 | 0.0 | 77.0 | -21.6418 | -6.2164 | True | True |
| LCA | 2014 | normal_year;nitrogen_stress_year;low_response_year | 198.8 | 0.0 | 0.0 | 82.0 | 189.2188 | -4.1078 | True | False |
| LCA | 2015 | normal_year;nitrogen_stress_year;low_response_year | 329.0 | 0.0 | 0.0 | 80.0 | 1.2561 | -6.0214 | True | False |
| LCA | 2016 | normal_year;nitrogen_stress_year;low_response_year | 296.4 | 0.0 | 0.0 | 75.0 | -8.3722 | -6.0837 | True | False |
| LCA | 2017 | dry_year;water_stress_year;nitrogen_stress_year | 175.2 | 1.0 | 0.1204 | 76.0 | 100.7922 | -5.6767 | True | True |
| LCA | 2018 | normal_year;nitrogen_stress_year;low_response_year | 200.1 | 0.0 | 0.0 | 69.0 | 29.7266 | -5.7027 | True | False |
| LCA | 2019 | normal_year;nitrogen_stress_year;low_response_year | 230.1 | 0.0 | 0.0 | 74.0 | 19.5862 | -5.8041 | True | False |
| LCA | 2020 | normal_year;nitrogen_stress_year;low_response_year | 293.5 | 0.0 | 0.0 | 73.0 | 28.6163 | -5.7138 | True | False |
| LCA | 2021 | wet_year;nitrogen_stress_year;low_response_year | 425.3 | 0.0 | 0.0 | 79.0 | -25.5237 | -6.2552 | True | False |
| LCA | 2022 | wet_year;nitrogen_stress_year;low_response_year | 451.3 | 0.0 | 0.0 | 79.0 | -31.5283 | -6.3153 | True | True |
| LCA | 2023 | wet_year;nitrogen_stress_year;low_response_year | 497.2 | 0.0 | 0.0 | 71.0 | 59.6588 | -5.4034 | True | True |
| SYA | 2005 | normal_year;nitrogen_stress_year;low_response_year | 569.2 | 0.0 | 0.0 | 99.0 | -416.4563 | -10.1646 | True | False |
| SYA | 2006 | normal_year;nitrogen_stress_year;low_response_year | 433.1 | 0.0 | 0.0 | 102.0 | -72.951 | -6.7295 | True | False |
| SYA | 2007 | normal_year;nitrogen_stress_year;low_response_year | 419.1 | 0.0 | 0.0 | 95.0 | 181.1444 | -4.6596 | True | False |
| SYA | 2008 | wet_year;nitrogen_stress_year;low_response_year | 584.4 | 0.0 | 0.0 | 86.0 | -745.5487 | -13.4555 | True | True |
| SYA | 2009 | normal_year;water_stress_year;nitrogen_stress_year | 332.0 | 9.0 | 0.5943 | 103.0 | 598.7427 | -3.5881 | True | False |
| SYA | 2010 | wet_year;nitrogen_stress_year;low_response_year | 796.6 | 0.0 | 0.0 | 106.0 | -149.8407 | -7.4984 | True | True |
| SYA | 2011 | normal_year;nitrogen_stress_year;low_response_year | 530.8 | 0.0 | 0.0 | 93.0 | -69.2279 | -6.6923 | True | False |
| SYA | 2012 | wet_year;nitrogen_stress_year;low_response_year | 708.0 | 0.0 | 0.0 | 97.0 | -739.9408 | -13.3994 | True | True |
| SYA | 2013 | normal_year;nitrogen_stress_year;low_response_year | 434.3 | 0.0 | 0.0 | 99.0 | 204.4397 | -3.9556 | True | False |
| SYA | 2014 | normal_year;water_stress_year;nitrogen_stress_year | 331.8 | 12.0 | 0.8005 | 97.0 | 922.9803 | -1.2761 | True | False |
| SYA | 2015 | normal_year;nitrogen_stress_year;low_response_year | 420.1 | 0.0 | 0.0 | 99.0 | 9.1632 | -6.9603 | True | False |
| SYA | 2016 | wet_year;nitrogen_stress_year;low_response_year | 713.6 | 0.0 | 0.0 | 99.0 | -38.927 | -6.3893 | True | True |
| SYA | 2017 | dry_year;water_stress_year;nitrogen_stress_year;irrigation_responsive_year | 301.7 | 13.0 | 0.8179 | 88.0 | 1058.0219 | 2.2807 | True | True |
| SYA | 2018 | normal_year;nitrogen_stress_year;low_response_year | 515.0 | 0.0 | 0.0 | 90.0 | -103.6926 | -7.0369 | True | False |
| SYA | 2019 | wet_year;nitrogen_stress_year;low_response_year | 623.4 | 0.0 | 0.0 | 99.0 | 102.8815 | -4.9712 | True | True |
| SYA | 2020 | normal_year;water_stress_year;nitrogen_stress_year | 570.7 | 3.0 | 0.6626 | 93.0 | 391.7474 | -2.787 | True | False |
| SYA | 2021 | normal_year;nitrogen_stress_year;low_response_year | 562.2 | 0.0 | 0.0 | 90.0 | -370.13 | -9.7013 | True | False |
| SYA | 2022 | wet_year;nitrogen_stress_year;low_response_year | 687.1 | 0.0 | 0.0 | 95.0 | -559.5819 | -11.9796 | True | True |
| SYA | 2023 | normal_year;nitrogen_stress_year;low_response_year | 491.7 | 0.0 | 0.0 | 92.0 | 277.1948 | -3.7685 | True | False |
| YCA | 2004 | dry_year;nitrogen_stress_year;irrigation_responsive_year | 27.6 | 0.0 | 0.0 | 95.0 | 6694.7911 | 56.2031 | True | True |
| YCA | 2005 | wet_year;nitrogen_stress_year;low_response_year | 551.4 | 0.0 | 0.0 | 87.0 | 8.4821 | -5.9152 | True | True |
| YCA | 2006 | normal_year;nitrogen_stress_year;low_response_year | 220.0 | 0.0 | 0.0 | 89.0 | 19.7913 | -5.9064 | True | False |

## Calibration / Validation Route

Use 2020-2021 as calibration candidates and 2022-2023 as validation candidates only where weather QC is ok. Actual cultivar calibration still requires observed yield and phenology records; weather alone is insufficient.

## PPO Route

Do not train unrestricted PPO. If the scenario pool has enough water-stress or irrigation-responsive years, first run all-year offline schedule search. Otherwise, prioritize cultivar calibration and nitrogen-management focus.