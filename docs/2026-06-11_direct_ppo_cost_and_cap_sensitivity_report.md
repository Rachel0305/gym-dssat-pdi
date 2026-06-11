# Direct PPO Cost and Cap Sensitivity Report

## Scope

This stage follows 006_17 direct action-safe PPO. It does not use RF, behavior cloning, imitation learning, offline schedule search, expert replay, constrained PPO fine-tuning, rainfall scaling, reward-structure redesign, episode-level profit reward, phenology-window action design, event action design, or multi-objective PPO.

Only four quantities were changed across scenarios: `water_cost`, `nitrogen_cost`, `season_irrigation_cap`, and `season_n_cap`.

## Tested Scenarios

| scenario | water_cost | nitrogen_cost | season_irrigation_cap | season_n_cap | n_cases | ok_cases |
| --- | --- | --- | --- | --- | --- | --- |
| C_cap_low | 0.1 | 0.25 | 100.0 | 180.0 | 45 | 45 |
| A_baseline | 0.1 | 0.25 | 160.0 | 250.0 | 45 | 45 |
| D_cost_high_cap_low | 0.3 | 0.5 | 100.0 | 180.0 | 45 | 45 |
| B_cost_high | 0.3 | 0.5 | 160.0 | 250.0 | 45 | 45 |

## Scenario Rollup

| scenario | water_cost | nitrogen_cost | season_irrigation_cap | season_n_cap | n_cases | ok_cases | cap_saturated_cases | cap_saturation_rate | recommended_cases | recommended_rate | mean_irrigation | mean_n | mean_final_grnwt | mean_final_topwt | mean_profit_simple | min_yield_fraction_vs_baseline | mean_yield_fraction_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_cap_low | 0.1 | 0.25 | 100.0 | 180.0 | 45 | 45 | 45 | 1.0 | 0 | 0.0 | 100.0 | 180.0 | 7833.7424 | 15962.2151 | 23.3374 | 0.5727 | 0.9598 |
| A_baseline | 0.1 | 0.25 | 160.0 | 250.0 | 45 | 45 | 45 | 1.0 | 0 | 0.0 | 160.0 | 250.0 | 8189.7598 | 16416.5481 | 3.3976 | 1.0 | 1.0 |
| D_cost_high_cap_low | 0.3 | 0.5 | 100.0 | 180.0 | 45 | 45 | 45 | 1.0 | 0 | 0.0 | 100.0 | 180.0 | 7833.5031 | 15962.0517 | -41.665 | 0.5735 | 0.9597 |
| B_cost_high | 0.3 | 0.5 | 160.0 | 250.0 | 45 | 45 | 45 | 1.0 | 0 | 0.0 | 160.0 | 250.0 | 8187.6201 | 16412.1227 | -91.1238 | 0.9896 | 0.9997 |

## Recommended Configuration

| scenario | water_cost | nitrogen_cost | season_irrigation_cap | season_n_cap | n_cases | ok_cases | cap_saturated_cases | cap_saturation_rate | recommended_cases | recommended_rate | mean_irrigation | mean_n | mean_final_grnwt | mean_final_topwt | mean_profit_simple | min_yield_fraction_vs_baseline | mean_yield_fraction_vs_baseline | recommendation_status | recommendation_reason | advantages | limitations | suitable_as_main_thesis_result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_cap_low | 0.1 | 0.25 | 100.0 | 180.0 | 45 | 45 | 45 | 1.0 | 0 | 0.0 | 100.0 | 180.0 | 7833.7424 | 15962.2151 | 23.3374 | 0.5727 | 0.9598 | fallback_for_mentor_discussion_not_final | No scenario met the strict non-saturation criterion; this fallback keeps yield near baseline with lower input caps, but still saturates its cap. | Keeps TOPWT/GRNWT workflow comparable to 006_17, but still saturates caps. | No tested parameter set fully solved cap saturation; not suitable as final thesis result yet. | False |

## Sensitivity Summary Preview

| scenario | station_code | year | total_irrigation | total_n | final_grnwt | profit_simple | decision_reasonableness_label | case_recommended | yield_fraction_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline | FQA | 2007 | 160.0 | 250.0 | 7078.9526 | -7.7105 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2008 | 160.0 | 250.0 | 7234.4885 | -6.1551 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2016 | 160.0 | 250.0 | 7358.6218 | -4.9138 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2020 | 160.0 | 250.0 | 7990.318 | 1.4032 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2023 | 160.0 | 250.0 | 8224.3243 | 3.7432 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2005 | 160.0 | 250.0 | 7339.7455 | -5.1025 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2011 | 160.0 | 250.0 | 5627.3279 | -22.2267 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2015 | 160.0 | 250.0 | 6931.5387 | -9.1846 | cap_saturated | False | 1.0 |
| A_baseline | FQA | 2019 | 160.0 | 250.0 | 7603.8507 | -2.4615 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2004 | 160.0 | 250.0 | 6024.1089 | -18.2589 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2020 | 160.0 | 250.0 | 6866.9489 | -9.8305 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2021 | 160.0 | 250.0 | 5059.6362 | -27.9036 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2022 | 160.0 | 250.0 | 7107.605 | -7.424 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2023 | 160.0 | 250.0 | 6077.9376 | -17.7206 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2012 | 160.0 | 250.0 | 0.0 | -78.5 | cap_saturated | False |  |
| A_baseline | HLA | 2013 | 160.0 | 250.0 | 5172.4792 | -26.7752 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2015 | 160.0 | 250.0 | 6969.3353 | -8.8066 | cap_saturated | False | 1.0 |
| A_baseline | HLA | 2018 | 160.0 | 250.0 | 6115.9259 | -17.3407 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2017 | 160.0 | 250.0 | 9008.5815 | 11.5858 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2020 | 160.0 | 250.0 | 7666.1603 | -1.8384 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2021 | 160.0 | 250.0 | 10986.4722 | 31.3647 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2022 | 160.0 | 250.0 | 9990.0183 | 21.4002 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2023 | 160.0 | 250.0 | 7888.3557 | 0.3836 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2006 | 160.0 | 250.0 | 8996.9904 | 11.4699 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2008 | 160.0 | 250.0 | 10768.1665 | 29.1817 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2009 | 160.0 | 250.0 | 9222.0728 | 13.7207 | cap_saturated | False | 1.0 |
| A_baseline | LCA | 2013 | 160.0 | 250.0 | 8756.2109 | 9.0621 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2009 | 160.0 | 250.0 | 12165.1257 | 43.1513 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2014 | 160.0 | 250.0 | 10363.6267 | 25.1363 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2017 | 160.0 | 250.0 | 10518.1335 | 26.6813 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2020 | 160.0 | 250.0 | 10000.9149 | 21.5091 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2022 | 160.0 | 250.0 | 10597.2278 | 27.4723 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2008 | 160.0 | 250.0 | 11073.4253 | 32.2343 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2010 | 160.0 | 250.0 | 8490.4968 | 6.405 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2012 | 160.0 | 250.0 | 10106.0101 | 22.5601 | cap_saturated | False | 1.0 |
| A_baseline | SYA | 2023 | 160.0 | 250.0 | 10738.2886 | 28.8829 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2004 | 160.0 | 250.0 | 8064.6497 | 2.1465 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2014 | 160.0 | 250.0 | 9381.2738 | 15.3127 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2015 | 160.0 | 250.0 | 9082.2668 | 12.3227 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2019 | 160.0 | 250.0 | 7598.8336 | -2.5117 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2022 | 160.0 | 250.0 | 8496.1908 | 6.4619 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2005 | 160.0 | 250.0 | 9111.3995 | 12.614 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2010 | 160.0 | 250.0 | 7246.4764 | -6.0352 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2012 | 160.0 | 250.0 | 7861.5112 | 0.1151 | cap_saturated | False | 1.0 |
| A_baseline | YCA | 2023 | 160.0 | 250.0 | 9577.1661 | 17.2717 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2007 | 160.0 | 250.0 | 7078.9972 | -102.21 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2008 | 160.0 | 250.0 | 7234.4812 | -100.6552 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2016 | 160.0 | 250.0 | 7358.5553 | -99.4144 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2020 | 160.0 | 250.0 | 7990.3204 | -93.0968 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2023 | 160.0 | 250.0 | 8224.3292 | -90.7567 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2005 | 160.0 | 250.0 | 7339.5404 | -99.6046 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2011 | 160.0 | 250.0 | 5627.3413 | -116.7266 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2015 | 160.0 | 250.0 | 6931.5393 | -103.6846 | cap_saturated | False | 1.0 |
| B_cost_high | FQA | 2019 | 160.0 | 250.0 | 7603.8507 | -96.9615 | cap_saturated | False | 1.0 |
| B_cost_high | HLA | 2004 | 160.0 | 250.0 | 6016.0406 | -112.8396 | cap_saturated | False | 0.9987 |
| B_cost_high | HLA | 2020 | 160.0 | 250.0 | 6866.9812 | -104.3302 | cap_saturated | False | 1.0 |
| B_cost_high | HLA | 2021 | 160.0 | 250.0 | 5059.5978 | -122.404 | cap_saturated | False | 1.0 |
| B_cost_high | HLA | 2022 | 160.0 | 250.0 | 7107.5415 | -101.9246 | cap_saturated | False | 1.0 |
| B_cost_high | HLA | 2023 | 160.0 | 250.0 | 6077.9376 | -112.2206 | cap_saturated | False | 1.0 |
| B_cost_high | HLA | 2012 | 160.0 | 250.0 | 0.0 | -173.0 | cap_saturated | False |  |
| B_cost_high | HLA | 2013 | 160.0 | 250.0 | 5172.5165 | -121.2748 | cap_saturated | False | 1.0 |
| B_cost_high | HLA | 2015 | 160.0 | 250.0 | 6969.433 | -103.3057 | cap_saturated | False | 1.0 |
| B_cost_high | HLA | 2018 | 160.0 | 250.0 | 6116.1224 | -111.8388 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2017 | 160.0 | 250.0 | 9008.5815 | -82.9142 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2020 | 160.0 | 250.0 | 7666.2921 | -96.3371 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2021 | 160.0 | 250.0 | 10986.4685 | -63.1353 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2022 | 160.0 | 250.0 | 9990.2423 | -73.0976 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2023 | 160.0 | 250.0 | 7888.4943 | -94.1151 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2006 | 160.0 | 250.0 | 8997.1887 | -83.0281 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2008 | 160.0 | 250.0 | 10768.3105 | -65.3169 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2009 | 160.0 | 250.0 | 9222.2205 | -80.7778 | cap_saturated | False | 1.0 |
| B_cost_high | LCA | 2013 | 160.0 | 250.0 | 8756.1823 | -85.4382 | cap_saturated | False | 1.0 |
| B_cost_high | SYA | 2009 | 160.0 | 250.0 | 12165.3748 | -51.3463 | cap_saturated | False | 1.0 |
| B_cost_high | SYA | 2014 | 160.0 | 250.0 | 10362.821 | -69.3718 | cap_saturated | False | 0.9999 |
| B_cost_high | SYA | 2017 | 160.0 | 250.0 | 10517.594 | -67.8241 | cap_saturated | False | 0.9999 |
| B_cost_high | SYA | 2020 | 160.0 | 250.0 | 10000.9186 | -72.9908 | cap_saturated | False | 1.0 |
| B_cost_high | SYA | 2022 | 160.0 | 250.0 | 10597.5574 | -67.0244 | cap_saturated | False | 1.0 |
| B_cost_high | SYA | 2008 | 160.0 | 250.0 | 11073.4424 | -62.2656 | cap_saturated | False | 1.0 |
| B_cost_high | SYA | 2010 | 160.0 | 250.0 | 8490.4968 | -88.095 | cap_saturated | False | 1.0 |
| B_cost_high | SYA | 2012 | 160.0 | 250.0 | 10105.9778 | -71.9402 | cap_saturated | False | 1.0 |

## Method Sources and Literature Support

- [Gautron et al. 2022 / gym-DSSAT](https://arxiv.org/abs/2207.03270): DSSAT can be wrapped as a Gym-style RL environment for crop management experiments.
- [Wu et al. 2022](https://arxiv.org/abs/2204.10394): Nitrogen management can be formulated as an RL problem using DSSAT crop simulations and input-cost tradeoffs.
- [Tao et al. 2022/2023](https://www.ijcai.org/proceedings/2023/691): Water and nitrogen management can be jointly optimized in DSSAT-based crop simulation settings.
- [Kallenberg et al. 2023](https://www.cambridge.org/core/journals/environmental-data-science/article/nitrogen-management-with-reinforcement-learning-and-crop-growth-models/358749FAFAA4990B1448DAB7F48D641C): CropGym demonstrates RL for crop-growth-model nitrogen management with yield and environmental tradeoffs.
- [Saikai et al. 2023](https://journals.plos.org/water/article?id=10.1371/journal.pwat.0000169): Deep RL can be used for irrigation scheduling with crop-model simulations and profit-oriented evaluation.

The literature supports the general formulation of crop management as simulator-based reinforcement learning and the use of input-cost tradeoffs. The specific finding here, namely whether the four tested cost/cap settings avoid cap saturation in the five Chinese station-year pool, is this project's experimental result and should not be attributed to the cited papers.