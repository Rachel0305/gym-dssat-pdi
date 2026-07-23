# 031_31 Four-site free-timing MaskablePPO readiness record

## Scope

- No training and no DSSAT rerun.
- This audit prepares HLA/FQA/LCA/YCA for the same free-timing discrete MaskablePPO workflow used in SYA.
- The SYA result is treated as completed reference, not retuned.

## Station-year inventory

| station_code | available_year_count | first_year | last_year | available_years | has_any_weather_after_2000 |
| --- | --- | --- | --- | --- | --- |
| FQA | 19 | 2005 | 2023 | 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 | True |
| HLA | 20 | 2004 | 2023 | 2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 | True |
| LCA | 19 | 2005 | 2023 | 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 | True |
| SYA | 19 | 2005 | 2023 | 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 | True |
| YCA | 20 | 2004 | 2023 | 2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023 | True |

## Proposed train-year readiness

| station_code | proposed_train_year | role | scenario_pool_row_count | weather_file | weather_exists | smoke_rows | smoke_ok_rows | smoke_final_grnwt | smoke_notes | ready_for_checkpoint_selection |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SYA | 2014 | reference_completed | 1 | Leave_One_experiments\wth_generated_qc\SYA\SYA2014.WTH | True | 2 | 2 | 10943.0957 |  | True |
| FQA | 2016 | target_next | 1 | Leave_One_experiments\wth_generated_qc\FQA\FQA2016.WTH | True | 2 | 2 | 7231.1621 |  | True |
| HLA | 2015 | target_next | 1 | Leave_One_experiments\wth_generated_qc\HLA\HLA2015.WTH | True | 2 | 2 | 6947.3767 |  | True |
| LCA | 2010 | target_next | 1 | Leave_One_experiments\wth_generated_qc\LCA\LCA2010.WTH | True | 2 | 2 | 8348.9923 |  | True |
| YCA | 2014 | target_next | 1 | Leave_One_experiments\wth_generated_qc\YCA\YCA2014.WTH | True | 2 | 2 | 8256.72 |  | True |

## Next task plan

| station_code | proposed_train_year | next_task | recommended_order | notes |
| --- | --- | --- | --- | --- |
| SYA | 2014 | already_completed_SYA_031_27_to_031_30 |  |  |
| FQA | 2016 | ready_for_checkpoint_selection | HLA,FQA,LCA,YCA | Use same frozen config as SYA; no tuning before first cross-station pass. |
| HLA | 2015 | ready_for_checkpoint_selection | HLA,FQA,LCA,YCA | Use same frozen config as SYA; no tuning before first cross-station pass. |
| LCA | 2010 | ready_for_checkpoint_selection | HLA,FQA,LCA,YCA | Use same frozen config as SYA; no tuning before first cross-station pass. |
| YCA | 2014 | ready_for_checkpoint_selection | HLA,FQA,LCA,YCA | Use same frozen config as SYA; no tuning before first cross-station pass. |