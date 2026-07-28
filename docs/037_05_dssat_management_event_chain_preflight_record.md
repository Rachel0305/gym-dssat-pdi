# 037_05：DSSAT 管理事件生效链路预检记录

## 结论先说

- 审计 snapshot 数：388
- 异常 snapshot 数：245
- 明细表：`benchmark_results/037_05_dssat_management_event_chain_preflight/tables/037_05_management_event_chain_audit.csv`
- 异常表：`benchmark_results/037_05_dssat_management_event_chain_preflight/tables/037_05_management_event_chain_issues.csv`

## 按站点/情景/状态汇总

| station_code | scenario | status | count |
| --- | --- | --- | --- |
| FQA | dssat_auto | event_chain_issue | 12 |
| FQA | dssat_auto | ok | 7 |
| FQA | null | ok | 19 |
| FQA | official_extension_expert | event_chain_issue | 19 |
| FQA | recorded_farmer_template | ok | 19 |
| HLA | dssat_auto | event_chain_issue | 12 |
| HLA | dssat_auto | ok | 8 |
| HLA | null | ok | 20 |
| HLA | official_extension_expert | event_chain_issue | 20 |
| HLA | recorded_farmer_template | event_chain_issue | 20 |
| LCA | dssat_auto | event_chain_issue | 18 |
| LCA | dssat_auto | ok | 1 |
| LCA | null | ok | 19 |
| LCA | official_extension_expert | event_chain_issue | 19 |
| LCA | recorded_farmer_template | event_chain_issue | 19 |
| SYA | dssat_auto | event_chain_issue | 17 |
| SYA | dssat_auto | ok | 2 |
| SYA | null | ok | 19 |
| SYA | official_extension_expert | event_chain_issue | 19 |
| SYA | recorded_farmer_template | event_chain_issue | 19 |
| YCA | dssat_auto | event_chain_issue | 11 |
| YCA | dssat_auto | ok | 9 |
| YCA | null | ok | 20 |
| YCA | official_extension_expert | event_chain_issue | 20 |
| YCA | recorded_farmer_template | event_chain_issue | 20 |

## 异常明细

| station_code | year | scenario | status | issues | planned_irrigation_event_count | inp_irrigation_event_count | mgmt_irrigation_event_count | planned_n_event_count | inp_n_event_count | mgmt_n_event_count | snapshot_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQA | 2005 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2005/official_extension_expert |
| FQA | 2006 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2006/official_extension_expert |
| FQA | 2007 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2007/dssat_auto |
| FQA | 2007 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2007/official_extension_expert |
| FQA | 2008 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 3 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2008/dssat_auto |
| FQA | 2008 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2008/official_extension_expert |
| FQA | 2009 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2009/dssat_auto |
| FQA | 2009 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2009/official_extension_expert |
| FQA | 2010 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2010/dssat_auto |
| FQA | 2010 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2010/official_extension_expert |
| FQA | 2011 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 2 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2011/dssat_auto |
| FQA | 2011 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2011/official_extension_expert |
| FQA | 2012 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2012/official_extension_expert |
| FQA | 2013 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2013/official_extension_expert |
| FQA | 2014 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2014/official_extension_expert |
| FQA | 2015 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2015/official_extension_expert |
| FQA | 2016 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 3 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2016/dssat_auto |
| FQA | 2016 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2016/official_extension_expert |
| FQA | 2017 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 2 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2017/dssat_auto |
| FQA | 2017 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2017/official_extension_expert |
| FQA | 2018 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 4 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2018/dssat_auto |
| FQA | 2018 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2018/official_extension_expert |
| FQA | 2019 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 6 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2019/dssat_auto |
| FQA | 2019 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2019/official_extension_expert |
| FQA | 2020 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2020/dssat_auto |
| FQA | 2020 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2020/official_extension_expert |
| FQA | 2021 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2021/dssat_auto |
| FQA | 2021 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2021/official_extension_expert |
| FQA | 2022 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2022/official_extension_expert |
| FQA | 2023 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 3 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2023/dssat_auto |
| FQA | 2023 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 6 | 1 | 1 | 5 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/FQA/2023/official_extension_expert |
| HLA | 2004 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 7 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2004/dssat_auto |
| HLA | 2004 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2004/official_extension_expert |
| HLA | 2004 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2004/recorded_farmer_template |
| HLA | 2005 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2005/dssat_auto |
| HLA | 2005 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2005/official_extension_expert |
| HLA | 2005 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2005/recorded_farmer_template |
| HLA | 2006 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2006/official_extension_expert |
| HLA | 2006 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2006/recorded_farmer_template |
| HLA | 2007 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 2 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2007/dssat_auto |
| HLA | 2007 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2007/official_extension_expert |
| HLA | 2007 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2007/recorded_farmer_template |
| HLA | 2008 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2008/official_extension_expert |
| HLA | 2008 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2008/recorded_farmer_template |
| HLA | 2009 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2009/dssat_auto |
| HLA | 2009 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2009/official_extension_expert |
| HLA | 2009 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2009/recorded_farmer_template |
| HLA | 2010 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 3 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2010/dssat_auto |
| HLA | 2010 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2010/official_extension_expert |
| HLA | 2010 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2010/recorded_farmer_template |
| HLA | 2011 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2011/dssat_auto |
| HLA | 2011 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2011/official_extension_expert |
| HLA | 2011 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2011/recorded_farmer_template |
| HLA | 2012 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2012/official_extension_expert |
| HLA | 2012 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2012/recorded_farmer_template |
| HLA | 2013 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2013/dssat_auto |
| HLA | 2013 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2013/official_extension_expert |
| HLA | 2013 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2013/recorded_farmer_template |
| HLA | 2014 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2014/official_extension_expert |
| HLA | 2014 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2014/recorded_farmer_template |
| HLA | 2015 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 2 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2015/dssat_auto |
| HLA | 2015 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2015/official_extension_expert |
| HLA | 2015 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2015/recorded_farmer_template |
| HLA | 2016 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 2 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2016/dssat_auto |
| HLA | 2016 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2016/official_extension_expert |
| HLA | 2016 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2016/recorded_farmer_template |
| HLA | 2017 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 1 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2017/dssat_auto |
| HLA | 2017 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2017/official_extension_expert |
| HLA | 2017 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2017/recorded_farmer_template |
| HLA | 2018 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2018/official_extension_expert |
| HLA | 2018 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2018/recorded_farmer_template |
| HLA | 2019 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2019/official_extension_expert |
| HLA | 2019 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2019/recorded_farmer_template |
| HLA | 2020 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2020/official_extension_expert |
| HLA | 2020 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2020/recorded_farmer_template |
| HLA | 2021 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2021/official_extension_expert |
| HLA | 2021 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2021/recorded_farmer_template |
| HLA | 2022 | dssat_auto | event_chain_issue | irrigation_event_count_inp_vs_mgmt;irrigation_amount_inp_vs_mgmt | 0 | 0 | 2 | 0 | 0 | 0 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2022/dssat_auto |
| HLA | 2022 | official_extension_expert | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp;nitrogen_event_drop_filex_to_inp;nitrogen_amount_filex_vs_inp | 8 | 1 | 1 | 6 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2022/official_extension_expert |
| HLA | 2022 | recorded_farmer_template | event_chain_issue | irrigation_event_drop_filex_to_inp;irrigation_amount_filex_vs_inp | 3 | 1 | 1 | 1 | 1 | 1 | benchmark_results/034_00_multisite_input_ic1_four_baseline_rebuild/snapshots/HLA/2022/recorded_farmer_template |

## 解释与边界

- 本次审计把 388 个 snapshot 全部读了一遍，其中 245 个出现 `event_chain_issue`。这个数字不能简单理解为“全部都是同一种 bug”。
- `official_extension_expert` 与 `recorded_farmer_template` 中大量异常属于真正危险信号：`fileX.MZX` 里有多次手工灌溉/施肥计划，但 `DSSAT48.INP` 与 `MgmtEvent.OUT` 中只出现第一条或明显更少的事件。LC2019 official expert 是典型例子：计划灌溉 6 次、施氮 5 次，但实际输入/执行只有灌溉 1 次、施氮 1 次。
- `dssat_auto` 的异常需要单独解释：自动管理事件可能由 DSSAT 运行时生成，因此 `DSSAT48.INP` 里没有预先写入的手工事件、`MgmtEvent.OUT` 中却出现自动灌溉，并不等同于“动作没有进入 DSSAT”。后续应为 auto 情景使用单独的自动管理审计规则。
- 因此，037_04 LC2019 及依赖 034_00 手工基线 snapshot 的五情景图/指标比较，目前只能作为 debug 结果，不能作为正式结果使用。

## 使用规则

- 后续正式训练、指标汇总、五情景图绘制前，必须先通过本类 preflight。
- 只要出现 `event_chain_issue`，该 snapshot 不进入正式比较，先修正管理事件生效链路。
