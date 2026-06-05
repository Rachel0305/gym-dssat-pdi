# 2026-06-05 observed year smoke test report

## 1. 本阶段目的
本阶段是环境体检，不是优化。目标是在进入 PPO 前验证 Phase 2c 推荐的真实试验年份是否能在 gym-DSSAT 中稳定跑完，并保存 daily CSV、响应图和汇总表。

## 2. 使用的站点和年份
测试年份来自 Phase 2c 的实测物候年份，共 14 个 station-year：FQA 2008/2010，HLA 2007/2011/2009，LCA 2010/2011/2008/2009，SYA 2014/2015/2012，YCA 2014/2008。

## 3. 固定策略
- null_zero: amir=0, anfer=0。
- fixed_low_input: DAP 1/30 施氮 30/20 kg ha-1。
- fixed_medium_input: DAP 1/30/60 施氮 50/50/50 kg ha-1，DAP 30/60 灌水 30/30 mm。
- fixed_high_input: DAP 1/30/60 施氮 80/80/90 kg ha-1，DAP 30/60/90 灌水 40/40/40 mm。

## 4. 最小 smoke test 结果
| station   |   year | policy_name   | run_status   | episode_completed   |   episode_length | daily_csv_path                                                                   |
|:----------|-------:|:--------------|:-------------|:--------------------|-----------------:|:---------------------------------------------------------------------------------|
| HLA       |   2007 | null_zero     | ok           | True                |              149 | Leave_One_experiments/smoke_tests/daily_outputs/HLA/HLA_2007_null_zero_daily.csv |

## 5. 56 次 episode 批量运行状态
| run_status   |   count |
|:-------------|--------:|
| timeout      |      29 |
| ok           |      27 |

按站点统计：
| station   |   ok |   timeout |
|:----------|-----:|----------:|
| FQA       |    8 |         0 |
| HLA       |   12 |         0 |
| LCA       |    4 |        12 |
| SYA       |    3 |         9 |
| YCA       |    0 |         8 |

## 6. 失败案例和错误信息
| station   |   year | policy_name        | run_status   | error_message      |
|:----------|-------:|:-------------------|:-------------|:-------------------|
| LCA       |   2010 | fixed_low_input    | timeout      | timeout_after_300s |
| LCA       |   2010 | fixed_medium_input | timeout      | timeout_after_300s |
| LCA       |   2010 | fixed_high_input   | timeout      | timeout_after_300s |
| LCA       |   2011 | fixed_low_input    | timeout      | timeout_after_300s |
| LCA       |   2011 | fixed_medium_input | timeout      | timeout_after_300s |
| LCA       |   2011 | fixed_high_input   | timeout      | timeout_after_300s |
| LCA       |   2008 | fixed_low_input    | timeout      | timeout_after_300s |
| LCA       |   2008 | fixed_medium_input | timeout      | timeout_after_300s |
| LCA       |   2008 | fixed_high_input   | timeout      | timeout_after_300s |
| LCA       |   2009 | fixed_low_input    | timeout      | timeout_after_300s |
| LCA       |   2009 | fixed_medium_input | timeout      | timeout_after_300s |
| LCA       |   2009 | fixed_high_input   | timeout      | timeout_after_300s |
| SYA       |   2014 | fixed_low_input    | timeout      | timeout_after_300s |
| SYA       |   2014 | fixed_medium_input | timeout      | timeout_after_300s |
| SYA       |   2014 | fixed_high_input   | timeout      | timeout_after_300s |
| SYA       |   2015 | fixed_low_input    | timeout      | timeout_after_300s |
| SYA       |   2015 | fixed_medium_input | timeout      | timeout_after_300s |
| SYA       |   2015 | fixed_high_input   | timeout      | timeout_after_300s |
| SYA       |   2012 | fixed_low_input    | timeout      | timeout_after_300s |
| SYA       |   2012 | fixed_medium_input | timeout      | timeout_after_300s |
| SYA       |   2012 | fixed_high_input   | timeout      | timeout_after_300s |
| YCA       |   2014 | null_zero          | timeout      | timeout_after_300s |
| YCA       |   2014 | fixed_low_input    | timeout      | timeout_after_300s |
| YCA       |   2014 | fixed_medium_input | timeout      | timeout_after_300s |
| YCA       |   2014 | fixed_high_input   | timeout      | timeout_after_300s |
| YCA       |   2008 | null_zero          | timeout      | timeout_after_300s |
| YCA       |   2008 | fixed_low_input    | timeout      | timeout_after_300s |
| YCA       |   2008 | fixed_medium_input | timeout      | timeout_after_300s |
| YCA       |   2008 | fixed_high_input   | timeout      | timeout_after_300s |

## 7. daily output 和图表完整性
- 成功 episode 数：27。
- daily CSV 数：27。
- episode 响应图数：135，按 5 张/成功 episode 计算应为 135。
- 站点级对比图数：25。

## 8. 固定策略响应对比
| station   | policy_name        |   final_grnwt_mean |   total_irrigation_mean |   total_n_mean |   mean_swfac |   mean_nstres |
|:----------|:-------------------|-------------------:|------------------------:|---------------:|-------------:|--------------:|
| FQA       | fixed_high_input   |           6484.29  |                     120 |            250 |    0.0017068 |   0.000157866 |
| FQA       | fixed_low_input    |           4917.51  |                       0 |             50 |    0.0105491 |   0.104264    |
| FQA       | fixed_medium_input |           6102.97  |                      60 |            150 |    0.0235046 |   0.00131217  |
| FQA       | null_zero          |           2292.17  |                       0 |              0 |    0         |   0.190715    |
| HLA       | fixed_high_input   |           7237.99  |                     120 |            250 |    0         |   0.0268745   |
| HLA       | fixed_low_input    |           1766.75  |                       0 |             50 |    0         |   0.274753    |
| HLA       | fixed_medium_input |           6515.7   |                      60 |            150 |    0         |   0.132381    |
| HLA       | null_zero          |            291.191 |                       0 |              0 |    0         |   0.375645    |
| LCA       | null_zero          |           2907.11  |                       0 |              0 |    0         |   0.221437    |
| SYA       | null_zero          |           2697.39  |                       0 |              0 |    0         |   0.197414    |

## 9. 变量缺失或异常
成功 episode 的 daily CSV 均包含 prompt 要求的核心列：dap、topwt、grnwt、xlai、totir、tofer、swfac、nstres、reward、real_action_amir、real_action_anfer、normalized_action_amir、normalized_action_anfer。若环境没有 tofer，则脚本使用施氮动作累计值补充。

## 10. 是否可以进入 PPO 训练脚本生成
否。YCA 全部超时，LCA/SYA 固定水氮策略大量超时；需要先排查这些环境输入或动作导致的卡顿。

## 11. 需要单独排错的站点或年份
- LCA: null_zero 全部成功，但 fixed_low/medium/high 全部 timeout，优先检查固定动作是否导致 DSSAT 交互进程等待或输出异常。
- SYA: null_zero 全部成功，但 fixed_low/medium/high 全部 timeout。
- YCA: null_zero 和全部固定策略均 timeout，优先检查 YCA 临时模板、WTH 命名、土壤/品种/管理组合和 DSSAT 日志。

## 12. 生成文件
- `Leave_One_experiments/smoke_tests/evaluation/smoke_test_summary.csv`
- `Leave_One_experiments/smoke_tests/evaluation/smoke_test_minimal_summary.csv`
- `Leave_One_experiments/smoke_tests/daily_outputs/`
- `Leave_One_experiments/smoke_tests/figures/`
- `Leave_One_experiments/smoke_tests/rendered_inputs/`
- `docs/2026-06-05_observed_year_smoke_test_report.md`
- `docs/2026-06-05_observed_year_smoke_test_report.pptx`
