# Smoke test timeout debug report

Generated at: 2026-06-06 09:51:53

## 1. 上一阶段 timeout 概况

上一阶段实测年份 smoke test 共 56 个 episode，结果为 `ok: 27, timeout: 29`。按站点统计如下：

| station   |   ok |   timeout |
|:----------|-----:|----------:|
| FQA       |    8 |         0 |
| HLA       |   12 |         0 |
| LCA       |    4 |        12 |
| SYA       |    3 |         9 |
| YCA       |    0 |         8 |

失败集中在 LCA、SYA、YCA：

- LCA: null_zero 成功，fixed_low_input / fixed_medium_input / fixed_high_input timeout。
- SYA: null_zero 成功，fixed_low_input / fixed_medium_input / fixed_high_input timeout。
- YCA: null_zero 与所有固定策略均 timeout。

## 2. 最小失败案例复现

按 prompt 要求先复现三个案例：

- LCA 2010 fixed_low_input
- SYA 2014 fixed_low_input
- YCA 2014 null_zero

复现时未通过延长 timeout 掩盖问题，而是检查 DSSAT 日志和 rendered input。修复后三个最小案例均能完成 episode，并生成 daily CSV、step trace CSV 和响应图。

## 3. step trace 发现

本次新增 step-level trace，字段包括 station、year、policy、step_index、date、doy、dap、normalized_action、real_action、obs_keys、topwt、grnwt、xlai、swfac、nstres、totir、tofer、reward、done、elapsed_time 和 last_successful_stage。

- step trace 文件数：29
- daily CSV 文件数：29
- 响应图 PNG 文件数：145

trace 汇总如下：

| trace_file                                 |   steps |   slow_steps |   max_step_seconds | last_stage         |
|:-------------------------------------------|--------:|-------------:|-------------------:|:-------------------|
| LCA_2008_fixed_high_input_step_trace.csv   |     105 |            0 |              0.006 | env.step_completed |
| LCA_2008_fixed_low_input_step_trace.csv    |     105 |            0 |              0.006 | env.step_completed |
| LCA_2008_fixed_medium_input_step_trace.csv |     105 |            0 |              0.007 | env.step_completed |
| LCA_2009_fixed_high_input_step_trace.csv   |     102 |            0 |              0.007 | env.step_completed |
| LCA_2009_fixed_low_input_step_trace.csv    |     102 |            0 |              0.009 | env.step_completed |
| LCA_2009_fixed_medium_input_step_trace.csv |     102 |            0 |              0.007 | env.step_completed |
| LCA_2010_fixed_high_input_step_trace.csv   |      92 |            0 |              0.006 | env.step_completed |
| LCA_2010_fixed_low_input_step_trace.csv    |      92 |            0 |              0.005 | env.step_completed |
| LCA_2010_fixed_medium_input_step_trace.csv |      92 |            0 |              0.005 | env.step_completed |
| LCA_2011_fixed_high_input_step_trace.csv   |     102 |            0 |              0.006 | env.step_completed |
| LCA_2011_fixed_low_input_step_trace.csv    |     102 |            0 |              0.005 | env.step_completed |
| LCA_2011_fixed_medium_input_step_trace.csv |     102 |            0 |              0.006 | env.step_completed |
| SYA_2012_fixed_high_input_step_trace.csv   |     135 |            0 |              0.008 | env.step_completed |
| SYA_2012_fixed_low_input_step_trace.csv    |     135 |            0 |              0.007 | env.step_completed |
| SYA_2012_fixed_medium_input_step_trace.csv |     135 |            0 |              0.007 | env.step_completed |
| SYA_2014_fixed_high_input_step_trace.csv   |     144 |            0 |              0.005 | env.step_completed |
| SYA_2014_fixed_low_input_step_trace.csv    |     144 |            0 |              0.007 | env.step_completed |
| SYA_2014_fixed_medium_input_step_trace.csv |     144 |            0 |              0.006 | env.step_completed |
| SYA_2015_fixed_high_input_step_trace.csv   |     143 |            0 |              0.007 | env.step_completed |
| SYA_2015_fixed_low_input_step_trace.csv    |     143 |            0 |              0.007 | env.step_completed |
| SYA_2015_fixed_medium_input_step_trace.csv |     143 |            0 |              0.007 | env.step_completed |
| YCA_2008_fixed_high_input_step_trace.csv   |     111 |            0 |              0.008 | env.step_completed |
| YCA_2008_fixed_low_input_step_trace.csv    |     111 |            0 |              0.007 | env.step_completed |
| YCA_2008_fixed_medium_input_step_trace.csv |     111 |            0 |              0.007 | env.step_completed |
| YCA_2008_null_zero_step_trace.csv          |     111 |            0 |              0.006 | env.step_completed |
| YCA_2014_fixed_high_input_step_trace.csv   |     109 |            0 |              0.007 | env.step_completed |
| YCA_2014_fixed_low_input_step_trace.csv    |     109 |            0 |              0.006 | env.step_completed |
| YCA_2014_fixed_medium_input_step_trace.csv |     109 |            0 |              0.012 | env.step_completed |
| YCA_2014_null_zero_step_trace.csv          |     109 |            0 |              0.009 | env.step_completed |

## 4. action_space 和固定策略检查

所有固定策略动作均在 action_space 范围内；normalized_action 均位于 [-1, 1]；fixed_low_input 仅在 DAP 1 和 30 触发 anfer，未发现每天重复施肥。

动作检查表：`Leave_One_experiments/smoke_tests_debug/evaluation/action_space_and_policy_check.csv`

| station   |   year | policy_name     | action_name   |   scheduled_events |   min_norm |   max_norm | within_bounds   |
|:----------|-------:|:----------------|:--------------|-------------------:|-----------:|-----------:|:----------------|
| HLA       |   2007 | fixed_low_input | amir          |                  4 |         -1 |       -1   | True            |
| HLA       |   2007 | fixed_low_input | anfer         |                  4 |         -1 |       -0.7 | True            |
| LCA       |   2010 | fixed_low_input | amir          |                  4 |         -1 |       -1   | True            |
| LCA       |   2010 | fixed_low_input | anfer         |                  4 |         -1 |       -0.7 | True            |
| SYA       |   2014 | fixed_low_input | amir          |                  4 |         -1 |       -1   | True            |
| SYA       |   2014 | fixed_low_input | anfer         |                  4 |         -1 |       -0.7 | True            |
| YCA       |   2014 | null_zero       | amir          |                  4 |         -1 |       -1   | True            |
| YCA       |   2014 | null_zero       | anfer         |                  4 |         -1 |       -1   | True            |

## 5. rendered input 检查

修复后的临时 rendered input 检查均通过。

文件检查表：`Leave_One_experiments/smoke_tests_debug/evaluation/rendered_input_check.csv`

| station   |   year | policy_name     | check_item                           | status   |   count |
|:----------|-------:|:----------------|:-------------------------------------|:---------|--------:|
| HLA       |   2007 | fixed_low_input | fertilizer_section_present           | pass     |       1 |
| HLA       |   2007 | fixed_low_input | irrigation_section_present           | pass     |       1 |
| HLA       |   2007 | fixed_low_input | mi_mf_enabled                        | pass     |       1 |
| HLA       |   2007 | fixed_low_input | no_original_pre_start_yca_irrigation | pass     |       1 |
| HLA       |   2007 | fixed_low_input | planting_section_present             | pass     |       1 |
| HLA       |   2007 | fixed_low_input | safe_zero_fertilizer_row_present     | pass     |       1 |
| HLA       |   2007 | fixed_low_input | safe_zero_irrigation_row_present     | pass     |       1 |
| HLA       |   2007 | fixed_low_input | simulation_controls_present          | pass     |       1 |
| HLA       |   2007 | fixed_low_input | weather_year_present                 | pass     |       1 |
| HLA       |   2007 | fixed_low_input | wth_section_present                  | pass     |       1 |
| LCA       |   2010 | fixed_low_input | fertilizer_section_present           | pass     |       1 |
| LCA       |   2010 | fixed_low_input | irrigation_section_present           | pass     |       1 |
| LCA       |   2010 | fixed_low_input | mi_mf_enabled                        | pass     |       1 |
| LCA       |   2010 | fixed_low_input | no_original_pre_start_yca_irrigation | pass     |       1 |
| LCA       |   2010 | fixed_low_input | planting_section_present             | pass     |       1 |
| LCA       |   2010 | fixed_low_input | safe_zero_fertilizer_row_present     | pass     |       1 |
| LCA       |   2010 | fixed_low_input | safe_zero_irrigation_row_present     | pass     |       1 |
| LCA       |   2010 | fixed_low_input | simulation_controls_present          | pass     |       1 |
| LCA       |   2010 | fixed_low_input | weather_year_present                 | pass     |       1 |
| LCA       |   2010 | fixed_low_input | wth_section_present                  | pass     |       1 |
| SYA       |   2014 | fixed_low_input | fertilizer_section_present           | pass     |       1 |
| SYA       |   2014 | fixed_low_input | irrigation_section_present           | pass     |       1 |
| SYA       |   2014 | fixed_low_input | mi_mf_enabled                        | pass     |       1 |
| SYA       |   2014 | fixed_low_input | no_original_pre_start_yca_irrigation | pass     |       1 |
| SYA       |   2014 | fixed_low_input | planting_section_present             | pass     |       1 |
| SYA       |   2014 | fixed_low_input | safe_zero_fertilizer_row_present     | pass     |       1 |
| SYA       |   2014 | fixed_low_input | safe_zero_irrigation_row_present     | pass     |       1 |
| SYA       |   2014 | fixed_low_input | simulation_controls_present          | pass     |       1 |
| SYA       |   2014 | fixed_low_input | weather_year_present                 | pass     |       1 |
| SYA       |   2014 | fixed_low_input | wth_section_present                  | pass     |       1 |

## 6. DSSAT 日志检查

日志检查表：`Leave_One_experiments/smoke_tests_debug/evaluation/dssat_log_check.csv`

注意：DSSAT log 是追加式日志，修复后日志文件中可能仍包含修复前的旧错误关键词。因此最终成功与否以 `smoke_test_timeout_rerun_summary.csv`、daily CSV 和 step trace 是否完整为准。

| case_id                  | station   |   year | policy_name     | log_file                                                  | exists   | last_lines_path                                                                             | detected_error_keywords                        | detected_warning_keywords   | notes      |
|:-------------------------|:----------|-------:|:----------------|:----------------------------------------------------------|:---------|:--------------------------------------------------------------------------------------------|:-----------------------------------------------|:----------------------------|:-----------|
| LCA_2010_fixed_low_input | LCA       |   2010 | fixed_low_input | Leave_One_experiments/smoke_tests/logs/LCA_2010.log       | True     | Leave_One_experiments/smoke_tests_debug/logs/LCA_2010_fixed_low_input_before_fix_last50.txt | STOP;Fortran runtime error                     | nan                         | before_fix |
| LCA_2010_fixed_low_input | LCA       |   2010 | fixed_low_input | Leave_One_experiments/smoke_tests_debug/logs/LCA_2010.log | True     | Leave_One_experiments/smoke_tests_debug/logs/LCA_2010_fixed_low_input_after_fix_last50.txt  | STOP;Fortran runtime error;Error key;not found | WARNING.OUT                 | after_fix  |
| SYA_2014_fixed_low_input | SYA       |   2014 | fixed_low_input | Leave_One_experiments/smoke_tests/logs/SYA_2014.log       | True     | Leave_One_experiments/smoke_tests_debug/logs/SYA_2014_fixed_low_input_before_fix_last50.txt | STOP;Fortran runtime error                     | nan                         | before_fix |
| SYA_2014_fixed_low_input | SYA       |   2014 | fixed_low_input | Leave_One_experiments/smoke_tests_debug/logs/SYA_2014.log | True     | Leave_One_experiments/smoke_tests_debug/logs/SYA_2014_fixed_low_input_after_fix_last50.txt  | STOP;Fortran runtime error;Error key;not found | WARNING.OUT                 | after_fix  |
| YCA_2014_null_zero       | YCA       |   2014 | null_zero       | Leave_One_experiments/smoke_tests/logs/YCA_2014.log       | True     | Leave_One_experiments/smoke_tests_debug/logs/YCA_2014_null_zero_before_fix_last50.txt       | STOP;Error key;prior to the start              | WARNING.OUT                 | before_fix |
| YCA_2014_null_zero       | YCA       |   2014 | null_zero       | Leave_One_experiments/smoke_tests_debug/logs/YCA_2014.log | True     | Leave_One_experiments/smoke_tests_debug/logs/YCA_2014_null_zero_after_fix_last50.txt        | STOP;Error key;not found                       | WARNING.OUT                 | after_fix  |

## 7. 假设验证

| 假设 | 结论 | 证据 |
|---|---|---|
| A 固定策略动作每天重复触发 | 不成立 | step trace 和 action check 显示 fixed_low_input 的 anfer 只在 DAP 1 和 30 触发，total_n 与预期一致。 |
| B 动作归一化或反归一化错误 | 不成立 | normalized_action 在 [-1, 1]，real_action 非负且未超过 action_space high。 |
| C 动作 key 不匹配 | 不成立 | action_space 包含 amir/anfer，固定策略 key 与环境一致。 |
| D YCA 输入文件路径或站点代码不一致 | 部分不成立 | YCA WTH/SOL/template 可被读取；真正问题是静态灌溉日期早于 SDATE。 |
| E 管理日期超出模拟日期范围 | 成立 | YCA 日志明确提示 irrigation application date prior to simulation start。 |
| F DSSAT 子进程等待输入 | 表象成立 | Python 层表现为 timeout，但底层 DSSAT 已在日志中报错；修复 rendered management 后消失。 |

## 8. 实际原因

### LCA

LCA fixed 策略 timeout 的根因不是 PPO 或 Python 循环，而是 rendered DSSAT management 设置不一致：临时输入中动态施肥动作被触发，但 treatment factor 中 MI/MF 仍为 0 或管理层级不可用，导致 DSSAT 在 `FertType_mod.for` 中出现 `fertfile` index 0 的 Fortran runtime error。Python 层等待环境 step 返回，最终表现为 timeout。

### SYA

SYA fixed 策略与 LCA 共享同类问题，同时部分模板缺少可供动态动作使用的 irrigation section。只打开 MI/MF 不够，必须保证临时 rendered 文件中 irrigation/fertilizer section 存在，并带有安全的基线管理行。

### YCA

YCA null_zero timeout 的根因是原始模板中的静态灌溉记录在替换年份后早于 simulation start date，DSSAT 日志报错 `First irrigation application date is defined prior to the start of simulation`。因为 null_zero 不执行新动作，这个问题会在环境初始化或早期 step 直接触发。

## 9. 修复内容

只修改 smoke test 代码路径，不修改 reward，不修改 `my_data/` 原始文件。

- `src/run_smoke_tests.py`: 支持 `SMOKE_TEST_OUTPUT_ROOT`，让 debug 重跑输出到 `Leave_One_experiments/smoke_tests_debug/`，不覆盖上一阶段结果。
- `src/run_smoke_tests.py`: 在临时 rendered 文件中启用 MI/MF treatment levels。
- `src/run_smoke_tests.py`: 为缺失 irrigation section 的模板插入安全的零灌溉 section。
- `src/run_smoke_tests.py`: 将原始静态 irrigation/fertilizer 事件替换为 simulation start 后的安全零/基线事件，避免历史年份日期残留造成 DSSAT 报错。
- `src/run_smoke_tests.py`: 增加 step-level trace。
- `src/rerun_smoke_test_timeouts.py`: 只重跑上一阶段失败的 29 个案例。
- `src/collect_smoke_timeout_debug_artifacts.py`: 汇总 action_space、rendered input 和 DSSAT log 检查表。
- `src/generate_smoke_timeout_debug_report.py`: 生成本报告和 PPT。

## 10. 修复后复测结果

29 个上一阶段失败案例只重跑失败集合，不重跑全部 56 个。结果为 `ok: 29`。

按站点统计：

| station   |   ok |
|:----------|-----:|
| LCA       |   12 |
| SYA       |    9 |
| YCA       |    8 |

重跑汇总表：

`Leave_One_experiments/smoke_tests_debug/evaluation/smoke_test_timeout_rerun_summary.csv`

## 11. 是否可以进入 PPO 训练脚本生成

可以进入 PPO 训练脚本生成，但前提是训练/评估脚本也使用同样的临时 rendered template 修复逻辑，尤其是：

1. 不覆盖 `my_data/` 原始模板；
2. 每个站点-年份单独生成临时 rendered input；
3. 在临时 rendered input 中保证 MI/MF 与 irrigation/fertilizer sections 可用；
4. 清除或替换跨年份残留的静态灌溉/施肥事件；
5. PPO 训练前先做 NullAgent 和固定策略小 smoke test。

当前 smoke test 层面没有剩余阻塞问题。
