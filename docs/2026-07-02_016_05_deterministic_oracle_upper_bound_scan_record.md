# 016_05 确定性水氮上界扫描记录

## 目的

不训练 DQN/PPO，仅用人工确定性水氮时序组合检查当前站点年份是否存在可达的高产/节水/节氮策略空间。

## 扫描组合

- 灌溉总量：[0.0, 60.0, 120.0]
- 施氮总量：[0.0, 150.0, 300.0]
- 时机模式：['early', 'critical']
- 单次灌溉上限 30 mm，单次施氮上限 100 kg/ha。

## 每个站点年份的最高产量组合

| site | year | scenario | status | message | final_gwad | final_cwad | irrigation_total | fertilizer_total | max_water_stress | max_nitrogen_stress | last_dap | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | 2007 | I0_N0_early | ok |  | 8109.00 | 14220.00 | 0.00 | 0.00 | 0.00 | 0.01 | 105.00 | DSSAT_auto_validation/deterministic_oracle_upper_bound_scan_016_05/runs/FQ/2007/I0_N0_early |
| FQ | 2008 | I60_N0_critical | ok |  | 8187.00 | 13802.00 | 60.00 | 0.00 | 0.00 | 0.01 | 104.00 | DSSAT_auto_validation/deterministic_oracle_upper_bound_scan_016_05/runs/FQ/2008/I60_N0_critical |
| FQ | 2016 | I60_N0_critical | ok |  | 8012.00 | 14058.00 | 60.00 | 0.00 | 0.00 | 0.04 | 96.00 | DSSAT_auto_validation/deterministic_oracle_upper_bound_scan_016_05/runs/FQ/2016/I60_N0_critical |
| YC | 2014 | I60_N150_early | ok |  | 9418.00 | 20514.00 | 60.00 | 150.00 | 0.00 | 0.01 | 104.00 | DSSAT_auto_validation/deterministic_oracle_upper_bound_scan_016_05/runs/YC/2014/I60_N150_early |

## 全部结果路径

- summary CSV: `DSSAT_auto_validation/deterministic_oracle_upper_bound_scan_016_05/016_05_deterministic_oracle_summary.csv`
- best CSV: `DSSAT_auto_validation/deterministic_oracle_upper_bound_scan_016_05/016_05_deterministic_oracle_best_by_year.csv`

## 关键判读

1. FQ2007/FQ2008 已经被抢救回可运行状态。旧问题不是年份不可用，而是 FQ 多年迁移时 treatment 2 的 `SDATE=yy153` 晚于 `ICDAT=yy152`，PDI/DSSAT 因此等待交互输入并 timeout。修正为 `SDATE=ICDAT=yy152` 后可完整运行。

2. FQ2007 在本次小扫描中最高产量就是 `I0/N0`，说明这个年份在当前输入条件下几乎没有可优化空间。它可以作为“无明显管理响应”的诊断年份，但不适合作为强化学习成功案例。

3. FQ2008 和 FQ2016 的最高产量均出现在 `I60/N0 critical` 或同等产量组合附近，说明主要响应来自灌溉，施氮在这组输入下没有带来明显产量收益。后续如果继续做 FQ，应优先确认为什么施氮响应弱，而不是直接堆训练步数。

4. YC2014 的最高产量在多种组合中达到同一平台，其中 `I60/N150` 已能达到最高产量。这说明 YC2014 存在“节水节氮仍达高产”的可达空间，也解释了为什么 DQN 在 YC 上更容易学出看起来合理的策略。

5. 016_05 是确定性上界/可达空间诊断，不是最终情景对比。它的作用是告诉我们哪些站点年份值得训练，哪些年份即使训练也可能没有明显收益。
