# 016_06 FQ 施氮响应弱原因诊断

固定灌溉为 `I60 critical`，扫描不同施氮总量。

| year | n_total | status | final_gwad | final_cwad | max_water_stress | max_nitrogen_stress | last_dap | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2008 | 0 | ok | 7890.000 | 13535.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N0 |
| 2008 | 25 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N25 |
| 2008 | 50 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N50 |
| 2008 | 75 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N75 |
| 2008 | 100 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N100 |
| 2008 | 150 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N150 |
| 2008 | 200 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N200 |
| 2008 | 250 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N250 |
| 2008 | 300 | ok | 7890.000 | 13536.000 | 0.000 | 0.012 | 104.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2008/I60_critical_N300 |
| 2016 | 0 | ok | 8012.000 | 14085.000 | 0.000 | 0.031 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N0 |
| 2016 | 25 | ok | 8012.000 | 14084.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N25 |
| 2016 | 50 | ok | 8012.000 | 14083.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N50 |
| 2016 | 75 | ok | 8012.000 | 14083.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N75 |
| 2016 | 100 | ok | 8012.000 | 14083.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N100 |
| 2016 | 150 | ok | 8012.000 | 14082.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N150 |
| 2016 | 200 | ok | 8012.000 | 14082.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N200 |
| 2016 | 250 | ok | 8012.000 | 14082.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N250 |
| 2016 | 300 | ok | 8012.000 | 14082.000 | 0.000 | 0.012 | 96.000 | DSSAT_auto_validation/fq_n_response_diagnostic_016_06/runs/2016/I60_critical_N300 |

## 初步判读

- FQ2008: 在固定 `I60 critical` 下，`N0=7890.0`, `N150=7890.0`, `N300=7890.0`, `best=7890.0`。
- FQ2016: 在固定 `I60 critical` 下，`N0=8012.0`, `N150=8012.0`, `N300=8012.0`, `best=8012.0`。

## 结论

1. `FQ2008` 的 `NSTRES` 从 `N0` 到 `N300` 基本恒定在 `0.012`，`GWAD` 也完全不变。这说明在当前输入和该灌溉方案下，系统几乎不缺氮，施氮不是可利用的增产杠杆。

2. `FQ2016` 在 `N0` 时 `NSTRES≈0.031`，加到 `N25` 后就降到 `0.012`，但 `GWAD` 仍然不变。这说明该年份存在一点轻微氮胁迫信号，但它不是主导最终产量的限制因子。

3. 因此，FQ 的“施氮响应弱”不是单纯训练问题，更像是站点-年份-当前管理窗口组合下的生理/环境事实：主要响应来自灌溉，氮追加没有转化成可见产量收益。

4. 这也意味着后续如果继续做 FQ，不应直接堆 RL 训练步数来逼出施氮动作；更合理的方向是先检查：
   - 是否需要换更缺氮的年份；
   - 是否需要改变施氮时机窗口；
   - 是否需要重新核对 FQ 的氮损失/土壤供氮设定。
