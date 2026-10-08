# 048 FQA WGEN 10K 留出天气两例诊断

[执行 prompt](../prompt_02/048_fqa_wgen_10k_heldout_pair_diagnostic.md)冻结 044 的 2K 接口 checkpoint 与 047 的多年份 10K checkpoint，分别在独立进程对 2007 年留出 WGEN seed 1081、1100 做确定性推断。未训练新模型。两组运行的 `ppo/reward/discrete_actions/action_safety` 合同与 047 相同；PPO seed 均为 0。来源及模型 SHA-256 见各自 `preflight.json`。

## 配对结果

逐日实际天气来自本次 DSSAT 状态捕获。两模型同一留出 seed 的天气哈希相同，1081 为 `BFA3F248...C8840BC2B2`，1100 为 `D142B32B...6CC96F1D4B`。完整哈希、逐日文件及 13 项审计见[配对表](../results/fqa_wgen_10k_heldout_diagnostic_048/paired_endpoints.csv)和[最终门槛](../results/fqa_wgen_10k_heldout_diagnostic_048/final_gate.json)。

| 2007 留出 seed | 模型 | 终态产量 kg/ha | wrapper 累计灌溉 mm | wrapper 累计施氮 kg/ha | PFP_N kg/kg |
| --- | --- | ---: | ---: | ---: | ---: |
| 1081 | 2K | 7034.04 | 225 | 240 | 29.31 |
| 1081 | 10K | 6861.59 | 135 | 240 | 28.59 |
| 1100 | 2K | 6253.34 | 210 | 240 | 26.06 |
| 1100 | 10K | 6124.56 | 135 | 240 | 25.52 |

10K 减 2K：seed 1081 产量 −172.45 kg/ha、灌溉 −90 mm、施氮 0、PFP_N −0.72；seed 1100 产量 −128.79 kg/ha、灌溉 −75 mm、施氮 0、PFP_N −0.54。这是两种单 seed checkpoint 在两个相同天气实现下的有限对照；2K checkpoint 原本只是 2007 年接口 smoke，10K 则来自九年训练，因此这些差异不能解释为纯训练步数效应，也不能推出 10K 策略总体优劣。

灌溉和施氮取安全 wrapper 的 episode 末累计执行量；本步未用 `Summary.OUT` 核对 DSSAT 全季实际施用闭合。这里的 PFP_N 只按终态产量除以该累计施氮计算。未做 ETCP 复跑，故没有 WP_ET 或 NUE。原生 FIELD 坐标 warning 的功能影响仍未确定。

## 运行与边界

2K 运行 204 天、峰值进程树 RSS 421.97 MB、6.13 秒；10K 运行 204 天、峰值 422.94 MB、5.39 秒。两边各归档两个 episode 的逐日天气和运行时 FileX/CLI/PDI seed 证据，逐份 SHA-256、天数、日期与气象物理检查通过。[最终门槛](../results/fqa_wgen_10k_heldout_diagnostic_048/final_gate.json)为 `PASS_PAIRED_DIAGNOSTIC_LIMITED`，即配对与归档成立，**不是**正式政策效果通过。未启动 100K 或其他 PPO seed。

这两个留出实现提示 10K checkpoint 在这两例中用水较少，但产量略低、PFP_N 也较低。后续若要判断是否值得正式训练，应先预先固定更充分的留出天气比较和资源/产量判据，并补足 DSSAT 实际施用及 ETCP 证据；不能只凭两例或训练 reward 放大实验。
