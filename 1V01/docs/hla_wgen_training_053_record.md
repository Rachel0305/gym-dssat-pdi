# 053 HLA WGEN 天气池与八种子训练记录

日期：2026-10-09。当前状态：**八种子 100K 训练、归档审计、十年历史天气验证与图表汇总均完成**。下文保留训练启动时的过程记录；最终结果见末节。

## 输入选择

用户选择先比较五种 111 天降水补缺方案，再以一套输入训练八个 PPO seed。024 的 330 个 D222 明示日配对显示，两种乘比值校正均使 MAE/RMSE 上升；GPCC_RAW 的 RMSE 9.846 mm 低于 CPC_RAW 的 10.791 mm，虽然 CPC_RAW 的 MAE 略低。053 五种单季 runtime 候选均通过基础完整性与物理筛查。按预先记录的 RMSE 优先与无需强度校正原则，本轮选 **GPCC_RAW**；机器决策见 `results/hla_wgen_8seed_053/scenario_selection.json`。此为可复现的实验输入选择，不宣称 111 天恢复了本站观测。

## 已完成门禁

- 五种 CLI 各生成 2007 年 seed101 限量候选，逐日天气与哈希、运行时 seed、CLI 一致；见 `limited_candidate_audit.json`。
- GPCC_RAW 的 80 个训练 seed（1001–1080，2004–2013 均衡年份调度）和 20 个留出 seed（1081–1100，2007 年）均已逐日归档。训练池 11,207 天、留出池 2,853 天，100 份天气哈希互不相同；进程树峰值 RSS 分别约 529 和 425 MiB。
- FQ 046 的气候筛查规则移植到 HL。HL 的共同作物季日窗按全部天气实际覆盖设为 5 月 5 日至 8 月 31 日，共 119 天；月雨量画像对应 5–8 月，其余阈值保持一致。`weather_pool/final_gate.json` 为 `PASS_INPUT_QC_ONLY`，所有结构与气候筛查均通过。这是启用 PPO 的输入筛查，不证明天气在所有指标上真实或无偏。
- PPO 第一次 432 步 smoke 在最后一日未完成 episode 因“单日零雨”被误当整季零雨而失败；完整失败目录保留。第二次仅对部分 episode 放宽这一项整季规则，432 步与 432 天归档闭合并可加载模型。独立的多年份训练 runner 432 步 smoke 也通过，峰值 RSS 约 432 MiB。

## 训练执行

每个 seed 从头训练 100K 环境步，保存 25K、50K、75K、100K checkpoint 和最终模型；逐 episode 保存真实天气 CSV、运行时 seed/CLI 证据、资源日志和失败现场。`run_seed_100k.py` 以单进程运行。seed0 已启动；`run_remaining_seeds.ps1` 等待 seed0 结束并审计，通过后依次运行 seed1–7，任何一步失败立即停止。状态见 `orchestrator_events.jsonl`；每 seed 结果见 `seed_XX/attempt_01/run_result.json` 与 `audit_gate.json`。

## 已知限制

032–037 的 native FIELD 坐标占位/转写问题未得到源码级修复，现有 runtime 在此限制下生成天气。此次训练结果须标注该限制，并与旧 HL 基线核对输入和 runtime 合同后才解释政策差异。2013 大雨沿用户决定保留 D222 原值；此前“量级未独立裁决”审计结论仍保留。当前尚无 2014–2023 验证、图表或八 seed 管理成效结论。

## 八种子完成与验证结果（2026-10-09 补记）

上述“seed0 已启动”和“尚无验证”是训练启动时的历史状态。seed0–7 均达到实际 100,080 步，精确 100,000 步 checkpoint、最终模型、逐 episode 随机天气和四档 checkpoint 均保存，八份 `audit_gate.json` 均为 `PASS_100K_ARCHIVE_ONLY`。原编排进程在 seed2 训练过程中退出，没有留下退出原因；seed2 训练结果完整并独立通过补审计，然后由 `resume_seed3_to7.ps1` 顺序完成 seed3–7，续跑记录见 `resume_events.jsonl`。未重训已完成的 seed。

验证使用每个 seed 的精确 100K checkpoint，在 2014–2023 固定历史天气上确定性回放；四个对照来自冻结的 054_03 HL 五情景结果。八个 seed 的逐年天气均按日期与冻结 Null 基线核对通过。保存 80 个逐季 DSSAT 快照及 `Summary.OUT`、每 seed 23 张图和六张 CSV 表。`WP_ET` 来自对应 `Summary.OUT` 的 ETCP；缺少作物氮吸收量，不报告 NUE。图表脚本与 FQ 052 同口径，HL 独立脚本为 `evaluate_100k.py` 和 `summarize_cohort.py`。

| 指标 | PPO 八 seed 十年均值的均值 ± SD | Null | Recorded | Auto + N | Expert |
|---|---:|---:|---:|---:|---:|
| 产量 (kg/ha) | 6308.42 ± 579.49 | 4653.40 | 5482.40 | 6060.70 | 6797.00 |
| WP_ET (kg/m³) | 1.40 ± 0.05 | 1.17 | 1.34 | 1.35 | 1.48 |
| PFP_N (kg grain/kg N) | 45.49 ± 39.63 | 未定义 | 33.22 | 123.72 | 22.89 |
| 灌溉 (mm) | 154.69 ± 91.93 | 0 | 30.00 | 164.30 | 266.00 |
| 施氮 (kg/ha) | 191.50 ± 73.65 | 0 | 165.00 | 45.00 | 297.00 |

PPO 产量的八 seed 均值高于 Null、Recorded 和 Auto + N，低于 Expert；仅 4/8 个 seed 的十年平均产量超过 Expert。WP_ET 八 seed 均低于 Expert。策略分化明显：seed0/1/2/4 平均灌溉 240 mm、十年均产量约 6826–6834 kg/ha；seed3/5/6/7 灌溉 45–87 mm，但均产量约 5584–6067 kg/ha。不能只凭整组均值宣称稳定的水氮双重优势。这些比较是描述性的，不作显著性推断。训练使用相同的 WGEN episode 调度，因此八个 seed 主要衡量 PPO 初始化与优化随机性，并非八套独立天气样本。

逐 seed 图在 `results/hla_wgen_8seed_053/055_03_five_scenario/HL/best_seed_seedX/figures/`；八 seed 对比表和两张 cohort 图在 `results/hla_wgen_8seed_053/055_03_five_scenario/HL/cohort_8seed_summary/`。原训练天气采用 GPCC_RAW 补缺情景；2014–2023 验证使用冻结历史天气。native FIELD 坐标限制仍须随结果说明，当前比较不消除这一限制。
