# HL/YC checkpoint rescue 审计（只读）

日期：2026-08-10。目的：判断 HLA（HL）与 YCA（YC）当前 100K PPO 的弱表现，是否首先应归因于 checkpoint 后期策略坍缩，而非立即启动大规模站点调参。本审计未训练、未回放、未改动既有脚本、奖励或 wrapper。

## 来源与口径

训练/检查点评估来源：

- HL：`benchmark_results/054_00_hla_lowIC_expanded_action_maskableppo/evaluation/054_00_checkpoint_validation_summary.csv`、`evaluation/054_00_validation_summary_by_station_checkpoint.csv`、`audits/054_00_formal_action_audit.csv`。
- YC：`benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/evaluation/055_00_checkpoint_validation_summary.csv`、`evaluation/055_00_validation_summary_by_station_checkpoint.csv`、`audits/055_00_formal_action_audit.csv`。
- 冻结的 100K 五情景链路：HL `054_03_hla_lowIC_054_00_hla_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/054_03_hla_five_scenario_season_summary.csv`；YC `055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_season_summary.csv`。

检查点评估覆盖 2014--2023 共 10 年、seed0、25K/50K/75K/100K。`mean_gap_yield_vs_four_max` 和 `mean_gap_pfp_n_vs_four_max` 是每年相对当年四基线最优值的差再取十年均值；`any_metric_win_four_count` 是十年中至少一个已计算指标获胜的年数。PFP 为逐年 `yield/N` 后的均值，非“均值产量/均值施氮”。25K--75K 原始 checkpoint 记录没有可重算的季节 ET/`WP_ET`；因此本审计不把 100K 五情景的 `WP_ET` 推断给早期 checkpoint。

## 检查点轨迹

|站点|checkpoint|平均产量 kg/ha|平均灌溉 mm|平均施氮 kg/ha|平均 PFP-N kg/kg|产量差 vs 四基线最优 kg/ha|PFP-N 差 kg/kg|至少一指标胜场|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|HL|25K|6844.4|225.0|240.0|28.52|+351.4|-6.67|8/10|
|HL|50K|6603.8|130.5|240.0|27.52|+110.9|-7.67|6/10|
|HL|75K|6331.9|99.0|240.0|26.38|-161.0|-8.81|6/10|
|HL|100K|6244.5|78.0|124.0|50.71|-248.5|+15.52|9/10|
|YC|25K|8201.6|228.0|240.0|34.17|-1.93|+1.08|10/10|
|YC|50K|6825.9|75.0|240.0|28.44|-1377.6|-4.65|5/10|
|YC|75K|6140.8|45.0|152.0|40.74|-2062.7|+7.64|7/10|
|YC|100K|6072.5|45.0|160.0|37.95|-2131.0|+4.86|7/10|

HL 从 25K 到 100K 平均产量减少 599.9 kg/ha（灌溉 -147 mm、施氮 -116 kg/ha）。这不是单纯“节水而不减产”：100K 的产量相对四基线最优已由 +351.4 转为 -248.5 kg/ha。其 PFP-N 增长来自施氮骤降，不能抵消产量劣势。

YC 的轨迹更强：25K 与四基线最优几乎持平（-1.9 kg/ha），但 100K 变为 -2131.0 kg/ha；平均产量减少 2129.1 kg/ha、灌溉减少 183 mm、施氮减少 80 kg/ha。故当前证据支持“后期策略坍缩是 YC 100K 弱表现的主要近因”。

## 动作传输与多样性

所有审计行均为 0 个 transmission mismatch，故不能把性能衰退归因于“动作未传入 DSSAT”。

|站点|checkpoint|平均正动作行/年|DAP1 后平均正动作行/年|年内最大非零动作对数|跨年动作签名数|新水平动作事件总数|
|---|---:|---:|---:|---:|---:|---:|
|HL|25K|16.4|15.4|4|2|154|
|HL|50K|10.7|9.7|3|1|97|
|HL|75K|8.0|7.0|4|2|70|
|HL|100K|2.1|1.1|2|1|11|
|YC|25K|10.2|9.2|5|3|94|
|YC|50K|5.0|4.0|3|1|10|
|YC|75K|3.8|2.8|3|1|18|
|YC|100K|4.0|3.0|3|1|20|

HL 100K 已退化为近乎 DAP1 后仅 1.1 次正动作/年；YC 100K 虽保留约 3 次 DAP1 后正动作，但跨十年只剩一个动作签名，且灌溉固定 45 mm。两站 25K 均保留多个非零动作对、多个跨年签名，且动作确实传输到环境。

## 极端年份检查

HL 25K 到 100K 的最大下降发生在 2016（-1812.6 kg/ha）、2017（-1507.3）和 2022（-1373.1）；2015 也下降 -1274.8。100K 仅在 2014、2018--2021 小幅不低于 25K，说明并非一个离群年份驱动结论。

YC 的明显坍缩年份是 2014（-7494.4 kg/ha）、2019（-4900.0）、2018（-3051.0）、2015（-2653.6）和 2023（-1661.6）。YC 2018/2019 的 25K 产量分别为 8324.8/7606.6 kg/ha，而 100K 为 5273.8/2706.6 kg/ha；因此这里不是“五情景共同 0 产量”的站点年份异常，而是 100K PPO 相对于早期 checkpoint 的明显退化。

## 与冻结 100K 五情景的关系

冻结五情景的 100K PPO 均值为：HL 6244.5 kg/ha、`WP_ET=1.447`、`PFP_N=50.71`，在五情景中各指标年度胜场均为 1/10；YC 6072.6 kg/ha、`WP_ET=1.879`、`PFP_N=37.95`，分别为产量 3/10、`WP_ET` 0/10、PFP-N 7/10。HL/YC 冻结 100K 表与 checkpoint 汇总的 PPO 产量/PFP 数值一致（四舍五入差异除外）。

## 决策与下一阶段

结论：两站均不应先启动大范围超参数扫描。HL 证据支持“存在后期漂移，25K 值得正式复核”；YC 则有强证据表明 100K 后期坍缩，优先级应是 checkpoint 选择/早停稳定性，而不是立刻改奖励或动作空间。

建议的下一阶段（仅在用户批准后执行）：

1. 用当前冻结五情景的同一基线、输入根、年份和渲染配置，对 HL-25K、YC-25K 各做完整十年五情景回放；先用一站一年 smoke 验证 checkpoint、输入根与渲染 provenance，再运行其余年份，绝不新训 100K。
2. 在该回放前预注册选择规则：仅在独立的验证年份/年份折叠上选择 checkpoint；目标为最大化平均产量，同时要求相对四基线最优的 `WP_ET` 与 PFP-N 均不劣于预设容忍度，并要求跨年动作签名数和 DAP1 后动作数达到预设下限。之后在未参与选择的测试年份报告五情景指标。
3. 若早期 checkpoint 在独立测试折叠仍不满足三指标，则将其作为“跨站迁移的失败/受限结果”报告，再对单一机制做 2K smoke；不得根据已经看到的十年结果直接宣称 25K 是最终论文 checkpoint。

## 未解决风险

- 25K--75K 目前没有与 `054_03/055_03` 完全同口径的五情景季节表，因而没有可审计的早期 checkpoint `WP_ET`、五情景胜场和完整外部自动氮对照。
- 所有现有 checkpoint 都来自同一 seed0 与同一十年集合；不具备种子稳健性，也不能将 25K 的优势直接称作泛化结论。
- 本结论是对已观察 checkpoint 的 post-hoc 审计，不是预注册的模型选择；只有按上述独立选择/测试规则复核后，才可进入论文主结论。
