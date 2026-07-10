# 018_02 官方农技推广 expert baseline 小试记录（clean）

## 目的

导师提出：“专家策略”应参考全国农技推广中心推文/ PDF 中的水肥一体化技术方案。此前项目中使用的 “expert / recorded” 更准确地说是“农民记录/实测管理实践”，不应继续直接等同于官方专家策略。

本轮只新增一个官方推广基线情景，不修改 DQN 奖励函数，不重新训练，不改原始输入文件。这样可以单独判断：如果把官方推广方案作为新 expert baseline，现有 DQN 结果的解释会怎么变化。

## 数据来源与单位换算

参考文件：

- `references/【农技推广】玉米大豆水肥一体化单产提升技术方案.pdf`

采用分区：

- HLA2010：东北及长城沿线春玉米区，对应 PDF 表 1。
- YC2014：华北黄淮和汾渭平原夏玉米区，对应 PDF 表 3。

单位换算：

- 氮肥：`kg/亩 × 15 = kg/ha`
- 灌水：`方/亩 × 1.5 = mm`

本轮采用各阶段推荐范围的中值，并用固定 DAP 映射到 DSSAT/gym-DSSAT 管理事件。没有使用实测生育期，也没有使用 DSSAT 事后模拟生育期，以避免引入“事后信息”。

## 本轮原则

- 不修改奖励函数。
- 不重新训练 DQN。
- 不覆盖已有结果。
- 只新增 `extension_expert_fixed_dap` 情景。
- HLA2010 使用 PDF 表 1。
- YC2014 使用 PDF 表 3。
- 灌溉事件若超过单次 50 mm，则拆成相邻日小事件，避免超出当前动作/管理事件约束。

## 输出文件

- schedule：`DSSAT_auto_validation/extension_expert_baseline_018_02/extension_expert_schedule.csv`
- extension summary：`DSSAT_auto_validation/extension_expert_baseline_018_02/018_02_extension_expert_summary.csv`
- clean comparison：`DSSAT_auto_validation/extension_expert_baseline_018_02/018_02_clean_hla2010_yc2014_comparison_with_extension_expert.csv`
- 原始 combined：`DSSAT_auto_validation/extension_expert_baseline_018_02/018_02_extension_expert_combined_comparison.csv`
- figure：`DSSAT_auto_validation/extension_expert_baseline_018_02/figures/018_02_extension_expert_summary.png`

注意：原始 combined 表中曾混入 HLA seed0/seed1 重复行，并漏合并 YC 的部分四情景结果。因此汇报时优先使用 clean comparison 表。

## 干净版对照结果

| 站点 | 年份 | 情景 | 籽粒产量 kg/ha | 生物量 kg/ha | 灌溉 mm | 施氮 kg/ha | 最大水分胁迫 | 最大氮胁迫 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| HLA | 2010 | Null | 6956.00 | 19344.00 | 0.00 | 0.00 | 0.919 | 0.157 |
| HLA | 2010 | Recorded expert | 7679.00 | 20665.00 | 30.00 | 165.00 | 0.811 | 0.016 |
| HLA | 2010 | DSSAT auto | 7854.00 | 20874.00 | 190.40 | 0.00 | 0.416 | 0.035 |
| HLA | 2010 | DQN best checkpoint | 7853.67 | 20886.03 | 120.00 | 0.00 | 0.416 | 0.062 |
| HLA | 2010 | Official extension expert fixed DAP | 7854.00 | 20571.00 | 266.10 | 300.00 | 0.421 | 0.016 |
| YC | 2014 | Null | 7825.00 | 17996.00 | 0.00 | 0.00 | 0.922 | 0.381 |
| YC | 2014 | Recorded/farmer practice | 9418.00 | 20514.00 | 120.00 | 374.00 | 0.000 | 0.013 |
| YC | 2014 | DSSAT auto | 8713.00 | 18945.00 | 86.50 | 0.00 | 0.000 | 0.436 |
| YC | 2014 | DQN best checkpoint | 9418.00 | 20513.00 | 120.00 | 250.00 | 0.000 | 0.013 |
| YC | 2014 | Official extension expert fixed DAP | 9417.00 | 20481.00 | 228.80 | 247.00 | 0.000 | 0.013 |

## 初步结论

### HLA2010

官方推广 expert、DSSAT auto 和 DQN seed0 best checkpoint 的产量几乎相同，约 7854 kg/ha。

但资源投入差异很大：

- 官方推广 expert：约 266.1 mm 灌溉 + 300 kg/ha 氮。
- DSSAT auto：约 190.4 mm 灌溉 + 0 kg/ha 氮。
- DQN best checkpoint：约 120 mm 灌溉 + 0 kg/ha 氮。

因此，在 HLA2010 上，新增官方推广 expert 之后，当前 DQN 叙事反而更清楚：DQN 没有明显提高产量，但它用更少灌溉、更少施氮达到了接近官方推广/auto 的产量平台。

### YC2014

官方推广 expert、recorded/farmer practice 和 DQN best checkpoint 的产量都约为 9417–9418 kg/ha。

资源投入差异：

- 官方推广 expert：约 228.8 mm 灌溉 + 247 kg/ha 氮。
- recorded/farmer practice：约 120 mm 灌溉 + 374 kg/ha 氮。
- DQN best checkpoint：约 120 mm 灌溉 + 250 kg/ha 氮。

因此，在 YC2014 上，DQN 相比官方推广 expert 明显节水，但施氮量略高；相比 recorded/farmer practice，DQN 产量相当且显著节氮。

## 汇报时建议怎么说

1. 以前的 “expert” 应改名为 “recorded/farmer practice”，因为它来自实测记录/农民管理，不是官方推广方案。
2. 新增官方推广 expert baseline 后，HLA2010 和 YC2014 的 DQN 结果并没有被推翻。
3. HLA2010 更支持“节水节氮达到高产平台”的叙事。
4. YC2014 支持“节水、相对农民记录节氮”的叙事；但相对官方推广 expert，氮投入并没有明显优势。
5. 当前不建议同时修改奖励函数。应先把“新增官方推广基线”这个变量单独汇报清楚，再决定是否加入氮淋洗惩罚或调整水氮成本权重。

## 下一步建议

短期：

- 先把 HLA2010 / YC2014 的官方推广 expert baseline 作为小试结果汇报给导师。
- 请导师确认：官方推广方案是否要作为所有站点年份的正式 expert baseline。
- 如果确认，则扩展到五站点候选年份，并保留原 recorded/farmer practice 作为另一个现实参考情景。

中期：

- 如果导师更重视“氮肥效率”，再讨论是否修改奖励函数，例如提高氮成本或加入氮淋洗惩罚。
- 如果导师更重视“达到高产且节水”，当前 HLA2010/YC2014 的结果已经可以支持继续扩展。
