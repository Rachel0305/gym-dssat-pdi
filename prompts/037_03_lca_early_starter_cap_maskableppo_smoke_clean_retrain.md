# 037_03：LCA early starter cap 自由时序 MaskablePPO clean retrain

## 背景

037_01 发现 036 主线 PPO 的主要措施问题不是“频繁小灌”，而是早期集中投入：

- 50 个验证年中 40 年存在 DAP1-10 early dump；
- 频繁/间隔问题只有 11 年；
- LCA 指标表现较好，但 10/10 年都属于 early dump，且平均灌溉/施氮事件数仅 1/1，说明最大事件数约束不是对症点。

文献中的 CropGym 也提醒：如果不限制动作空间，agent 可能在季节开始一次性施完氮肥，使 timing problem 退化为 amount-only problem。因此本任务测试一个最小 early starter cap，目标是保留自由时序，同时阻止季初集中用完资源。

037_02 曾运行一次，但第二次执行复用了已有 checkpoint，`run_status=ok_existing`，验证结果不能作为 cap 是否有效的科学结论。因此 037_03 使用新的输出目录进行 clean retrain。

## 唯一改动

相对 036 主线配置，只新增 DAP1-10 累计早期 starter 上限：

- DAP1-10 累计灌溉量 ≤ 30 mm；
- DAP1-10 累计施氮量 ≤ 40 kg/ha。

其余全部不变：

- 数据源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`；
- `IC=1`；
- `IRRIG=L, FERTI=L`；
- 自由时序 MaskablePPO；
- 灌溉档位 `{0,15,30,45}` mm；
- 施氮档位 `{0,40,80,120}` kg/ha；
- 灌溉/施氮最小间隔 7 天；
- 单季水氮上限；
- DAP90 后禁氮；
- 原 032_22 stress-aware reward；
- PPO 超参数不变。

## 站点与训练/验证

- 站点：LCA。
- 训练年份：沿用 036 half-split，LCA 2005-2013。
- 验证年份：沿用 036 half-split，LCA 2014-2023。
- seed：0。
- 训练步数：100K。
- checkpoint：25K/50K/75K/100K。

## 预期检查

本任务不直接证明最终成功，只回答：

1. early starter cap 是否明显减少 LCA 验证年的 DAP1-10 集中投入；
2. 是否没有把 LCA 原本较好的指标表现明显打崩；
3. 是否产生新的不合理行为，例如过晚施氮、频繁补偿性操作或产量明显下降。

## 停止线

- 只做 LCA 单站点 smoke，不扩展到五站点；
- 不修改 reward；
- 不调 PPO 超参数；
- 不根据中间 checkpoint 事后修改 early cap；
- 若结果明显劣化，记录为阴性，不现场换阈值。
