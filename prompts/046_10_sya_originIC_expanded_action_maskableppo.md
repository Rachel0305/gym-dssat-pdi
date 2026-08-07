# 046_10 SYA originIC 扩展动作空间 MaskablePPO

## 研究问题

在 `046_02` 已完成的 originIC、原始 observation、无天气预报、无 observation normalization 基准上，PPO 的表现是否主要受二值剂量档位限制？

本轮只改变动作空间：

```text
灌溉：[0, 15, 30, 45] mm
施氮：[0, 40, 80, 120] kg/ha
组合动作：16
```

## 冻结项

- 输入：`originIC`，即 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013`。
- observation：沿用 `046_02` 的原始 25 维状态；不归一化、不附加天气或未来天气。
- 训练/验证年份：SYA 2005–2013 训练，2014–2023 验证。
- 算法、reward、MaskablePPO 安全层、最小操作间隔、DAP90 后禁氮、季节上限和后期灌溉 reserve mask：均沿用 `042_15 → 046_02`。
- `recorded_farmer_template` 冻结；不调整其日程或数值。

## 执行顺序

1. 先运行 dry-run，确认输入路径、年份划分、16 个组合动作及冻结项。
2. 再运行 2K smoke；输出写入独立 `*_smoke2k` 目录。
3. smoke 的硬性证据包括：所有已执行动作都落在指定档位、正动作能与季节累计量变化闭合、最终 2K checkpoint 的验证轨迹不全部为 DAP1，且至少出现两个非零动作组合并至少使用一个新增档位（I15/I30/N40/N120）。
4. 只有 smoke 通过且得到明确人工确认后，才使用 `--formal` 启动正式 100K。该入口会重新读取 smoke 的结果与 manifest；只要 smoke 缺失、动作契约不一致或 gate 未通过，就拒绝训练。

## 解释边界

- 2K smoke 只验证链路、动作可达性与早期塌缩风险，不能用于性能优劣结论或 checkpoint 选择。
- 100K 若启动，必须与 `046_02` 同表比较，不得同时改变 reward、观测、天气信息或安全约束。
