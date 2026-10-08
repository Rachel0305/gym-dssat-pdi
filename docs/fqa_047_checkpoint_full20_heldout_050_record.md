# 050 FQA 同一训练 5K/10K checkpoint：20 个留出天气配对评估

[执行 prompt](../prompt_02/050_fqa_047_checkpoint_full20_heldout_evaluation.md)对 047 同一次 PPO seed 0 训练保存的 5K 与 10K checkpoint，在 2007 年留出的 WGEN seed 1081–1100 上分别进行确定性评估。两 checkpoint 使用相同 seed 和逐日天气，彼此独立运行；没有再训练或改动 PPO 合同。

## 结果

| 指标 | 5K | 10K | 10K − 5K |
| --- | ---: | ---: | ---: |
| 平均产量（kg/ha） | 6411.5 | 6349.6 | −61.9 |
| 产量中位数（kg/ha） | 6346.6 | 6298.8 | −47.8 |
| 平均 wrapper 灌溉量（mm） | 112.50 | 110.25 | −2.25 |
| 平均 wrapper 施氮量（kg/ha） | 240.0 | 240.0 | 0 |
| 平均 PFP_N（wrapper N，kg/kg） | 26.714 | 26.457 | −0.258 |
| 平均 episode reward | 0.5844 | 0.6072 | +0.0229 |
| 平均正灌溉日数 | 5.50 | 3.35 | −2.15 |
| 平均正施氮日数 | 4.00 | 4.00 | 0 |
| 每个 episode 平均动作类别数 | 5.25 | 5.35 | +0.10 |

10K 在 20 个配对 seed 中有 4 个产量较高、16 个较低；平均产量比 5K 低约 1.0%。两个 checkpoint 的动作频数分布明显改变：在 2077 个决策日中，动作索引 5 从 40 次降为 0 次、索引 7 从 20 次降为 0 次，索引 2 和 8 分别从 0 次增至 40 次；动作索引 0 仍占绝大多数（1962 次与 1950 次）。10K 的正灌溉日减少，但累计灌溉仅平均减少 2.25 mm，累计施氮都为 240 kg/ha。episode reward 平均略升，但产量和 PFP_N 平均下降，因此不能单独用 reward 上升判定管理效果变好。

## 结论

5K 到 10K 之间策略确实发生变化，所以不能说 PPO 完全没有更新；但在这组 20-seed 留出评估中，变化没有转化为产量或 PFP_N 的总体改善。结果也不能证明 10K 一定太短、模型已经收敛，或继续到 100K 会变好。**本轮不建议仅凭“10K 可能太少”直接启动 100K。**如果继续投入训练，先设置一个较小的中间预算和同一组留出 seed 检查点门槛，再看是否值得扩大。

## 审计、资源与限制

- 最终门槛：`PASS_CHECKPOINT_FULL20_PAIRED_ARCHIVE_ONLY`。两个 checkpoint 各完成 20 个 episode、2077 个逐日轨迹行与 2077 个天气日；同 seed 天气 SHA-256 一致，seed 1081 和 1100 也与 048/049 已有天气归档一致。天气日期连续、物理筛查和动作/episode 闭合通过。
- 峰值进程树 RSS：5K 为 423.35 MB，10K 为 423.25 MB；运行时间分别为 20.40 秒和 20.68 秒。
- 审计第一次运行曾因检查器错误地要求固定 YAML bootstrap seed 与每个 episode 的运行时 `_rseed1` 相同而未通过。047–049 的既有约定是验证 YAML bootstrap 已配置，并单独验证运行时 `_rseed1` 等于该 episode seed。修正这一检查后通过；未重跑模拟，失败审计保存在 `audit_attempt01_failed_gate.json`。
- 这里的产量来自模拟终态，灌溉和氮用量来自 wrapper 累计，尚未与 `Summary.OUT` 实际施用量闭合。PFP_N 按 wrapper 累计 N 计算。没有 ETCP replay，因此不报告 WP_ET 或 NUE。
- 结论仅覆盖一个历史年（2007）和 20 个合成 WGEN 留出天气实现；没有配对传统管理对照，也不作跨年泛化、显著性或收敛声明。

完整逐 seed 配对差值见 [`paired_checkpoint_full20.csv`](../results/fqa_047_checkpoint_full20_heldout_050/paired_checkpoint_full20.csv)，归档门槛见 [`final_gate.json`](../results/fqa_047_checkpoint_full20_heldout_050/final_gate.json)。40 份逐日天气、运行时 seed 证据、动作轨迹和资源记录均在 `results/fqa_047_checkpoint_full20_heldout_050/`。SHA-256 清单为 [`fqa_wgen_050_file_manifest.csv`](fqa_wgen_050_file_manifest.csv)。
