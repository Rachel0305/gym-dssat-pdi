# 049 FQA PPO 5K/10K checkpoint 学习信号 smoke

[执行 prompt](../prompt_02/049_fqa_047_checkpoint_learning_signal_smoke.md)比较 047 同一 PPO seed 0 训练过程中保存的 5K 与 10K checkpoint。两个模型都在 2007 年、留出 WGEN seed 1081 与 1100 上做确定性推断；每组运行使用独立进程。模型 SHA、相同 PPO/reward/action/safety 合同和环境来源见各自 `preflight.json`。

## 配对结果

| 留出天气 seed | checkpoint | 终态产量 kg/ha | wrapper 灌溉 mm | wrapper 施氮 kg/ha | PFP_N（wrapper N） | episode reward | 动作类别数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1081 | 5K | 7130.1 | 120 | 240 | 29.71 | 0.7504 | 5 |
| 1081 | 10K | 6926.1 | 120 | 240 | 28.86 | 0.7181 | 6 |
| 1100 | 5K | 6069.1 | 120 | 240 | 25.29 | 0.5827 | 5 |
| 1100 | 10K | 6137.3 | 120 | 240 | 25.57 | 0.5935 | 6 |

两组模型在每个天气 seed 上都复现了相同的逐日天气 SHA-256，且与 048 的对应天气文件相同。5K 到 10K 的确定性动作分布发生了变化：动作类别数从 5 增到 6，正灌溉日从 6 天变为 4 天，正施氮日仍是 4 天；累计灌溉与施氮总量不变。

## 结论与边界

可以排除“5K 到 10K 策略完全没有任何变化”：同轨迹 checkpoint 的动作分布变了，两个产量端点也一升一降。但这**不能证明发生了有用学习**，更不能证明 10K 已收敛或能泛化。两例的平均产量变化约为 −68 kg/ha，样本太少，且没有固定管理基线比较。047 完整训练 episode 的前 15 个平均 reward 为 0.5246，最后 15 个为 0.5127；两段年份和天气组成不同，所以这不是同条件学习曲线，也没有显示明确的整体上升证据。

因此 10K 仍属于短程预演：可能已有策略调整，但现有证据不足以判断它是否学到有价值的管理策略。当前不据此启动 100K。下一步可用全部 20 个留出 WGEN seed 对 5K/10K 做同条件比较，再决定是否增加训练预算。

水氮数值来自 safety wrapper 的 episode 累计，不是 `Summary.OUT` 核对后的 DSSAT 实际施用量；PFP_N 使用 wrapper 累计 N。没有 ETCP replay，因此不报告 WP_ET 或 NUE。两例天气均为 2007 年，不能代替跨年验证。逐日 action/state trace、天气归档、运行时 seed 证据和 15 项审计在 `results/fqa_047_checkpoint_learning_signal_049/`。
