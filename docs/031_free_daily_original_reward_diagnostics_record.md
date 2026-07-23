# 031 自由日尺度原始 reward 诊断记录

更新时间：2026-07-20

## 为什么补这份 docs 记录

031 系列原始输出保存在 `benchmark_results/`，但为了后续汇报和复查，实验记录需要在 `docs/` 下有一份容易定位的总览。以后保持这个规则：

- `benchmark_results/`：保存原始 CSV、JSON、模型、日值表、图和任务内记录。
- `docs/`：保存可读的实验总记录、方法边界、结论口径和下一步建议。

## 当前问题

导师要求放开 expert DAP 窗口，让 RL 在日尺度上自由决定是否灌溉/施肥。031 系列因此测试：

- 不使用 expert 固定决策窗口；
- 不保留 7 天最小操作间隔；
- 只保留季节上限、日上限、DAP90 后禁氮；
- reward 改回原始形式：

```text
reward_t = delta_GRNWT_t - 1.0 * irrigation_t - 5.0 * nitrogen_t
```

无 terminal bonus，无 `/1000` 缩放，无 TOPWT 项。

## 031_01：PPO 512步 smoke

位置：

- prompt: `prompts/031_01_free_daily_original_reward_smoke.md`
- record: `benchmark_results/031_01_free_daily_original_reward_smoke/031_01_free_daily_original_reward_smoke_record.md`
- daily: `benchmark_results/031_01_free_daily_original_reward_smoke/daily_outputs/SYA/2014_ppo_daily.csv`

结果：

| 算法 | 年份 | 步数 | 产量 | 灌溉 | 施氮 | 首次灌溉 | 首次施氮 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO | SY2014 | 512 | 10385.68 | 160 | 250 | DAP1 | DAP1 |

解释边界：

512 步不到 4 个完整 season，不能说 PPO 已经“学会”了早期打满预算，只能说流程跑通，短 smoke 下出现 I160/N250 触顶。

## 031_02：DQN 512步 smoke

位置：

- prompt: `prompts/031_02_free_daily_original_reward_dqn_smoke.md`
- record: `benchmark_results/031_02_free_daily_original_reward_dqn_smoke/031_02_free_daily_original_reward_dqn_smoke_record.md`
- daily: `benchmark_results/031_02_free_daily_original_reward_dqn_smoke/daily_outputs/SYA/2014_eval_dqn_daily.csv`
- comparison: `benchmark_results/031_02_free_daily_original_reward_dqn_smoke/evaluation/031_02_ppo_vs_dqn_smoke_comparison.csv`

DQN 必须离散动作，因此使用 3×3 网格：

- irrigation `{0,20,40}` mm/day
- nitrogen `{0,40,80}` kg/ha/day

结果：

| 算法 | 年份 | 步数 | 产量 | 灌溉 | 施氮 | 首次灌溉 | 首次施氮 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN | SY2014 | 512 | 10367.32 | 160 | 250 | DAP10 | DAP8 |

解释边界：

这不是完全纯粹的“只换算法”，因为 PPO 是连续动作，DQN 是离散动作。512步结果只能说明 DQN 路径能跑通，并同样触顶，不能说明 DQN 稳定优于或劣于 PPO。

## 031_03：无训练随机/no-op对照

位置：

- prompt: `prompts/031_03_free_daily_original_reward_random_baseline.md`
- record: `benchmark_results/031_03_free_daily_original_reward_random_baseline/031_03_free_daily_original_reward_random_baseline_record.md`
- comparison: `benchmark_results/031_03_free_daily_original_reward_random_baseline/evaluation/031_03_ppo_dqn_random_noop_comparison.csv`

重要修正：

连续动作中，归一化动作 `0` 不是物理 no-op，而是动作空间中点。真正 no-op 应使用 `action_space.low`。031_03 最终记录已按正确 no-op 重跑。

结果：

| 策略 | 是否训练 | 产量 | 灌溉 | 施氮 | 首次灌溉 | 首次施氮 |
|---|---:|---:|---:|---:|---:|---:|
| continuous no-op | 否 | 2729.50 | 0 | 0 | — | — |
| DQN no-op | 否 | 2729.50 | 0 | 0 | — | — |
| continuous random | 否 | 10365.56 | 160 | 250 | DAP1 | DAP1 |
| DQN random | 否 | 10316.41 | 160 | 250 | DAP1 | DAP1 |

关键结论：

随机策略不训练也能触顶 I160/N250。因此，031_01/031_02 的触顶不能解释为“模型已经学出了打满上限的稳定策略”。更准确的说法是：自由日尺度 + 原始 reward 设置本身容易让任何愿意行动的策略触顶。

## 031_04：PPO/DQN 5000步训练

位置：

- prompt: `prompts/031_04_free_daily_original_reward_5k_train.md`
- record: `benchmark_results/031_04_free_daily_original_reward_5k_train/031_04_free_daily_original_reward_5k_train_record.md`
- comparison: `benchmark_results/031_04_free_daily_original_reward_5k_train/031_04_ppo_dqn_5k_vs_baselines.csv`
- PPO daily: `benchmark_results/031_04_free_daily_original_reward_5k_train/ppo/daily_outputs/SYA/2014_ppo_daily.csv`
- DQN daily: `benchmark_results/031_04_free_daily_original_reward_5k_train/dqn/daily_outputs/SYA/2014_eval_dqn_daily.csv`

结果：

| 策略 | 步数 | 产量 | 灌溉 | 施氮 | 首次灌溉 | 首次施氮 | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| PPO | 5000 | 10387.86 | 160 | 250 | DAP1 | DAP1 | 与 random continuous 几乎同型，仍然早期触顶 |
| DQN | 5000 | 6926.16 | 160 | 250 | DAP102 | DAP83 | 学到很晚操作，产量明显下降 |

031_04 相对 031_03 的重点：

- PPO 5k 与 random continuous 很接近：产量差约 22.3 kg/ha，收益差约 22.3，均小于 0.3%。
- DQN 5k 比 DQN random 明显更差：产量从约 10316 降到 6926，首次灌溉推迟到 DAP102，首次施氮推迟到 DAP83。

关键结论：

单纯把训练步数从 512 增加到 5000，没有让自由日尺度原始 reward 学出合理时序。

## 对 Claude 最新意见的采纳口径

Claude 的核心提醒是对的：

1. 不能把 512步或 5000步触顶直接说成“模型学会了打满上限”。
2. PPO 5k 和 random continuous 几乎一样，说明当前 reward 对“聪明时机”和“随机触顶”的区分力很弱。
3. DQN 5k 出现负向优化，不能只看终点指标，应记录为 DQN 在当前自由日尺度原始 reward 下的训练稳定性风险。
4. 不宜现在直接扩展到全站点全年份。

## 当前最稳妥结论

在 SY2014 seed0 上：

> 完全自由日尺度 + 原始 reward 能运行，但目前没有证据表明 PPO 或 DQN 在该设置下学到了比随机触顶更合理的管理时序。PPO 5k 仍近似随机触顶；DQN 5k 则学到晚施水肥的低产策略。

## 下一步建议

不要继续盲目加训练步数。下一步更适合做以下二选一：

1. 如果坚持自由日尺度：
   - 重新预注册 reward/约束设计；
   - 明确加入能区分“合理时机”和“随机触顶”的信号；
   - 先在一个时机敏感年份做单站点多 seed 验证。

2. 如果当前目标是给导师汇报已有可用结果：
   - 将 031 系列定位为“自由时序原始 reward 的诊断性失败/边界测试”；
   - 不把它混入此前阶段型 MaskablePPO 的正面结果。

## 031_05：SY2014 固定时机规则敏感性检查

位置：

- prompt: `prompts/031_05_sy2014_free_timing_rule_sensitivity.md`
- docs record: `docs/031_05_sy2014_free_timing_rule_sensitivity_record.md`
- result folder: `benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/`
- summary: `benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/evaluation/031_05_rule_timing_sensitivity_summary.csv`
- figure: `benchmark_results/031_05_sy2014_free_timing_rule_sensitivity/figures_031_05_rule_timing_summary.png`

目的：

在设计新的自由时序 RL reward 前，先确认 SY2014 在同一 I160/N250 上限下是否真的对操作时机敏感。

结果：

| rule | 产量 | 灌溉 | 施氮 | 简化收益 | 首次灌溉 | 首次施氮 |
|---|---:|---:|---:|---:|---:|---:|
| early_dump | 10091.84 | 160 | 250 | 8681.84 | DAP1 | DAP1 |
| uniform_spread | 10908.60 | 160 | 250 | 9498.60 | DAP1 | DAP1 |
| expert_window_budget | 10908.60 | 160 | 250 | 9498.60 | DAP1 | DAP1 |
| stress_triggered | 9761.71 | 120 | 160 | 8841.71 | DAP1 | DAP41 |
| delayed_late | 6926.16 | 160 | 250 | 5516.16 | DAP102 | DAP83 |

关键结论：

- 固定规则之间产量范围为 3982.44 kg/ha，说明 SY2014 明显存在时机敏感性。
- `uniform_spread/expert_window_budget` 明显优于 `early_dump` 和 `delayed_late`。
- 因此，自由时序 RL 值得做；031_01--031_04 的问题不是“时机本身不重要”，而是当前原始 reward 没有把时机差异有效传给学习算法。

新的下一步口径：

> 不应直接扩大全站点训练，也不应继续单纯加步数。更合理的下一步是重新预注册自由时序 reward/约束，让奖励明确区分“合理分散时机”和“随机/早期/晚期触顶”。

## 文件管理规则

从 031_04 之后，所有新实验都应同时保存：

- `prompts/<task_id>_*.md`
- `benchmark_results/<task_id>*/...`
- `docs/<task_id或系列名>_*.md`

避免只把记录放在 `benchmark_results/` 里导致后续不好找。
