# 044_03 DQN 结果整理：导师汇报用数据和图表

## 目的

整理当前 SYA lowIC 自由时序框架下已经完成的 DQN 相关结果，形成导师汇报可直接引用的数据表、图表和中文说明。

本任务只读取已有结果，不重新训练，不重新运行 DSSAT。

## 纳入范围

纳入三类 DQN 结果：

1. `040_01_sya_lowIC_free_timing_dqn`
   - 普通 SB3 DQN + 外部动作安全修正。
   - 不是严格意义上的 MaskableDQN。
2. `040_02_sya_lowIC_strict_maskable_dqn`
   - 自定义严格 MaskableDQN。
   - 训练和评估时对非法动作做 Q 值屏蔽。
3. `044_00–044_02`
   - Demo-DQN / DQfD 风格尝试。
   - 用 lowIC teacher trajectory 作为示范经验，检查能否把非零 teacher 动作学成高 Q 值。

## 输出

输出目录：

```text
benchmark_results/044_03_dqn_results_for_advisor_summary/
```

需要生成：

- DQN checkpoint 均值指标汇总表；
- 各方法最佳 checkpoint 的逐年验证表；
- Demo-DQN / DQfD 的 Q 排序诊断表；
- checkpoint 指标对比图；
- 最佳 checkpoint 逐年产量图；
- Demo-DQN / DQfD Q 排序诊断图；
- 中文实验记录 `docs/044_03_dqn_results_for_advisor_summary_record.md`。

## 判读边界

本整理不能宣称“DQN 完全不可行”，只能宣称：

- 在当前 SYA lowIC 自由时序设置、当前动作空间、当前奖励和当前训练预算下，普通 DQN、严格 MaskableDQN、Demo-DQN/DQfD 都没有给出可作为主线正向结果的稳定策略；
- Demo-DQN/DQfD 的问题不是代码完全没接上，而是示范非零动作没有被学成 Q 值最高动作；
- 这批结果适合放在汇报中作为“为什么当前主线优先回到 PPO，而不是继续无限修 DQN”的证据。

