# 044_05 DQN 坏例子图：导师汇报补充材料

## 目的

为 `044_03` 的 DQN 汇总结论补充更具体的例子图，方便向导师解释：

- “普通 DQN 策略比较固定”具体是什么意思；
- “严格 MaskableDQN 训练越久越退化”具体体现在哪里；
- “Demo-DQN / DQfD 仍然偏 no-op”具体证据是什么；
- “teacher 非零动作没有被学成高 Q 值”如何从 Q 排序诊断看出来。

## 输入

只读取已有结果：

- `040_01_sya_lowIC_free_timing_dqn`
- `040_02_sya_lowIC_strict_maskable_dqn`
- `044_00_sya_lowIC_binary_forecast_demo_dqn_smoke_smoke2k`
- `044_01_sya_lowIC_demo_dqn_q_ranking_audit`
- `044_02_sya_lowIC_demo_only_pretrain_dqn_audit`

不训练，不重新运行 DSSAT。

## 输出

输出目录：

```text
benchmark_results/044_05_dqn_bad_case_examples_for_advisor/
```

输出内容：

- 普通 DQN 最佳 checkpoint 的逐年措施与产量坏例子图；
- 严格 MaskableDQN checkpoint 退化坏例子图；
- Demo-DQN / DQfD no-op 坏例子图；
- Q 排序坏例子图；
- 中文记录 `docs/044_05_dqn_bad_case_examples_for_advisor_record.md`。

## 判读边界

这些图用于展示“当前 DQN 结果为什么不能作为主线成功结果”，不能扩展成“DQN 理论上不可能成功”。

