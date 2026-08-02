# 044_05 DQN 坏例子图：导师汇报补充材料

## 任务性质

本任务只读取已有 DQN / DQfD 结果，不重新训练，不重新运行 DSSAT。

## 具体坏例子

### 1. 普通 DQN 040_01：策略固定且平均产量偏低

- 最佳 checkpoint：75000。
- 验证年平均产量：7721.7 kg/ha。
- 平均总灌溉：150.0 mm；平均总施氮：240.0 kg/ha。
- 动作序列类型数：2；最常见序列覆盖 9 个验证年。

### 2. 严格 MaskableDQN 040_02：训练越久越退化

- 25K 平均产量：6427.4 kg/ha。
- 100K 平均产量：1527.3 kg/ha。
- 100K 平均施氮为 0.0 kg/ha，已明显偏向少操作/no-op。

### 3. Demo-DQN / DQfD 044_00：加入示范经验后仍 no-op

- 2K smoke 平均产量：1515.3 kg/ha。
- 2K smoke 平均灌溉：0.0 mm；平均施氮：0.0 kg/ha。
- Q 排序诊断显示，在线 Demo-DQN 的非零 teacher 动作 argmax 比例为 0；Demo-only 预训练后也降为 0。

## 图件

- `benchmark_results\044_05_dqn_bad_case_examples_for_advisor\figures\044_05_bad_case_ordinary_dqn_fixed_strategy.png`
- `benchmark_results\044_05_dqn_bad_case_examples_for_advisor\figures\044_05_bad_case_strict_maskable_dqn_training_collapse.png`
- `benchmark_results\044_05_dqn_bad_case_examples_for_advisor\figures\044_05_bad_case_demo_dqn_noop.png`
- `benchmark_results\044_05_dqn_bad_case_examples_for_advisor\figures\044_05_bad_case_demo_dqn_q_ranking.png`

## 汇报口径

这组图可以用来说明：我们不是只说 DQN 不好，而是具体看到三类失败形态：普通 DQN 动作模板化且产量不够，严格 MaskableDQN 随训练步数增加退化，Demo-DQN/DQfD 虽然加入了优质示范经验，但非零示范动作没有真正成为 Q 值最高动作。因此当前阶段 DQN 更适合作为算法对照保留，主线继续优化 PPO。
