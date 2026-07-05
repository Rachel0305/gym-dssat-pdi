# 015_09 FQ2016 正式 DQN 长训练 prompt

## 目标

在已经筛选出的候选里，优先对 FQ2016 做一轮正式 DQN 长训练，判断它是否能在统一奖励和约束逻辑下，给出比当前基线更稳定、更可解释的水氮联合策略。

## 背景

当前主线筛选结论是：

- HLA2010 / HLA2015：稳定成功案例；
- YC2014：高产平顶案例，不建议继续追更高产；
- FQ2016：最值得继续推进正式训练。

因此这轮不再做站点筛选，而是对 FQ2016 做正式长训练复核。

## 训练原则

1. 只保留一套奖励函数和一套动作语义，不要在不同站点之间切换目标定义。
2. 保持 IC=1、天气、土壤、品种、管理模板一致。
3. 先做低成本 smoke test，再决定是否进入正式 50K / 更长训练。
4. 每次只改一个变量，不要同时改奖励、动作窗口和预算。
5. 记录 seed0 / seed1 的差异，不要只报告单个最好结果。

## 需要输出

1. 四情景对比图：
   - null
   - expert / recorded
   - DSSAT auto
   - DQN

2. 日值表格：
   - rainfall
   - soil water stress
   - soil nitrogen stress
   - irrigation
   - fertilization
   - grain yield
   - biomass
   - cumulative reward proxy

3. 汇总表：
   - irrigation total
   - fertilizer total
   - final yield
   - biomass
   - max water stress
   - max nitrogen stress
   - seed0 / seed1 comparison

## 判断标准

- 若 FQ2016 能稳定优于 null，并且输出有意义的水氮操作，则可以作为正式长训练对象继续推进。
- 若只能追平专家或 auto，但没有稳定超越空间，则要把它写成“可解释但未证明显著优越”。
- 若 seed 间波动仍然很大，则先停在诊断，不要扩大训练步数。

## 输出路径

建议放到：

```text
DSSAT_auto_validation/fq2016_formal_dqn_015_09/
```

## 备注

这轮 prompt 不要求一次性解决所有站点，只针对 FQ2016。
