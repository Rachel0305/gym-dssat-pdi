# 020_04 HLA2010 Double DQN seed1 严格单变量对照

## 历史核对

- 当前标准DQN seed1正式结果来自015_12：固定规则选中25K，I60/N0，产量7573 kg/ha。
- 020_03证明seed1/seed2形成相似的低灌溉Q值排序，而seed0形成I120/N0策略族。
- 项目曾在HLA2015旧economic-reward、4动作、5K条件下测试Double DQN并得到负结果；本轮reward、9动作和50K框架不同，因此属于当前正式框架的首次严格对照，而不是重复旧实验。

## 唯一改动

标准DQN TD target：

```text
r + gamma * max_a Q_target(s', a)
```

Double DQN TD target：

```text
a* = argmax_a Q_online(s', a)
r + gamma * Q_target(s', a*)
```

其他全部保持与015_12 seed1一致：

- HLA2010、IC=1、相同天气/土壤/品种/MZX；
- seed=1；
- baseline-relative reward；
- 9动作，I∈{0,15,30}，N∈{0,50,100}；
- I≤120、N≤300、最小操作间隔7天；
- learning_rate=1e-4、buffer_size=10000、learning_starts=50、batch_size=32；
- train_freq=1、gradient_steps=1、gamma=0.99；
- exploration_fraction=0.35、epsilon 1.0→0.05；
- 50K，每5K保存和确定性评估。

## 固定判定规则

1. checkpoint按`total_reward`最大选择；并列时取最早；
2. 选定后再计算DSSAT原生WP_ET、NLCM并比较auto和官方expert；
3. 不根据产量或图形人工换checkpoint；
4. 严格成功沿用020_01/020_02口径。

## 执行

1. 500-step smoke，仅检查兼容性、动作链和内存；
2. smoke通过后seed1 50K；
3. 与标准DQN seed1固定规则结果严格比较；
4. 如果没有改善，关闭当前Double DQN路线，不追加seed。

## 输出

```text
DSSAT_auto_validation/HLA_2004/hla2010_double_dqn_seed1_020_04/
docs/2026-07-10_020_04_hla2010_double_dqn_seed1_record.md
```

不修改`site-packages`、015_12脚本或旧结果。
