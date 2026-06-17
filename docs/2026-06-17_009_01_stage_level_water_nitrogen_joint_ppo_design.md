# 009_01 阶段级水氮联合 PPO 设计说明

## 1. 为什么要做 009_01

目前 008 系列已经证明：

- HLA 2004 和 FQA 2016 这两个水分胁迫年份中，PPO 能产生非零、非饱和、可解释的灌溉决策；
- 但这两个成功结果都是在固定充分氮肥 `N150` 条件下得到的；
- 因此当前结果更准确地说是“固定氮肥背景下的灌溉优化”，还不能完全支撑“水氮联合优化”。

009_01 的目的就是把氮肥重新纳入 PPO 决策，但不能回到最早那种每日自由水氮动作，因为那种设计已经被证明容易出现：

- 打满水氮上限；
- 第一天乱灌乱施；
- PPO 不施氮或过量施氮；
- 决策过程无法解释。

所以 009_01 的核心思想是：

> 保留 008 系列已经验证有效的阶段级、天气预报、胁迫诊断和 soft reward 框架，只把“氮肥追加量”逐步交还给 PPO。

## 2. 当前推荐框架

### 2.1 动作频率

不采用每日动作。

继续采用 DAP 阶段动作：

- S1：建苗期
- S2：营养生长期前期
- S3：中期快速生长期
- S4：后期水分风险阶段
- S5：成熟/终端阶段

PPO 每个阶段只决策一次。

### 2.2 灌溉设计

沿用 008 的思路：

- S1 禁止灌溉；
- S2-S5 根据天气预报/胁迫 gate 决定是否允许灌溉；
- 不再使用 hard minimum irrigation；
- PPO 自己决定实际灌溉量；
- 每阶段最大灌溉量为 40 mm。

也就是：

| 阶段 | 灌溉基础量 | PPO 可控灌溉上限 |
|---|---:|---:|
| S1 | 0 | 0 |
| S2 | 0 | 40 |
| S3 | 0 | 40 |
| S4 | 0 | 40 |
| S5 | 0 | 40 |

### 2.3 氮肥设计

氮肥不再完全固定为 150 kg/ha，但也不完全交给 PPO 从零开始决定。

采用“基础氮 + PPO 追加氮”：

| 阶段 | 基础氮 | PPO 追加上限 | 阶段最大氮 |
|---|---:|---:|---:|
| S1 | 50 | 0 | 50 |
| S2 | 50 | 0 | 50 |
| S3 | 0 | 50 | 50 |
| S4 | 0 | 20 | 20 |
| S5 | 0 | 20 | 20 |

全季氮肥范围：

- 最低：100 kg/ha
- 最高：190 kg/ha

这样做的意义：

- S1/S2 保证基本农学供氮，避免 PPO 一开始学成“不施氮”；
- S3 是主要追加氮窗口，PPO 可以决定是否追肥；
- S4/S5 只给较小追加空间，避免后期过量施氮；
- PPO 仍然真实参与氮肥决策，因为总氮不再固定。

## 3. 奖励函数建议

第一版 009_02 smoke test 不建议大规模调参。

建议沿用 008_17 的 stronger soft-stress reward：

```text
R_stage =
  0.001 * ΔTOPWT
+ 0.020 * ΔGRNWT
+ 0.010 * terminal_GRNWT
- 0.050 * irrigation
- nitrogen_cost * nitrogen
- soft_SWFAC_penalty
- soft_NSTRES_penalty
```

其中：

```text
soft_SWFAC_penalty =
  3.0 * sum(max(SWFAC - 0.05, 0))
+ 1.0 * number_of_days(SWFAC > 0.05)
```

因为 009 开始把一部分氮肥决策交给 PPO，如果只惩罚“施氮成本”，不惩罚“缺氮胁迫”，PPO 可能会为了省成本而少施氮，导致 NSTRES 偏高。因此 009_02 需要同步加入一个较弱的 soft NSTRES 惩罚：

```text
soft_NSTRES_penalty =
  1.5 * sum(max(NSTRES - 0.05, 0))
+ 0.5 * number_of_days(NSTRES > 0.05)
```

这个惩罚暂时设为 SWFAC 惩罚的一半左右，目的不是让 PPO 盲目多施氮，而是防止 PPO 用严重氮胁迫来换取较低施肥成本。

第一版建议：

```text
nitrogen_cost = 0.030
```

这个值不要一开始就扫很多组。009_02 先看一个 HLA 2004 小步数 smoke test 是否能跑通。

## 4. 009_02 应该如何判断成败

### 成功信号

HLA 2004 seed0、5k timesteps 下，如果出现：

- 灌溉非零；
- 灌溉不打满；
- 总氮在 110-190 kg/ha 之间；
- PPO 追加氮非零，或者 PPO 选择低氮但 NSTRES 胁迫天数 < 30 天；
- 如果 NSTRES 胁迫天数 >= 30 天，需要标记为 warning，除非产量仍然保持很高；
- GRNWT 不崩溃；
- S1 不灌溉；
- 水氮动作集中在 S3/S4/S5 等合理阶段；

则说明水氮联合 PPO 的动作空间设计初步可行。

### 失败信号

如果出现：

- 总灌溉为 0；
- 只施基础氮 100 kg/ha 且 NSTRES 很高；
- total_ppo_extra_n = 0 且 NSTRES 胁迫天数 >= 30 天；
- 总氮总是打满 190 kg/ha；
- 灌溉打满；
- GRNWT 明显崩溃；
- S1 灌溉；

则先检查动作空间和 gate，不要立刻调 reward。

## 5. 需要记录的指标

009 后续脚本必须记录：

- total irrigation；
- total nitrogen；
- base nitrogen；
- PPO extra nitrogen；
- soft SWFAC penalty；
- soft NSTRES penalty；
- PPO irrigation；
- 每阶段 raw irrigation action；
- 每阶段 raw nitrogen action；
- 每阶段实际灌溉；
- 每阶段实际施氮；
- SWFAC stress days；
- NSTRES stress days；
- final GRNWT；
- reward components；
- stage action evolution during training。

这些指标以后会直接用于论文里的训练曲线和策略解释。

## 6. 当前结论

009_01 的设计结论是：

> 下一阶段不是重新开始，而是在 008 已经验证有效的天气/胁迫辅助阶段 PPO 框架上，谨慎地把氮肥追加量纳入 PPO 决策。这样既回应“水氮联合优化”的科学问题，又避免回到最初不稳定的 daily unrestricted PPO。
