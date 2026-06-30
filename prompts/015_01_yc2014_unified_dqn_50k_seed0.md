# 015_01 YC2014 统一 DQN 正式训练 seed0

## 目标

启动一条新的“正式 DQN 水氮联合优化”实验线，先在禹城站 2014 年验证统一框架是否能稳定产生可解释、可写入论文的 DQN 管理策略。

本轮只跑一个站点年份和一个 seed：

- 站点：Yucheng / YC
- 年份：2014
- 算法：DQN
- 训练步数：50,000
- seed：0

## 统一框架

### 奖励函数

采用阶段经济型奖励：

```text
R_t = max(0, GRNWT_t - GRNWT_{t-1}) - 1.0 × I_t - 5.0 × N_t
```

其中：

- `GRNWT_t`：当前籽粒产量；
- `I_t`：当前步实际灌溉量，单位 mm；
- `N_t`：当前步实际施氮量，单位 kg/ha；
- 水成本系数 `1.0`；
- 氮成本系数 `5.0`。

该奖励函数不再混用 HLA baseline-relative 终季 reward，而是使用统一的阶段增量 reward，便于后续跨站点、跨年份比较。

### 动作空间

采用 9 个离散动作：

```text
I ∈ {0, 15, 30} mm
N ∈ {0, 50, 100} kg/ha
```

组合为 9 个动作：

| action | irrigation | nitrogen |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 15 | 0 |
| 2 | 30 | 0 |
| 3 | 0 | 50 |
| 4 | 15 | 50 |
| 5 | 30 | 50 |
| 6 | 0 | 100 |
| 7 | 15 | 100 |
| 8 | 30 | 100 |

### 硬约束

```text
seasonal irrigation ≤ 120 mm
seasonal nitrogen ≤ 300 kg/ha
daily irrigation action ≤ 30 mm
daily nitrogen action ≤ 100 kg/ha
minimum operation interval = 7 days
```

### 管理输入

继续使用已经校正过的 YC2014 输入包和 PDI/gym-DSSAT 运行链路。

DQN 动态动作情景必须使用：

```text
IRRIG = L
FERTI = L
```

并保留原始管理表/指针，避免链接动作无法写入 DSSAT 管理事件。

## 执行要求

1. 不覆盖旧的 013/014 结果。
2. 新输出目录使用：

```text
DSSAT_auto_validation/yc2014_unified_dqn_formal_015_01/
```

3. 保存：

- 训练日志
- 评估日值 CSV
- 汇总 CSV
- 模型文件
- PDI 评估快照
- 中文实验记录

4. 只跑 `seed0` 和单一统一 DQN 情景，不同时跑多个窗口版本。
5. 如果训练中断或 OOM，立即停止并记录错误，不自动扩大实验。

## 判断标准

本轮先看：

- DQN 是否明显优于 null；
- DQN 是否接近或超过 recorded expert / DSSAT auto；
- 水氮投入是否合理；
- 灌溉和施肥事件是否真正进入 MgmtEvent.OUT；
- 日值胁迫、产量、生物量轨迹是否合理。

