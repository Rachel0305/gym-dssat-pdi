# 017_12 LC2010 baseline-relative DQN smoke

## 目标

在 017_11 已修通 LC 输入链路并筛出 LC2010 有明确优化空间之后，先做一个低成本 DQN smoke，不进入长训练。

核心判断：

1. DQN 是否能在 LC2010 上获得接近或超过 recorded expert / DSSAT auto 的产量；
2. DQN 是否能使用更少或不明显更多的灌溉与施氮；
3. DQN 的操作是否能通过水分胁迫、氮胁迫和管理事件解释；
4. 如果 seed0 5K 已经完全不合理，则不要继续 seed1 或长训练。

## 输入基础

使用 017_11 已确认的 LC 临时输入修复逻辑，不修改原始 LC 输入包：

- 土壤 ID 修正为 `LC990012007`；
- IC/SDATE/planting/management event 日期按年份对齐；
- 使用逐年 FileX 名称；
- 传入全部 `CNLC*.WTH`；
- DQN 情景使用 `IRRIG=L, FERTI=L`，动态动作进入 DSSAT/PDI。

目标年份：

- LC2010

基线：

- null: 8051 kg/ha
- recorded expert: 8732 kg/ha, I=130 mm, N=250 kg/ha
- DSSAT auto: 8738 kg/ha, I=138.5 mm, N=0 kg/ha

## DQN 设置

- 算法：Stable-Baselines3 DQN
- timesteps：5000
- seed：0
- checkpoint：每 1000 步
- 动作空间：9 个离散动作，I ∈ {0, 15, 30} mm，N ∈ {0, 50, 100} kg/ha
- 预算：I ≤ 120 mm，N ≤ 300 kg/ha
- 单次上限：I ≤ 30 mm，N ≤ 100 kg/ha
- 最小操作间隔：7 天
- 操作窗口：DAP 1–120

奖励函数：

```text
reward = max(0, GWAD_final - GWAD_null_LC2010) - 1.0 * irrigation - 5.0 * nitrogen
```

说明：

- 这是与 HLA/YC/FQ/SY 当前统一方向一致的 baseline-relative DQN 奖励；
- null baseline 使用 LC2010 本地 null，而不是其他站点的 baseline；
- 本轮不调初始水氮，不为了结果改 IC。

## 输出

保存到：

```text
DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12
```

必须保存：

- checkpoint summary CSV
- DQN evaluation daily CSV
- DQN management event CSV
- checkpoint 模型 zip
- checkpoint 诊断图
- 中文实验记录 MD

## 决策规则

若某个 checkpoint 满足以下任意一种，则进入下一步 seed1 复核：

1. 产量 ≥ recorded/auto，且 I/N 不明显高于二者；
2. 产量接近 recorded/auto（差距 < 100 kg/ha），但明显节水或节氮；
3. 明显优于 null，且操作过程合理，可作为继续训练候选。

若 DQN 只会打满资源但产量不超过 expert/auto，或操作完全不可解释，则不继续长训练，先做动作/奖励诊断。
