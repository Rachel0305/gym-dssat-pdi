# 014_08 HLA2010 action-space sensitivity probe

## 背景

014_07 说明：

- baseline-relative reward 仍然没有把 HLA2010 的 DQN 从 no-op 拉回来；
- 因此下一步不继续大幅改奖励，而是诊断动作空间是否过粗。

## 目标

只做 HLA2010 的最小诊断，不训练长模型。

对比两种离散动作粒度在同一 reward 下的 smoke 行为：

1. 4 动作版本：当前 YC/FQ/HLA 统一动作空间
2. 9 动作版本：加密动作空间

先各跑 200 step smoke，比较是否更容易出现有意义的水氮操作。

## 奖励

沿用 014_07 的 baseline-relative reward：

```text
reward = (Yield_policy - Yield_null_baseline) - 1.0 * irrigation - 5.0 * nitrogen
```

## 4 动作版本

- 0：不操作
- 1：灌溉 30 mm
- 2：施氮 100 kg/ha
- 3：灌溉 30 mm + 施氮 100 kg/ha

## 9 动作版本

灌溉 `{0, 15, 30}` mm × 施氮 `{0, 50, 100}` kg/ha，共 9 个动作。

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_action_space_sensitivity_probe_014_08/`
- `014_08_hla2010_action_space_summary.csv`
- `014_08_hla2010_action_space_daily.csv`
- `docs/2026-06-30_014_08_hla2010_action_space_sensitivity_probe_record.md`

## 原则

- 不训练 5K；
- 只做 smoke；
- 如果 9 动作也没有明显改善，就不要继续往动作粒度上加码。
