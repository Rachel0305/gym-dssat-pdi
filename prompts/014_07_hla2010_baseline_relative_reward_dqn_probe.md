# 014_07 HLA2010 baseline-relative reward DQN 探针

## 背景

014_06 证明了：

- Tao et al. 2023 风格的绝对产量终季奖励，在 HLA2010 里仍然会退化为 no-op；
- 原因是 HLA2010 的 null 产量本身不低，绝对产量奖励会天然偏袒 no-op。

本轮改成“相对 null 基线”的奖励，测试是否能打破这个偏袒。

## 目标

只做 HLA2010，低成本探针，不直接扩站点。

1. 先 200 step smoke，确认动作链路正常；
2. 再跑 5K seed0；
3. 看模型是否还会退化成 no-op。

## 奖励思想

把奖励改成“比 null 基线多赚多少”：

```text
reward = (Yield_policy - Yield_null_baseline) - water_cost × irrigation - nitrogen_cost × nitrogen
```

其中 `Yield_null_baseline` 来自 HLA2010 的 null 情景终产量。

## 环境设置

- 年份：HLA2010
- 初始条件：IC=1
- 管理方式：沿用 014_03 的 linked DQN 动作链路
- 动作：
  - 0：不操作
  - 1：灌溉 30 mm
  - 2：施氮 100 kg/ha
  - 3：灌溉 30 mm + 施氮 100 kg/ha
- 预算：
  - I ≤ 120 mm
  - N ≤ 300 kg/ha
  - 单次 I≤30 mm，N≤100 kg/ha
  - 最小操作间隔 7 天
- 窗口：
  - free_daily：DAP 1–120 都允许

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_baseline_relative_reward_dqn_probe_014_07/`
- `dqn_baseline_rel_eval_daily.csv`
- `event_summary.json`
- `014_07_hla2010_baseline_relative_reward_summary.csv`
- `docs/2026-06-30_014_07_hla2010_baseline_relative_reward_dqn_probe_record.md`

## 运行原则

- 不训练三站；
- 先 smoke，后 5K；
- 不覆盖旧结果；
- 如果 5K 仍然 no-op，就说明 baseline-relative 也不足以修复，需要回到动作/时间窗/奖励尺度再诊断。
