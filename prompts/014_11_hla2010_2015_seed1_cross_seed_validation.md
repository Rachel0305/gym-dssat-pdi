# 014_11 HLA2010/HLA2015 跨 seed 稳定性验证

## 目标

在已经确认 HLA2010 与 HLA2015 存在“成功信号”的前提下，只做最小成本的跨 seed 验证，检查 seed1 是否还能复现：

- 明显优于 null 的产量；
- 有意义的水氮操作；
- 不退化成 no-op。

## 采用设置

- 年份：HLA2010 与 HLA2015
- 初始条件：IC=1
- 动作空间：9 动作
  - 灌溉：0 / 15 / 30 mm
  - 施氮：0 / 50 / 100 kg/ha
- 奖励：baseline-relative reward
  - `reward = (Yield_policy - Yield_null_baseline) - 1.0 * irrigation - 5.0 * nitrogen`

## 执行顺序

1. 先跑 HLA2010 seed1，确认是否还能保持 seed0 的“高产 + 有意义动作”；
2. 再跑 HLA2015 seed1，检查是否能稳定复现；
3. 对比 seed0 / seed1 的产量、灌溉、施氮、最大胁迫与动作轨迹。

## 结论判据

- 若 seed1 仍明显优于 null 且动作合理，则可把对应年份定为可复现成功案例；
- 若 seed1 明显退化到 no-op 或接近 null，则说明当前年份仍不具备跨 seed 稳定性，只能算成功候选。

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_success_strategy_validation_014_09/`
- `DSSAT_auto_validation/HLA_2004/hla2015_success_strategy_validation_014_10/`
- `docs/2026-06-30_014_11_hla2010_2015_seed1_cross_seed_validation_record.md`
