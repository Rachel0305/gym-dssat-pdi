# 014_09 HLA2010 成功策略验证

## 目标

按照导师当前判据，验证 HLA2010 是否能找到一条“成功策略”：

- 明显优于 null；
- 能稳定输出有意义的水氮操作；
- 不是一次 smoke 偶然撞出来。

## 采用设置

- 年份：HLA2010
- 初始条件：IC=1
- 动作空间：9 动作
  - 灌溉：0 / 15 / 30 mm
  - 施氮：0 / 50 / 100 kg/ha
- 奖励：baseline-relative reward
  - `reward = (Yield_policy - Yield_null_baseline) - 1.0 * irrigation - 5.0 * nitrogen`

## 执行顺序

1. 先确认 014_08 的 9 动作 smoke 结果；
2. 直接在同一设置下跑 5K seed0；
3. 观察：
   - 是否仍然 no-op；
   - 是否能稳定比 null 更优；
   - 是否能保持有意义的灌溉/施肥输出。

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2010_action_space_sensitivity_probe_014_08/`
- `DSSAT_auto_validation/HLA_2004/hla2010_success_strategy_validation_014_09/`
- `docs/2026-06-30_014_09_hla2010_success_strategy_validation_record.md`

## 备注

- 不扩展到其他站点；
- 不改奖励形式；
- 先看 HLA2010 能否达成导师定义的“成功策略”。
