# 014_11 HLA2010 / HLA2015 跨 seed 稳定性验证记录

## 目的

在已经确认 HLA2010 与 HLA2015 具有“成功信号”的前提下，只做最小成本跨 seed 验证：

- 检查 seed1 是否还能复现 seed0 的高产结果；
- 检查水氮动作是否仍然有意义；
- 判断当前“成功候选案例”能否进一步升级为“可复现成功案例”。

## 统一设置

- 初始条件：IC=1
- 动作空间：9 动作
  - 灌溉：0 / 15 / 30 mm
  - 施氮：0 / 50 / 100 kg/ha
- 奖励：baseline-relative reward
  - `reward = (Yield_policy - Yield_null_baseline) - 1.0 * irrigation - 5.0 * nitrogen`

## 实验结果

### HLA2010

| seed | final_grnwt | irrigation total | nitrogen total | max WSPD | max NSTRES | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 7853.665 | 120.0 | 300.0 | 0.4158 | 0.0158 | 成功候选 |
| 1 | 7853.665 | 120.0 | 300.0 | 0.4158 | 0.0158 | 与 seed0 几乎完全一致 |

### HLA2015

| seed | final_grnwt | irrigation total | nitrogen total | max WSPD | max NSTRES | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 7626.842 | 120.0 | 300.0 | 0.0000 | 0.0145 | 成功候选 |
| 1 | 7651.901 | 120.0 | 300.0 | 0.0000 | 0.0145 | 与 seed0 一致，且产量略高 |

## 解释

1. HLA2010 的 seed1 与 seed0 在总灌溉、总施氮、最大胁迫和最终产量上都几乎一致，说明这条线具备很强的跨 seed 稳定性。
2. HLA2015 的 seed1 也复现了相同的管理结构，并且产量略高于 seed0，说明这条线同样具备稳定复现性。
3. 因此，HLA2010 与 HLA2015 已经不只是“成功信号”，而是可以更稳妥地称为“可复现的成功案例”。

## 结论

当前这套 `9-action + baseline-relative reward + IC=1` 配置下：

- HLA2010：跨 seed 稳定；
- HLA2015：跨 seed 稳定；
- 两者都可以作为后续主实验年份继续推进。

## 文件位置

- `prompts/014_11_hla2010_2015_seed1_cross_seed_validation.md`
- `DSSAT_auto_validation/HLA_2004/hla2010_success_strategy_validation_014_09/2010/action9_seed0_5000steps/`
- `DSSAT_auto_validation/HLA_2004/hla2010_success_strategy_validation_014_09/2010/action9_seed1_5000steps/`
- `DSSAT_auto_validation/HLA_2004/hla2015_success_strategy_validation_014_10/2015/baseline_relative_seed0_5000steps/`
- `DSSAT_auto_validation/HLA_2004/hla2015_success_strategy_validation_014_10/2015/baseline_relative_seed1_5000steps/`
