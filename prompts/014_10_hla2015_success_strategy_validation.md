# 014_10 HLA2015 成功策略复现

## 目标

把 HLA2010 已经跑通的“成功策略”迁移到 HLA2015，验证是否也能：

- 明显优于 null；
- 稳定输出有意义的水氮操作；
- 不退化成 no-op。

## 采用设置

- 年份：HLA2015
- 初始条件：IC=1
- 输入包：HLA2015 文献对齐输入包
  - `DSSAT_auto_validation/HLA_2004/hla2015_literature_aligned_dqn_012_15/2015/literature_aligned_seed0_5000steps/input`
- 动作空间：9 动作
  - 灌溉：0 / 15 / 30 mm
  - 施氮：0 / 50 / 100 kg/ha
- 奖励：baseline-relative reward
  - `reward = (Yield_policy - Yield_null_baseline) - 1.0 * irrigation - 5.0 * nitrogen`

## 执行顺序

1. 先 200 step smoke；
2. 再 5K seed0；
3. 判断是否能稳定优于 null。

## 输出

- `DSSAT_auto_validation/HLA_2004/hla2015_success_strategy_validation_014_10/`
- `docs/2026-06-30_014_10_hla2015_success_strategy_validation_record.md`

## 备注

- 不扩展到其他站点；
- 不改 reward；
- 先看 2015 能不能复现 HLA2010 的成功策略。
