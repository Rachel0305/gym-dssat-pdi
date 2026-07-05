# 015_13 HLA2010/HLA2015 baseline-relative DQN seed1 稳定性复核

## 目的

补做 HLA2010/HLA2015 在统一 baseline-relative DQN 框架下的 seed1 复核，用于判断 seed0 中出现的成功策略是否可以跨随机种子复现。

## 固定框架

奖励函数：

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

固定设置：

- 算法：DQN
- seed：1
- timesteps：50K
- checkpoint：每 5K 保存和评估一次
- 动作空间：9 actions
  - 灌溉：0 / 15 / 30 mm
  - 施氮：0 / 50 / 100 kg/ha
- 预算：I <= 120 mm, N <= 300 kg/ha
- 单次上限：I <= 30 mm, N <= 100 kg/ha
- 最小操作间隔：7 days
- 每个站点年份使用自己的 null 产量作为 baseline。

## 输入和输出

- Prompt: `prompts/015_13_hla2010_2015_baseline_relative_seed1_validation.md`
- Script: `src/run_hla_baseline_relative_dqn_checkpoint_015_12.py`
- HLA2010 seed1:
  - `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/checkpoint_summary.csv`
  - `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/dqn_eval_daily.csv`
- HLA2015 seed1:
  - `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps/checkpoint_summary.csv`
  - `DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps/dqn_eval_daily.csv`

## HLA2010 seed0 vs seed1

HLA2010 基线：

- null: GWAD 6956 kg/ha, I 0, N 0
- recorded/expert 平移: GWAD 7679 kg/ha, I 30, N 165
- DSSAT auto: GWAD 7854 kg/ha, I 190.4, N 0

| seed | best reward checkpoint | I mm | N kg/ha | GWAD kg/ha | CWAD kg/ha | max water stress | max N stress | reward | 解释 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 15000/25000/35000 | 120 | 0 | 7854 | 20678-20886 | 0.416 | 0.038-0.062 | 777.7 | 追平 DSSAT auto 产量，较 auto 节水约 70.4 mm，N=0 |
| 1 | 25000 | 60 | 0 | 7573 | 20255 | 0.751 | 0.173 | 557.0 | 明显优于 null，但低于 recorded 和 auto |

### HLA2010 判断

HLA2010 seed1 没有完全复现 seed0 的 `I120/N0/GWAD7854` 策略。它仍能学到节水、零氮、增产策略，但产量只到 7573 kg/ha，低于 recorded 和 auto。

因此 HLA2010 当前结论应写成：

```text
HLA2010 显示 baseline-relative DQN 的优化潜力，但追平 DSSAT auto 的高产策略尚未跨 seed 稳定复现。
```

## HLA2015 seed0 vs seed1

HLA2015 基线：

- null: GWAD 6486 kg/ha, I 0, N 0
- recorded/expert 平移: GWAD 7296 kg/ha, I 30, N 165
- DSSAT auto: GWAD 7648 kg/ha, I 141.5, N 0

| seed | best reward checkpoint | I mm | N kg/ha | GWAD kg/ha | CWAD kg/ha | max water stress | max N stress | reward | 解释 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 45000 | 75 | 0 | 7653 | 19077 | 0.000 | 0.108 | 1092.2 | 略高于 auto，较 auto 节水约 66.5 mm，N=0 |
| 1 | 40000 | 75 | 0 | 7653 | 19077 | 0.000 | 0.076 | 1092.1 | 与 seed0 高度一致 |
| 1 | 50000 | 45 | 50 | 7653 | 19081 | 0.000 | 0.015 | 872.2 | 同产量，更少水但用 50 kg/ha N |

### HLA2015 判断

HLA2015 的关键策略跨 seed 复现较好：

```text
GWAD≈7653, I≈75, N=0
```

与 DSSAT auto 相比：

- 产量：7653 vs 7648 kg/ha，略高；
- 灌溉：75 vs 141.5 mm，少约 66.5 mm；
- 施氮：0 vs 0 kg/ha，相同；
- 水分胁迫：DQN best checkpoint 为 0，与 auto 类似；
- 氮胁迫：seed0/seed1 略有差异，但没有导致产量下降。

与 recorded/expert 相比：

- 产量更高；
- 氮肥更少；
- 灌溉更多。

因此 HLA2015 当前可以作为最强 HLA 成功案例：

```text
HLA2015 在统一 baseline-relative DQN 框架下，seed0/seed1 均能找到略高于 DSSAT auto 且明显节水的策略。
```

## 给导师汇报时的简明结论

1. HLA2010：有优化潜力，但 seed 稳定性不足；不能作为最强正式案例。
2. HLA2015：目前是 HLA 中最清楚的成功案例；seed0/seed1 都能达到约 7653 kg/ha，且比 DSSAT auto 少用约 66.5 mm 水。
3. 当前框架不应只看最终 50K 模型；需要 checkpoint selection，因为训练后期可能退化到 null 或资源浪费策略。
4. 下一步应把同一套 baseline-relative DQN 框架扩展到 FQ，并把 YC/FQ/HLA 的结果整理成统一比较表。

