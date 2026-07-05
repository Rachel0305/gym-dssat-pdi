# 015_15 当前 DQN 跨站点训练结果汇总：导师讨论版

## 统一框架

当前用于比较的正式 DQN 框架为 baseline-relative reward：

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

说明：

- 每个站点年份使用自己的 null 产量作为 baseline；
- 奖励公式、成本系数、动作空间、预算和算法保持一致；
- 不是每个站点单独调 reward；
- 当前训练均使用 checkpoint selection，不直接使用最终 50K 模型。

固定动作和约束：

| 项目 | 设置 |
|---|---|
| 算法 | DQN |
| 灌溉动作 | 0 / 15 / 30 mm |
| 施氮动作 | 0 / 50 / 100 kg/ha |
| 季节灌溉预算 | I <= 120 mm |
| 季节施氮预算 | N <= 300 kg/ha |
| 最小操作间隔 | 7 days |
| 训练步数 | 50K |
| checkpoint | 每 5K 评估一次 |

## 当前最重要结论

目前最适合作为“DQN 可行性成功案例”的是 **HLA2015**。

HLA2015 在 seed0 和 seed1 下都出现了相同类型的策略：

```text
GWAD≈7653 kg/ha, I≈75 mm, N=0 kg/ha
```

与 DSSAT auto 相比：

- 产量略高：7653 vs 7648 kg/ha；
- 灌溉更少：75 vs 141.5 mm；
- 施氮相同：0 vs 0 kg/ha。

这说明在 HLA2015 这个站点年份上，当前统一 DQN 框架已经可以跨 seed 复现“略高产 + 明显节水”的策略。

## 各站点年份结果汇总

| 站点年份 | 训练状态 | 最好结果 | 与 auto 比较 | 与 recorded 比较 | 当前判断 |
|---|---|---|---|---|---|
| HLA2010 seed0 | 50K 完成 | GWAD 7854, I120, N0 | 产量持平，节水 70.4 mm | 产量更高、少氮，但多水 | 有潜力 |
| HLA2010 seed1 | 50K 完成 | GWAD 7573, I60, N0 | 低于 auto | 低于 recorded | 未稳定复现 seed0 高产策略 |
| HLA2015 seed0 | 50K 完成 | GWAD 7653, I75, N0 | 略高于 auto，节水 66.5 mm | 产量更高、少氮，但多水 | 成功候选 |
| HLA2015 seed1 | 50K 完成 | GWAD 7653, I75, N0 | 略高于 auto，节水 66.5 mm | 产量更高、少氮，但多水 | 当前最强成功证据 |
| YC2014 seed0 | 50K 完成 | 高产：GWAD 9418, I120, N300；高 reward：GWAD 8657, I120, N0 | 高产需 N300；高 reward 产量下降 | 不能同时证明高产和节氮 | 诊断案例，不建议主推 |
| FQ2016 seed0 | 50K 完成 | 高产：GWAD 8316, I120, N300；高 reward：GWAD 7779, I60, N0 | 高产需 N300；高 reward 低于 auto | 高 reward 低于 recorded | 不建议继续 seed1，除非导师要求 |

## HLA 详细结果

### HLA2010

基准：

| 情景 | GWAD kg/ha | I mm | N kg/ha |
|---|---:|---:|---:|
| null | 6956 | 0 | 0 |
| recorded/expert | 7679 | 30 | 165 |
| DSSAT auto | 7854 | 190.4 | 0 |

DQN：

| seed | best reward checkpoint | GWAD kg/ha | I mm | N kg/ha | reward | 判断 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 15K/25K/35K | 7854 | 120 | 0 | 777.7 | 追平 auto，节水 |
| 1 | 25K | 7573 | 60 | 0 | 557.0 | 优于 null，但低于 recorded/auto |

结论：

```text
HLA2010 显示优化潜力，但追平 auto 的策略尚未跨 seed 稳定复现。
```

### HLA2015

基准：

| 情景 | GWAD kg/ha | I mm | N kg/ha |
|---|---:|---:|---:|
| null | 6486 | 0 | 0 |
| recorded/expert | 7296 | 30 | 165 |
| DSSAT auto | 7648 | 141.5 | 0 |

DQN：

| seed | best reward checkpoint | GWAD kg/ha | I mm | N kg/ha | reward | 判断 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 45K | 7653 | 75 | 0 | 1092.2 | 略高于 auto，明显节水 |
| 1 | 40K | 7653 | 75 | 0 | 1092.1 | 与 seed0 高度一致 |

结论：

```text
HLA2015 是当前最强 DQN 成功案例：跨 seed 复现，产量略高于 auto，灌溉明显少于 auto，施氮不增加。
```

## YC2014 结果

基准：

| 情景 | GWAD kg/ha |
|---|---:|
| null | 7825 |

baseline-relative DQN seed0：

| checkpoint | GWAD kg/ha | I mm | N kg/ha | reward | 解释 |
|---:|---:|---:|---:|---:|---|
| 5K | 9418 | 120 | 300 | -27.1 | 高产，但用氮过高且 reward 为负 |
| 25K | 8657 | 120 | 0 | 711.9 | reward 最高，但产量明显下降 |
| 50K | 8751 | 90 | 50 | 585.9 | 折中，但不构成强成功案例 |

结论：

```text
YC2014 在 baseline-relative reward 下不能同时给出“专家级高产 + 少氮”的稳定策略。目前更适合作为 reward 诊断案例。
```

## FQ2016 结果

基准：

| 情景 | GWAD kg/ha | I mm | N kg/ha |
|---|---:|---:|---:|
| null | 7066 | 0 | 0 |
| recorded_shifted | 7933 | 75 | 144 |
| DSSAT auto | 8012 | 59.9 | 0 |

baseline-relative DQN seed0：

| checkpoint | GWAD kg/ha | I mm | N kg/ha | reward | 解释 |
|---:|---:|---:|---:|---:|---|
| 5K | 8316 | 120 | 300 | -370.4 | 产量超过 auto，但用水/氮过高，reward 为负 |
| 25K | 7779 | 60 | 0 | 652.6 | reward 最高，但低于 recorded 和 auto |
| 50K | 7066 | 0 | 0 | 0.1 | 退化到 null |

结论：

```text
FQ2016 seed0 没有形成理想策略。高产依赖 N300；节约资源时产量不足。因此暂不建议继续 seed1，除非导师要求用它作为“失败/边界案例”。
```

## 建议给导师讨论的问题

1. 是否认可 HLA2015 作为当前主成功案例？
2. 是否接受 checkpoint selection 作为正式方法，而不是使用最终 50K 模型？
3. 是否需要继续把 FQ2016 做 seed1，还是把它作为当前框架的失败案例？
4. 下一步是否应该：
   - 继续在 HLA2015 上做更长训练或 seed2；
   - 或扩展到新站点年份；
   - 或调整 reward，使目标直接相对 DSSAT auto，而不是相对 null？

## 文件索引

- HLA seed1 prompt: `prompts/015_13_hla2010_2015_baseline_relative_seed1_validation.md`
- HLA seed1 record: `docs/2026-07-01_015_13_hla2010_2015_baseline_relative_seed1_validation_record.md`
- HLA script: `src/run_hla_baseline_relative_dqn_checkpoint_015_12.py`
- FQ prompt: `prompts/015_14_fq2016_baseline_relative_dqn_checkpoint.md`
- FQ script: `src/run_fq2016_baseline_relative_dqn_checkpoint_015_14.py`
- FQ record: `docs/2026-07-01_015_14_fq2016_baseline_relative_dqn_checkpoint_record.md`
- YC record: `docs/2026-07-01_015_10_yc2014_baseline_relative_dqn_record.md`

