# 012_08 HLA2015 固定 N0 水量扫描记录

## 目的

012_07 中，HLA2015 economic DQN seed1 学到的策略是：

- 灌溉 60 mm；
- 施氮 0 kg/ha；
- 产量约 7632 kg/ha；
- economic reward 约 7572。

这个结果看起来比固定 I120/N0 更“经济”，因为固定 I120/N0 产量略高，但多用了 60 mm 水，reward 反而更低。

因此本实验补充一个固定 N0 水量扫描，判断 seed1 的 I60/N0 是否接近当前 economic reward 下的固定水量最优。

## 实验性质

确定性反事实扫描，不训练模型。

## 扫描组合

| 情景 | 灌溉总量 | 施氮总量 |
|---|---:|---:|
| fixed_I0_N0 | 0 mm | 0 kg/ha |
| fixed_I30_N0 | 30 mm | 0 kg/ha |
| fixed_I60_N0 | 60 mm | 0 kg/ha |
| fixed_I90_N0 | 90 mm | 0 kg/ha |
| fixed_I120_N0 | 120 mm | 0 kg/ha |

奖励函数：

```text
R_t = ΔGRNWT_t - 1.0 × I_t - 5.0 × N_t
```

由于本扫描中 N=0，因此实际比较的是：

```text
R_t = ΔGRNWT_t - 1.0 × I_t
```

## 运行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla2015_fixed_water_n0_scan_012_08.py"
```

绘图命令：

```bash
python src/plot_hla2015_fixed_water_n0_scan_012_08.py
```

## 输出文件

- 运行脚本：

```text
src/run_hla2015_fixed_water_n0_scan_012_08.py
```

- 绘图脚本：

```text
src/plot_hla2015_fixed_water_n0_scan_012_08.py
```

- 汇总表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_fixed_water_n0_scan_012_08/hla2015_fixed_water_n0_scan_summary.csv
```

- 日值表：

```text
DSSAT_auto_validation/HLA_2004/hla2015_fixed_water_n0_scan_012_08/hla2015_fixed_water_n0_scan_daily.csv
```

- 汇总图：

```text
DSSAT_auto_validation/HLA_2004/hla2015_fixed_water_n0_scan_012_08/figures/hla2015_fixed_water_n0_scan_summary.png
```

## 结果

| 情景 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | economic reward | 最大水分胁迫 | 最大氮胁迫 |
|---|---:|---:|---:|---:|---:|---:|
| fixed_I0_N0 | 6486 | 0 | 0 | 6485.77 | 1.000 | 0.093 |
| fixed_I30_N0 | 7017 | 30 | 0 | 6987.07 | 0.896 | 0.115 |
| fixed_I60_N0 | 7625 | 60 | 0 | 7564.80 | 0.195 | 0.101 |
| fixed_I90_N0 | 7645 | 90 | 0 | 7555.12 | 0.000 | 0.057 |
| fixed_I120_N0 | 7645 | 120 | 0 | 7525.12 | 0.000 | 0.059 |

## 关键结论

1. 在 HLA2015、N0 条件下，产量从 I60 到 I120 只增加约 20 kg/ha。
2. 因为 water_cost=1.0，I90 和 I120 多用的水不能被微小产量增益抵消。
3. 固定扫描中 economic reward 最高的是 I60/N0。
4. DQN seed1 学到的 I60/N0 与固定扫描最优方向一致。

## 对 012_07 的解释

012_07 中，DQN seed1 的行为不应被解释为“没有打满水，所以不够好”。

更准确的解释是：

> 在当前 economic reward 下，HLA2015 的经济最优固定水量接近 I60/N0；DQN seed1 正好学到了这个方向。

但 seed0 仍退化为 null，因此算法稳定性仍然没有通过。

## 当前判断

HLA2015 的情况可以总结为：

- 管理增益来自灌溉，不来自施氮；
- 经济奖励下，60 mm 水已经接近最优；
- DQN seed1 学到了合理策略；
- DQN seed0 没学到，说明 seed 稳定性仍是主要问题。

## 下一步建议

现在不建议立刻改 reward。

建议下一步二选一：

1. HLA2015 再跑 seed2，仍然 5K。
   - 目的：判断 seed0 是偶然失败，还是 5K 训练本身高方差。

2. HLA2015 seed0 延长到 20K。
   - 目的：判断 seed0 是否只是训练不足。
   - 代价高于 seed2。

当前更推荐先跑 seed2，因为更省算力，也能直接回答 seed 稳定性问题。

