# 013_01 封丘/禹城新参数前向筛选与优化空间检查

## 背景

HLA2015 的 DQN 算法线已经定位到：水分决策可以学习，但混合水氮任务中氮动作 Q-value 高估和 seed 敏感性仍然存在。为了判断这个问题是否是 HLA 特例，需要扩展到其他站点，但不能直接训练。

用户已经提供封丘 FQ、禹城 YC 的新调参输入文件：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC
```

每个站点包含：

- `.MZX`
- `MZCER048.CUL`
- `SOIL.SOL`
- 2000–2023 单年 `.WTH`

## 新参数校准信息

### HWAM

| 编号 | 站点 | 年份 | 观测 HWAM | 模拟 HWAM | 误差 | 相对误差 |
|---|---|---:|---:|---:|---:|---:|
| YC1 | Yucheng | 2008 | 7852 | 7324 | -528 | -6.7% |
| YC2 | Yucheng | 2014 | 9718 | 8494 | -1224 | -12.6% |
| FQ1 | Fengqiu | 2007 | 7888 | 8072 | +184 | +2.3% |
| FQ2 | Fengqiu | 2010 | 6312 | 6595 | +283 | +4.5% |

### CWAM

| 站点 | 年份 | 观测 CWAM | 模拟 CWAM | 误差 | 相对误差 |
|---|---:|---:|---:|---:|---:|
| Yucheng | 2008 | 17999.2 | 17751 | -248.2 | -1.4% |
| Yucheng | 2014 | 18965.2 | 19090 | +124.8 | +0.7% |
| Fengqiu | 2007 | 14823 | 14121 | -702 | -4.7% |
| Fengqiu | 2010 | 12616 | 12690 | +74 | +0.6% |

初步判断：

- FQ 参数更稳；
- YC 可用，但 YC2014 籽粒产量偏低；
- 这些年份只是调参/校准年份，不必固定为后续 RL 年份；
- 如果校准年份没有优化空间，可以把专家策略迁移到其他年份做筛选。

## 本轮目标

本轮只做前向筛选，不训练 PPO/DQN。

目标是回答：

1. FQ/YC 新参数和输入包是否能在 PDI/gym-DSSAT 4.8.0 中正常运行；
2. 在校准年份和候选年份中，哪些年份存在水氮管理优化空间；
3. 专家策略迁移到其他年份后，是否能产生合理增产/减胁迫；
4. 是否有适合后续 DQN/PPO 的候选站点年份。

## 输入文件

### FQ

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0801.MZX
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/MZCER048.CUL
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/SOIL.SOL
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0001.WTH ... CNFQ2301.WTH
```

MZX 中已有年份：

- 2007
- 2008
- 2010

### YC

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0801.MZX
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/MZCER048.CUL
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/SOIL.SOL
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0001.WTH ... CNYC2301.WTH
```

MZX 中已有年份：

- 2008
- 2014

## 筛选策略

### 第一阶段：校准年份前向复核

先用 MZX 中已有年份跑：

- FQ：2007、2008、2010
- YC：2008、2014

情景：

1. null：不灌溉、不施肥；
2. expert/original：MZX 中已有记录管理；
3. DSSAT auto irrigation + auto-N attempt：保留作为参考，但如果 auto-N 不触发，只如实记录；

输出：

- HWAM/GWAD；
- CWAM/CWAD；
- 水分胁迫 WSPD/SWFAC；
- 氮胁迫 NSTD/NSTRES；
- 降雨；
- 灌溉事件；
- 施肥事件；
- 管理总量。

判断是否有优化空间：

- null 是否显著低于 expert/original；
- expert 是否仍有水分或氮胁迫；
- 管理投入是否明显过量或不足；
- DSSAT auto irrigation 是否能触发并有效降低水分胁迫；
- auto-N 不触发则记录，不作为必须条件。

### 第二阶段：候选年份筛选

如果校准年份没有明显优化空间，则迁移专家策略到其他年份。

候选年份优先考虑：

- 2010–2016；
- 2000–2023 中降雨偏少或产量潜在受限年份；
- 避免明显天气缺测年份。

每个站点先筛 3–5 个年份，不要全量训练。

情景：

1. null；
2. expert-shifted：将该站点已有专家策略平移到目标年份；
3. DSSAT auto irrigation + auto-N attempt；
4. 如有必要，再补固定水氮扫描，而不是直接训练。

## 图表输出

每个站点年份输出：

1. 过程图：
   - 降雨柱状图；
   - 水分胁迫曲线；
   - 氮胁迫曲线；
   - 灌溉/施肥事件；
   - 籽粒产量和生物量曲线；
2. 汇总表：
   - 情景；
   - 总灌溉；
   - 总施氮；
   - HWAM/GWAD；
   - CWAM/CWAD；
   - 最大水分胁迫；
   - 最大氮胁迫；
   - 成熟/终止 DAP；
3. 日值 CSV；
4. 中文实验记录。

图的风格尽量沿用之前 `ufga_rain_irrigation_fertilizer_stress_all_treatments.png` 类似的清晰风格：

- 横轴 DAP；
- 降雨柱状；
- 胁迫线颜色区分明显；
- 管理事件用竖线/三角点；
- 产量/生物量单独或分面显示，避免过挤。

## 执行约束

- 使用 Docker 容器 `b2fd6726c8c1`；
- 使用 Python `/opt/gym_dssat_pdi/bin/python`；
- 不训练 RL；
- 不覆盖原始输入文件；
- 所有改写后的临时输入、脚本、输出都放入新目录；
- 保存所有 CSV、图、日志、MD；
- 严格节省算力和内存。

## 输出目录建议

```text
DSSAT_auto_validation/multisite_new_cultivar_forward_screening_013_01/
```

## 最终判断

本轮结束后给出：

1. FQ/YC 输入是否可用于 PDI/gym-DSSAT；
2. 哪些站点年份有明显水氮优化空间；
3. 哪些年份适合后续 DQN；
4. 哪些年份不建议训练；
5. 是否需要先做固定水氮扫描。
