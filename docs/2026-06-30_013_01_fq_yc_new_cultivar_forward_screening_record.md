# 013_01 封丘/禹城新参数前向筛选记录

## 目的

本轮只做前向模拟，不训练 RL。目标是检查 FQ/YC 新品种参数和输入包是否能在 PDI/gym-DSSAT 4.8.0 中正常运行，并初步判断校准年份是否存在水氮管理优化空间。

输入目录：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC
```

输出目录：

```text
DSSAT_auto_validation/multisite_new_cultivar_forward_screening_013_01
```

脚本：

```text
src/run_fq_yc_new_cultivar_forward_screening_013_01.py
```

## 运行状态

| 站点 | 年份 | null | recorded | dssat_auto | 备注 |
|---|---:|---|---|---|---|
| FQ | 2007 | 成功 | 成功 | 成功 | 可用 |
| FQ | 2008 | 超时 | 超时 | 超时 | 暂不纳入判断，需单独诊断 |
| FQ | 2010 | 成功 | 成功 | 成功 | 可用 |
| YC | 2008 | 成功 | 成功 | 成功 | 可用 |
| YC | 2014 | 成功 | 成功 | 成功 | 可用 |

说明：FQ2008 三个情景都在 240 秒内未完成，可能与该 treatment/天气/生长终止条件有关。本轮先不继续消耗算力排查。

## 重要解析说明

本轮同时读取了 `Summary.OUT` 和 `PlantGro.OUT`。由于 `Summary.OUT` 中部分列在当前解析下与日值最终结果不一致，本轮主要判断使用 `PlantGro.OUT` 的最终日值：

- `final_gwad_daily` 作为籽粒产量；
- `final_cwad_daily` 作为地上部生物量；
- `max_wspd` / `max_nstd` 作为水分/氮胁迫最大值。

管理事件来自 `MgmtEvent.OUT`，已对重复事件去重，并支持 `kg[N]/ha` 施肥单位。

## 核心结果（日值最终值）

| 站点 | 年份 | 情景 | GWAD kg/ha | CWAD kg/ha | 灌溉 mm | 施氮 kg/ha | max WSPD | max NSTD | final DAP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| FQ | 2007 | null | 8109 | 14220 | 0 | 0 | 0.000 | 0.012 | 105 |
| FQ | 2007 | recorded | 8071 | 14118 | 75 | 165 | 0.000 | 0.012 | 105 |
| FQ | 2007 | dssat_auto | 8109 | 14220 | 19.6 | 0 | 0.000 | 0.012 | 105 |
| FQ | 2010 | null | 6646 | 12801 | 0 | 0 | 0.000 | 0.012 | 98 |
| FQ | 2010 | recorded | 6595 | 12690 | 100 | 144 | 0.000 | 0.012 | 98 |
| FQ | 2010 | dssat_auto | 6646 | 12803 | 38.0 | 0 | 0.000 | 0.012 | 98 |
| YC | 2008 | null | 7930 | 18131 | 0 | 0 | 0.000 | 0.293 | 106 |
| YC | 2008 | recorded | 8160 | 18772 | 120 | 303 | 0.000 | 0.013 | 106 |
| YC | 2008 | dssat_auto | 7930 | 18131 | 0 | 0 | 0.000 | 0.293 | 106 |
| YC | 2014 | null | 7825 | 17996 | 0 | 0 | 0.922 | 0.381 | 104 |
| YC | 2014 | recorded | 9418 | 20514 | 120 | 374 | 0.000 | 0.013 | 104 |
| YC | 2014 | dssat_auto | 8713 | 18945 | 86.5 | 0 | 0.000 | 0.436 | 104 |

## 初步判断

### FQ 2007 / 2010

FQ 两个成功年份都显示：

- null 已经接近或略高于 recorded；
- 水分胁迫 max WSPD = 0；
- 氮胁迫 max NSTD 仅约 0.012；
- recorded 施肥/灌溉没有带来增产，反而略低。

因此，FQ2007 和 FQ2010 不适合作为下一步 RL 优化年份。至少在当前 IC 和新参数下，它们缺少明显水氮优化空间。

### YC 2008

YC2008 显示：

- null 产量 7930；
- recorded 产量 8160，增加约 230 kg/ha；
- recorded 明显降低氮胁迫：max NSTD 从 0.293 降到 0.013；
- 水分胁迫为 0，主要不是水分限制。

因此，YC2008 有一定氮管理响应，但增产幅度中等，且 recorded 使用 303 kg/ha 氮，投入较高。可以作为次级候选，但不是最优候选。

### YC 2014

YC2014 是本轮最有价值的候选：

- null 产量 7825；
- recorded 产量 9418，增加约 1593 kg/ha；
- null 水分胁迫 max WSPD = 0.922，氮胁迫 max NSTD = 0.381；
- recorded 同时把水分和氮胁迫都降到很低；
- dssat_auto 只自动灌溉 86.5 mm、不施氮，产量 8713，高于 null 但低于 recorded。

这说明 YC2014 同时存在明显水分和氮素管理空间，适合作为下一步算法/固定扫描候选年份。

## 关于 DSSAT auto

本轮再次观察到：

- 自动灌溉可以触发，尤其 YC2014；
- 自动施肥仍然没有触发；
- auto-N 不触发不影响本轮筛选目的，只记录为 DSSAT 原生自动施肥链路问题。

## 输出文件

```text
DSSAT_auto_validation/multisite_new_cultivar_forward_screening_013_01/013_01_run_status.csv
DSSAT_auto_validation/multisite_new_cultivar_forward_screening_013_01/013_01_fq_yc_forward_summary.csv
DSSAT_auto_validation/multisite_new_cultivar_forward_screening_013_01/013_01_fq_yc_forward_daily.csv
DSSAT_auto_validation/multisite_new_cultivar_forward_screening_013_01/013_01_fq_yc_forward_events.csv
DSSAT_auto_validation/multisite_new_cultivar_forward_screening_013_01/figures/
```

已生成过程图：

```text
FQ_2007_forward_process.png
FQ_2010_forward_process.png
YC_2008_forward_process.png
YC_2014_forward_process.png
```

## 下一步建议

不建议马上训练。建议下一步做：

```text
013_02 YC2014 固定水氮扫描
```

目的：

1. 确认 YC2014 的产量响应面；
2. 判断 recorded 的 120 mm / 374 kg N 是否明显过量；
3. 找到低氮/适量灌溉是否也能接近 recorded 产量；
4. 如果存在类似 HLA2015 的“少氮高产”空间，再考虑 DQN。

FQ2008 可以之后单独诊断，但不应阻塞 YC2014 的下一步。
