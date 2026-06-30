# HLA 2004-2023 IC=0 null 与 IC=1 null 并排比较记录

## 1. 本轮问题

用户检查 HLA 2004-2023 IC=1 结果后认为总体形态比 IC=0 更合理，但明确要求不要马上把全部实验切到 IC=1，而是先和 IC=0 多年结果并排比较，尤其关注 HLA 2004、2012 等异常低产年份，判断 IC=1 是否只是改变了初始土壤水氮，还是已经改变了研究情景。

本轮只做离线分析：

- 不训练 PPO；
- 不重新运行 DSSAT；
- 只读取已有 IC=0 和 IC=1 null 输出；
- 生成年度汇总表、日值表和对比图。

## 2. 输入数据

IC=0 对照：

- 路径：`DSSAT_auto_validation/HLA_2004/run_CNHL0407_DSSAT480_IC0_null_2004_2023/`
- 模型/环境：Windows DSSAT 4.8.0
- 情景：HLA 2004-2023，IC=0，null，无灌溉、无施肥

IC=1 对照：

- 路径：`DSSAT_auto_validation/HLA_2004/hla_ic1_yearly_diagnostics_2004_2023/null/`
- 模型/环境：PDI/gym DSSAT 4.8.0
- 情景：将 2004 年初始土壤水氮剖面平移到 2004-2023 每一年，null，无灌溉、无施肥

注意：此前已经验证过 Windows DSSAT 4.8.0 与 PDI/gym DSSAT 4.8.0 在相同输入下基本一致，因此本轮重点不是模型传输问题，而是 IC=0 与 IC=1 初始条件情景差异。

## 3. 新增脚本与输出

新增脚本：

- `src/compare_hla_ic0_ic1_null_yearly.py`

输出目录：

- `DSSAT_auto_validation/HLA_2004/analysis_hla_ic0_vs_ic1_null_2004_2023/`

主要输出：

- `hla_ic0_vs_ic1_null_summary_wide_2004_2023.csv`
- `hla_ic0_vs_ic1_null_summary_long_2004_2023.csv`
- `hla_ic0_vs_ic1_null_daily_values_2004_2023.csv`
- `hla_ic0_vs_ic1_null_yield_2004_2023.png`
- `hla_2004_ic0_vs_ic1_null_daily_stress_yield.png`
- `hla_2012_ic0_vs_ic1_null_daily_stress_yield.png`

说明：IC=1 的部分 `Summary.OUT` 存在空白固定宽度字段，直接按空格解析容易错位。因此本轮年度产量/生物量汇总以 `PlantGro.OUT` 的最终日值状态为准，避免 Summary 解析错位。

## 4. 年度产量对比

年度产量对比显示，IC=1 不是对 IC=0 的小幅修正，而是大幅改变了 null 情景的产量水平。

主要观察：

- IC=0 null 下，HLA 2004-2023 多数年份籽粒产量只有几百 kg/ha。
- IC=1 null 下，除 2004 和 2012 外，多数年份籽粒产量达到约 5-8 t/ha。
- 这说明 2004 初始剖面平移后，初始水氮供给强到足以让无管理情景也产生较高产量。

关键异常年份：

| 年份 | IC=0 籽粒产量 HWAM | IC=1 籽粒产量 HWAM | IC=0 生物量 CWAM | IC=1 生物量 CWAM | 解释 |
|---|---:|---:|---:|---:|---|
| 2004 | 412 kg/ha | 0 kg/ha | 757 kg/ha | 4486 kg/ha | IC=1 下植株生长显著增强，但没有形成籽粒，低产机制从“弱生长/缺素”变成“强营养生长但零籽粒”。 |
| 2012 | 0 kg/ha | 0 kg/ha | 93 kg/ha | 1645 kg/ha | IC=1 仍未形成籽粒，但生物量明显高于 IC=0。 |

全年图：

- `DSSAT_auto_validation/HLA_2004/analysis_hla_ic0_vs_ic1_null_2004_2023/hla_ic0_vs_ic1_null_yield_2004_2023.png`

## 5. HLA 2004 与 2012 的过程差异

HLA 2004：

- IC=0：NSTD 较高，籽粒最终约 412 kg/ha。
- IC=1：前中期 WSPD 出现强水分胁迫，NSTD 接近 0，生物量达到 4486 kg/ha，但籽粒产量为 0。
- 解释：IC=1 并不是简单“修正氮胁迫”，而是把生长路径改成了更强营养生长、但未能形成籽粒的机制。

HLA 2012：

- IC=0 与 IC=1 籽粒产量均为 0。
- IC=1 生物量明显更高，说明初始剖面仍改变了作物生长潜力。
- 但两者都未形成籽粒，说明该年份异常低产不能只用初始水氮不足解释，还需要继续看物候、温度、辐射或生殖期过程。

过程图：

- `DSSAT_auto_validation/HLA_2004/analysis_hla_ic0_vs_ic1_null_2004_2023/hla_2004_ic0_vs_ic1_null_daily_stress_yield.png`
- `DSSAT_auto_validation/HLA_2004/analysis_hla_ic0_vs_ic1_null_2004_2023/hla_2012_ic0_vs_ic1_null_daily_stress_yield.png`

## 6. 当前判断

当前证据支持：

1. HLA 多年 IC=1 结果在“模型能正常响应初始剖面”这一点上是可解释的。
2. 但把 2004 初始土壤水氮剖面平移到 2004-2023 每一年，并不只是轻微改变初始水氮，而是显著改变了 null 情景的整体生产潜力。
3. 因此，IC=1 暂时不能直接替代全部主实验默认设置；它更适合作为一个新的“统一初始剖面敏感性情景”或“初始条件修正情景”来讨论。
4. 如果未来要正式切换到 IC=1，需要重新定义主实验问题：不再是原来的 IC=0 低初始供给背景，而是“给定统一初始水氮剖面下的水氮管理优化”。

## 7. 下一步建议

暂不把全部 PPO / 四情景实验直接切到 IC=1。

建议先做两件低成本核查：

1. 对 HLA 2004 和 2012 的 IC=1 null 做更细的过程解释：重点查看花期、成熟日期、LAI、生物量、WSPD/NSTD，解释为什么生物量增加但籽粒仍为 0。
2. 若导师认可“统一初始剖面情景”，再考虑只在 HLA 上先跑一套小规模 IC=1 四情景对照；不要立即扩展到所有站点。

