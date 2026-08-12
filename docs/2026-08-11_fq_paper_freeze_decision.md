# FQ originIC：论文结果冻结与 2018 敏感性决定（2026-08-11）

## 决定

**冻结 `051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000` 作为 FQ 的正式、可复现实验结果。**它应在论文中作为跨站 transfer 的“PPO 没有形成明确综合优势”的站点案例，而不是作为 PPO 成功站点或继续为了让 PPO 获胜而调参的起点。

冻结含义是固定 051_03 的 100K checkpoint、五情景基线、originIC 输入和统计口径；不更改 `recorded_farmer_template`，不修改 reward/observation/weather/动作网格，不进行新的 100K。2018 同时保留在原始十年结果中，并预先报告“剔除 2018”的共同异常敏感性表，不能在选择 checkpoint 或比较算法时临时删年。

## 数据范围与来源

- 正式五情景季节表：`benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/051_03_fqa_five_scenario_season_summary.csv`。
- 100K 动作审计：`benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo/audits/051_00_formal_action_audit.csv`。
- 160-cap 隔离 2K：`benchmark_results/062_00_fqa_originIC_expanded_action_maskableppo_ncap160_smoke_smoke2k/062_00_ncap160_gate.json`，manifest 和有效覆盖记录位于同目录。
- 2018 DSSAT 证据：051_03 PPO、051_02 四基线和 051_01 auto-N 的各自 `snapshots/FQA/2018/*/{Summary,WARNING,PlantGro}.OUT`。

所有表中单位为：产量 kg/ha、灌溉 mm、N kg/ha、`WP_ET` kg/m3、`PFP_N` kg/kg。N=0 时 `PFP_N` 为 N/A，均值不以零替代。

## 051_03 正式结果：含 2018 与剔除 2018

下表直接由同一份季节表按年份求平均；“剔除2018”仅删除每个情景的 2018 行，未改变其他年、情景或指标定义。

| 统计范围 / 情景 | 产量 | WP_ET | PFP_N | 灌溉 | N |
|---|---:|---:|---:|---:|---:|
| 含2018 / DSSAT auto external-N | 7178.2 | 2.125 | N/A | 34.7 | 0.0 |
| 含2018 / 官方专家 | 7311.9 | 2.079 | 33.156 | 212.9 | 241.7 |
| 含2018 / 农户模板 | 7201.7 | 2.145 | 55.578 | 75.0 | 144.0 |
| 含2018 / PPO | 7315.3 | 1.996 | 33.867 | 219.0 | 240.0 |
| 剔除2018 / DSSAT auto external-N | 7975.8 | 2.361 | N/A | 32.2 | 0.0 |
| 剔除2018 / 官方专家 | 8124.3 | 2.310 | 33.156 | 219.0 | 245.0 |
| 剔除2018 / 农户模板 | 8001.9 | 2.383 | 55.578 | 75.0 | 144.0 |
| 剔除2018 / PPO | 8128.1 | 2.218 | 33.867 | 223.3 | 240.0 |

相对于同年其余四情景的指标最大值，PPO 的比较不因剔除异常年而反转：

| 范围 | PPO 产量均值差 / 胜出年数 | PPO WP_ET 均值差 / 胜出年数 | PPO PFP_N 均值差 / 胜出年数 |
|---|---:|---:|---:|
| 含2018（10年） | -65.6 / 3胜、1平 | -0.195 / 1胜、1平 | -21.711 / 0胜（9个有效N年） |
| 剔除2018（9年） | -72.9 / 3胜、0平 | -0.217 / 1胜、0平 | -21.711 / 0胜 |

因此，2018 会拉低全部情景的绝对均值，但不是造成 FQ PPO 相对弱势的原因。论文应同时给出两种范围，正文以包含全部十年为主，补充材料给出剔除2018敏感性。

## 2018 的共同异常证据与表述边界

五情景 2018 产量均为 0，但生物量约 3804--3835 kg/ha，管理投入从 null 的 0/0 到 PPO 的 180 mm/240 kg N/ha 不等。这排除“PPO 未执行动作”作为零产量主因。

五个 `Summary.OUT` 均使用 `CNFQ1801`、`FQ99001200`，播种 DOY 162、成熟/收获 DOY 240；各 `WARNING.OUT` 均出现 JD238 “slowed grain filling”，以及 JD240 的 `CYCRDin`、`CXCRDin`、`CELEVin` FIELD 变量传递错误并置零。可称其为**共享的 DSSAT/站点—年份运行或输入异常**，不应归咎于 PPO。

根因尚未通过重新渲染的单年独立复现精确定位到 WTH、MZX 作物历或 FIELD 接口；论文不得声称已找到根因，也不得把剔除2018解释为对 PPO 的有利筛选。

## 160-cap 2K 分支：停止，不并入正式比较

`062_00` 是隔离的机制 smoke，不是可与 100K 051_03 对等的最终模型。唯一改变为季节 N safety cap `250 -> 160 kg/ha`；其余合同已由 manifest/dry-run 核验一致。

- N cap 与传输机制通过：最大累计 N=160，request--safe 差异=0，safe--DSSAT mismatch=0。
- 存在 4 个非零动作对、96 个 DAP1 后正动作。
- 但 checkpoint 2000 的十个验证年 `cross_year_distinct_signatures=1`：每年均为 `I0/N40; I15/N0; I30/N0; I45/N0`。
- 每年均把 N 用至 160 kg/ha；2K 仅机制证据，不导出 WP_ET，不能宣称最终性能改善。

故 062 在预注册的“不能退化为跨年单一签名”门槛失败，`next_step_allowed=false`，停止分支且不批准 100K。它不改变或替代 051_03 的论文主冻结结果。

## 论文建议表述

建议以如下边界陈述 FQ：

> 在 FQ originIC 十年验证中，100K PPO 的平均产量与官方专家接近（7315 vs 7312 kg ha-1），但以更高的资源投入运行（219 mm 灌溉、240 kg N ha-1），未优于农户模板的水分生产率（1.996 vs 2.145 kg m-3）或氮肥偏生产力（33.867 vs 55.578 kg kg-1）。PPO 仅在 3/10 年的产量和 1/10 年的 WP_ET 上超过其余情景的年度最优值，未在 PFP_N 上胜出。

同时描述行为而不夸大自适应性：

> 100K PPO 在十个验证年均使用相同的三个非零剂量组合（I0/N80、I15/N0、I45/N80），且正动作多数发生在 DAP1 之后并已通过 DSSAT 传输审计；然而其剂量组合跨年未发生变化，结合长期 240 kg N ha-1 投入，表明该站点尚未形成可证明的资源高效跨年适应策略。

结果部分应紧随 2018 注记：

> 2018 年所有管理情景均为零粒重并共享 DSSAT 运行警告；因此该年被保留在主分析中，并以预先定义的剔除2018敏感性分析检验结论稳健性。两种统计范围下，PPO 均未显示综合效率优势。

