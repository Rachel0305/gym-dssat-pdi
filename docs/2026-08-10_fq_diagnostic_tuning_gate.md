# FQ originIC PPO 诊断与单因素调参闸门（2026-08-10）

## 结论

FQ/FQA 当前的弱势不是 100K checkpoint 才出现的后期坍缩：25K--100K 的平均产量只在 7,310.8--7,317.0 kg/ha 间变动，而每个 checkpoint 的平均投入均为 219 mm 灌溉和 240 kg N/ha。策略持续采用高投入、低氮效率的固定式模式，且动作对覆盖从 25K 的 5 对收缩到 100K 的 3 对。故**值得做一次、且仅做一次以季节氮硬上限为唯一变化的 FQ 2K smoke**；不建议当前进行奖励、观察、天气、动作网格或 PPO 超参数的多因素扫描，也不授权 100K 训练。

推荐的唯一第一因素是将 PPO 的季节总氮设为 **160 kg/ha 硬上限**（保留现有 `[0,15,30,45] x [0,40,80,120]` 动作网格、原始观测、奖励、seed、训练/验证年份和灌溉安全逻辑）。这直接检验 PPO 是否只是依赖 240 kg/ha 的早期氮投入；比先调 reward 更容易保持与冻结主线的可归因性。它不是为了强行令 PPO 获胜。

## 已核对的正式链路与数据来源

| 用途 | 已核对来源 |
|---|---|
| PPO 正式训练 | `benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo/051_00_formal_result.json`；FQA/FQ、originIC、2005--2013 训练、2014--2023 验证、16 动作、100K |
| checkpoint 汇总 | `benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo/evaluation/051_00_validation_summary_by_station_checkpoint.csv` |
| 动作审计 | `benchmark_results/051_00_fqa_originIC_expanded_action_maskableppo/audits/051_00_formal_action_audit.csv` |
| 五情景正式结果 | `benchmark_results/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/051_03_fqa_five_scenario_season_summary.csv`、`051_03_result.json` |
| 原有实验说明 | `docs/051_00_fqa_originIC_expanded_action_maskableppo_record.md`、`docs/051_03_fqa_originIC_051_00_fqa_originIC_expanded_action_maskableppo_five_scenario_figures_ckpt100000_record.md` |

正式训练记录表明 2K 原始 smoke 已通过，且正式 100K 动作传输、网格合法性和非 DAP1 正动作检查均通过。本次诊断没有运行新的 smoke 或训练。

## 逐 checkpoint 比较

数值是 2014--2023 十年平均；`I/N 事件`为平均有正灌溉/正施氮事件数，`正动作`为每天记录中任一正投入动作的平均行数。动作对覆盖为十年并集，分母为声明的 16 个组合动作。`WP_ET` 没有被 051_00 的 checkpoint 汇总或日记录导出，且 25K/50K/75K 没有同口径 snapshot/ETCP 文件；不能用 100K 的 WP_ET 反推，故如实标为未提供。100K 的 WP_ET 来自独立、完整五情景季节表。

| checkpoint | 产量 kg/ha | WP_ET kg/m3 | PFP_N kg/kg | 灌溉 mm | N kg/ha | I/N 事件 | 正动作 | 动作对覆盖 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 25K | 7,310.8 | 未提供 | 30.46 | 219.0 | 240.0 | 12.5 / 3.0 | 13.5 | 5/16：I0/N40、I15/N0、I15/N120、I30/N0、I45/N80 |
| 50K | 7,313.9 | 未提供 | 30.47 | 219.0 | 240.0 | 11.6 / 4.0 | 13.6 | 4/16：I0/N40、I15/N0、I30/N80、I45/N80 |
| 75K | 7,317.0 | 未提供 | 30.49 | 219.0 | 240.0 | 12.6 / 4.0 | 15.6 | 4/16：I0/N40、I0/N80、I15/N0、I45/N80 |
| 100K | 7,315.4 | 1.996 | 30.48 | 219.0 | 240.0 | 12.6 / 3.0 | 14.6 | 3/16：I0/N80、I15/N0、I45/N80 |

备注：100K 五情景表中的 PPO 数值四舍五入后为 7,315.3 kg/ha、219.0 mm、240.0 kg N/ha、WP_ET 1.996、PFP_N 33.867；PFP_N 与 checkpoint 评估表略有口径差异，前者应作为论文五情景报告值。

## 高投入与 Pareto 诊断

1. 所有 checkpoint 总氮均恰为 240 kg/ha；25K--100K 无任何下降，氮全部集中在 DAP 1--30。灌溉亦稳定在 219 mm，分阶段平均为 DAP 1--30：75 mm、31--60：75 mm、61--90：43.5 mm、91+：25.5 mm。
2. 100K 五情景的十年均值中，PPO 的产量 7,315.3 kg/ha 仅略高于官方专家 7,311.9 和农户模板 7,201.7，却使用 240 kg N/ha；农户模板以 144 kg N/ha 获得 PFP_N 55.578，而 PPO 为 33.867。PPO 的 WP_ET 1.996 亦低于四个非 PPO 情景中的最高均值 2.145（农户模板）。
3. 年度比较中，100K PPO 相对其他四情景最优值只在 3/10 年产量、1/10 年 WP_ET、0/10 年 PFP_N 胜出。故没有“以额外投入换取明确产量收益”的证据。
4. 051_00 的已存安全配置显示灌溉季节上限为 240 mm、N 软限制为 250 kg/ha；PPO 的灌溉已接近该上限、N 接近该软限制。正式日记录的季节累计 N 为 240 kg/ha，但现有归档不足以证明 240 是 N 的硬掩码上限；这里的结论是“长期贴近高投入边界”，而非声称存在已核实的 240 kg/ha N 硬上限。

这是一项基于已有结果的静态 Pareto 诊断；没有执行新的 fixed-action/action-grid 仿真，以避免在没有已配置的、仅改一个因素的 FQ 执行清单时产生不可追溯的额外运行。

## 2018 年零产量：共同站点--年份/DSSAT异常，不归因于 PPO

2018 年五情景的结果均为零粒重，但仍有约 3,804--3,835 kg/ha 生物量，管理投入彼此不同：null 0/0、DSSAT auto external-N 57/0、农户模板 75/144、官方专家 158/212、PPO 180/240（单位依次为 mm/kg N ha-1）。因此不是 PPO 的动作失败。

五个 2018 `Summary.OUT` 的作物/气象/土壤来源都写为 `CNFQ1801`、`FQ99001200`，播种 DOY 162、成熟/收获 DOY 240；五个 `WARNING.OUT` 均记录：

- `MZ_GRO ... JD 238 due to slowed grain filling`；
- JD 240 的 `CYCRDin`、`CXCRDin`、`CELEVin` FIELD 变量转移错误并被置零；
- 同一层 7 土壤有机质 C:N=9.0 警告。

对应文件位于五情景各自的 `snapshots/FQA/2018/*/{Summary,WARNING,PlantGro}.OUT`。该共同终止/变量传递警告与管理无关，足以把 2018 标为**共享的 DSSAT/站点--年份输入或运行异常**，不得计为 PPO 劣势证据。更深的根因（WTH、MZX 作物历还是 DSSAT FIELD 接口）尚未通过重新渲染的单年独立复现定位；在论文中应预注册为敏感性分析：主结果同时报告包含与排除 2018 的版本，并明确排除是共同异常而非按算法删样本。

## 单因素 2K smoke 闸门

只允许创建新的、隔离的 FQ 运行目录；不得覆盖 051_00、既有 reward、wrapper、输入、基线或冻结结果。训练前必须把以下合同写入 manifest：唯一变化为 `season_n_hard_cap_kg_ha: 160`，其余 observation、reward、action grid、seed、input root、训练/验证年份、灌溉安全、checkpoint 节点均与 051_00 一致。

2K 通过条件：

1. 使用项目 WSL/container 的 gym 环境，单进程/单环境执行；保留运行日志和内存异常记录。若出现 OOM、DSSAT 运行失败或 renderer/input provenance 不一致，立即停止，不重试长训练。
2. manifest、渲染输入和日记录证明只有该 N 硬上限变化；全部施氮动作仍在原 16 动作网格，累计 N 从不超过 160 kg/ha，且请求--安全--DSSAT 传输不匹配为 0。
3. 验证回放中须存在至少 3 个非零动作对、至少一个正动作在 DAP1 之后，且不退化为每年相同的单一 DAP1 动作签名；同时报告每年总 I/N、事件数和动作对。
4. 2K 只检验机制，不用其产量宣称优越。只有上述传输和多样性都通过，才可另行批准一次 100K；若策略仍在所有验证年把 N 推到 160 kg/ha，或动作收缩为单一签名，则停止该分支，不再把 reward/天气/超参数叠加进去。

若未来 100K 获准，其成功判据需预注册为：相对于冻结 051_03 的 PPO，十年平均 N 明显降低且产量不恶化到低于农户模板，同时 WP_ET、PFP_N 至少有一项改善；2018 要作为共同异常单独处理，而不用于选择 checkpoint 或调参。

