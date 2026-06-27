# gym-DSSAT / PDI 与 Windows DSSAT 模拟和数据传输验证总结

日期：2026-06-27  
项目目录：`C:\Users\DELL\gym_workspace\gym_dssat_pdi_bingo\gym-dssat-pdi`

## 1. 一句话结论

经过 HLA（海伦）、FQZ（封丘）、YC（禹城）和 UFGA 官方示例的多轮对比，当前证据支持：

1. **gym-DSSAT wrapper 的 post-state 读取链路没有发现数据传输错误**：gym 读到的 `swfac/nstres/topwt/grnwt` 与 PDI/DSSAT 原始输出一致。
2. **PDI/gym DSSAT 4.8.0 与 Windows DSSAT 4.8.0 在相同输入下基本一致**：HLA、FQZ、YC 的 IC=0 null 逐年对比中，`WSPD/NSTD` 差异为 0 或接近 0，`CWAD/GWAD` 只出现 0–2 kg/ha 量级差异。
3. **HLA 2004 IC=1 的异常不是 gym 传输问题**：Windows DSSAT 4.8.0 与 PDI/gym DSSAT 4.8.0 对 `CNHL0408`（自动灌溉）和 `CNHL0409`（无灌溉无施肥）给出完全一致的日值、产量和胁迫结果。
4. **DSSAT 胁迫指数方向在当前输出中应解释为：数值越大，胁迫越强；接近 0 表示胁迫低或无胁迫。**
5. **HLA 2004 null 早熟零产是情景本身导致的**：IC=1 + 无灌溉 + 无施肥 + 生育期前段无雨时，Windows DSSAT 4.8.0 和 PDI/gym 都模拟出早熟、籽粒产量 0；自动灌溉后 WSPD 被压低并正常高产。

## 2. 为什么做这一系列验证

导师质疑的核心问题有两个：

1. **DSSAT 模拟本身是否合理？**  
   例如：无灌溉无施肥时是否应该出现水分/氮素胁迫；灌溉后胁迫指数是否应该响应。

2. **gym-DSSAT 与底层 DSSAT/PDI 之间的数据传输是否有问题？**  
   例如：gym 中的 `swfac/nstres` 是否真来自 DSSAT 原始输出，是否存在未反转、读取滞后、post-state 不同步等问题。

因此验证被拆成两条链：

- **链 A：raw → post-state 抽取链路验证**  
  检查 PDI/DSSAT 原始 `PlantGro.OUT` 与 gym wrapper 读取到的 post-state 是否一致。

- **链 B：Windows DSSAT raw vs PDI/gym DSSAT raw 对比**  
  在相同 MZX、天气、土壤、品种和管理条件下，对比 Windows DSSAT 4.8.0 输出与 PDI/gym DSSAT 4.8.0 输出。

## 3. CNHL040X 文件与验证用途

| 文件/目录 | 主要设置 | 验证目的 | 关键结果 | 结论 |
|---|---|---|---|---|
| `CNHL0401.MZX` / `run_CNHL0401` | HLA 2004，尝试 DSSAT 原生自动施氮 `FERTI=A` | 检查 DSSAT 自动施氮是否触发 | 自动施氮参数可读入，但未触发有效施氮 | 自动施氮链路在该配置下未成功触发 |
| `CNHL0402.MZX` / `run_CNHL0402` | HLA 2004，继续调整自动施氮指针 | 第二次排查 `FERTI=A` | 仍未成功触发自动施氮 | 原生自动施氮不作为主实验方案 |
| `CNHL0403.MZX` / `run_CNHL0403` | HLA 2004，`FERTI=R`，指定日期施肥阳性对照 | 验证 DSSAT 能否执行明确施肥事件 | `NI#M>0`、施肥量与指定值对应，产量和吸氮上升 | DSSAT 执行施肥事件本身正常，问题集中在自动触发逻辑 |
| `CNHL0404.MZX` / `run_CNHL0404` | HLA 2004，IC=1，Windows/XBuild 可打开的单年文件 | 初步测试 IC=1 与 Windows DSSAT 输出 | Windows 4.8.5 可完整成熟，HWAM 约 2038 kg/ha | 说明 IC=1 输入在 Windows 4.8.5 下可运行 |
| `CNHL0404` 的 gym/PDI IC=1 验证 | 同文件进入 PDI/gym | 检查 PDI/gym IC=1 是否一致 | PDI/gym 早熟、HWAM=0；后来发现与 Windows 4.8.0 null 结果一致 | 不能直接拿 Windows 4.8.5 和 PDI 4.8.0 混比 |
| `CNHL0405.MZX` / `run_CNHL0405(_IC0_null)` | HLA 2000–2023，多年 Windows DSSAT 结果 | 检查多年 null 胁迫指数是否普遍异常 | 多年胁迫与降雨/生长过程有关，不是单年偶然 | 促成后续 HLA 多年 Windows 4.8.0 vs PDI 对比 |
| `CNHL0406.MZX` / `run_CNHL0406_*` | HLA 2004 IC=0 null，Windows 4.8.0 与 PDI 4.8.0 对比 | 控制 DSSAT 版本，判断 PDI/gym 是否可信 | Windows 4.8.0 与 PDI/gym 基本一致 | 排除 gym/PDI 在 HLA 2004 IC=0 null 下的传输/模型差异 |
| `CNHL0407.MZX` / `run_CNHL0407_DSSAT480_IC0_null_2004_2023` | HLA 2004–2023，Windows DSSAT 4.8.0，IC=0 null | 多年版本控制对比 | HLA 多年 Windows 4.8.0 与 PDI/gym IC=0 null 对比图和表生成 | IC=0 多年层面未发现 gym/PDI 系统性传输错误 |
| `CNHL0408.MZX` / `run_CNHL0408_DSSAT480_2004` | HLA 2004，IC=1，自动灌溉 `IRRIG=A`，无实际施氮 | 检查自动灌溉是否能降低 WSPD，并与 PDI/gym 对比 | Windows 与 PDI/gym 完全一致；最终 GWAD=8492；灌溉总量 377.3 mm；WSPD 差异 0 | 自动灌溉有效，gym/PDI 与 Windows 4.8.0 一致 |
| `CNHL0409.MZX` / `run_CNHL0409_DSSAT480_2004` | HLA 2004，IC=1，`IRRIG=N`，`FERTI=N` | 检查无灌溉无施肥的 IC=1 null 情景 | Windows 与 PDI/gym 完全一致；最终 GWAD=0；灌溉 0；WSPD 差异 0 | 早熟零产是 DSSAT 4.8.0 在该情景下的模拟结果，不是 gym 错误 |

## 4. HLA 0408/0409 的关键数值

对比文件：

- `DSSAT_auto_validation/HLA_2004/analysis_hla0408_0409_windows480_vs_pdi_ic1/summary_windows480_vs_pdi_gym_hla0408_0409_ic1.csv`
- 生成脚本：`src/compare_hla0408_0409_windows480_vs_pdi_ic1.py`

| 情景 | Windows final GWAD | PDI/gym final GWAD | max abs diff WSPD | max abs diff NSTD | Windows irrigation | PDI/gym irrigation |
|---|---:|---:|---:|---:|---:|---:|
| `CNHL0408_IC1_auto_irrig` | 8492 | 8492 | 0.0 | 0.0 | 377.3 mm | 377.3 mm |
| `CNHL0409_IC1_null` | 0 | 0 | 0.0 | 0.0 | 0.0 mm | 0.0 mm |

这组对比是目前最关键证据：

- 同一 MZX、同一天气、同一土壤、同一品种、同一 DSSAT 4.8.0 版本；
- Windows DSSAT 与 PDI/gym 逐日 `WSPD/NSTD/CWAD/GWAD` 完全一致；
- 因此 HLA 2004 IC=1 下的早熟零产不能归因于 gym wrapper 传输错误。

## 5. HLA IC=1 多年诊断

对比文件：

- `DSSAT_auto_validation/HLA_2004/hla_ic1_yearly_diagnostics_2004_2023/hla_ic1_yearly_summary.csv`
- 生成脚本：`src/run_hla_ic1_yearly_diagnostics.py`

设置：

- `auto_irrig`：IC=1 + 自动灌溉 + 无实际施氮；
- `null`：IC=1 + 无灌溉 + 无施肥；
- 注意：该诊断把 2004 年初始土壤水氮剖面平移到 2004–2023 各年份，因此是诊断用途，不是最终正式论文设置。

结果摘要：

- 40 个单年模拟全部成功，无超时；
- 2004 年 `null`：final GWAD=0，final DAP=108，max WSPD=1.0；
- 2004 年 `auto_irrig`：final GWAD=8492，irrigation=377.3 mm，max WSPD=0.0；
- 2005 年 `null`：final GWAD=7310，rain=481.8 mm，irrigation=0，max WSPD=0.0；
- 这说明 2004 年 null 早熟零产并不是 IC=1 必然异常，而是与当年有效生育期内无雨/水分限制有关。

## 6. FQZ / YC / UFGA 的独立验证

### 6.1 封丘 FQZ

目录：

- `DSSAT_auto_validation/HLA_2004/run_CNFQ0802_DSSAT480_standalone/run_CNFQ0802_DSSAT480_standalone/analysis_windows480_vs_pdi_yearly_ic0`

脚本：

- `src/compare_fqz_windows480_vs_pdi_yearly_ic0.py`
- `src/plot_yc_fqz_rain_stress_yield.py`

结果：

- 2007–2023 单年拆分对比成功；
- `max_abs_WSPD_diff=0.0`；
- `max_abs_NSTD_diff` 最大约 0.001；
- `CWAD/GWAD` 最大差异约 0–2 kg/ha。

结论：

- FQZ IC=0 null 下，Windows DSSAT 4.8.0 与 PDI/gym DSSAT 4.8.0 基本一致。

### 6.2 禹城 YC

目录：

- `DSSAT_auto_validation/run_CNYC0802_DSSAT480_IC0_null_2000_2023/analysis_windows480_vs_pdi_yearly_ic0`

脚本：

- `src/compare_yc_windows480_vs_pdi_yearly_ic0.py`
- `src/plot_yc_fqz_rain_stress_yield.py`

结果：

- 2008–2023 单年拆分对比成功；
- `max_abs_WSPD_diff=0.0`；
- `max_abs_NSTD_diff` 最大约 0.001；
- `CWAD/GWAD` 最大差异约 0–1 kg/ha。

结论：

- YC IC=0 null 下，Windows DSSAT 4.8.0 与 PDI/gym DSSAT 4.8.0 基本一致。

### 6.3 UFGA 官方示例

目录：

- `DSSAT_auto_validation/UFGA_Windows480_vs_gym_pdi/analysis_ufga_windows480_vs_pdi_with_rain`

脚本：

- `src/analyze_ufga_windows480_vs_pdi.py`
- `src/plot_ufga_management_stress.py`

结果：

- 官方示例 6 个 treatment 中，灌溉处理 run 3/4 的 WSPD 全程为 0；
- 雨养处理出现 WSPD 波动；
- 高氮处理产量高于低氮处理；
- Windows DSSAT 4.8.0 与 PDI/gym 4.8.0 胁迫指数基本一致。

结论：

- DSSAT 模型本身对灌溉、施肥、雨养差异有合理响应；
- “WSPD=0”在有灌溉且水分充足时是合理结果。

## 7. raw → post-state 抽取链路验证

相关脚本：

- `src/demo_gym_pdi_state_flow.py`
- `src/diagnose_hla2004_raw_post_state.py`
- `src/diagnose_pdi_field_coordinates_ic0_null.py`

验证逻辑：

1. 运行 gym-DSSAT 环境；
2. 保存 PDI 临时运行目录快照；
3. 读取 PDI/DSSAT 原始输出，例如 `PlantGro.OUT`；
4. 同时保存 gym wrapper 每一步读取到的 post-state；
5. 对比原始 `PlantGro.OUT` 与 gym post-state 的 `topwt/grnwt/swfac/nstres`。

结论：

- 在已验证情景下，gym post-state 与 PDI/DSSAT 原始输出一致；
- 未发现 wrapper 读取滞后或变量未更新问题；
- 因此当前主要问题不在 gym wrapper 的数据传输。

## 8. 胁迫指数解释

本轮验证后，对当前分析图和 CSV 采用如下解释：

- `WSPD` / gym 中对应 `swfac`：水分胁迫指数；
- `NSTD` / gym 中对应 `nstres`：氮素胁迫指数；
- 在当前输出与图表中，**数值越大表示胁迫越强，接近 0 表示胁迫低或无胁迫**。

证据：

- `CNHL0408` 自动灌溉后 WSPD 全程 0，产量高；
- `CNHL0409` 无灌溉时 WSPD 最高到 1，早熟零产；
- UFGA 官方示例中，灌溉 treatment 的 WSPD 为 0，雨养 treatment 出现 WSPD 波动。

## 9. 当前可以向导师汇报的结论

可以较稳妥地说：

1. 已经做了多站点、多年份、官方示例和 HLA IC=1 的控制验证；
2. Windows DSSAT 4.8.0 与 PDI/gym DSSAT 4.8.0 在相同输入下基本一致；
3. gym wrapper 的 post-state 读取链路没有发现传输错误；
4. HLA 2004 null 早熟零产不是 gym 错误，而是该天气/管理/初始条件组合下 DSSAT 4.8.0 的模拟结果；
5. 自动灌溉能明显降低 WSPD 并提高产量，说明模型对水分管理有合理响应；
6. 未来若要正式使用 IC=1，需要为每个站点/年份提供真实可解释的初始土壤水氮剖面，而不能简单把 HLA 2004 的 IC 剖面平移到所有年份。

## 10. 仍需注意的限制

1. HLA IC=1 多年诊断使用的是 2004 初始土壤剖面平移，不是逐年真实 IC 数据。
2. FQZ、YC 目前没有确认真实可用的 IC=1 初始条件剖面，因此不建议直接硬改为 IC=1。
3. `DSSAT_auto_validation/` 下的大量原始输出没有纳入 Git 提交，主要保留在本地；GitHub 备份的是报告和复现实验脚本。
4. 当前验证的是模型与传输一致性，不等同于证明 PPO 策略已经最优。

## 11. 备份和恢复说明

本次整理前，旧工作区脏状态已保存为 Git stash：

```text
stash@{0}: backup before gym-dssat validation report cleanup 2026-06-27
```

如需恢复旧脏状态，可先确认当前工作区无重要未提交改动，再执行：

```bash
git stash show --stat stash@{0}
git stash apply stash@{0}
```

本次建议提交到 GitHub 的内容仅包括：

- 本报告；
- gym-DSSAT / Windows DSSAT 对比脚本；
- 绘图和 post-state 诊断脚本。

