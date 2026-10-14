# HL/FQ 历史天气 PPO 8-seed baseline 进度与跨站点策略诊断

## 1. 任务目的与边界

本轮完成 HL seeds 0–7（8/8）与 FQ seeds 0–6（7/8）的历史天气 MaskablePPO 100K baseline，使用 2014–2023 十个独立验证年份。seed 0 复用已有正式结果；只对缺失 seed 运行正式训练与验证。FQ seed 7 因其 2K smoke action-response gate 未通过，被正式 runner 的 preflight 拒绝，未开始 100K 训练，因此本轮整体为 15/16 个 seed 正式结果。YC 仅读取现有 004_15 结果作描述性对照。未生成 WGEN 天气，未修改 YC、reward、action mask、动作空间、IC 或天气/训练年份配置。

比较 YC 时需注意：YC PPO 训练来源为 RANDOM_WEATHER_WGEN；本轮 HL/FQ 均为各站历史 .WTH 年份池。因此是跨站点、跨训练天气来源的描述性策略对照，不是严格同天气处理效应比较。

## 2. 配置审计

| Site | Seeds | PPO/步数/选点 | Reward | 动作空间 | 安全约束 | IC 与 MZX | 训练天气/年份 | 验证年份 | WGEN |
|---|---|---|---|---|---|---|---|---|---|
| HL | 0,1,2,3,4,5,6,7 | MaskablePPO / 100000 / fixed 100000 environment-step checkpoint | 040_28 reward_04026 minus 50*max(SWFAC_after-0.05,0)*reward_scale; unchanged across seed configs | 16 joint actions; I=[0,15,30,45] mm x N=[0,40,80,120] kg/ha | DAP<=30:75; DAP<=60:150; DAP<=90:195; season:240 mm; N season <=250 kg/ha; 间隔 7 days separately for repeat irrigation and repeat nitrogen applications | lowIC; soil `HL99001200`; cultivar `HY0006 Haiyu No006`; `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL/CNHL0701_corrected_IC123.MZX` | `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL`，2004-2013（WTH） | 2014-2023 | 否 |
| FQ | 0,1,2,3,4,5,6,7 | MaskablePPO / 100000 / fixed 100000 environment-step checkpoint | 040_28 reward_04026 minus 50*max(SWFAC_after-0.05,0)*reward_scale; unchanged across seed configs | 16 joint actions; I=[0,15,30,45] mm x N=[0,40,80,120] kg/ha | DAP<=30:75; DAP<=60:150; DAP<=90:195; season:240 mm; N season <=250 kg/ha; 间隔 7 days separately for repeat irrigation and repeat nitrogen applications | originIC; soil `FQ99001200`; cultivar `FQ0985 Zhengdan No985`; `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ/CNFQ0801.MZX` | `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/FQ`，2005-2013（WTH） | 2014-2023 | 否 |

HL 的土壤、品种、IC 来源锁定在 `CNHL0701_corrected_IC123.MZX` 与 lowIC profile；FQ 锁定在 `CNFQ0801.MZX` 与 originIC profile。逐年天气文件清单和 MZX SHA-256 在 `results/hl_fq_8seed_cross_site/analysis/configuration_audit.csv`。每个 seed 的 config snapshot、checkpoint 路径/hash、正式日志状态在 `run_metadata.json`。

配置 JSON 的 `reward_and_safety` 字段文字标记 `inherit_042_15_via_046_02_unchanged`；但本次实际 051/054 runner 的 import/delegation 链是 `042_10 -> 040_36 -> 040_28`。因此当前 reward 为 `reward_04026 - 50 × max(SWFAC_after - 0.05, 0) × reward_scale`；040_36 在继承的 staged/guardrail wrapper 上新增 DAP≤90 累计灌溉≤195 mm late-reserve mask。逐日输出 flag 与参数均用于核验生效状态。报告按实际执行代码与输出记录审计，未更改该声明或实际配置。

每日输出还核对 action level、mask 输出标记、DAP 30/60/90 与全季灌溉上限、全季 N 上限及分别计算的灌溉/施氮事件间隔。实际代码设定 I 与 N 各自至少 7 天。全量逐年结果在 `mask_and_closure_audit.csv`。
逐日 action trace 安全审计状态：FAIL（10 个 seed-year 行未通过）。其中事件间隔不符的逐年行数为 10；前若干行：HL seed5 2014 I=7d/N=6d; HL seed5 2015 I=7d/N=6d; HL seed5 2016 I=7d/N=6d; HL seed5 2017 I=7d/N=6d; HL seed5 2018 I=7d/N=6d; HL seed5 2019 I=7d/N=6d; HL seed5 2020 I=7d/N=6d; HL seed5 2021 I=7d/N=6d; HL seed5 2022 I=7d/N=6d; HL seed5 2023 I=7d/N=6d。HL seed 5 的所有十个验证年均在记录 DAP 上呈施氮间隔 6 天（DAP 2→8），尽管现行代码配置要求 7 天且日志记录动作未被裁剪；此项保留为实测配置/输出不一致，未修改 action mask，也不宣称其通过。
逐日日志存在 DAP 从 5 回到 1 的一次重置记录；完整 action hash 按 CSV 原始行序计算，不按 DAP 重排。上述 DAP 重置和 HL seed 5 间隔差异均应在解读 action-sequence 时保留。

## 3. Seeds、正式训练与验证完整性

标准 seed 列表：`0,1,2,3,4,5,6,7`。HL seed 0、FQ seed 0 复用既有 100K 正式 checkpoint；HL 1–7 与 FQ 1–6 均各正式运行一次且 exit 0。已完成 seed 均有 100K checkpoint、十个 2014–2023 日过程 CSV、十条 100K 验证汇总行，且水氮总量/事件数与逐日 action trace 闭合。全量核验见 `seed_completion.csv`。

FQ seed 7 的 2K smoke 输出保留在 `benchmark_results/8fq07_fq_seed7_8seed_smoke2k`，`smoke_gate_summary.csv` 记为 fail。其余 smoke 检查通过，但 `not_all_positive_actions_at_dap1=false`；FQ 正式 runner 的 `verify_completed_smoke()` 因此返回 `next_step_allowed=false`，正式训练在 preflight 阶段明确拒绝（本轮尝试退出码 1），没有 100K checkpoint，也没有正式验证结果。为遵守项目 smoke gate，本轮不绕过该 preflight。seed 7 保留为 `NOT_EVALUATED_SMOKE_GATE_BLOCKED`，FQ 站点层级 8-seed 判断标记为 coverage 不完整。

## 4. 指标口径与限制

产量、灌溉、施氮、事件数、PFP_N、episode return 取固定 100K 的正式验证汇总，并与逐日记录闭合。PFP_N 按现有正式表口径报告；N=0 时为 N/A。当前 051/054 验证产物没有精确作物吸氮量，因此 NUE 不估算；没有本轮各 seed 的 Summary.OUT/ETCP 精确回放，因此 WUE/WP_ET 不从灌溉量推算，均记 N/A。YC 的 WP_ET 只沿用既有有精确 ETCP 回放的 seed。

## 5. 年际 action-sequence 判定

标准化序列以每个验证年的实际非零管理事件构成，事件字段为 DAP、灌溉量、施氮量；逐日完整 action index/水氮剂量序列也另存于 `normalized_action_sequences.json`。对年际管理措施分类时忽略不改变管理量的收获后/no-op 行，避免生育期长度差异造成虚假策略差异；总量相同但日期不同仍视为不同序列。分类阈值固定为：全十年事件序列完全相同为 C；否则若同一完整序列至少覆盖 7/10 年为 B；其他为 A。

A=`YEAR_SPECIFIC_POLICY`；B=`MOSTLY_STABLE_WITH_VARIATION`；C=`IDENTICAL_ACROSS_YEARS`。逐 seed 结果如下：

| Site | Seed | 管理事件分类 | 匹配事件序列年数/10 | 唯一事件序列数 | 完整逐日 action 是否十年全同 | 唯一完整逐日序列数 | 平均 yield | 平均 irrigation | 平均 N | 平均 PFP_N |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| YC | 0 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | None | None | 5348.29 | 45.00 | 40.00 | 133.71 |
| YC | 1 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | None | None | 6303.99 | 45.00 | 120.00 | 52.53 |
| YC | 2 | MOSTLY_STABLE_WITH_VARIATION | 7/10 | 2 | None | None | 6821.76 | 75.00 | 188.00 | 36.74 |
| YC | 3 | YEAR_SPECIFIC_POLICY | 5/10 | 2 | None | None | 6901.89 | 75.00 | 100.00 | 72.67 |
| YC | 4 | YEAR_SPECIFIC_POLICY | 5/10 | 3 | None | None | 8202.89 | 214.50 | 240.00 | 34.18 |
| YC | 5 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | None | None | 8148.66 | 165.00 | 240.00 | 33.95 |
| YC | 6 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | None | None | 3552.30 | 45.00 | 0.00 | N/A |
| YC | 7 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | None | None | 5348.29 | 45.00 | 40.00 | 133.71 |
| HL | 0 | MOSTLY_STABLE_WITH_VARIATION | 9/10 | 2 | False | 10 | 6244.47 | 78.00 | 124.00 | 50.71 |
| HL | 1 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | False | 10 | 6034.91 | 75.00 | 200.00 | 30.17 |
| HL | 2 | YEAR_SPECIFIC_POLICY | 5/10 | 3 | False | 10 | 6833.45 | 240.00 | 240.00 | 28.47 |
| HL | 3 | YEAR_SPECIFIC_POLICY | 3/10 | 6 | False | 10 | 6834.09 | 240.00 | 180.00 | 38.57 |
| HL | 4 | YEAR_SPECIFIC_POLICY | 2/10 | 9 | False | 10 | 6833.39 | 240.00 | 240.00 | 28.47 |
| HL | 5 | YEAR_SPECIFIC_POLICY | 4/10 | 6 | False | 10 | 6800.48 | 240.00 | 240.00 | 28.34 |
| HL | 6 | MOSTLY_STABLE_WITH_VARIATION | 9/10 | 2 | False | 10 | 6825.42 | 226.50 | 240.00 | 28.44 |
| HL | 7 | YEAR_SPECIFIC_POLICY | 6/10 | 3 | False | 10 | 6653.20 | 129.00 | 160.00 | 41.58 |
| FQ | 0 | YEAR_SPECIFIC_POLICY | 4/10 | 4 | False | 9 | 7315.37 | 219.00 | 240.00 | 30.48 |
| FQ | 1 | YEAR_SPECIFIC_POLICY | 4/10 | 4 | False | 9 | 7318.41 | 219.00 | 240.00 | 30.49 |
| FQ | 2 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | False | 9 | 7118.43 | 45.00 | 40.00 | 177.96 |
| FQ | 3 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | False | 9 | 7188.58 | 45.00 | 80.00 | 89.86 |
| FQ | 4 | YEAR_SPECIFIC_POLICY | 6/10 | 2 | False | 10 | 7206.13 | 75.00 | 216.00 | 33.26 |
| FQ | 5 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | False | 9 | 6804.34 | 45.00 | 0.00 | N/A |
| FQ | 6 | IDENTICAL_ACROSS_YEARS | 10/10 | 1 | False | 9 | 7118.43 | 45.00 | 40.00 | 177.96 |
| FQ | 7 | NOT_EVALUATED_SMOKE_GATE_BLOCKED | 0/10 | 0 | None | None | N/A | N/A | N/A | N/A |

逐年水氮措施与指标完整表见 `metrics_per_seed_year.csv` 和 `management_events_per_seed_year.csv`。以下列出所有已完成正式运行的 seed-year 完整事件序列；FQ seed 7 没有正式评估序列：

### HL 每年水氮措施

| Seed | Year | Yield kg/ha | Irrigation mm | N kg/ha | I/N events | PFP_N | DAP 与联合措施序列 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2014 | 7340.00 | 75.00 | 120.00 | 2/2 | 61.17 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2015 | 6378.31 | 105.00 | 160.00 | 3/3 | 39.86 | DAP1 I45/N80; DAP8 I30/N40; DAP31 I30/N40 |
| 0 | 2016 | 5716.99 | 75.00 | 120.00 | 2/2 | 47.64 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2017 | 4695.58 | 75.00 | 120.00 | 2/2 | 39.13 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2018 | 6573.99 | 75.00 | 120.00 | 2/2 | 54.78 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2019 | 5883.71 | 75.00 | 120.00 | 2/2 | 49.03 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2020 | 7557.69 | 75.00 | 120.00 | 2/2 | 62.98 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2021 | 5411.88 | 75.00 | 120.00 | 2/2 | 45.10 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2022 | 6559.26 | 75.00 | 120.00 | 2/2 | 54.66 | DAP1 I45/N80; DAP8 I30/N40 |
| 0 | 2023 | 6327.29 | 75.00 | 120.00 | 2/2 | 52.73 | DAP1 I45/N80; DAP8 I30/N40 |
| 1 | 2014 | 7339.87 | 75.00 | 200.00 | 2/2 | 36.70 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2015 | 4261.65 | 75.00 | 200.00 | 2/2 | 21.31 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2016 | 5729.81 | 75.00 | 200.00 | 2/2 | 28.65 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2017 | 4695.42 | 75.00 | 200.00 | 2/2 | 23.48 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2018 | 6573.99 | 75.00 | 200.00 | 2/2 | 32.87 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2019 | 5883.70 | 75.00 | 200.00 | 2/2 | 29.42 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2020 | 7557.68 | 75.00 | 200.00 | 2/2 | 37.79 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2021 | 5411.85 | 75.00 | 200.00 | 2/2 | 27.06 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2022 | 6568.04 | 75.00 | 200.00 | 2/2 | 32.84 | DAP1 I45/N120; DAP8 I30/N80 |
| 1 | 2023 | 6327.06 | 75.00 | 200.00 | 2/2 | 31.64 | DAP1 I45/N120; DAP8 I30/N80 |
| 2 | 2014 | 7275.04 | 240.00 | 240.00 | 14/6 | 30.31 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N0; DAP39 I0/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2015 | 7649.02 | 240.00 | 240.00 | 14/6 | 31.87 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2016 | 7526.94 | 240.00 | 240.00 | 14/6 | 31.36 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2017 | 6201.02 | 240.00 | 240.00 | 14/6 | 25.84 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2018 | 6564.24 | 240.00 | 240.00 | 14/6 | 27.35 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP30 I0/N40; DAP31 I15/N0; DAP37 I0/N40; DAP38 I15/N0; DAP44 I0/N40; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2019 | 5792.00 | 240.00 | 240.00 | 14/6 | 24.13 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2020 | 7496.75 | 240.00 | 240.00 | 14/6 | 31.24 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N0; DAP39 I0/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2021 | 5384.80 | 240.00 | 240.00 | 14/6 | 22.44 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N0; DAP39 I0/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2022 | 7928.01 | 240.00 | 240.00 | 14/6 | 33.03 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 2 | 2023 | 6516.65 | 240.00 | 240.00 | 14/6 | 27.15 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP31 I15/N40; DAP38 I15/N0; DAP39 I0/N40; DAP45 I15/N0; DAP46 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2014 | 7275.23 | 240.00 | 160.00 | 14/2 | 45.47 | DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP41 I0/N120; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2015 | 7649.06 | 240.00 | 160.00 | 14/2 | 47.81 | DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP46 I0/N120; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2016 | 7526.92 | 240.00 | 200.00 | 15/2 | 37.63 | DAP1 I30/N80; DAP8 I15/N0; DAP15 I15/N0; DAP22 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP49 I0/N120; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2017 | 6200.98 | 240.00 | 200.00 | 15/2 | 31.00 | DAP1 I30/N80; DAP8 I15/N0; DAP15 I15/N0; DAP22 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP49 I0/N120; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2018 | 6566.93 | 240.00 | 200.00 | 15/2 | 32.83 | DAP1 I30/N80; DAP8 I15/N0; DAP15 I15/N0; DAP22 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP44 I0/N120; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2019 | 5790.03 | 240.00 | 160.00 | 14/2 | 36.19 | DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP46 I0/N120; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2020 | 7493.03 | 240.00 | 200.00 | 15/2 | 37.47 | DAP1 I30/N80; DAP8 I15/N0; DAP15 I15/N0; DAP22 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP49 I0/N120; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2021 | 5394.22 | 240.00 | 200.00 | 15/2 | 26.97 | DAP1 I30/N80; DAP8 I15/N0; DAP15 I15/N0; DAP22 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP47 I0/N120; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2022 | 7927.81 | 240.00 | 160.00 | 14/2 | 49.55 | DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP46 I0/N120; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 3 | 2023 | 6516.65 | 240.00 | 160.00 | 14/2 | 40.73 | DAP1 I45/N40; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP42 I0/N120; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2014 | 7274.87 | 240.00 | 240.00 | 14/3 | 30.31 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP32 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP51 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2015 | 7649.02 | 240.00 | 240.00 | 14/3 | 31.87 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP38 I15/N0; DAP39 I0/N120; DAP45 I15/N0; DAP52 I15/N0; DAP54 I0/N40; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2016 | 7526.94 | 240.00 | 240.00 | 14/3 | 31.36 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP33 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP55 I0/N40; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2017 | 6201.03 | 240.00 | 240.00 | 14/3 | 25.84 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP34 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP54 I0/N40; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2018 | 6564.23 | 240.00 | 240.00 | 14/3 | 27.35 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP30 I0/N120; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP50 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2019 | 5791.75 | 240.00 | 240.00 | 14/3 | 24.13 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP34 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP54 I0/N40; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2020 | 7496.72 | 240.00 | 240.00 | 14/3 | 31.24 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP34 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP57 I0/N40; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2021 | 5384.69 | 240.00 | 240.00 | 14/3 | 22.44 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP35 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP53 I0/N40; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2022 | 7928.00 | 240.00 | 240.00 | 14/3 | 33.03 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP36 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP53 I0/N40; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 4 | 2023 | 6516.65 | 240.00 | 240.00 | 14/3 | 27.15 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I15/N0; DAP33 I0/N120; DAP38 I15/N0; DAP45 I15/N0; DAP51 I0/N40; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 5 | 2014 | 7214.03 | 240.00 | 240.00 | 6/2 | 30.06 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP47 I45/N0; DAP54 I45/N0; DAP61 I45/N0; DAP91 I45/N0 |
| 5 | 2015 | 7636.01 | 240.00 | 240.00 | 6/2 | 31.82 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP49 I45/N0; DAP56 I45/N0; DAP63 I45/N0; DAP91 I45/N0 |
| 5 | 2016 | 7486.39 | 240.00 | 240.00 | 6/2 | 31.19 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP49 I45/N0; DAP56 I45/N0; DAP63 I45/N0; DAP91 I45/N0 |
| 5 | 2017 | 6201.56 | 240.00 | 240.00 | 6/2 | 25.84 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP49 I45/N0; DAP56 I45/N0; DAP63 I45/N0; DAP91 I45/N0 |
| 5 | 2018 | 6525.28 | 240.00 | 240.00 | 6/2 | 27.19 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP45 I45/N0; DAP52 I45/N0; DAP61 I45/N0; DAP91 I45/N0 |
| 5 | 2019 | 5753.93 | 240.00 | 240.00 | 6/2 | 23.97 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP50 I45/N0; DAP57 I45/N0; DAP64 I45/N0; DAP91 I45/N0 |
| 5 | 2020 | 7480.92 | 240.00 | 240.00 | 6/2 | 31.17 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP49 I45/N0; DAP56 I45/N0; DAP63 I45/N0; DAP91 I45/N0 |
| 5 | 2021 | 5322.77 | 240.00 | 240.00 | 6/2 | 22.18 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP48 I45/N0; DAP55 I45/N0; DAP62 I45/N0; DAP91 I45/N0 |
| 5 | 2022 | 7872.15 | 240.00 | 240.00 | 6/2 | 32.80 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP48 I45/N0; DAP55 I45/N0; DAP62 I45/N0; DAP91 I45/N0 |
| 5 | 2023 | 6511.76 | 240.00 | 240.00 | 6/2 | 27.13 | DAP1 I45/N0; DAP2 I0/N120; DAP8 I15/N120; DAP46 I45/N0; DAP53 I45/N0; DAP61 I45/N0; DAP91 I45/N0 |
| 6 | 2014 | 7254.12 | 225.00 | 240.00 | 9/4 | 30.23 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2015 | 7649.64 | 225.00 | 240.00 | 9/4 | 31.87 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2016 | 7526.65 | 225.00 | 240.00 | 9/4 | 31.36 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2017 | 6198.75 | 225.00 | 240.00 | 9/4 | 25.83 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2018 | 6558.89 | 240.00 | 240.00 | 9/4 | 27.33 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I30/N0; DAP45 I30/N0; DAP61 I30/N0; DAP91 I30/N0; DAP98 I30/N0 |
| 6 | 2019 | 5791.92 | 225.00 | 240.00 | 9/4 | 24.13 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2020 | 7491.18 | 225.00 | 240.00 | 9/4 | 31.21 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2021 | 5347.52 | 225.00 | 240.00 | 9/4 | 22.28 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2022 | 7918.89 | 225.00 | 240.00 | 9/4 | 33.00 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 6 | 2023 | 6516.65 | 225.00 | 240.00 | 9/4 | 27.15 | DAP1 I45/N120; DAP8 I15/N0; DAP9 I0/N40; DAP15 I15/N0; DAP16 I0/N40; DAP23 I0/N40; DAP31 I15/N0; DAP38 I15/N0; DAP45 I30/N0; DAP61 I30/N0; DAP68 I30/N0; DAP91 I30/N0 |
| 7 | 2014 | 7321.54 | 120.00 | 160.00 | 4/2 | 45.76 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80 |
| 7 | 2015 | 7041.72 | 135.00 | 160.00 | 5/2 | 44.01 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80; DAP38 I15/N0 |
| 7 | 2016 | 7270.31 | 135.00 | 160.00 | 5/2 | 45.44 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80; DAP38 I15/N0 |
| 7 | 2017 | 5501.98 | 135.00 | 160.00 | 5/2 | 34.39 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80; DAP38 I15/N0 |
| 7 | 2018 | 6571.23 | 120.00 | 160.00 | 4/2 | 41.07 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP32 I45/N80 |
| 7 | 2019 | 5844.24 | 135.00 | 160.00 | 5/2 | 36.53 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80; DAP38 I15/N0 |
| 7 | 2020 | 7535.45 | 120.00 | 160.00 | 4/2 | 47.10 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80 |
| 7 | 2021 | 5402.79 | 135.00 | 160.00 | 5/2 | 33.77 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80; DAP38 I15/N0 |
| 7 | 2022 | 7528.97 | 135.00 | 160.00 | 5/2 | 47.06 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80; DAP38 I15/N0 |
| 7 | 2023 | 6513.81 | 120.00 | 160.00 | 4/2 | 40.71 | DAP1 I45/N80; DAP8 I15/N0; DAP15 I15/N0; DAP31 I45/N80 |

### FQ 每年水氮措施

| Seed | Year | Yield kg/ha | Irrigation mm | N kg/ha | I/N events | PFP_N | DAP 与联合措施序列 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2014 | 8686.65 | 240.00 | 240.00 | 14/3 | 36.19 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 0 | 2015 | 7927.06 | 225.00 | 240.00 | 13/3 | 33.03 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 0 | 2016 | 7907.10 | 210.00 | 240.00 | 12/3 | 32.95 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0 |
| 0 | 2017 | 7432.21 | 225.00 | 240.00 | 13/3 | 30.97 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 0 | 2018 | 0.00 | 180.00 | 240.00 | 10/3 | 0.00 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0 |
| 0 | 2019 | 8837.21 | 240.00 | 240.00 | 14/3 | 36.82 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 0 | 2020 | 9449.29 | 225.00 | 240.00 | 13/3 | 39.37 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 0 | 2021 | 7333.89 | 225.00 | 240.00 | 13/3 | 30.56 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 0 | 2022 | 6416.97 | 210.00 | 240.00 | 12/3 | 26.74 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0 |
| 0 | 2023 | 9163.33 | 210.00 | 240.00 | 12/3 | 38.18 | DAP1 I45/N80; DAP8 I15/N0; DAP9 I0/N80; DAP15 I15/N0; DAP16 I0/N80; DAP31 I15/N0; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0 |
| 1 | 2014 | 8688.67 | 240.00 | 240.00 | 14/6 | 36.20 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 1 | 2015 | 7928.72 | 225.00 | 240.00 | 13/6 | 33.04 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 1 | 2016 | 7906.35 | 210.00 | 240.00 | 12/6 | 32.94 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0 |
| 1 | 2017 | 7432.21 | 225.00 | 240.00 | 13/6 | 30.97 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 1 | 2018 | 0.00 | 180.00 | 240.00 | 10/6 | 0.00 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0 |
| 1 | 2019 | 8847.66 | 240.00 | 240.00 | 14/6 | 36.87 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0; DAP105 I15/N0 |
| 1 | 2020 | 9449.02 | 225.00 | 240.00 | 13/6 | 39.37 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 1 | 2021 | 7336.04 | 225.00 | 240.00 | 13/6 | 30.57 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0; DAP98 I15/N0 |
| 1 | 2022 | 6431.56 | 210.00 | 240.00 | 12/6 | 26.80 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0 |
| 1 | 2023 | 9163.83 | 210.00 | 240.00 | 12/6 | 38.18 | DAP1 I45/N40; DAP8 I15/N40; DAP15 I15/N40; DAP22 I0/N40; DAP29 I0/N40; DAP31 I15/N0; DAP36 I0/N40; DAP38 I15/N0; DAP45 I15/N0; DAP52 I15/N0; DAP59 I15/N0; DAP66 I15/N0; DAP73 I15/N0; DAP80 I15/N0; DAP91 I15/N0 |
| 2 | 2014 | 8028.91 | 45.00 | 40.00 | 1/1 | 200.72 | DAP1 I45/N40 |
| 2 | 2015 | 8003.84 | 45.00 | 40.00 | 1/1 | 200.10 | DAP1 I45/N40 |
| 2 | 2016 | 7811.16 | 45.00 | 40.00 | 1/1 | 195.28 | DAP1 I45/N40 |
| 2 | 2017 | 7704.49 | 45.00 | 40.00 | 1/1 | 192.61 | DAP1 I45/N40 |
| 2 | 2018 | 0.00 | 45.00 | 40.00 | 1/1 | 0.00 | DAP1 I45/N40 |
| 2 | 2019 | 7856.85 | 45.00 | 40.00 | 1/1 | 196.42 | DAP1 I45/N40 |
| 2 | 2020 | 8523.44 | 45.00 | 40.00 | 1/1 | 213.09 | DAP1 I45/N40 |
| 2 | 2021 | 7462.66 | 45.00 | 40.00 | 1/1 | 186.57 | DAP1 I45/N40 |
| 2 | 2022 | 6693.69 | 45.00 | 40.00 | 1/1 | 167.34 | DAP1 I45/N40 |
| 2 | 2023 | 9099.31 | 45.00 | 40.00 | 1/1 | 227.48 | DAP1 I45/N40 |
| 3 | 2014 | 8295.55 | 45.00 | 80.00 | 1/1 | 103.69 | DAP1 I45/N80 |
| 3 | 2015 | 8005.10 | 45.00 | 80.00 | 1/1 | 100.06 | DAP1 I45/N80 |
| 3 | 2016 | 7808.34 | 45.00 | 80.00 | 1/1 | 97.60 | DAP1 I45/N80 |
| 3 | 2017 | 7704.42 | 45.00 | 80.00 | 1/1 | 96.31 | DAP1 I45/N80 |
| 3 | 2018 | 0.00 | 45.00 | 80.00 | 1/1 | 0.00 | DAP1 I45/N80 |
| 3 | 2019 | 7859.85 | 45.00 | 80.00 | 1/1 | 98.25 | DAP1 I45/N80 |
| 3 | 2020 | 8961.38 | 45.00 | 80.00 | 1/1 | 112.02 | DAP1 I45/N80 |
| 3 | 2021 | 7462.83 | 45.00 | 80.00 | 1/1 | 93.29 | DAP1 I45/N80 |
| 3 | 2022 | 6689.17 | 45.00 | 80.00 | 1/1 | 83.61 | DAP1 I45/N80 |
| 3 | 2023 | 9099.11 | 45.00 | 80.00 | 1/1 | 113.74 | DAP1 I45/N80 |
| 4 | 2014 | 8286.79 | 75.00 | 240.00 | 2/4 | 34.53 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40; DAP23 I0/N40 |
| 4 | 2015 | 7970.47 | 75.00 | 240.00 | 2/4 | 33.21 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40; DAP23 I0/N40 |
| 4 | 2016 | 7902.00 | 75.00 | 200.00 | 2/3 | 39.51 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40 |
| 4 | 2017 | 7702.27 | 75.00 | 200.00 | 2/3 | 38.51 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40 |
| 4 | 2018 | 0.00 | 75.00 | 200.00 | 2/3 | 0.00 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40 |
| 4 | 2019 | 8043.66 | 75.00 | 240.00 | 2/4 | 33.52 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40; DAP23 I0/N40 |
| 4 | 2020 | 8960.49 | 75.00 | 240.00 | 2/4 | 37.34 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40; DAP23 I0/N40 |
| 4 | 2021 | 7462.51 | 75.00 | 200.00 | 2/3 | 37.31 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40 |
| 4 | 2022 | 6636.50 | 75.00 | 200.00 | 2/3 | 33.18 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40 |
| 4 | 2023 | 9096.67 | 75.00 | 200.00 | 2/3 | 45.48 | DAP1 I45/N120; DAP8 I30/N0; DAP9 I0/N40; DAP16 I0/N40 |
| 5 | 2014 | 7324.34 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2015 | 7769.01 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2016 | 7731.72 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2017 | 7630.88 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2018 | 0.00 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2019 | 7708.15 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2020 | 7518.49 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2021 | 7306.05 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2022 | 6571.22 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 5 | 2023 | 8483.58 | 45.00 | 0.00 | 1/0 | N/A | DAP1 I45/N0 |
| 6 | 2014 | 8028.91 | 45.00 | 40.00 | 1/1 | 200.72 | DAP1 I45/N40 |
| 6 | 2015 | 8003.84 | 45.00 | 40.00 | 1/1 | 200.10 | DAP1 I45/N40 |
| 6 | 2016 | 7811.16 | 45.00 | 40.00 | 1/1 | 195.28 | DAP1 I45/N40 |
| 6 | 2017 | 7704.49 | 45.00 | 40.00 | 1/1 | 192.61 | DAP1 I45/N40 |
| 6 | 2018 | 0.00 | 45.00 | 40.00 | 1/1 | 0.00 | DAP1 I45/N40 |
| 6 | 2019 | 7856.85 | 45.00 | 40.00 | 1/1 | 196.42 | DAP1 I45/N40 |
| 6 | 2020 | 8523.44 | 45.00 | 40.00 | 1/1 | 213.09 | DAP1 I45/N40 |
| 6 | 2021 | 7462.66 | 45.00 | 40.00 | 1/1 | 186.57 | DAP1 I45/N40 |
| 6 | 2022 | 6693.69 | 45.00 | 40.00 | 1/1 | 167.34 | DAP1 I45/N40 |
| 6 | 2023 | 9099.31 | 45.00 | 40.00 | 1/1 | 227.48 | DAP1 I45/N40 |

## 6. 跨 seed 稳定性与站点判断

站点层级分类使用已完成 seed 之间按年份计算的管理事件集合 Jaccard 相似度中位数：≥0.80 为 `SEED_STABLE`，≤0.35 为 `SEED_SENSITIVE`，其余为 `MIXED_SEED_BEHAVIOR`。FQ 仅完成 7/8，因此其最终 8-seed 站点判断不下结论，标记 `INCOMPLETE_EIGHT_SEED_COVERAGE`，表内相似度与 provisional assessment 只描述已完成的 7 个 seed。这描述的是策略输出相似度，不代表产量最优或因果稳健性。

| Site | Seeds | A | B | C | seed 间中位相似度 | 站点判断 |
|---|---:|---:|---:|---:|---:|---|
| HL | 8 | 5 | 2 | 1 | 0.00 | SEED_SENSITIVE |
| FQ | 7 | 3 | 0 | 4 | 0.00 | INCOMPLETE_EIGHT_SEED_COVERAGE |
| YC | 8 | 2 | 1 | 5 | 0.00 | SEED_SENSITIVE |

## 7. 与 YC 现有结果对照

YC 的每个 seed 同样用 observed-weather 2014–2023 事件表作分类；未训练、未回放、未改动 YC。YC 使用 random-weather 训练来源，HL/FQ 使用各自历史 WTH，因此该对照用于判断‘完全一致’是否在站点间普遍出现，不用于宣称任一固定策略更优。逐 seed 对照表见 `site_seed_action_classification.csv`。

## 8. 图表索引

每个有正式结果的 HL/FQ seed 输出 6 张与 YC 004_15 同类图：年度指标、年度水氮总量、reward、年度趋势、DAP action calendar、逐年剂量图。FQ seed 7 没有正式逐年输出，因此不绘制 seed 图；站点图将其灰显为 smoke gate blocked。站点 seed 分布和 A/B/C 汇总图也一并生成。完整清单见 `figure_manifest.csv`。

- `HL seed 0`：[seed0_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_0/seed0_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 0`：[seed0_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_0/seed0_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 0`：[seed0_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_0/seed0_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 0`：[seed0_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_0/seed0_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 0`：[seed0_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_0/seed0_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 0`：[seed0_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_0/seed0_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed 1`：[seed1_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_1/seed1_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 1`：[seed1_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_1/seed1_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 1`：[seed1_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_1/seed1_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 1`：[seed1_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_1/seed1_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 1`：[seed1_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_1/seed1_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 1`：[seed1_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_1/seed1_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed 2`：[seed2_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_2/seed2_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 2`：[seed2_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_2/seed2_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 2`：[seed2_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_2/seed2_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 2`：[seed2_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_2/seed2_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 2`：[seed2_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_2/seed2_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 2`：[seed2_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_2/seed2_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed 3`：[seed3_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_3/seed3_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 3`：[seed3_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_3/seed3_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 3`：[seed3_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_3/seed3_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 3`：[seed3_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_3/seed3_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 3`：[seed3_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_3/seed3_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 3`：[seed3_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_3/seed3_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed 4`：[seed4_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_4/seed4_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 4`：[seed4_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_4/seed4_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 4`：[seed4_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_4/seed4_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 4`：[seed4_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_4/seed4_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 4`：[seed4_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_4/seed4_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 4`：[seed4_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_4/seed4_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed 5`：[seed5_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_5/seed5_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 5`：[seed5_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_5/seed5_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 5`：[seed5_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_5/seed5_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 5`：[seed5_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_5/seed5_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 5`：[seed5_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_5/seed5_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 5`：[seed5_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_5/seed5_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed 6`：[seed6_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_6/seed6_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 6`：[seed6_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_6/seed6_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 6`：[seed6_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_6/seed6_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 6`：[seed6_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_6/seed6_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 6`：[seed6_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_6/seed6_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 6`：[seed6_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_6/seed6_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed 7`：[seed7_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_7/seed7_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `HL seed 7`：[seed7_annual_management.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_7/seed7_annual_management.png) — Annual irrigation and nitrogen totals
- `HL seed 7`：[seed7_reward.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_7/seed7_reward.png) — Common reward and canonical PPO return on separate axes
- `HL seed 7`：[seed7_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_7/seed7_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `HL seed 7`：[seed7_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_7/seed7_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `HL seed 7`：[seed7_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/hl/seed_7/seed7_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `FQ seed 0`：[seed0_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_0/seed0_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `FQ seed 0`：[seed0_annual_management.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_0/seed0_annual_management.png) — Annual irrigation and nitrogen totals
- `FQ seed 0`：[seed0_reward.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_0/seed0_reward.png) — Common reward and canonical PPO return on separate axes
- `FQ seed 0`：[seed0_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_0/seed0_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `FQ seed 0`：[seed0_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_0/seed0_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `FQ seed 0`：[seed0_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_0/seed0_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `FQ seed 1`：[seed1_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_1/seed1_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `FQ seed 1`：[seed1_annual_management.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_1/seed1_annual_management.png) — Annual irrigation and nitrogen totals
- `FQ seed 1`：[seed1_reward.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_1/seed1_reward.png) — Common reward and canonical PPO return on separate axes
- `FQ seed 1`：[seed1_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_1/seed1_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `FQ seed 1`：[seed1_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_1/seed1_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `FQ seed 1`：[seed1_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_1/seed1_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `FQ seed 2`：[seed2_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_2/seed2_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `FQ seed 2`：[seed2_annual_management.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_2/seed2_annual_management.png) — Annual irrigation and nitrogen totals
- `FQ seed 2`：[seed2_reward.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_2/seed2_reward.png) — Common reward and canonical PPO return on separate axes
- `FQ seed 2`：[seed2_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_2/seed2_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `FQ seed 2`：[seed2_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_2/seed2_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `FQ seed 2`：[seed2_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_2/seed2_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `FQ seed 3`：[seed3_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_3/seed3_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `FQ seed 3`：[seed3_annual_management.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_3/seed3_annual_management.png) — Annual irrigation and nitrogen totals
- `FQ seed 3`：[seed3_reward.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_3/seed3_reward.png) — Common reward and canonical PPO return on separate axes
- `FQ seed 3`：[seed3_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_3/seed3_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `FQ seed 3`：[seed3_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_3/seed3_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `FQ seed 3`：[seed3_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_3/seed3_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `FQ seed 4`：[seed4_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_4/seed4_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `FQ seed 4`：[seed4_annual_management.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_4/seed4_annual_management.png) — Annual irrigation and nitrogen totals
- `FQ seed 4`：[seed4_reward.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_4/seed4_reward.png) — Common reward and canonical PPO return on separate axes
- `FQ seed 4`：[seed4_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_4/seed4_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `FQ seed 4`：[seed4_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_4/seed4_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `FQ seed 4`：[seed4_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_4/seed4_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `FQ seed 5`：[seed5_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_5/seed5_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `FQ seed 5`：[seed5_annual_management.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_5/seed5_annual_management.png) — Annual irrigation and nitrogen totals
- `FQ seed 5`：[seed5_reward.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_5/seed5_reward.png) — Common reward and canonical PPO return on separate axes
- `FQ seed 5`：[seed5_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_5/seed5_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `FQ seed 5`：[seed5_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_5/seed5_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `FQ seed 5`：[seed5_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_5/seed5_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `FQ seed 6`：[seed6_annual_metrics.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_6/seed6_annual_metrics.png) — YC-style annual yield, WP_ET, PFP_N, NUE panels; unavailable metrics marked N/A
- `FQ seed 6`：[seed6_annual_management.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_6/seed6_annual_management.png) — Annual irrigation and nitrogen totals
- `FQ seed 6`：[seed6_reward.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_6/seed6_reward.png) — Common reward and canonical PPO return on separate axes
- `FQ seed 6`：[seed6_annual_trends.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_6/seed6_annual_trends.png) — Annual yield, common reward, irrigation, N, and PFP_N trends
- `FQ seed 6`：[seed6_action_calendar.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_6/seed6_action_calendar.png) — Event timing by DAP and validation year; circles=I, squares=N
- `FQ seed 6`：[seed6_management_action_doses_by_year.png](../results/hl_fq_8seed_cross_site/figures/fq/seed_6/seed6_management_action_doses_by_year.png) — Management event dose and DAP for each validation year
- `HL seed all`：[hl_seed_metric_distributions.png](../results/hl_fq_8seed_cross_site/figures/hl_seed_metric_distributions.png) — Per-seed distribution of 10-year mean yield, irrigation, N and PFP_N
- `HL seed all`：[hl_seed_policy_classifications.png](../results/hl_fq_8seed_cross_site/figures/hl_seed_policy_classifications.png) — Seed-wise A/B/C year-to-year management classification
- `FQ seed all`：[fq_seed_metric_distributions.png](../results/hl_fq_8seed_cross_site/figures/fq_seed_metric_distributions.png) — Per-seed distribution of 10-year mean yield, irrigation, N and PFP_N
- `FQ seed all`：[fq_seed_policy_classifications.png](../results/hl_fq_8seed_cross_site/figures/fq_seed_policy_classifications.png) — Seed-wise A/B/C year-to-year management classification
- `HL/FQ seed all`：[hl_fq_year_to_year_classification_counts.png](../results/hl_fq_8seed_cross_site/figures/hl_fq_year_to_year_classification_counts.png) — Station-level A/B/C counts

## 9. 当前能得出的结论

- HL 8/8 和 FQ 7/8 已完成 seed 均固定选择 100K checkpoint，并在独立验证年逐年记录管理措施；训练天气仍是历史 WTH。FQ seed 7 受 smoke preflight gate 阻断，没有正式结果。
- 是否普遍出现‘十年措施完全一致’由上表 C 计数直接判定；仅在事件日期与剂量完全相同时记为 C。
- YC 若 C 数明显高于 HL/FQ，只能称为当前站点/训练天气情境中更突出的现象；不能据此断言固定策略正确或最优。
- NUE 与本轮 WUE/WP_ET 没有精确来源，均不作推断；yield-only 不能替代资源效率结论。

## 10. 当前不能得出的结论与下一步

本结果不能证明跨站点统计显著性、模型最优性或随机天气鲁棒性；YC 与 HL/FQ 的训练天气来源不同。若后续需要 WUE/WP_ET 或 NUE，须先取得每个 seed-year 精确 ETCP/Summary.OUT 或作物吸氮量来源并核对闭合。WGEN 版本留给后续独立任务，本轮未生成。

完整 machine-readable 表、图和 provenance 位于 `results/hl_fq_8seed_cross_site/analysis_v1`。

> `CROSS_SITE_MULTI_SEED_DIAGNOSIS = BLOCKED_BY_CONFIGURATION_OR_RUNTIME_ERROR`

## 11. Git 与终端摘要

本轮正式训练覆盖 HL 8/8、FQ 7/8。FQ seed 7 被已生效的 smoke preflight 阻断；同时 HL seed 5 的十个验证年存在 DAP 2→8 的 6 天施氮间隔记录，与有效配置的 7 天间隔不一致。因此未满足原任务“完成并通过基本完整性检查后”提交与推送的前置条件，本轮未创建 commit，也未 push；所有本地结果、失败 smoke 证据和日志均保留。

```text
HL training seeds completed: 8/8
FQ training seeds completed: 7/8
HL station-level assessment: SEED_SENSITIVE
FQ station-level assessment: INCOMPLETE_EIGHT_SEED_COVERAGE
HL identical-across-years seeds: 1/8
FQ identical-across-years seeds: 4/8 confirmed; seed 7 not evaluated
YC modified: NO
reward modified: NO
action mask modified: NO
docs generated: YES
plots generated: YES (95 files)
git commit: NONE
git push: NOT_ATTEMPTED
CROSS_SITE_MULTI_SEED_DIAGNOSIS = BLOCKED_BY_CONFIGURATION_OR_RUNTIME_ERROR
```
