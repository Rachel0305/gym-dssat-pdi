# 027_05 松弛成功标准下五情景、日值证据审计与决策合理性图表

## 1. 任务目标

对项目中已经出现过成功或接近成功结果的 DQN 与阶段型 MaskablePPO 候选进行严格证据审计，整理并生成可复查的五情景对照数据、日值表格和过程图，用于判断强化学习策略的水氮决策是否具有农学合理性。

本任务不是重新宣称所有历史结果正式有效，也不是无条件重跑全部模型。必须先核对实验记录、模型、checkpoint、输入、五情景结果和日值来源，再补齐缺口。

## 2. 纳入范围

### 2.1 历史 DQN 候选

- HLA2010
- YC2014
- FQ2016
- LC2010
- SY2014

这些 DQN 结果必须统一标记为 `historical_candidate_provisional`。原因是 021_05 已确认旧分段训练协议压缩了探索率日程；它们可用于历史候选策略和决策过程审计，但不能冒充修复训练协议后的正式科学结论。

### 2.2 当前阶段型 MaskablePPO 候选

- SY2014：复用 026_03/026_02/026_04 预注册选择的三个冻结模型；优先以 seed0 作为代表图，同时保留三 seed 汇总。
- HLA2010：复用 027_02/027_03 已有模型；优先以严格原主判据通过的 seed0 作为代表图，同时保留 seed1/2 汇总。

如审计发现其他站点当前没有阶段型 PPO 正式训练结果，不得把 smoke 结果冒充正式模型。只有在已有站点专属 readiness/smoke 已通过、预注册协议明确允许训练时，才能串行启动必要训练；否则记录为 `not_trained_not_authorized_in_027_05`。

## 3. 五情景定义

每个站点年份的五情景必须是：

1. `null`
2. `recorded_farmer`
3. `dssat_auto`
4. `official_extension_expert`
5. `rl_candidate`，并明确 `algorithm=DQN` 或 `algorithm=MaskablePPO`

不得把 recorded farmer 称为 expert。不得把不同输入、不同 IC、不同年份或不同站点的基线拼接成同一张图。

## 4. 审计优先级与执行边界

1. 先生成证据清单，逐项记录：站点、年份、算法、seed、checkpoint、模型路径、SHA256、输入来源、IC/treatment、汇总文件、日值文件、五情景完整性、证据状态。
2. 已有模型但缺日值时，只做确定性冻结重评估，不调用 `learn()`。
3. 已有完整日值时优先直接复用，不重复运行 DSSAT。
4. 确无模型时，先记录缺口，再检查站点专属 preregistration/readiness/smoke；只有获得本任务授权且链路已通过时，才允许串行训练。
5. 所有 DSSAT 或训练运行必须串行，防止 OOM。
6. 不修改 reward、IC、DSSAT 原始输入、动作空间或已冻结 checkpoint 选择协议。
7. 不覆盖任何已有结果；新增输出统一写入 `benchmark_results/027_05/`。

## 5. 日值数据合同

每个可用情景至少输出以下字段；不存在的原生字段必须记为 NA，并说明来源限制，禁止猜测或伪造：

- site, year, algorithm, seed, scenario
- date 或 year+doy，DAP
- rainfall_mm
- tmax_c, tmin_c
- soil_water 指标及其原始字段名
- water_stress 指标及其原始字段名/方向定义
- nitrogen_stress 指标及其原始字段名/方向定义
- irrigation_requested_mm（如有）
- irrigation_executed_mm
- nitrogen_requested_kg_ha（如有）
- nitrogen_executed_kg_ha
- grain_yield_kg_ha（过程值或终值）
- biomass_kg_ha（过程值或终值）
- step_reward
- cumulative_reward
- reward_provenance

若基线情景没有原生 RL reward，可按同一站点年份冻结公式计算 `counterfactual_same_formula_reward`，但必须在 `reward_provenance` 中明确标记，不得称为该基线的原生奖励。

## 6. 图表合同

### 6.1 五情景终值对照图

至少展示：

- grain yield 与 biomass
- irrigation 与 nitrogen
- WP_ET 与 PFP_N（N=0 时 PFP_N 必须为 NA，不得设为无穷或 0）
- total/counterfactual reward，必须清楚标注 reward 口径

### 6.2 日值过程图

每个站点年份/算法代表策略生成同源 PNG 和 SVG，至少包含：

1. 降雨与 Tmax/Tmin
2. 土壤水分/水分胁迫/氮胁迫
3. 灌溉措施
4. 施氮措施
5. 籽粒产量与生物量
6. 累积奖励

颜色、线型和标记至少双重编码；灌溉不得与降雨使用难以区分的同色同形；标题保持中性，不在图标题中写“成功”“最优”等结论性词语。

## 7. 数据一致性与视觉 QA

必须自动检查：

- 图中终值与 summary CSV 一致；
- 日值累计灌溉/施氮与 summary 一致；
- 日值最终 grain/biomass 与 summary 一致；
- scenario、algorithm、seed、checkpoint 映射唯一；
- 0 降雨不是读取错误；天气列必须追溯到实际 WTH 或环境 observation；
- PNG/SVG 均生成且非空；
- 至少逐类目视检查一张渲染图，确认图例、坐标轴、单位、文字和数据可读。

## 8. 输出

统一写入：

`benchmark_results/027_05/`

至少包括：

- `027_05_evidence_manifest.csv`
- `027_05_gap_matrix.csv`
- `027_05_five_scenario_summary.csv`
- `027_05_daily_values.csv`
- `027_05_data_consistency_checks.csv`
- `figures/*.png`
- `figures/*.svg`
- `docs/2026-07-17_027_05_relaxed_success_five_scenario_daily_evidence_record.md`

## 9. 停止条件

- 模型/checkpoint/输入身份无法唯一确认：停止该条目，不猜测。
- 五情景不在同一输入口径：停止合图，记录缺口。
- 日值字段方向或单位无法确认：保留原字段并标 NA/unknown，不自行解释。
- readiness/smoke 未通过：不得训练。
- 发现已有结果与记录不一致：优先记录冲突并追溯，不覆盖旧文件。

