# 043_01 SYA lowIC 042_15/042_12 冻结 PPO 输入敏感性审计

## 背景

042_15 冻结的是 SYA lowIC binary-timing MaskablePPO 流程，核心候选 checkpoint 来自 042_11 的 25K 模型，并由 042_12/042_14/042_15 记录为阶段性最好结果。

当前结果的优点是指标较好、动作比早期连续小剂量方案更可解释；当前仍需解释的问题是：PPO 是否真的根据年份、土壤水氮状态、天气状态改变动作，而不是主要执行一套模板化时序。

本任务服务于论文叙事中的下一步问题：

> 如果当前模型对已有天气/土壤状态输入的动作响应不足，那么引入“历史天气下的完美天气预报/短期天气窗口”就不是为了调参凑结果，而是为了改变 RL 的决策信息结构，让模型有机会根据未来水分供给风险做更合理的自由时序决策。

## 固定对象

- 站点：SYA
- 输入数据：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 模型：`benchmark_results/042_11_sya_lowIC_binary_timing_training_length_curve/models/SYA/SYA_half_split_stress_aware_maskableppo_seed0_ckpt25000.zip`
- 算法：MaskablePPO
- 动作空间：binary timing
  - 灌溉：0 或 45 mm
  - 施氮：0 或 80 kg/ha
- 本任务不训练、不改 reward、不改 action mask、不改 DSSAT 输入。

## 审计方法

1. 复用 042_10 的环境配置和 wrapper，创建与 042_15 一致的 SYA lowIC binary-timing PPO 环境。
2. 用冻结 PPO 在 2014–2023 验证年份中确定性回放，抽取预注册 DAP 的真实 policy observation 与 action mask。
3. 在同一 action mask 下，对 observation 中已经存在的变量做局部扰动，检查 PPO 动作概率和 deterministic argmax 是否改变。
4. 同时明确列出 042_15 当前 observation 中不存在、因此当前模型无法直接响应的变量，例如 rain、tmin、past/future rain window 等。

## 预注册抽样 DAP

固定审计以下 DAP：

- DAP1
- DAP2
- DAP31
- DAP38
- DAP51
- DAP61
- DAP91

这些 DAP 覆盖早期固定动作、中期可变动作、后期固定灌溉等关键节点。

## 扰动类型

只扰动 policy observation 中真实存在的变量。若变量不存在，记录为 not_in_policy_observation，不强行伪造。

- `high_swfac`：提高水分胁迫指标
- `high_nstres`：提高氮胁迫指标
- `dry_soil`：将土壤水分相关维度调到经验低分位
- `wet_soil`：将土壤水分相关维度调到经验高分位
- `hot_tmax`：若 tmax 存在，增加 5°C
- `low_srad`：若 srad 存在，降低 20%
- `high_srad`：若 srad 存在，提高 20%

## 输出

- `docs/043_01_sya_lowIC_04215_policy_input_sensitivity_audit_record.md`
- `benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_policy_input_sensitivity_detail.csv`
- `benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_policy_input_sensitivity_by_scenario.csv`
- `benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_policy_observation_variables.csv`
- `benchmark_results/043_01_sya_lowIC_04215_policy_input_sensitivity_audit/tables/043_01_unobservable_weather_variables.csv`

## 判定分支

- A：动作对水氮/天气扰动有清晰响应，可继续用当前 observation 设计解释性审计。
- B：只在部分变量或部分 DAP 响应，说明当前模型已有一定状态敏感性，但仍需加强天气窗口/预报信息。
- C：deterministic action 基本不变，说明当前模型更像模板化策略；下一步应引入明确的天气/预报输入与天气响应 guardrail，而不是继续只看终值指标。

## 运行边界

- `--dry-run` 只检查文件、路径和配置，不跑 DSSAT。
- 正式运行会逐年回放冻结 PPO 以获取真实 observation，但不训练模型、不做 DSSAT 反事实、不改变任何输入文件。
