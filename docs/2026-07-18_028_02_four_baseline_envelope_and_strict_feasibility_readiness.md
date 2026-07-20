# 028_02 四基线包络与严格可行性准备审计记录

## 结论先行

本任务复用了 7 个当前已有完整四基线源，并按 DSSAT 实际 Summary 施氮量重新计算 WP_ET、PFP_N 和四情景包络。其中 Tier A 有 6 个，Tier B 有 1 个。Tier A 尚有 10 个年份没有当前可直接复用的完整四基线。

对现有 17 个 PPO 模型—年份结果按 028 的四情景三指标严格门槛重判后，严格联合通过为 **3**。这只是当前选中模型的重判，不是可行性搜索；严格通过为 0 也不能说明目标不存在。

## 执行边界

- RL训练：0；
- DSSAT运行：0；
- 旧结果修改：0；
- 指标统一为 `WP_ET=yield/(10×ETCP)`、`PFP_N=yield/Summary实际N`；
- N=0 的 PFP_N 为不可比较。
- 旧表报告值与统一重算值的最大绝对差：WP_ET=0.005363 kg/m³，PFP_N=0.084063 kg/kg；因此本任务的门槛和胜负一律使用统一重算值，不使用旧表舍入值。

## 当前 selected PPO 严格重判

| 站点 | 年份 | 模型数 | 严格联合通过数 | 是否达到2/3 |
|---|---:|---:|---:|---|
| FQ | 2016 | 1 | 0 | False |
| HLA | 2010 | 3 | 0 | False |
| LC | 2010 | 1 | 0 | False |
| SY | 2012 | 3 | 0 | False |
| SY | 2014 | 3 | 2 | True |
| SY | 2015 | 3 | 1 | False |
| YC | 2014 | 3 | 0 | False |

## Tier A 待补四基线

- FQ2007：documented_original_treatment_needs_028_revalidation；旧四基线状态 `partial_historical_not_028_complete`
- FQ2008：documented_original_treatment_needs_028_revalidation；旧四基线状态 `partial_historical_not_028_complete`
- FQ2010：documented_original_treatment_needs_028_revalidation；旧四基线状态 `partial_historical_not_028_complete`
- HLA2007：needs_variant_lock_before_028；旧四基线状态 `complete_historical_for_prepared_variant`
- HLA2009：documented_not_revalidated_under_028；旧四基线状态 `not_revalidated_as_four_complete_baselines`
- HLA2011：documented_not_revalidated_under_028；旧四基线状态 `not_revalidated_as_four_complete_baselines`
- LC2008：documented_soil_and_ic_adapter_required；旧四基线状态 `partial_historical_not_028_complete`
- LC2009：documented_soil_and_ic_adapter_required；旧四基线状态 `partial_historical_not_028_complete`
- LC2011：documented_soil_and_ic_adapter_required；旧四基线状态 `partial_historical_not_028_complete`
- YC2008：documented_not_revalidated_under_028；旧四基线状态 `historical_evidence_needs_four_baseline_recompute`

## 方法边界

1. 当前 selected PPO 的 028 严格通过数不等于 RL 框架最终能力；旧 reward 和选模协议并未针对四情景三指标联合目标设计。
2. 本任务尚未运行 deterministic feasibility，因此严格可优化年份数仍为 0 个“已认证”，不是 0 个“实际存在”。
3. FQ2016 虽有当前完整四基线，但属于 Tier B 派生天气回放，必须单列。

## 输出

- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_baseline_metrics_recomputed.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_strict_target_envelopes.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_current_selected_rl_strict_comparison.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_missing_tier_a_four_baselines.csv`
- `benchmark_results/028_02_four_baseline_envelope_readiness/028_02_result.json`

## 下一步

先为缺失 Tier A 年份在新目录补齐四基线；随后对全部 valid envelope 年份使用相同阶段、动作、预算和 mask 做 deterministic strict-feasibility 搜索。完成前不启动全局 RL 调参。
