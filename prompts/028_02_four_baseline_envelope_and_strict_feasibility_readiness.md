# 028_02 四基线包络与严格可行性准备审计

## 1. 任务目的

在 028_01 注册表基础上，先复用当前已有的完整四基线 CSV，按 028 统一公式重算每个就绪站点—年份的 yield、WP_ET、PFP_N 包络和严格数值门槛；同时重新判定当前已选阶段型 MaskablePPO 模型是否已经达到四情景三指标严格目标。

本任务仍是零训练、零 DSSAT 的准备审计。它只确定哪些年份可以直接进入下一轮 deterministic feasibility，哪些年份必须先补齐四基线。

## 2. 统一指标

```text
WP_ET = grain_yield_kg_ha / (10 * ETCP_mm)
PFP_N = grain_yield_kg_ha / actual_summary_N_kg_ha
```

- `actual_summary_N` 必须来自 DSSAT Summary 实际执行量；
- N=0 时 PFP_N 为 not_comparable；
- 若 requested/management N 与 Summary N 不一致，保留差值并以 Summary N 作为正式效率分母；
- 四基线目标与 028_00 完全一致；
- 严格数值门槛为 `Y_target+1.0`、`WP_target+0.01`、`PFP_target+0.1`。

## 3. 当前允许复用的完整四基线源

- HLA2010：`benchmark_results/027_01_attempt2/027_01_hla2010_four_baselines.csv`
- SY2012/2014/2015：`benchmark_results/026_07_attempt2/026_07_syYYYY_four_baselines.csv`
- YC2014、FQ2016、LC2010：各站 027_07 attempt2 `four_baseline_fresh_rerun.csv`

FQ2016 必须标记 Tier B 派生天气回放，不能并入 Tier A 权威主结论。

## 4. 审计内容

1. 验证每个源恰好包含 null、recorded、auto、expert 四情景；
2. 使用原始 yield、ETCP、Summary N 重算 WP_ET 与 PFP_N；
3. 保存四情景包络、胜出情景和严格目标；
4. 对已有 selected PPO 逐 seed 重新判定 `strict_joint_pass`；
5. 从 028_01 列出 Tier A 中尚无当前完整四基线的年份；
6. 不把旧 local-primary、recorded pass 或单项获胜升级为严格联合通过。

## 5. 输出

```text
benchmark_results/028_02_four_baseline_envelope_readiness/
  028_02_baseline_metrics_recomputed.csv
  028_02_strict_target_envelopes.csv
  028_02_current_selected_rl_strict_comparison.csv
  028_02_missing_tier_a_four_baselines.csv
  028_02_source_hashes.csv
  028_02_result.json

docs/2026-07-18_028_02_four_baseline_envelope_and_strict_feasibility_readiness.md
```

## 6. 停止规则

- 任一源缺情景、数值非有限或正施氮基线缺失：该年份记 invalid，不补猜测；
- 不启动 RL，不启动 DSSAT；
- 不修改旧 CSV 或输入；
- 只有 baseline envelope valid 的年份才能进入后续 deterministic feasibility；
- 其余 Tier A 年份必须先在新目录补四基线，不能直接调参。
