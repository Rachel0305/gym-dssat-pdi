# 042_14 SYA lowIC binary-timing PPO checkpoint 综合 guardrail 审计

## 背景

042_11 给出了 binary-timing PPO 的训练长度曲线：

- 1K：动作多样，但部分年份产量较低；
- 2K/5K/10K：出现明显坍缩或低投入；
- 25K：动作多样性恢复，指标较好。

042_12 证明 25K 在 2014–2023 上指标表现较好；042_13 进一步证明 25K 不是完全固定模板，但仍有固定 DAP1/DAP2/DAP91 等模板成分。

因此下一步需要把“指标”和“动作合理性”放在同一个 checkpoint 审计表里，避免以后只按 reward 或单一指标选择 checkpoint。

## 本任务目标

本任务不训练、不调参，只读取 042_11 已有 checkpoint 结果，比较 1K/2K/5K/10K/25K：

1. 产量表现：
   - 平均产量；
   - 相对 official expert 平均产量比例；
   - 低于 official expert 90% 的验证年份数。
2. 胁迫表现：
   - 平均 WSPD>0.05 天数；
   - 严重水分失败年份数，定义为 WSPD>0.05 天数 ≥20。
3. 动作多样性：
   - validation action signature 数；
   - 灌溉/施氮事件槽位固定数；
   - 事件 DAP 最大跨年范围。
4. 输出一个候选 checkpoint guardrail 草案。

## 候选 guardrail 草案

本任务中的规则只是“诊断性草案”，不是最终预注册规则：

- `unique_action_signatures >= 5`
- `mean_yield_ratio_vs_official_expert >= 0.90`
- `low_yield_years_below_90pct_expert <= 2`
- `severe_swfac_failure_years <= 2`
- `fixed_event_slots <= 3`

满足上述条件者标记为 `candidate_pass`。

## 输出

- `docs/042_14_sya_lowIC_binary_timing_checkpoint_guardrail_audit_record.md`
- `benchmark_results/042_14_sya_lowIC_binary_timing_checkpoint_guardrail_audit/tables/`
- `benchmark_results/042_14_sya_lowIC_binary_timing_checkpoint_guardrail_audit/figures/`

