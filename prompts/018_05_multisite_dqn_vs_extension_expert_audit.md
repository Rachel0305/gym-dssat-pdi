# 018_05 五站点代表年份 DQN vs 官方推广 expert 审计

## 背景

018_03/018_04 已经把官方农技推广 expert baseline 加入五个站点代表年份：

- HLA2010
- SY2014
- YC2014
- FQ2016
- LC2010

导师已经认可使用官方推广方案作为 expert baseline。现在需要把现有证据整理成“DQN 是否真的有提升、哪些站点可信、下一步先复核哪里”的表格，而不是马上继续训练。

## 目标

基于已有合并表：

`DSSAT_auto_validation/extension_expert_baseline_018_03/018_03_clean_multisite_comparison_with_extension_expert.csv`

生成五站点代表年份审计结果：

1. DQN 相对 DSSAT auto：
   - 产量差
   - 灌溉差
   - 施氮差
   - 是否追平/超过 auto
   - 是否节水/节氮

2. DQN 相对 official extension expert：
   - 产量差
   - 灌溉差
   - 施氮差
   - 是否追平/超过官方推广 expert
   - 是否节水/节氮

3. DQN 相对 recorded/farmer practice：
   - 产量差
   - 灌溉差
   - 施氮差

4. 给每个站点一个当前可信等级：
   - `strong_candidate`：DQN 追平或超过 auto / official expert，且明显节水或节氮。
   - `promising_but_needs_seed`：单 seed 结果好，但跨 seed 稳定性不足。
   - `near_plateau_tradeoff`：产量接近但不完全超过，资源效率有优势。
   - `needs_reward_or_setup_review`：DQN 未能稳定优于关键基线。

5. 给出下一步 seed 稳定性复核优先级。

## 限制

- 不训练。
- 不重跑 DSSAT。
- 不修改奖励函数。
- 不修改任何原始输入文件。
- 只读已有 CSV，生成审计表、图和中文记录。

## 输出

写入：

`DSSAT_auto_validation/extension_expert_baseline_018_03/018_05_multisite_dqn_vs_extension_audit/`

至少包括：

- `018_05_site_level_audit.csv`
- `018_05_pairwise_dqn_advantage.csv`
- `figures/018_05_dqn_advantage_summary.png`
- `docs/2026-07-09_018_05_multisite_dqn_vs_extension_expert_audit_record.md`

## 汇报口径

要特别谨慎：

- 可以说“某些站点 seed0/checkpoint 显示 DQN 有实质节水节氮潜力”。
- 不要说“所有站点已经稳定成功”。
- 对 LC/SY/HLA 等单 seed 或 seed 不稳定案例，要明确写“需要 seed1/seed2 复核”。
- 对 YC/FQ 的产量和资源权衡，要诚实标注。

