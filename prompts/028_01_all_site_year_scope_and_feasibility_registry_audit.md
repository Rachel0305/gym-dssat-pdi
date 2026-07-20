# 028_01 五站点全部年份范围与可行性证据注册表审计

## 1. 任务目的

在不训练 RL、不运行大规模 DSSAT 的前提下，建立 028 总协议的唯一站点—年份范围表，回答：

- 哪些年份具有权威 treatment/IC；
- 哪些只是派生天气回放或 WTH-only；
- 哪些已有完整四基线；
- 哪些已有 deterministic feasibility/upper-bound 证据；
- 哪些已有最新阶段型 MaskablePPO 同年训练或固定权重跨年证据；
- 哪些年份允许进入下一阶段严格可行性认证。

本任务不得把旧 primary 通过改写成四情景三指标严格通过。

## 2. 只读证据源

至少核对：

- `prompts/020_11_hla_five_scenario_expert_completion_and_nstep_freeze.md`
- `prompts/026_06_sy_all_authoritative_years_frozen_stage_ppo_validation.md`
- `benchmark_results/026_07_attempt2/026_07_result.json`
- `benchmark_results/027_03/027_03_result.json`
- `benchmark_results/027_07_site_specific_stage_maskable_ppo_attempt2/027_07_result.json`
- `docs/2026-06-30_014_01_fq_all_year_screen_and_dqn_transfer_record.md`
- `docs/2026-07-02_016_05_deterministic_oracle_upper_bound_scan_record.md`
- `docs/2026-07-07_018_01_multisite_dqn_potential_reward_ic_audit.md`
- `docs/2026-07-13_021_05_dqn_training_protocol_correction.md`
- 五站点当前输入目录、MZX 和 WTH 文件名。

## 3. 每行必须包含的字段

```text
site, year, evidence_tier, treatment_status, weather_status,
input_provenance_status, four_baseline_status,
strict_feasibility_status, optimization_space_status,
latest_rl_algorithm, latest_same_year_training_status,
latest_fixed_weight_crossyear_status,
strict_all_four_three_metric_status,
next_phase_eligibility, source_paths, notes
```

`evidence_tier` 至少区分：

- `A_authoritative_treatment`
- `B_derived_weather_replay`
- `C_weather_only`
- `excluded_invalid`

不得仅因目录里存在 WTH 就把年份写成权威可训练年。

## 4. 当前结果判定规则

- 只有明确保存 null/recorded/auto/expert 四情景并能复算指标，才记 `four_baseline_status=complete`；
- 只有与未来 RL 相同阶段/动作/预算的确定性搜索找到严格联合达标序列，才记 `strict_feasibility_status=certified`；
- 旧 DQN oracle、旧 primary 或任一指标获胜均不能自动升级为 `strict_all_four_three_metric_status=pass`；
- 当前没有证据时必须写 `not_run` 或 `not_certified`，禁止猜测；
- 历史 DQN 正式性必须注明 021_05 探索率协议影响；
- 派生天气回放与权威 treatment 分表汇总。

## 5. 执行约束

- 不调用 `learn()`；
- 不启动 DSSAT；
- 不修改 MZX/WTH/SOL/CUL/IC/reward/旧脚本/旧结果；
- 只读取和哈希证据文件；
- 所有新文件写入新目录；
- 若年份来源冲突，保留冲突并标 `needs_provenance_review`，不得自行裁决。

## 6. 输出

```text
benchmark_results/028_01_all_site_year_scope_registry/
  028_01_station_year_registry.csv
  028_01_current_crossyear_evidence.csv
  028_01_source_file_hashes.csv
  028_01_result.json

docs/2026-07-18_028_01_all_site_year_scope_and_feasibility_registry_audit.md
```

记录必须明确回答：

1. 最新 PPO 到底在哪些站点做过真正固定权重跨年验证；
2. 当前有多少 Tier A、Tier B 和 WTH-only 年份；
3. 当前有多少年份已经严格可行性认证；
4. 下一阶段实际需要补哪些四基线和 deterministic feasibility；
5. 当前是否已经存在“覆盖五站点全部可优化年份”的统一配置。

## 7. 停止条件

任何关键证据文件缺失、年份来源冲突或无法区分权威 treatment 与派生回放时，任务仍输出 partial 注册表和冲突清单，但不得进入训练。
