# YC 004_10：action-mask 规则归因审计

## 结论

- `mask_reconstruction = EXACT`：48768 个 action-mask 单元逐个与 004_09 实录一致，差异 0 个。
- `blocked_raw_argmax_attribution = COMPLETE`：1933 个被屏蔽 raw argmax 均有至少一个明确规则；其中 663 步同时满足多条禁用条件。
- `mask_rule_reasonableness = CONDITIONAL`：mask 按源码和冻结限制一致执行；这些来自 SYA 的阶段上限在 YC 上是否最合适，现有结果不能验证。
- Seed 5 的 411 个有合法替代动作且 raw 首选被挡的决策中，330 个（80.3%）由继承的早/中期灌溉上限首先触发。这些上限实质参与形成了当前用水策略。
- `next_step = document effective YC mask contract and DAP0 semantics before any rule change`。

## 有效规则

- 实际调用链为 `004_03 ScheduledEpisodeEnv -> LateIrrigationReserveMaskWrapper -> I240SwfacGuardrailWrapper -> I240StagedIrrigationReserveMaskWrapper -> StressAwareDiscreteWrapper`。040_28 的 SWFAC guardrail 修改 reward，不新增 mask 条件。
- 基础限制：灌溉 DAP [1, 120]、施氮 DAP [1, 90]；I/N 季节上限 240 mm / 250 kg/ha；最小间隔 I/N 各 7 / 7 天。
- 继承阶段限制：有效 DAP ≤30 时累计 I ≤75 mm；≤60 时 ≤150 mm；≤90 时 ≤195 mm。
- `action_safety` 冻结 JSON 明列了季节上限、间隔、时窗和晚期 195 mm 预留，但没有直接列出早期 75 mm / 中期 150 mm 阈值；它们在 040_26 继承类中生效。源码是有效契约的一部分，配置单独阅读会遗漏这两条。
- 每日 I/N 上限用于安全裁剪；此 16 动作网格最高剂量刚好等于上限，不构成 `_action_is_legal_without_clipping` 的单独禁用规则。no-op 始终合法。

## 逐 seed 统计

| Seed | 被屏蔽/全步 | 被屏蔽/有其他合法动作 | 仅一个合法动作的 step |
|---:|---:|---:|---:|
| 3 | 206/1016 | 21/816 (2.6%) | 200 |
| 4 | 966/1016 | 113/163 (69.3%) | 853 |
| 5 | 761/1016 | 411/666 (61.7%) | 350 |

- 重建的决策前累计 I/N 与 policy observation 的 `totir/cumsumfert` 最大绝对差为 0 mm / 1.52588e-05 kg/ha。

## 首个禁用规则

| 规则 | 含义 | 源码首个禁用 | 离线独立同时触发 |
|---|---|---:|---:|
| N_SEASON_BUDGET | 季节施氮上限 | 689 | 817 |
| I_MIN_INTERVAL | 灌溉最小间隔 | 502 | 502 |
| N_MIN_INTERVAL | 施氮最小间隔 | 330 | 707 |
| I_MID_CAP | 中期灌溉上限 | 230 | 290 |
| I_EARLY_CAP | 早期灌溉上限 | 114 | 338 |
| N_DAP_WINDOW | 施氮 DAP 时窗 | 68 | 68 |
| I_DAP_WINDOW | 灌溉 DAP 时窗 | 0 | 0 |
| I_SEASON_BUDGET | 季节灌溉上限 | 0 | 0 |
| I_LATE_RESERVE | 晚期灌溉预留 | 0 | 141 |

- 被挡 raw argmax 总计 1933 步，其中仍有合法替代动作 545 步。`blocked_rule_hits.csv` 的并发条件是离线独立判定；源码会在首个失败条件处返回，不能把后续条件描述为运行时逐条执行。
- 晚期预留可作为并发条件出现，哪怕它不是首个返回的禁用原因；首规则计数为 0 不表示该条件从未满足。

## 被挡动作类型

| raw argmax 类型 | 被挡次数 |
|---|---:|
| I-only | 189 |
| N-only | 924 |
| I+N | 820 |
| no-op | 0 |

- 每步完整 seed/year/timestep、原始与有效 DAP、动作剂量/类型、预算和上次施用 DAP、首规则和所有独立触发条件见 `blocked_raw_argmax_decisions.csv`；逐 seed/year 见 `seed_year_rule_summary.csv`。

## 关键 DAP

| 原始 DAP | 被挡总步数 | 其中有合法替代 | 首规则:次数 |
|---:|---:|---:|---|
| 0 | 120 | 0 | I_MIN_INTERVAL:80, N_MIN_INTERVAL:40 |
| 15 | 20 | 10 | I_EARLY_CAP:13, N_MIN_INTERVAL:7 |
| 31 | 10 | 10 | N_SEASON_BUDGET:10 |
| 61 | 10 | 0 | N_SEASON_BUDGET:10 |

- Seed 5 的 DAP 15 在全部十年都出现同一个关键归因：raw 首选 action 9（I30/N40），先前累计 I 已是 75 mm，增加 30 mm 会超过有效早期 75 mm 上限；该步实际选择 action 2（I0/N80）。这里有多个合法动作，阻断不是 forced no-op。
- 原始 DAP 0 共 150 个 step，其中 120 个 raw argmax 被挡。源码 `_dap()` 将非正值映射为有效 DAP 1；因此配置的时窗起点 1 并不会禁止记录为 DAP 0 的首日施用。DAP 0 当日后续 step 使用同一个有效 DAP，最小间隔会阻止再次施用。这是明确的时序契约差异，需要文档化；现有证据不能单独认定为错误。

## 合理性判断

- 机械层面：所有 mask 均可用现行源码和决策前累计状态解释，记录的动作合法且剂量与离散网格一致；没有未知规则造成的额外屏蔽。
- 约束目的：时窗、最小间隔、季节上限和灌溉预留具有明确的资源管理意图；大量 raw 首选被挡说明它们实质改变了策略输出，但无法据此推断放开后会提高产量或 reward。
- YC 适用性：早/中期预留阈值沿用 SYA 链路，冻结配置没有单独展开；它们挡住了 seed 5 多数有替代选择时的 raw 灌溉首选。因此 165 mm 年用水不能仅解释为 actor 自发偏好。DAP 0→1 的实现语义也与结果表面 DAP 不同。本次没有同状态的替代动作结果，结论是条件合理，尚不足以给出农艺最优保证。先把有效 mask 契约和 DAP 口径写清，再决定是否需要受控 evaluation-only 规则敏感性测试。

## 图表和数据

- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/figures/primary_rule_by_seed_choice_capable.png`
- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/figures/primary_rule_by_seed_all_blocked.png`
- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/figures/blocked_choice_rate_by_dap.png`
- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/decision_mask_reconciliation.csv`：全 3048 个决策的状态重建和闭合。
- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/blocked_raw_argmax_decisions.csv`：被屏蔽的逐步原始偏好。
- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/blocked_rule_hits.csv`：每条同时成立的禁用条件。
- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/seed_year_rule_summary.csv`、`dap_rule_summary.csv`、`action_type_summary.csv`：汇总。
- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/analysis_summary.json`、`provenance.json`、`manifest.json`：分母、结论及哈希。

## 局限

- 004_09 未提供 calendar date/DOY，故用原始 DAP 和 timestep；本审计不推测日历日期。
- 只复算当前规则在已发生轨迹上的布尔值；不更改任何规则，也没有未选动作的 DSSAT 产量、水氮效率或 reward 结果。
- 未 commit、未 push。
