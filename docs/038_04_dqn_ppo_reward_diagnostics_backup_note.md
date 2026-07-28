# 038_04 历史诊断线备份说明：031/035 系列

记录时间：2026-07-28

## 这份文件的作用

本文件说明为什么要备份 031/035 系列，以及这些文件后续该如何使用。

031/035 系列不是当前最终可信结果线，而是 DQN/PPO/reward 机制诊断线。它们记录了在发现 IC 与 DSSAT 管理事件链问题前后，项目围绕自由时序强化学习、操作事件成本、reward 与指标对齐、施氮边际收益等问题做过的探索。

## 031 系列本批内容

本批只纳入两个 031 小任务：

- `031_41_lc2010_operation_event_cost_maskableppo_smoke`
- `031_42_sample_ppo_five_scenario_daily_process_audit`

它们的价值是：

- 记录 LC2010 上对操作事件成本的 smoke；
- 记录样例 PPO 五情景日过程审计；
- 为后续讨论“频繁小灌”“措施合理性”和“累积奖励图口径”提供历史依据。

## 035 系列本批内容

035 系列集中在 FQA2014 linked free-timing 设置下的 reward 与施氮机制探索，包括：

- 固定规则 probe；
- reward 与评价指标是否对齐；
- checkpoint guardrail；
- reward v2；
- project-profit reward；
- 氮边际收益审计；
- 氮成本调整后的 PPO guardrail。

这些内容用于复盘：

- 为什么单纯 reward 调整不应被当作最终修复；
- 为什么必须检查施氮边际收益；
- 为什么需要区分“最终指标好看”和“管理措施合理”；
- 为什么后续转向更严格的 DSSAT 管理事件链审计。

## 使用边界

031/035 系列可以用于：

- 方法演进说明；
- DQN/PPO/reward 失败或中间尝试的证据链；
- 查找旧脚本和参数；
- 解释为什么后续需要 036/037 的修正。

031/035 系列不建议用于：

- 最终可信站点年结果汇报；
- 与 037 修正后基线直接混合比较；
- 证明 PPO 或 DQN 最终成功；
- 作为论文主结果图表来源。

## 本批建议提交内容

本批只提交小体积、可复盘文件：

- `prompts/031_41*`
- `prompts/031_42*`
- `docs/031_41*`
- `docs/031_42*`
- `src/*031_41*`
- `src/*031_42*`
- `experiments/ppo_observed_years/config_031_41*.yaml`
- `prompts/035_*`
- `docs/035_*`
- `src/*035*`

本批不提交：

- `benchmark_results/031_*`
- `benchmark_results/035_*`
- 模型；
- runtime；
- snapshot；
- DSSAT 中间输出。

## 推荐 commit message

```text
exp031-035: add historical reward and management diagnostics
```

