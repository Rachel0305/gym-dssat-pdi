# 022_06 SY2014 支持集内 Q 排序与预算动作别名离线审计

## 1. 目的与边界

022_05 在训练全过程加入固定网格行为支持 mask 后，仍只有 1/4 checkpoint 通过联合主判据，四个 checkpoint 均达到 N300。本任务区分两个候选解释：

1. 支持集内部的 Q 排序仍然错误；
2. N300 只是请求动作被预算裁剪后形成的动作别名或记账假象。

本任务为纯离线分析：训练次数为 0、DSSAT 调用次数为 0；只读取 022_03 的 288 条固定网格数据和 022_05 的四个 checkpoint、阶段动作记录。

## 2. 数据一致性检查

- replay observation/action/target 形状分别为 288×25、288、288；
- NPZ 中的动作和 target 与 transition manifest 逐行一致；
- 022_05 共 24 个 checkpoint 阶段决策；
- DAP1 的 48 个标准化观测完全相同，最大绝对差为 0，因此 DAP1 可作同状态动作比较；
- DAP30 以后各方案已有不同管理历史，平均 target 只能作描述性排序，不能当成严格反事实因果结果。

## 3. 支持集内 Q 排序

### 3.1 总体

| 指标 | 结果 |
|---|---:|
| checkpoint×阶段比较数 | 24 |
| 平均 Q 最优动作与经验平均 target 最优动作一致 | 12/24 |
| checkpoint 实际动作与经验平均 target 最优动作一致 | 12/24 |
| 阶段内 Q-target Spearman 中位数 | 0.40 |

这说明去除未覆盖动作后，已覆盖动作内部仍没有形成稳定一致的回报排序。

### 3.2 DAP1 同状态证据

DAP1 的经验平均 target 最优动作是 action4（I15/N50，scaled target=6.0814），其次是 action3（I0/N50，6.0563）。

- season15 选择 action3，与平均 target 首选不一致；
- season30/45/60 选择 action4，与平均 target 首选一致；
- 因此 DAP1 为 3/4 一致，并非四个 checkpoint 全部错误。

这条证据表明排序错位真实存在，但仅凭 DAP1 不能解释全部 N300 行为。

### 3.3 DAP65 描述性强信号

四个 checkpoint 在 DAP65 的平均 Q 首选均为 action7（I15/N100），而固定网格经验平均 target 首选均为 action1（I15/N0）；Spearman 分别为 -0.657、-0.600、-0.600、-0.600。

这与 DQN 在 DAP65 将氮预算用满的行为一致，是支持集内部施氮排序错位的强诊断信号。但由于 DAP65 状态已经包含此前管理历史，不能把分组平均 target 差异写成 action7 相对 action1 的严格因果效应。

## 4. 预算裁剪与动作别名

| 指标 | 结果 |
|---|---:|
| checkpoint 阶段决策 | 24 |
| 请求动作发生裁剪 | 3 |
| 所选执行量存在多个请求动作别名 | 5 |
| 决策前氮预算已经耗尽 | 8 |
| 四个 checkpoint 最终实际施氮 | 300 kg/ha |
| 首次达到 N300 | 全部为 DAP65 |

具体例子：

- season15 DAP65 请求 N100，但剩余氮预算只有 N50，实际执行 N50；action7 与 action4 此时映射为同一执行量；
- season15/30 DAP85 在氮预算已耗尽后仍请求 N50，实际执行 N0；
- season45/60 DAP85 虽请求 N0，但 action1 与 action4 在氮预算耗尽后具有相同执行量。

因此动作别名确实会掩盖网络“本来想请求什么”，但不能解释季节 N300：四个 checkpoint 都在 DAP65 以前及当日通过正的实际施氮达到 N300。N300 是真实执行总量，不是后期裁剪造成的记账假象。

## 5. 预注册判定

按任务书判为 **A：支持集内 Q 排序错位为主要诊断信号**。

需要收紧解释：该判定不是说 24 个比较全部错误，也不是严格证明每个后期动作的因果排序错误。它说明：

- 未覆盖动作外推不是唯一问题；
- 预算别名不是 N300 的主因；
- 已覆盖动作内部的 Q 排序仍不稳定，尤其在 DAP65 对施氮动作表现出一致的反向描述性排序。

## 6. 对下一步的约束

下一步不应直接增加训练季数，也不应仅加入 budget-exact mask并期待解决 N300。更有针对性的下一步是预注册一个**离线支持动作排序学习测试**：

- 只使用 288 条固定网格数据；
- 先检验能否在不调用 DSSAT 的情况下，让网络在留出场景上恢复 Monte Carlo target 排序；
- 处理 DAP30 以后历史状态混杂，不能简单把阶段动作平均 target 当作监督真值；
- 若离线留出验证都不能改善排序，则停止该分支；若能改善，才考虑一次短在线 smoke。

## 7. 输出

- `prompts/022_06_sy2014_supported_q_ranking_and_budget_alias_offline_audit.md`
- `src/audit_sy2014_supported_q_ranking_budget_alias_022_06.py`
- `benchmark_results/022_06/022_06_checkpoint_stage_action_q_target_summary.csv`
- `benchmark_results/022_06/022_06_stage_ranking_summary.csv`
- `benchmark_results/022_06/022_06_dap1_identical_state_q_target_audit.csv`
- `benchmark_results/022_06/022_06_checkpoint_budget_alias_detail.csv`
- `benchmark_results/022_06/022_06_checkpoint_budget_alias_summary.csv`
- `benchmark_results/022_06/022_06_supported_q_vs_target_ranking.png`
- `benchmark_results/022_06/022_06_supported_q_vs_target_ranking.svg`
- `benchmark_results/022_06/022_06_result.json`
