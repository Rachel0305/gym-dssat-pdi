# 143E2 SY 双通道天气读取 5K 验证记录

## 核心结论

- 新运行2K精确复现：`True`；5K天气响应通过：`True`。
- E2 5K相对no-forecast平均产量差：`60.81` kg/ha；逐年产量胜出：`4/10`。
- E2/no-forecast PFP_N：`41.9591` / `41.7057`；动作序列种类：`9` / `1`。
- 保留进入独立seed验证：`True`；未运行10K或更长训练。

## 5K天气响应

```json
{
  "decision_states": 1381,
  "swap_changed_rows": 25,
  "swap_action_change_rate": 0.018102824040550327,
  "shuffled_changed_rows": 18,
  "shuffled_action_change_rate": 0.013034033309196235,
  "swap_irrigation_changed_rows": 25,
  "swap_nitrogen_changed_rows": 0,
  "all_actions_legal": true,
  "response_passed_gt_1pct": true
}
```

## 严格动作审计

```json
{
  "checkpoint_step": 5000,
  "validation_rows": 10,
  "daily_rows": 1381,
  "transmission_mismatch_fields": 0,
  "off_grid_fields": 0,
  "positive_rows_after_dap1": 60,
  "unique_nonzero_pairs": 3,
  "nonzero_pairs": [
    "I15/N0",
    "I30/N80",
    "I45/N0"
  ],
  "passed": true
}
```

## 解释边界

5K仍为单seed诊断。天气交换只评估同状态策略敏感性，反事实动作未执行进DSSAT。WP_ET因缺少有效ETCP保持unavailable。
