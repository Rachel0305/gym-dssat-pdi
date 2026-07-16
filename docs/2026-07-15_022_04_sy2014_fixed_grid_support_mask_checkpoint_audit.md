# 022_04 SY2014 固定网格行为支持 mask checkpoint 审计

## 结论

预注册判定为 **B_partial_support_effect**。未覆盖动作 Q 外推被确认是 022_03 失败的参与机制，但不是充分解释。

## 方法

- 没有训练或更新参数。
- Control 复用 022_03 已保存的四个确定性评估。
- Treatment 加载同一 checkpoint，只允许选择 48 个固定网格在相应阶段实际覆盖过的动作。
- 支持集由全部 288 条好坏 transition 自动计算，不使用成功标签筛选。

行为支持集：

| DAP | 支持动作 |
|---:|---|
| 1 | 0,1,3,4 |
| 30 | 4,5,7,8 |
| 50 | 4,5,7,8 |
| 65 | 1,2,4,5,7,8 |
| 85 | 1,2,4,5 |
| 110 | 0,1 |

每个阶段都同时包含来自主判据成功和失败场景的经验；支持集是原始合法动作集的子集。

## Treatment 结果

| checkpoint | HWAM | I | N | WP_ET | PFP_N | 主判据 | 严格判据 | 动作序列 |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 15 | 11198.94 | 90 | 300 | 2.29 | 37.3 | True | False | [3,7,7,7,5,1] |
| 30 | 11198.61 | 105 | 300 | 2.24 | 37.3 | False | False | [4,7,7,7,5,1] |
| 45 | 11202.06 | 90 | 300 | 2.24 | 37.3 | False | False | [4,7,7,7,1,1] |
| 60 | 11202.06 | 90 | 300 | 2.24 | 37.3 | False | False | [4,7,7,7,1,1] |

24 个 checkpoint×阶段状态中，原始 argmax 有 8 次落在行为支持集之外。加入支持约束后，season15 从未覆盖的 DAP1 动作7改为已覆盖动作3，并通过产量、WP_ET、PFP_N联合主判据。这是支持外推确实参与失败的直接证据。

但仅 1/4 checkpoint 通过，season60 未通过，且 0/4 达到严格 I90/N250 等级，因此不能声称支持 mask 已解决问题，也不能把 season15 单点包装成稳定成功策略。

## 下一步建议

如果继续，应另立 022_05：从训练第一步开始，在探索、贪心选择和评估中统一使用同一固定行为支持 mask；其他设置与 022_03 完全一致。该实验检验“训练期间避免未覆盖动作外推”是否能让成功从单个 post-hoc checkpoint 变成稳定轨迹。

本任务按 B 分支停止，没有自动启动新训练。

## 输出

- `prompts/022_04_sy2014_fixed_grid_support_mask_checkpoint_audit.md`
- `src/audit_sy2014_fixed_grid_support_mask_checkpoints_022_04.py`
- `benchmark_results/022_04/022_04_stage_action_support.json/.csv`
- `benchmark_results/022_04/022_04_control_treatment_summary.csv`
- `benchmark_results/022_04/022_04_treatment_stage_actions.csv`
- `benchmark_results/022_04/022_04_treatment_daily_values.csv`
- `benchmark_results/022_04/022_04_q_support_audit.csv`
- `benchmark_results/022_04/022_04_support_mask_checkpoint_audit.png/.svg`
- `benchmark_results/022_04/022_04_result.json`

未执行 Git commit 或 push。
