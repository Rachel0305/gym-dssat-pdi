# 021_27 SY2014 示范 return-to-go 信用分配离线 A/B

## 边界

本任务完全离线。Control与Treatment唯一差异是示范长期目标：5-step return对比终止前完整折扣return-to-go。即时reward、1-step target、标准化、网络、PER和loss权重均未改变。

## 非零动作目标审计

| transition_index | dap | action | immediate_reward | control_5step_return | treatment_full_return_to_go | control_horizon | treatment_horizon | treatment_terminal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 22 | 1 | -15.0 | -15.0 | 908.788513 | 5 | 118 | True |
| 49 | 29 | 4 | -265.0 | -265.0 | 991.119873 | 5 | 111 | True |
| 62 | 42 | 7 | -515.0 | -515.0 | 1431.441406 | 5 | 98 | True |
| 76 | 56 | 4 | -265.0 | -265.0 | 2240.518799 | 5 | 84 | True |
| 99 | 79 | 1 | -15.0 | -15.0 | 3157.098877 | 5 | 61 | True |

## 学习曲线

| arm | milestone_updates | noop_accuracy | nonzero_action_recall | positive_expert_margin_fraction_nonzero | mean_expert_margin_nonzero | median_total_gradient | gradient_clip_fraction | milestone_gate_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_5step | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | False |
| control_5step | 10 | 0.354839 | 0.0 | 0.0 | -0.186807 | 1.675263 | 0.0 | False |
| control_5step | 25 | 0.877419 | 0.0 | 0.0 | -0.188598 | 1.434753 | 0.0 | False |
| control_5step | 50 | 0.987097 | 0.0 | 0.0 | -0.251126 | 1.199703 | 0.0 | False |
| control_5step | 100 | 1.0 | 0.0 | 0.0 | -0.389789 | 1.017283 | 0.0 | False |
| control_5step | 250 | 1.0 | 0.0 | 0.0 | -0.65891 | 0.72447 | 0.0 | False |
| control_5step | 500 | 1.0 | 0.0 | 0.0 | -0.93199 | 0.574471 | 0.0 | False |
| control_5step | 1000 | 1.0 | 0.0 | 0.0 | -3.898948 | 0.569999 | 0.0 | False |
| treatment_full_return_to_go | 0 | 0.154839 | 0.0 | 0.0 | -0.197212 | nan | nan | False |
| treatment_full_return_to_go | 10 | 0.354839 | 0.0 | 0.0 | -0.18463 | 2.780484 | 0.0 | False |
| treatment_full_return_to_go | 25 | 0.864516 | 0.0 | 0.0 | -0.183434 | 2.534765 | 0.0 | False |
| treatment_full_return_to_go | 50 | 1.0 | 0.0 | 0.0 | -0.269113 | 2.435537 | 0.0 | False |
| treatment_full_return_to_go | 100 | 1.0 | 0.0 | 0.0 | -0.493644 | 2.468065 | 0.0 | False |
| treatment_full_return_to_go | 250 | 1.0 | 0.0 | 0.0 | -0.897725 | 0.830138 | 0.0 | False |
| treatment_full_return_to_go | 500 | 1.0 | 0.0 | 0.0 | -0.785405 | 0.343395 | 0.0 | False |
| treatment_full_return_to_go | 1000 | 1.0 | 0.0 | 0.0 | -1.013211 | 0.752318 | 0.0 | False |

## 预注册判定

- control passing milestones: `[]`
- treatment passing milestones: `[]`
- treatment consecutive pass: `False`
- branch: `C_full_return_to_go_not_sufficient`

该结果只证明或否定离线示范信用目标的作用，不代表在线DQN产量或稳定性；不自动进入在线训练。

## 1-step 与长期TD梯度方向审计

Treatment已把五个动作的长期目标从负成本改为 `908.79–3157.10` 的正回报，但1-step目标仍保留即时成本。对相同初始化网络逐事件计算Smooth L1对所选Q值的导数后发现：五个事件的1-step斜率全部为`+1`（梯度下降会压低动作Q），完整return斜率全部为`-1`（梯度下降会抬高动作Q）；两者均进入Huber饱和区，等权相加均严格为0。

因此完整return-to-go没有奏效，并不是长期目标没有生效，而是它与保留的1-step成本TD在关键动作上发生了精确梯度抵消。当前margin监督还要在155个no-op状态和共享网络参数的背景下单独承担稀疏动作学习，最终未能让任何非零动作成为argmax。

这把下一步进一步收紧为：若继续DQfD路线，应先离线检验“示范样本只使用完整return TD + margin、取消示范上的冲突1-step TD”这一单变量，而不是调margin数值或直接在线训练。

## 输出

- `benchmark_results/021_27/021_27_target_audit.csv`
- `benchmark_results/021_27/021_27_target_validation.json`
- `benchmark_results/021_27/021_27_learning_curve.csv`
- `benchmark_results/021_27/021_27_nonzero_event_predictions.csv`
- `benchmark_results/021_27/021_27_update_diagnostics.csv`
- `benchmark_results/021_27/021_27_nonzero_td_gradient_conflict_audit.csv`
- `benchmark_results/021_27/021_27_td_gradient_conflict_summary.json`
- `benchmark_results/021_27/021_27_summary.json`
- `benchmark_results/021_27/021_27_return_to_go_credit_assignment_ab.png/.svg`
