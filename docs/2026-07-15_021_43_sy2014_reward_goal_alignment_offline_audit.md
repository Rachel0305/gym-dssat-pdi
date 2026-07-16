# 021_43 SY2014 reward与导师目标一致性审计记录

## 当前公式

`R = max(0, Y - 5408) - 1*I - 5*N`。本轮只重算已有checkpoint，不训练、不改reward。

## 结果

| online_seed | reward_best_checkpoint | reward_best_yield | reward_best_irrigation | reward_best_nitrogen | reward_best_score | reward_best_passes_expert_gate | best_passing_checkpoint | best_passing_score | reward_advantage_of_failing_choice | nitrogen_cost_break_even | objective_mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 750 | 11205.0 | 90.0 | 200.0 | 4707.0 | True | 750 | 4707.0 | 0.0 | nan | False |
| 1 | 1000 | 10787.0 | 90.0 | 150.0 | 4539.0 | False | 250 | 4192.0 | 347.0 | 2.6867 | True |
| 2 | 750 | 10787.0 | 90.0 | 150.0 | 4539.0 | False | 250 | 4177.0 | 362.0 | 2.5867 | True |

预注册分支：**A**。seed1/2均出现当前reward偏好低于expert产量的低氮策略，存在系统性目标错位。

## 含义与边界

如果低氮策略虽然未达到expert产量，却获得更高reward，那么DQN选择它不一定是训练失败，而可能是正确优化了当前标量目标。盈亏平衡系数只用于解释当前权衡，不授权现场把氮成本改成该数值；是否采用“产量硬门槛+门槛内资源效率”需要单独预注册。
