# 021_44 SY2014 产量可行性优先reward离线审计记录

## 候选公式

`R_candidate = R_current + 1620 * I[Y >= local official expert yield]`，其中1620严格由冻结预算与成本`1*120+5*300`推导，不按本次结果拟合。

## 结果

| online_seed | old_best_checkpoint | old_best_passes_gate | old_best_reward | candidate_best_checkpoint | candidate_best_passes_gate | candidate_best_reward | candidate_best_yield | candidate_best_irrigation | candidate_best_nitrogen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 750 | True | 4707.0 | 750 | True | 6327.0 | 11205.0 | 90.0 | 200.0 |
| 1 | 1000 | False | 4539.0 | 250 | True | 5812.0 | 11175.0 | 75.0 | 300.0 |
| 2 | 750 | False | 4539.0 | 250 | True | 5797.0 | 11175.0 | 90.0 | 300.0 |

预注册分支：**A**。候选可行性优先公式在三个在线seed上均把最高分排序转向expert-gate可行策略。

## 边界

这只是反事实排序检查，现有reward代码没有修改，也未授权训练。下一步若执行，必须单独实现terminal bonus、先做单元/smoke验证，再进行1K单变量A/B。
