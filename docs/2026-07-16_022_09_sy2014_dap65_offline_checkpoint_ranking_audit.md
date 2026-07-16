# 022_09 SY2014 DAP65 离线checkpoint专项排序审计

## 1. 为什么补做

022_08 的测试 recorded-action Spearman约0.80，证明网络能预测已记录动作在不同场景中的价值变化；但DQN实际决策依赖同一状态下候选动作的相对Q排序。022_06定位DAP65为N300形成前的关键错位阶段，因此在warm-start在线训练前专项检查该阶段。

## 2. 方法边界

- 直接读取022_08已保存的seed×stage×action Q-target CSV；
- 不训练、不调用DSSAT、不重新计算checkpoint；
- 固定比较DAP65可估计的actions 1/2/4/5/7/8；
- 固定检查action7（I15/N100）相对action1（I15/N0）的margin；
- DAP65不同动作样本来自不同前期管理历史，经验target均值只作描述性证据，不是同一状态因果反事实。

## 3. 结果

| seed | DAP65 Q-target Spearman | Q首选 | 经验target首选 | Q7−Q1 | Target7−Target1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.257 | 7 | 1 | +0.014 | −0.485 |
| 1 | 0.314 | 7 | 1 | +0.347 | −0.485 |
| 2 | 0.143 | 7 | 1 | +0.185 | −0.485 |

相对022_06约−0.60的DAP65相关，三个seed均转为正相关，说明离线训练带来部分改善；但：

- 0/3 seed达到预注册Spearman≥0.50；
- 0/3 seed满足Q(action1)>Q(action7)；
- 三个checkpoint仍全部把action7排为DAP65首选；
- seed1的关键反向margin最大（+0.347），因此尤其不适合直接进入warm-start在线实验。

## 4. 判定

预注册判为 **B：整体部分改善，但关键动作对仍反向**。

Claude提出的专项核查是必要的。它避免了只看总体0.80相关就误以为关键N300病灶已经修好。

同时需要纠正“同状态排序”的表述：本审计的DAP65经验target来自不同历史状态，不能严格证明action1在任意同一状态下必然优于action7。因此当前结论不是“应该强制action1”，而是“seed1初始化尚未给出足够证据，不能直接在线扩展”。

## 5. 对warm-start草案的处理

原022_09 seed1 warm-start任务书在执行前被用户中断，现保留并顺延为：

- `prompts/022_10_sy2014_offline_mc_q_warmstart_online_ab_seed1_draft.md`

状态为未执行、暂停。没有产生在线训练、DSSAT评估或checkpoint。

## 6. 下一步

先做固定管理历史的DAP65受控动作交换：

1. 固定DAP1/30/50/85/110动作；
2. 仅把DAP65在action1与action7之间交换；
3. 保持同一DSSAT输入，比较产量、WP、PFP和资源量；
4. 至少选择一个与warm-start策略接近、且预算未在DAP65前耗尽的固定前缀；
5. 若受控结果确认action7劣于action1，再设计有依据的排序约束；若不确认，则不能用历史混杂均值纠正Q。

## 7. 输出

- `prompts/022_09_sy2014_dap65_offline_checkpoint_ranking_audit.md`
- `src/audit_sy2014_dap65_offline_checkpoint_ranking_022_09.py`
- `benchmark_results/022_09/022_09_dap65_seed_action_q_target.csv`
- `benchmark_results/022_09/022_09_dap65_seed_summary.csv`
- `benchmark_results/022_09/022_09_dap65_offline_ranking.png`
- `benchmark_results/022_09/022_09_dap65_offline_ranking.svg`
- `benchmark_results/022_09/022_09_result.json`
