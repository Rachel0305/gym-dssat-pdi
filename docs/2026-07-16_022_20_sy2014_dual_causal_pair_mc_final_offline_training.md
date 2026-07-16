# 022_20 SY2014 双因果pair+MC唯一最终离线训练

## 1. 硬停止背景

本任务是预先约定的最后三步中的第二步。022_19只有在梯度兼容时才允许本轮唯一离线训练；本轮少于2/3 seed最终同时满足两组因果排序，则立即停止SY阶段型DQN训练线，不进入022_21在线验证，也不再添加loss、权重或训练步数补丁。

## 2. 固定配置

```text
L = L_MC
  + 0.003002436945892297 * L_DAP65
  + 0.06764715488665898  * L_DAP110
```

- 起点：022_08三个seed checkpoint；
- Adam重新初始化，lr=1e-4；
- 每seed3000次全批次更新；
- checkpoint为0/500/1500/3000；
- 不含软锚定，不扫描权重，不延长训练，不挑中间模型；
- 总离线更新9000次，DSSAT调用0次，在线交互0次。

## 3. 预注册成功判据

最终单seed必须同时满足：

1. DAP65三个因果状态全部正确；
2. DAP110八个因果状态全部正确；
3. MC loss相对起点增加不超过1%；
4. 数值有限。

至少2/3 seed通过才允许022_21。

## 4. 结果

| Seed | 最终DAP65正确 | 最终DAP110正确数 | MC loss变化 | 最终通过 |
|---:|---|---:|---:|---|
| 0 | 3/3 | 6/8 | -41.60% | 否 |
| 1 | 3/3 | 5/8 | -42.39% | 否 |
| 2 | 3/3 | 5/8 | -43.22% | 否 |

三seed在500步时已经全部纠正DAP65，并保持到3000步；但DAP110在任何观察checkpoint均未与DAP65同时达到全部正确：

- seed0：DAP110为4/8、5/8、6/8；
- seed1：4/8、4/8、5/8；
- seed2：4/8、4/8、5/8。

seed2在起点DAP110为8/8正确，但当DAP65于500步被纠正时，DAP110已下降为4/8，与022_18“保护区与DAP65修复区不重叠”的结论一致。

DAP110平均SmoothL1最终较中期下降，但仍存在负margin。这说明组内平均loss改善不等于8个状态逐一满足排序约束。

## 5. 判定

**C分支：0/3 seed最终通过。**

可以确认：

- DAP65因果pair可被稳定、跨seed学习；
- 当前统一梯度预算下，DAP110八状态约束未能逐一满足；
- MC数值回归继续改善，但不能保证两组因果动作排序同时正确；
- 从梯度单步兼容到多步训练成功之间仍存在明确差距；
- 没有任何中间checkpoint可作为被遗漏的成功模型。

不能声称：

- 不能说DQN算法在所有可能设计下绝对不可能成功；
- 不能说确定性优良策略不存在，022_01已证明其存在；
- 不能把本轮失败事后归因于某一个未经对照的机制。

## 6. 最终停止决定

按照运行前已经锁定的规则：

- 不启动022_21在线验证；
- 022_13 warm-start草案永久保持未执行状态；
- 不追加DAP110权重、不改为逐样本max loss、不增加训练步数、不加入蒸馏或正则化；
- SY阶段型DQN当前调试线在022_20结束。

当前可向导师报告的准确结论为：

> 固定生育阶段动作空间中存在18个严格达标的确定性策略，证明目标可达；DQN能够稳定学习DAP65局部因果排序，但在三seed统一配置下无法同时保持八个DAP110因果排序，因此当前阶段型DQN训练方案尚未形成稳定、全面优于expert与auto的策略。

是否启动完全不同的算法/数据路线，应作为新的导师决策，而不是022系列继续补丁。

## 7. 输出

- `prompts/022_20_sy2014_dual_causal_pair_mc_final_offline_training.md`
- `src/run_sy2014_dual_causal_pair_mc_final_offline_022_20.py`
- `benchmark_results/022_20/022_20_training_loss.csv`
- `benchmark_results/022_20/022_20_checkpoint_metrics.csv`
- `benchmark_results/022_20/022_20_causal_margin_trajectory.csv`
- `benchmark_results/022_20/022_20_descriptive_stage_changes.csv`
- `benchmark_results/022_20/022_20_seed_final_status.csv`
- `benchmark_results/022_20/022_20_result.json`
- `benchmark_results/022_20/022_20_final_offline_training.png/.svg`
- `benchmark_results/022_20/checkpoints/`

当前未执行Git commit或push。
