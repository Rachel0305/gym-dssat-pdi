# 022_15 SY2014 固定 pairwise+MC 唯一一次离线训练

## 1. 目的

检验 022_12/022_14 推导出的统一 pairwise 约束，能否在一次固定的离线训练中纠正 DAP65 action1/action7 排序，同时保持原 MC 回归和其他阶段的策略结构。

## 2. 冻结配置

- 起点：022_08 seed0/1/2 的 2000-update checkpoint；
- `lambda_pairwise=0.0060048738917845`；
- `loss=L_MC+lambda_pairwise*L_pair`；
- Adam，`lr=1e-4`；022_08未保存优化器状态，因此明确重新初始化Adam；
- 每seed固定3000次全批次更新；
- 观察点为0/500/1500/3000；
- 中间checkpoint只观察轨迹，不允许事后挑选；
- DQN离线更新9000次，DSSAT调用0次，在线交互0次。

三个pairwise状态中的 `W120_critical__N200_early` 来自022_08原测试场景。因此，原12场景指标不再具有完整场景独立性；主要guardrail使用未受pairwise监督的其余11个测试场景。

## 3. 预注册成功标准

单seed必须在最终3000步同时满足：

1. 三个受控DAP65状态的 `Q(a1)-Q(a7)>0`；
2. 原MC训练loss相对起点增加不超过1%；
3. 11个未污染测试场景中DAP1与DAP110支持集argmax改变数均为0；
4. 数值有限。

至少2/3 seed最终通过才进入A分支。中间通过、最终回退不能挑中间checkpoint报成功。

## 4. 结果

### 4.1 Pairwise排序

三个seed在500步时，三个受控前缀的margin已经全部由负转正，并一直保持到3000步。

| Seed | 起点margin范围 | 500步margin范围 | 3000步margin范围 |
|---:|---:|---:|---:|
| 0 | -0.081至-0.015 | 0.357至0.477 | 0.481至0.499 |
| 1 | -0.408至-0.293 | 0.357至0.405 | 0.479至0.512 |
| 2 | -0.237至-0.108 | 0.334至0.369 | 0.464至0.516 |

这证明pairwise项有足够推力，且022_14基于一次SGD式虚拟步的局部外推明显偏保守。正式训练使用Adam并重新初始化优化器，实际轨迹不能由单步局部线性/几何估算精确预测。

### 4.2 MC回归

三个seed的MC训练loss均没有恶化，反而下降：

| Seed | MC loss相对变化 |
|---:|---:|
| 0 | -41.50% |
| 1 | -42.34% |
| 2 | -42.88% |

因此失败不是MC数值拟合变差。

### 4.3 其他阶段策略扰动

DAP1在三个seed均保持0个argmax改变，但DAP110未能保持：

| Seed | DAP1改变数 | DAP110改变数 | 最终通过 |
|---:|---:|---:|---|
| 0 | 0 | 6 | 否 |
| 1 | 0 | 2 | 否 |
| 2 | 0 | 3 | 否 |

此外，DAP30、DAP50、DAP65等阶段也存在明显支持集argmax变化；DAP85保持稳定。

## 5. 判定

**C分支：0/3 seed最终通过。**

准确结论是：

- 受控pairwise约束成功、快速并跨seed一致地纠正了目标DAP65局部排序；
- 但继续离线训练没有保持其他阶段的策略结构，违反预注册DAP110零改变guardrail；
- 因此当前checkpoint不能作为warm-start成功模型，022_13继续暂停；
- 不得挑选500或1500步checkpoint，因为它们同样存在DAP110改变，且任务已规定中间点不能用于事后择优。

## 6. 因果边界

本任务没有设置“从同一022_08 checkpoint继续3000步、但只使用MC loss”的对照。因此，当前数据只能证明 **MC+pairwise联合继续训练后的整体结果不满足guardrail**，不能把所有跨阶段漂移单独归因于pairwise loss。

同样不能因为MC loss下降就认为策略结构更正确：回归误差下降与支持集argmax稳定是不同指标。

## 7. 停止线与下一步

按照预注册规则，本轮不能自动进入warm-start或在线DQN，也不能现场降低训练步数、挑中间checkpoint或修改lambda。

若继续追究机制，最小且有判别力的下一步不是调pairwise参数，而是先由用户决定是否允许一个 **纯离线MC-only继续训练对照**，用于区分：

- 原MC网络继续优化本身就会改变DAP110排序；还是
- pairwise共享梯度额外造成了跨阶段漂移。

该对照属于新的问题，不在022_15授权范围内，本任务不自动执行。

## 8. 文件

- `prompts/022_15_sy2014_fixed_pairwise_mc_offline_training.md`
- `src/run_sy2014_fixed_pairwise_mc_offline_training_022_15.py`
- `benchmark_results/022_15/022_15_training_loss.csv`
- `benchmark_results/022_15/022_15_checkpoint_metrics.csv`
- `benchmark_results/022_15/022_15_pair_margin_trajectory.csv`
- `benchmark_results/022_15/022_15_stage_argmax_trajectory.csv`
- `benchmark_results/022_15/022_15_seed_final_status.csv`
- `benchmark_results/022_15/022_15_result.json`
- `benchmark_results/022_15/022_15_offline_pairwise_training.png/.svg`
- `benchmark_results/022_15/checkpoints/`

当前未执行Git commit或push。
