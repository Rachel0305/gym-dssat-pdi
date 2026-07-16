# 022_16 SY2014 MC-only续训跨阶段漂移对照

## 1. 目的

022_15在MC+pairwise继续训练后纠正了DAP65目标排序，但DAP110等阶段的支持集argmax发生漂移。本任务只改变一个变量：移除pairwise项，其他起点、优化器重置、学习率、3000次更新和检查点完全一致，以判断pairwise是否为漂移发生的必要条件。

本任务不负责解释漂移的完整机制。

## 2. 预注册分支

- A：MC-only最终至少2/3 seed出现DAP110漂移，说明pairwise不是必要条件；
- B：MC-only为0/3，说明pairwise在当前设置下是必要条件之一；
- C：仅1/3，证据混合；
- D：实现或数值失败。

## 3. DAP110数据来源核对

DAP110合法动作支持为`{0,1}`，动作数确实最少，但原固定网格数据仍有48条DAP110转移：

- action0：40条；
- action1：8条。

因此不能把DAP110预先描述为“转移样本量最少”。动作类别不平衡和Q差距敏感性可以审计，但不是本任务预设的原因。

## 4. 执行设置

- 022_08 seed0/1/2 checkpoint；
- Adam重新初始化，`lr=1e-4`；
- 只优化原216条训练转移的SmoothL1 MC回归；
- 每seed3000次全批次更新；
- checkpoint 0/500/1500/3000；
- 使用与022_15相同的11个未受pairwise状态污染的测试场景；
- 总离线更新9000次，DSSAT调用0次，在线交互0次。

## 5. 结果

### 5.1 DAP110漂移

| Seed | MC-only最终改变数 | MC+pairwise最终改变数 | 两者均改变数 | 两模型最终动作相同数/11 |
|---:|---:|---:|---:|---:|
| 0 | 5 | 6 | 5 | 10/11 |
| 1 | 2 | 2 | 2 | 11/11 |
| 2 | 3 | 3 | 3 | 11/11 |

MC-only在3/3 seed均出现DAP110漂移，且与022_15的逐状态结果高度一致。

### 5.2 全阶段比较

最终支持集argmax改变数：

| DAP | MC-only seed0/1/2 | MC+pairwise seed0/1/2 |
|---:|---|---|
| 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| 30 | 8 / 11 / 11 | 11 / 11 / 8 |
| 50 | 3 / 5 / 5 | 3 / 5 / 5 |
| 65 | 11 / 11 / 11 | 11 / 9 / 11 |
| 85 | 0 / 0 / 0 | 0 / 0 / 0 |
| 110 | 5 / 2 / 3 | 6 / 2 / 3 |

两个训练条件的跨阶段重排结构高度相似。

### 5.3 MC loss

MC-only的MC loss相对起点下降：

- seed0：41.57%；
- seed1：42.55%；
- seed2：43.23%。

这再次表明：MC数值回归继续改善，不保证支持集argmax相对旧checkpoint保持不变。

### 5.4 近似平局描述

DAP110起点`|Q(a0)-Q(a1)|`中位数约为：

- seed0：0.0593；
- seed1：0.0582；
- seed2：0.0463。

最终发生改变的状态起点差距中位数为0.0364，未改变状态为0.0538，提示近似平局可能提高敏感性；但两组范围高度重叠，且发生改变的状态最大起点差距达到0.190，因此不能仅凭“接近平局”解释全部漂移。

## 6. 判定

**A分支：pairwise不是DAP110漂移发生的必要条件。**

可以确认：

- 022_15观察到的跨阶段漂移在MC-only续训中同样跨seed出现；
- pairwise成功纠正DAP65受控排序，但不是造成整体argmax重排的唯一或必要原因；
- 从022_08 checkpoint继续全批次优化MC回归，本身足以产生类似重排；
- 因此不能用022_15的guardrail失败直接判定pairwise方法有缺陷。

不能确认：

- 尚未确定漂移是优化器重置、继续拟合、动作均值排序脆弱、数据不平衡或其他因素中的哪一个造成；
- 不能由本任务直接决定DAP110 guardrail是否应该放宽；
- “pairwise非必要”不等于pairwise完全没有额外影响，seed0仍比MC-only多改变1个DAP110状态。

## 7. 对主线的影响

022_15的局部结果仍然成立：pairwise能快速、跨seed纠正三个受控DAP65排序。其“0/3最终通过”是因为采用了“必须保持旧checkpoint DAP110 argmax完全不变”的严格guardrail，而022_16证明该guardrail连MC-only续训也无法满足。

这说明下一步若要评价pairwise是否可用，不能继续把“相对旧checkpoint零argmax改变”当作未经校准的唯一标准；但也不能现场删除该标准。应先单独审计这些改变后的DAP110动作，比较它们在受控DSSAT回放中的真实季节回报，判断“改变”究竟是性能退化还是旧checkpoint本身的排序被继续训练修正。该审计必须另立任务，且不需要训练。

022_13 warm-start仍未启动。

## 8. 输出

- `prompts/022_16_sy2014_mc_only_continuation_drift_control.md`
- `src/run_sy2014_mc_only_continuation_drift_control_022_16.py`
- `benchmark_results/022_16/022_16_checkpoint_metrics.csv`
- `benchmark_results/022_16/022_16_stage_argmax_trajectory.csv`
- `benchmark_results/022_16/022_16_pair_margin_descriptive.csv`
- `benchmark_results/022_16/022_16_dap110_state_q_trajectory.csv`
- `benchmark_results/022_16/022_16_dap110_mc_only_vs_pairwise_comparison.csv`
- `benchmark_results/022_16/022_16_result.json`
- `benchmark_results/022_16/022_16_mc_only_drift_control.png/.svg`
- `benchmark_results/022_16/checkpoints/`

当前未执行Git commit或push。
