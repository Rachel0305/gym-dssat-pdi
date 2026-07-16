# 022_05 SY2014 行为支持约束固定网格 replay DQN seed1

## 1. 目的

022_03 将 48 个固定网格场景的 288 条好坏经验预填充到 replay 后，DQN 仍会选择数据未覆盖的动作。022_04 对保存 checkpoint 进行零训练支持约束评估时，season15 曾通过联合主判据，但其余 checkpoint 未通过。

本实验检验：从在线训练第一步开始，把随机探索、贪心选择和 checkpoint 评估都限制在固定网格真实覆盖的阶段动作支持集内，能否把 022_04 的单点改善变成稳定训练结果。

## 2. 单变量与冻结条件

相对 022_03，唯一科学变化是加入阶段行为支持 mask。支持集由全部 48 个固定网格场景自动推导，未按成功标签筛选：

| DAP | 支持动作索引 |
|---:|---|
| 1 | 0, 1, 3, 4 |
| 30 | 4, 5, 7, 8 |
| 50 | 4, 5, 7, 8 |
| 65 | 1, 2, 4, 5, 7, 8 |
| 85 | 1, 2, 4, 5 |
| 110 | 0, 1 |

其余条件冻结：SY2014、IC=2、25 维固定标准化、terminal-complete Monte Carlo return/1000、288 条 uniform replay 预填充、不做离线预训练、seed1、60 个在线季、336 次梯度更新、I120/N300 预算及原裁剪逻辑。没有同时加入 budget-exact mask、demo、DQfD、PER、target network 或 TD bootstrap。

## 3. 训练前检查

所有预注册检查均通过：

- 022_03 数据集验证状态为 passed；
- replay 预填充严格为 288 条；
- 六阶段支持集与预注册值完全一致；
- 每个阶段的支持集都来自同时包含成功和失败的场景，而非只保留成功经验；
- 所有支持动作均满足原阶段合法性；
- 随机探索和贪心选择单元检查均未越出支持集；
- DAP110 支持集中不存在施氮动作。

## 4. 结果

| checkpoint | HWAM (kg/ha) | 灌溉 (mm) | 施氮 (kg/ha) | WP_ET | PFP_N | 主判据 | 严格判据 | 动作序列 |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 15 | 11198.94 | 90 | 300 | 2.29 | 37.3 | 通过 | 未通过 | 3,7,7,7,5,1 |
| 30 | 11198.29 | 105 | 300 | 2.24 | 37.3 | 未通过 | 未通过 | 4,4,7,7,5,1 |
| 45 | 11201.76 | 90 | 300 | 2.24 | 37.3 | 未通过 | 未通过 | 4,4,7,7,1,1 |
| 60 | 11201.76 | 90 | 300 | 2.24 | 37.3 | 未通过 | 未通过 | 4,4,7,7,1,1 |

主判据阈值为产量不低于 11077 kg/ha、WP_ET 不低于 2.26、PFP_N 不低于 36.9。严格判据还要求 I≤90 mm、N≤250 kg/ha。

四个 checkpoint 都达到产量和 PFP_N 阈值；season30/45/60 因 WP_ET=2.24 低于 2.26 而未通过主判据。四个 checkpoint 均使用 N300，因此没有严格成功。

## 5. 预注册分支判定

结果为 **B：有信号但不稳定**：

- 主判据通过 1/4；
- 严格判据通过 0/4；
- season60 未通过；
- 不满足 A 分支所要求的至少 3/4 主判据通过、season60 通过且至少一个严格成功。

因此按预注册规则停止自动扩展，不补 seed、不加训练季数、不修改阈值。

## 6. 科学解释

1. 训练期行为支持约束确实能复现 022_04 的 season15 单点改善，说明未覆盖动作的 Q 外推是问题参与因素。
2. 该约束没有使改善稳定化：后续 checkpoint 仍未满足联合判据。
3. 当前更明显的剩余问题是所有 checkpoint 都把氮预算用到 N300；这不是“产量学不会”，而是尚未稳定学到产量与资源效率的联合最优权衡。
4. 支持 mask 只能阻止数据完全未覆盖的动作外推，不能保证在已覆盖动作内部学出正确的 Q 排序，故不能把它写成最终修复。

## 7. 输出文件

- `prompts/022_05_sy2014_support_constrained_grid_replay_dqn_seed1.md`
- `src/run_sy2014_support_constrained_grid_replay_dqn_seed1_022_05.py`
- `benchmark_results/022_05/022_05_pretraining_checks.json`
- `benchmark_results/022_05/022_05_behavior_support_sources.csv`
- `benchmark_results/022_05/022_05_training_seasons.csv`
- `benchmark_results/022_05/022_05_training_stage_actions.csv`
- `benchmark_results/022_05/022_05_update_log.csv`
- `benchmark_results/022_05/022_05_checkpoint_evaluation_summary.csv`
- `benchmark_results/022_05/022_05_checkpoint_stage_actions.csv`
- `benchmark_results/022_05/022_05_checkpoint_daily_values.csv`
- `benchmark_results/022_05/022_05_vs_022_03_training_diagnostics.png`
- `benchmark_results/022_05/022_05_vs_022_03_training_diagnostics.svg`
- `benchmark_results/022_05/022_05_result.json`
- `benchmark_results/022_05/checkpoints/season_015.pt`、`season_030.pt`、`season_045.pt`、`season_060.pt`

## 8. 下一步边界

本分支不能自动继续训练或扫描参数。若继续追求“DQN 同时超过 expert 与 auto 的产量和水氮效率”，下一步应先利用现有数据做零训练诊断，区分：

- 已覆盖动作内部的 Q 排序是否与固定网格 Monte Carlo target 一致；
- N300 是否来自动作请求与预算裁剪的别名问题；
- 是否需要把“请求动作”改成预算余量下的精确可执行动作集合。

在上述诊断前，不应直接再加训练步数或调 reward 系数。
