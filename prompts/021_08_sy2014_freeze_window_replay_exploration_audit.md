# 021_08 SY2014 两个冻结窗口的 replay 与探索差异离线审计

## 目标

在不训练、不调用 DSSAT、不修改任何配置的前提下，对比：

- 5K→10K：target 不变、策略仍保持高产；
- 15K→20K：target 不变、氮策略发生坍缩。

判断两个窗口在探索率、replay buffer 请求动作分布、氮动作样本占比和标量 reward 分布上是否存在可区分差异，为后续单变量实验选择提供依据。

## 必须复用

- 021_05 SY2014 seed0 修复探索协议后的 5K/10K/15K/20K checkpoint；
- 各 checkpoint 已保存的 replay buffer；
- 021_06/021_07 已保存的固定状态 Q 和动作映射。

## 边界

- 不启动训练；
- 不调用 DSSAT；
- 不改 reward、IC、target interval、探索参数、动作空间、预算或观测空间；
- 不覆盖已有结果；
- 不把 replay 中保存的请求动作冒充 wrapper 实际执行动作；
- 若历史数据没有保存“探索动作/贪心动作来源”或“实际执行动作”，必须标记为不可恢复，不得推断；
- 标量 reward 可以审计分布，但没有组件日志时不得伪造水成本、氮成本和产量增益的逐项分解。

## 方法

1. 从 endpoint replay buffer 的环形位置精确提取两个窗口新增的 transition，而不是比较整个 buffer 快照；
2. 核对 start/end buffer 的 `pos`、`full`、容量和新增 transition 数；
3. 对每个窗口统计：
   - 9个请求动作的频数和比例；
   - N0/N50/N100、I0/I15/I30比例；
   - no-op、正氮、正灌溉、水氮同时操作比例；
   - 动作熵；
   - reward 均值、中位数、正/负/零比例、P95、最大值；
   - episode 终止样本比例；
4. 记录窗口起止 checkpoint 的 exploration rate；明确 replay 没有保存每个动作究竟来自随机探索还是 greedy；
5. 可以用窗口 endpoint 网络计算“保存动作是否等于 endpoint greedy argmax”，但必须标记为事后对照，不代表动作生成时的真实探索来源；
6. 生成 CSV、JSON、PNG 和中文实验记录。

## 判定纪律

- 若第二窗口正氮动作占比明显下降且探索率接近下限，只能记为与坍缩同期的候选机制，不能直接写成因果；
- 若两个窗口动作分布接近，则削弱“replay 动作构成单独解释坍缩”的假设；
- target interval、探索率或 reward 的任何训练对照必须另写 prompt，先 smoke，并保持单变量。

## 输出

- `src/audit_sy2014_freeze_window_replay_021_08.py`
- `benchmark_results/021_08/021_08_window_summary.csv`
- `benchmark_results/021_08/021_08_action_distribution.csv`
- `benchmark_results/021_08/021_08_replay_audit_summary.json`
- `benchmark_results/021_08/021_08_replay_exploration_comparison.png`
- `docs/2026-07-14_021_08_sy2014_freeze_window_replay_exploration_audit.md`
