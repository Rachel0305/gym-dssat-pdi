# 021_07 SY2014 target 更新、离线 TD 残差与三档氮排序审计

## 状态

- 新训练：**未进行**
- DSSAT 调用：**未进行**
- checkpoint 保存与加载核查：**completed**
- target 更新计数核查：**completed**
- 离线 Bellman/TD 残差重算：**completed**
- 固定 replay buffer 交叉评估：**completed**
- N0/N50/N100 完整排序汇总：**completed**
- 历史逐步训练 loss 恢复：**不可完成，原训练未保存该序列**
- 坍缩因果根因：**尚未确认**

## 目的与边界

021_06 已确认 SY2014 seed0 在 15K→20K 期间同时存在 online Q 排序漂移和 wrapper 动作混叠。本轮只复用 021_05 已保存的 5K、10K、15K、20K、25K checkpoint、replay buffer 和 021_06 固定状态，不改 reward、IC、动作空间、预算、target interval 或观测空间，也不启动训练。

本轮回答三个问题：

1. 15K 与 20K 的 target Q 完全相同，是正常更新窗口还是 checkpoint 保存/加载错误；
2. online 网络漂移时，固定经验上的 Bellman 拟合是否同步恶化；
3. 15K→20K 期间，N0、N50、N100 三档之间是否发生比“有氮/无氮”二分指标更广泛的排序变化。

## 方法

### 1. checkpoint 与 target 更新审计

对 5K–25K checkpoint 提取：

- `model.zip`、online 参数和 target 参数的 SHA256；
- `num_timesteps`、`_n_calls`、`_n_updates`；
- `target_update_interval`、探索率和内部总训练步数；
- 保存模型和当前运行时的 Stable-Baselines3 版本。

相邻 checkpoint 同时比较 online 与 target 参数最大绝对差，避免把同一文件被重复加载误判成 target 正常冻结。

### 2. 离线 Bellman/TD 残差

从每个 checkpoint 对应的 `NStepReplayBuffer` 重算：

`TD target = n-step reward + (1-done) × discount × max_a Q_target(s', a)`

`TD error = TD target - Q_online(s, a_executed)`

该指标是**保存时网络对保存时经验的事后 Bellman 残差**，不是训练期间逐次优化实际记录的 loss。

为排除“不同 checkpoint 的 replay buffer 内容变化”这一混杂因素，额外做固定 buffer 交叉评估：分别将 15K 和 20K 的同一份 buffer 交给 15K 与 20K 网络计算残差。

### 3. 三档氮排序

复用 021_06 的 18 个真实固定状态。对每个状态、每个灌溉档位，分别比较 online/target 网络在 15K 与 20K 时的 `N0/N50/N100` 完整排序，并单独统计 N50 与 N100 的相对次序变化。I120/N300 是全季累计资源量，不是单步 Q 动作。

## 结果一：checkpoint 不同，target 冻结窗口与计数一致

五个 `model.zip` 的哈希均不相同，online 参数在每个相邻区间均发生变化；保存模型与运行时均为 SB3 2.8.0。因此没有证据表明 15K/20K 意外加载了同一个 checkpoint。

| checkpoint | `_n_calls` | 已完成 target 更新数 | 距下一次更新调用数 | target 相对上一 checkpoint |
|---:|---:|---:|---:|---|
| 5K | 4,999 | 0 | 5,001 | — |
| 10K | 9,998 | 0 | 2 | 不变 |
| 15K | 14,997 | 1 | 5,003 | 改变 |
| 20K | 19,996 | 1 | 4 | 不变 |
| 25K | 24,995 | 2 | 5,005 | 改变 |

`target_update_interval=10000` 以 `_n_calls` 计数。由于每个 5K 训练段的绝对步数 callback 在边界停止，`_n_calls` 每段相对 `num_timesteps` 少 1 次，因此 10K 和 20K checkpoint 都分别停在 target 更新点之前 2 次和 4 次调用。target 在 10K→15K、20K→25K 改变，在 5K→10K、15K→20K 不变，与该计数严格一致。

因此，15K→20K target 不变是当前训练协议和 target interval 共同形成的正常冻结窗口，不是保存/加载复用错误。但“冻结窗口与策略坍缩同时出现”仍只是时间重合，不能据此证明 target interval 导致坍缩。

## 结果二：固定经验上的 Bellman 拟合在 20K 网络中恶化

### 各 checkpoint 自身 buffer

| checkpoint | 平均绝对 TD error | 中位数 | P95 | 平均 Huber loss | 有符号 TD error 均值 |
|---:|---:|---:|---:|---:|---:|
| 5K | 65.91 | 26.22 | 88.78 | 65.43 | -1.16 |
| 10K | 67.67 | 28.95 | 113.12 | 67.20 | -0.95 |
| 15K | 70.96 | 28.04 | 183.31 | 70.48 | -8.88 |
| 20K | 83.41 | 23.69 | 358.16 | 82.93 | -29.64 |
| 25K | 101.32 | 20.11 | 513.77 | 100.84 | -72.39 |

自身 buffer 的平均绝对残差和尾部残差持续增大，但 buffer 内容也在变化，单独看这张表不能把恶化归因于网络更新。

### 固定 buffer 交叉评估

| 固定经验 | 15K 网络平均绝对 TD error | 20K 网络平均绝对 TD error | 变化 | 15K 网络 P95 | 20K 网络 P95 |
|---|---:|---:|---:|---:|---:|
| 15K buffer | 70.96 | 80.22 | +9.26（+13.0%） | 183.31 | 285.55 |
| 20K buffer | 69.77 | 83.41 | +13.64（+19.6%） | 218.59 | 358.16 |

在两份完全固定的经验上，20K 网络的平均绝对残差和 P95 均高于 15K 网络。因此，15K→20K 的数值拟合恶化不能仅由 replay buffer 分布变化解释；online 网络变化本身与恶化有关。由于 15K 与 20K 的 target 网络相同，这个交叉结果主要反映 online 预测和动作选择变化。

这仍不等于历史训练 loss 曲线，也不能单独证明 Q 过估计、reward 尺度或 target interval 是根因。

## 结果三：online 的 N0/N50/N100 完整排序广泛改变

| 指标（15K→20K） | online | target |
|---|---:|---:|
| 状态×灌溉档位比较数 | 54 | 54 |
| 完整三档氮排序改变 | 42 | 0 |
| N50 与 N100 相对次序改变 | 15 | 0 |
| N0 与 N50 相对次序改变 | 35 | 0 |
| N0 与 N100 相对次序改变 | 21 | 0 |
| 无动作混叠组合中的完整排序改变 | 21/27 | 0/27 |

完整排序变化不仅是“有氮动作与 N0 互换”。15/54 个组合还发生 N50 与 N100 之间的相对次序改变。即使限定到 27 个 9 个请求动作均可区分的无混叠组合，仍有 21 个出现 online 完整排序变化，而 target 排序完全不变。

这进一步支持 021_06 的判断：动作混叠真实存在，但不能单独解释氮策略坍缩；15K→20K 期间 online 网络对氮动作的相对价值发生了广泛漂移。

## 补充结果：第一个与第二个 target 冻结窗口对比

根据外部复核建议，本轮进一步复用同一批固定状态 Q 值，对比两个 target 参数均保持不变的窗口：5K→10K（10K 仍为高产候选）和 15K→20K（发生氮策略坍缩）。该补充仍然没有训练或 DSSAT 调用。

| 指标 | 5K→10K | 15K→20K |
|---|---:|---:|
| online 完整三档氮排序改变 | 36/54 | 42/54 |
| N50 与 N100 次序改变 | 22/54 | 15/54 |
| 有氮/无氮 margin 符号翻转 | 22/54 | 28/54 |
| robust margin flip | 16/54 | 19/54 |
| 无混叠组合完整排序改变 | 21/27 | 21/27 |
| 9动作全局 argmax 改变 | 17/18 | 15/18 |
| 三档氮 Q 平均绝对漂移 | 8.60 | 4.29 |
| target 完整排序改变 | 0/54 | 0/54 |

两个冻结窗口都出现广泛 online 排序漂移。第二窗口的完整排序改变和有氮/无氮翻转仅略高，但第一窗口的 N50/N100 次序变化、全局 argmax 改变和 Q 绝对漂移反而更大。由于第一窗口结束时 10K 策略仍然保持高产，**“冻结窗口内存在大量 Q 排序变化”本身不足以区分是否会发生氮策略坍缩**。

这一结果削弱了“target 冻结时间过长是坍缩主要原因”的直接证据。它不排除 target 更新频率参与训练动力学，但说明 021_08 不能以“缩短 interval 已经有明确机制依据”为前提；即便单变量缩短 interval 后短期结果改善，也必须检查是否只是推迟坍缩。

## 历史 loss 的证据边界

现有 `training_log.jsonl` 只保存 checkpoint 级元数据，没有保存训练期间每次 gradient update 的 loss、TD error、Q 均值或动作分组 Q。因此：

- 不能恢复 15K→20K 期间真实历史 loss 曲线；
- 不能把本轮离线残差称为“训练 loss”；
- 后续若需要验证一次单变量训练，应在不改算法逻辑的前提下前瞻性记录这些诊断量，而不是事后伪造。

## 综合结论

021_07 可以确认：

1. 15K 与 20K checkpoint 是两个独立存档，target 网络被正常保存和加载；
2. 15K→20K target 不变与 `_n_calls` 尚未跨过下一次 10,000 调用更新点完全一致；
3. 同期 online 网络发生广泛的 N0/N50/N100 排序漂移；
4. 在固定 replay buffer 上，20K 网络的 Bellman 残差较 15K 网络明显恶化；
5. 该现象与 SY2014 氮动作从 N300 坍缩到 N0 同期出现，但现有证据仍不足以确定因果根因。

当前最安全的表述是：**探索率时间轴修复后，SY2014 的剩余问题被定位为 target 冻结窗口内 online Q 排序漂移和 Bellman 拟合恶化，并受到后期动作混叠/部分可观测性的共同影响。尚不能断言应修改 target interval、reward 或观测空间中的哪一项。**

## 下一步建议

不立即进行长训练，也不同时修改多个变量。

1. 先保留 021_05–021_07 为修复前后及机制诊断证据；
2. 两个冻结窗口的离线对比显示排序漂移并非坍缩窗口独有，因此不直接把缩短 target interval 当作首选修复；若仍做该单变量 smoke，应把它定位为因果排除实验，并前瞻记录 loss/TD/Q、持续检查是否只是推迟坍缩；
3. “补充累计/剩余氮预算和距上次操作天数”属于 MDP/观测定义修复，应单独交由用户和导师决定，不能与 target 对照混做；
4. 在上述决定前，五站点历史 DQN 结论继续标记为 provisional，不恢复为正式论文结论。

## 输出文件

- `prompts/021_07_sy2014_target_td_full_n_ranking_audit.md`
- `src/audit_sy2014_target_td_full_n_ranking_021_07.py`
- `benchmark_results/021_07/021_07_target_update_audit.csv`
- `benchmark_results/021_07/021_07_network_parameter_differences.csv`
- `benchmark_results/021_07/021_07_td_residual_summary.csv`
- `benchmark_results/021_07/021_07_td_residual_by_action.csv`
- `benchmark_results/021_07/021_07_td_cross_evaluation.csv`
- `benchmark_results/021_07/021_07_full_nitrogen_ranking.csv`
- `benchmark_results/021_07/021_07_full_nitrogen_ranking_changes.csv`
- `benchmark_results/021_07/021_07_freeze_window_comparison.csv`
- `benchmark_results/021_07/021_07_freeze_window_comparison_summary.json`
- `benchmark_results/021_07/021_07_freeze_window_comparison.png`
- `benchmark_results/021_07/021_07_target_td_nitrogen_ranking_summary.png`
- `benchmark_results/021_07/021_07_audit_summary.json`

## Git

- 当前用户已决定稍后手动提交；本任务不重复尝试写入 `.git`。
- 本轮没有执行 push。
- 建议与尚未提交的 021_06 一并选择性提交，避免加入旧模型、临时目录和无关研究文件。
