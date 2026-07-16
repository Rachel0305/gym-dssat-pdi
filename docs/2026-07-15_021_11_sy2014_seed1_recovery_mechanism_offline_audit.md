# 021_11 SY2014 seed1 中期下降与25K恢复机制离线审计

## 状态

- 新训练：**未进行**
- DSSAT调用：**未进行**
- 固定状态Q排序：**completed**
- target更新核查：**completed**
- replay新增经验审计：**completed**
- 固定buffer TD残差：**completed**
- 唯一因果根因：**尚未确认**

## 目的

021_10延长探索配置的seed1在10K为N300/11046 kg/ha，15K降至N100/9620，20K降至N50/7352，25K又恢复N300/10794。本轮不训练、不调用DSSAT，只复用checkpoint、replay buffer和021_06固定观测，判断下降与恢复伴随哪些网络和经验变化。

## 方法与证据边界

1. 对10K、15K、20K、25K读取online/target参数哈希、`_n_calls`和epsilon；
2. 将021_06的18个SY2014真实固定观测输入四个seed1网络，对同一状态比较9动作Q值；
3. 比较每个灌溉档位下N0/N50/N100完整排序和全局argmax；
4. 从各endpoint环形buffer中精确提取10K–15K、15K–20K、20K–25K新增的4999条transition；
5. 用每个窗口起点和终点网络分别评估同一份endpoint buffer，计算事后Bellman残差。

固定观测来自seed0参考轨迹，仅用于“同状态网络比较”，不冒充seed1实际评估轨迹；replay动作是请求动作，不等于wrapper实际执行动作；事后TD残差不是历史训练loss。

## 运行异常记录

首次脚本读取固定状态清单时使用了不存在的`actual_dap`列，实际字段名为`dap`，程序在生成结论前安全停止。修正该单一字段映射后重新运行成功；期间没有训练、DSSAT调用或旧结果覆盖。

## 结果一：target更新与下降和恢复都同时出现

| checkpoint | `_n_calls` | epsilon | target哈希状态 |
|---:|---:|---:|---|
| 10K | 9,998 | 0.729 | A |
| 15K | 14,997 | 0.593 | B（相对10K改变） |
| 20K | 19,996 | 0.457 | B（相对15K不变） |
| 25K | 24,995 | 0.321 | C（相对20K改变） |

10K→15K target更新时，确定性策略从N300降到N100；20K→25K target再次更新时，策略从N50恢复到N300。同一个“target更新”事件既伴随下降也伴随恢复，因此不能把target同步本身简单定性为好或坏。15K→20K target冻结期间，策略仍从N100继续降至N50。

## 结果二：恢复窗口的Q排序重组最强

| 窗口 | online完整氮排序改变 | online全局argmax改变 | target完整氮排序改变 | target argmax改变 |
|---|---:|---:|---:|---:|
| 10K–15K | 25/54 | 8/18 | 50/54 | 16/18 |
| 15K–20K | 17/54 | 6/18 | 0/54 | 0/18 |
| 20K–25K | 39/54 | 12/18 | 29/54 | 9/18 |

20K→25K恢复伴随online和target动作排序的大规模重组；15K→20K进一步下降时target完全不变、online变化相对较少。该结果说明25K恢复确实对应网络动作偏好的重新排列，但不能证明是哪一项训练机制触发了有利重排。

## 结果三：replay正氮请求先降后小幅恢复

| 窗口 | 正灌溉请求 | 正氮请求 | N0 | N50 | N100 | 水氮同时为正 | 动作熵 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10K–15K | 69.6% | 64.9% | 35.1% | 31.7% | 33.2% | 44.8% | 0.991 |
| 15K–20K | 70.9% | 58.3% | 41.7% | 26.0% | 32.2% | 39.1% | 0.977 |
| 20K–25K | 67.5% | 60.9% | 39.1% | 26.9% | 34.0% | 41.3% | 0.978 |

中期下降时正氮请求减少6.6个百分点，恢复窗口又增加2.7个百分点；正灌溉比例一直较高。方向与确定性策略一致，但幅度远小于季节确定性施氮N300→N100→N50→N300的变化。这表明随机探索产生的buffer总体分布和确定性greedy策略不能直接等同。

## 结果四：25K恢复时Bellman残差反而最差

固定endpoint buffer交叉比较：

| 固定buffer | 起点网络平均绝对TD残差 | 终点网络平均绝对TD残差 | 变化 |
|---|---:|---:|---:|
| 15K buffer：10K→15K | 60.14 | 69.03 | +8.89 |
| 20K buffer：15K→20K | 69.91 | 101.40 | +31.49 |
| 25K buffer：20K→25K | 106.23 | 148.98 | +42.75 |

TD残差持续恶化，25K恢复高产和N300时残差反而最大。因此“25K恢复是因为Q网络整体Bellman拟合改善”不成立。恢复更可能来自部分关键状态的动作排序发生有利重组，而不是全局数值误差下降。

## 综合结论

021_11支持以下判断：

1. 延长探索使seed1没有永久锁死在低氮策略，但其greedy策略仍会大幅振荡；
2. target更新不是单向原因：它既与下降同步，也与恢复同步；
3. replay中的氮请求先降后回升，与策略方向一致，但不足以单独解释剧烈的确定性结果变化；
4. 策略恢复不依赖整体TD残差改善；
5. 当前更像是bootstrapped Q排序在部分状态上不稳定重组，并受到隐藏预算/操作间隔、动作混叠和经验分布反馈共同影响。

最安全的通俗表述是：**延长探索让DQN“跌倒后还能爬起来”，但没有让它稳定地走直线。**

## 下一步建议

不建议马上堆seed2或继续扫探索参数。021_06已经确认agent看不到累计/剩余施氮量和`last_operation_dap`，但这些隐藏变量会决定请求动作是否被裁剪，破坏了状态的Markov完整性。下一步应先由用户/导师决定是否建立独立的“观测空间Markov修复”实验：

- 增加剩余灌溉预算、剩余氮预算、距上次操作天数；
- reward、动作、预算、DQN超参数全部不变；
- 先做环境单元测试和smoke，再做单seed短对照；
- 不把它描述为为了结果调参，而是让agent能够观察决定动作实际执行结果的必要状态。

若暂不允许改变观测定义，则应接受当前结论：延长探索降低永久坍缩风险，但现有POMDP式wrapper下仍存在跨checkpoint和跨seed策略不稳定。

## 输出

- `prompts/021_11_sy2014_seed1_recovery_mechanism_offline_audit.md`
- `src/audit_sy2014_seed1_recovery_021_11.py`
- `benchmark_results/021_11/021_11_network_audit.csv`
- `benchmark_results/021_11/021_11_fixed_state_q_values.csv`
- `benchmark_results/021_11/021_11_nitrogen_ranking_changes.csv`
- `benchmark_results/021_11/021_11_argmax_changes.csv`
- `benchmark_results/021_11/021_11_q_change_summary.csv`
- `benchmark_results/021_11/021_11_replay_window_summary.csv`
- `benchmark_results/021_11/021_11_replay_action_distribution.csv`
- `benchmark_results/021_11/021_11_fixed_buffer_td_comparison.csv`
- `benchmark_results/021_11/021_11_seed1_recovery_audit.png`
- `benchmark_results/021_11/021_11_audit_summary.json`
- 本记录。

## Git

- 用户计划稍后手动提交；本轮未执行commit或push。
