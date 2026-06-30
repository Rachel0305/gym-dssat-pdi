# 012_19 HLA2015 文献式 DQN warm-start / imitation 探针

## 背景

012_15 文献式 DQN 能让 seed0 学会灌溉，但最终施氮 N150。

012_16/012_18 的 Q-value 诊断显示：

- reward 离线复评分并不偏好 N150；
- Q 网络在关键状态中高估含氮动作；
- n-step return 改善了灌溉排序，但没有解决氮动作高估。

SB3 2.8.0 当前没有现成 PrioritizedReplayBuffer。因此本轮不手搓复杂 PER，先做更可控的 warm-start / imitation 探针。

## 目的

测试：如果先用固定扫描中较优的 I60/N0 思路给 Q 网络一个“少氮、适量灌溉”的初始排序，后续 DQN 训练是否仍然漂回 N150。

## 设计

与 012_17 相比，保持以下设置不变：

- HLA2015；
- IC=1；
- 文献式 terminal reward；
- 25 个水氮离散动作；
- I120/N150 总预算；
- 操作窗口和 7 天间隔；
- `n_steps=5`；
- seed0；
- 5000 timesteps。

新增一步：

1. 初始化 DQN；
2. 生成一条启发式 warm-start 轨迹：
   - 目标总量约 I60/N0；
   - 在灌溉窗口内选择若干 `I12_N0` 动作；
   - 不施氮；
3. 用这条轨迹对 Q 网络做少量监督式预训练，使目标动作 Q 值高于其他动作；
4. 再正常运行 DQN 5K。

## 注意

本轮不是最终方案，只是诊断：

- 如果 warm-start 后仍然回到 N150，说明在线 DQN 更新会覆盖初始排序；
- 如果 warm-start 能保持少氮高产，说明问题确实主要在初始 Q 排序和探索/回放，而不是环境或 reward。

## 执行要求

1. 新建脚本，不覆盖旧结果；
2. 先 200 steps smoke test；
3. smoke 通过后跑 5000 steps；
4. 保存 daily CSV、event_summary.json、debug log、model、warm-start 数据和中文记录。
