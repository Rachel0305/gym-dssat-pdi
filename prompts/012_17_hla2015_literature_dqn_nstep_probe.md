# 012_17 HLA2015 文献对齐版 DQN n-step return 探针

## 背景

012_15 使用文献式终端产量奖励 + 投入成本 + 25 个水氮离散动作，HLA2015 seed0 5K 结果为：

- 产量约 7652 kg/ha；
- 灌溉 102 mm；
- 施氮 150 kg/ha。

012_16 的 Q-value 诊断显示：

- reward 离线复评分并不偏好 N150；
- 但是 DQN 的 Q 网络在关键状态里把含氮动作排高；
- 问题更像是终端奖励信用分配/Q 排序不稳定，而不是 reward 公式本身错误。

## 目的

在不改环境、不改 reward、不改动作空间的前提下，只加入 DQN 的 `n_steps=5`，测试多步回报能否让终端产量奖励更快回传到前中期水氮动作，从而减少错误的 N150 策略。

## 单变量原则

与 012_15 相比，本轮只改：

```text
n_steps: 1 -> 5
```

其他保持不变：

- HLA2015；
- IC=1；
- 文献式 reward；
- 25 个水氮动作；
- 窗口与预算约束；
- seed0；
- 5K timesteps；
- policy net_arch = [256, 256, 256]；
- learning_rate = 1e-5；
- batch_size = 1024；
- exploration 设置不变。

## 执行要求

1. 新建脚本，不覆盖 012_15；
2. 先 200 steps smoke test；
3. smoke 通过后跑 seed0 5000 steps；
4. 保存 daily CSV、event_summary.json、debug log、model、PDI snapshot；
5. 记录中文实验 MD。

## 判断标准

重点不是单看产量，而是看：

1. 是否仍然 N150；
2. 是否接近 012_08 固定扫描中较优的 I60/N0；
3. 是否比 012_15 的 I102/N150 更合理；
4. 如果 n-step 仍失败，说明只靠多步回报不足以解决 Q 排序问题，下一步才考虑 prioritized replay 或 warm-start。
