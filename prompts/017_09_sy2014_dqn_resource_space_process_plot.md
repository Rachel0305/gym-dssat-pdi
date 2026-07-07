# 017_09 SY2014 DQN 资源使用与四情景过程图

## 目标

在不重新训练、不增加算力消耗的前提下，复用 `017_08` 已生成的 SY2014 结果，完成以下四件事：

1. 绘制 SY2014 最佳 DQN checkpoint 的四情景过程图：
   - null
   - recorded expert
   - DSSAT auto
   - DQN transfer_SY2014_ckpt15000
2. 检查 DQN 为什么使用满 I120/N300 预算：
   - 对比四情景产量、生物量、灌溉总量、施氮总量、水分胁迫、氮胁迫。
3. 判断 SY2014 是否存在“节水节氮且高产”的空间：
   - 目前只基于已有四情景结果作诊断，不进行新训练。
4. 输出可复核结果：
   - 四情景日值表；
   - 四情景管理事件表；
   - 四情景汇总表；
   - 高对比过程图；
   - 中文实验记录 MD。

## 约束

- 不训练新模型。
- 不覆盖 `017_08` 原始结果。
- 不修改原始输入数据。
- 使用已有最佳模型结果：
  `transfer_SY2014_ckpt15000`。
- 绘图使用 Python/matplotlib。
- 图中必须包含：
  - 降雨；
  - 水分胁迫；
  - 氮胁迫；
  - 灌溉与施肥事件；
  - GWAD 与 CWAD；
  - 统一 DQN reward proxy 的累计曲线。

## Reward proxy 定义

为便于四情景公平比较，统一使用与 DQN 训练目标一致的 proxy：

```text
daily_proxy_reward = -1.0 * irrigation_mm - 5.0 * fertilizer_kg_ha
terminal_bonus = max(0, final_GWAD - null_GWAD)
cumulative_proxy_reward = sum(daily_proxy_reward) + terminal_bonus_at_final_day
```

注意：

- 这个 proxy 不是 DSSAT 内置 reward。
- 它用于比较四情景在同一训练目标下的相对得分。

## 判断口径

- 如果 DQN 产量超过 recorded/auto，但水氮投入也更高，则判定为“高产型成功”，不是“节水节氮型成功”。
- 如果 DQN 在更少或相近水氮投入下超过 recorded/auto，才可判定为“节水节氮型成功”。
- 若 DQN 用满 I120/N300，说明当前 reward 与预算条件下，高产策略仍倾向资源上限；下一步应做资源成本/预算敏感性诊断，而不是直接加长训练。
