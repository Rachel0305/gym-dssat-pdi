# 019_06 FQ2016 leaching-cost sensitivity smoke

## 目的

在 019_05 已证明 leaching-aware reward 链路可运行后，做一个低成本 `leaching_cost` 系数敏感性测试。

## 实验设置

站点年份：FQ2016。

算法：DQN。

训练步数：500 steps。

seed：0。

奖励函数：

```text
reward = max(0, final_grnwt - local_null_yield)
         - water_cost * irrigation
         - nitrogen_cost * nitrogen
         - leaching_cost * delta_cleach
```

测试系数：

- `leaching_cost = 0`
- `leaching_cost = 20`
- `leaching_cost = 50`
- `leaching_cost = 100`

## 原则

- 不修改 DSSAT/PDI 模板。
- 不改已有主线训练脚本。
- 不做长训练。
- 本轮只判断 reward 链路和系数方向，不用 500-step 结果判断最终策略优劣。

## 输出

- 每个系数一套 500-step smoke 结果。
- 汇总 CSV。
- 中文实验记录 MD。

## 判断问题

1. `leaching_cost` 增大是否能降低 `final_cleach` 或 `sum_delta_cleach`？
2. 是否会导致 DQN 完全不施氮或产量明显崩坏？
3. 是否值得进入 5K 短训练？
