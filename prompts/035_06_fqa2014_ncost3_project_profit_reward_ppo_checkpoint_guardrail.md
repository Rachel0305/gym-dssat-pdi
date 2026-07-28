# 035_06 FQA2014 linked 自由时序 PPO：提高氮惩罚到 3.0

## 背景

035_04 将训练 reward 改为项目 simple_profit 口径：

```text
final_GRNWT - 1.1 * irrigation - 1.58 * nitrogen
```

但 50K 内最好的 checkpoint 仍然施氮 N240，产量略低于 expert，PFP_N 明显偏低。

035_05 在固定 I45、固定早期分期施氮时机下做了氮边际收益审计，结果显示：

- N0→N80 有明显增产；
- N80 后继续加氮几乎不增产；
- N200→N240 每 kg N 只带来约 0.025 kg/kg 的边际产量，远低于当前 nitrogen_cost=1.58。

因此本任务不限制 PPO 的动作空间，只做最简单的 reward 权重敏感性测试：把 `nitrogen_cost` 提高到 3.0。

## 任务目标

测试：在完整自由动作空间仍然保留的情况下，提高施氮惩罚是否能让 PPO 自己减少 N 用量，同时保持产量和 WP_ET。

## 固定设置

- 站点：FQA
- 年份：2014
- 算法：MaskablePPO
- seed：0
- 总训练步数：50K
- checkpoint：10K / 20K / 30K / 40K / 50K
- DSSAT 管理模式：动态 RL treatment 必须为 `IRRIG=L, FERTI=L`
- 动作空间、动作安全层、总水氮上限、7天最小间隔、后期禁氮等全部沿用 035_04
- 不做多系数扫描
- 不加 seed
- 不扩展站点年份

## reward

训练时每一步的即时 reward：

```text
reward = -1.1 * irrigation - 3.0 * nitrogen
```

终止步额外加入：

```text
+ final_GRNWT
```

整体乘以：

```text
reward_scale = 0.001
```

也就是未缩放季节总 reward 等价于：

```text
final_GRNWT - 1.1 * total_irrigation - 3.0 * total_nitrogen
```

## guardrail

每个 checkpoint 用 DSSAT 确定性回放后评估：

1. interface pass 必须为 True；
2. 产量不低于 official expert；
3. 产量、WP_ET、PFP_N 三者至少一个不低于/超过 official expert；
4. 排序优先级：guardrail_pass、训练同口径 `ncost3_profit`、施氮少、灌溉少。

同时继续报告项目原口径：

```text
project_simple_profit = yield - 1.1 * I - 1.58 * N
```

## 停止线

本任务只跑 `nitrogen_cost=3.0` 这一组。若失败，不现场继续加到 5/7，也不加训练步数。
