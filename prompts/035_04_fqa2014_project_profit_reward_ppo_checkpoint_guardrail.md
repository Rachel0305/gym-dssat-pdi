# 035_04 FQA2014 linked 自由时序 PPO：项目 simple_profit reward 检验

## 背景

035_02 证明：在 linked action 接口修复后，FQA2014 MaskablePPO 50K 的 10K/20K/30K/40K/50K checkpoints 全部没有通过 guardrail，策略持续接近 I150/N240 打满。

035_03 进一步做了纯离线 reward 对齐审计：

- 当前 logged reward 排名第一是 `noop`，说明它和最终指标不对齐；
- 项目现有 `simple_profit = yield - 1.1 * irrigation - 1.58 * nitrogen` 能把 `water_saving_n160`、`critical_i90_n200`、`split_moderate_n160` 排在前列；
- 历史代理 `Y - I - 5N` 会把 `noop` 排第一，不适合作为当前 FQA2014 自由时序训练主目标。

因此 035_04 只改 reward 口径，不继续盲目加训练步数。

## 任务目标

测试：在 FQA2014 linked 自由时序设置中，把训练 reward 改为与项目综合指标一致的季末 simple_profit 口径后，MaskablePPO 是否能在 50K 内产生通过 guardrail 的 checkpoint。

## 固定设置

- 站点：FQA
- 年份：2014
- 算法：MaskablePPO
- seed：0
- 总训练步数：50K
- checkpoint：10K / 20K / 30K / 40K / 50K
- DSSAT 管理模式：动态 RL treatment 必须为 `IRRIG=L, FERTI=L`
- 动作空间、动作安全层、总水氮上限、7天最小间隔、后期禁氮等全部沿用 034_05 / 035_02
- 不做超参数扫描
- 不加 seed
- 不扩展站点年份

## reward

训练时每一步的即时 reward：

```text
reward = -1.1 * irrigation - 1.58 * nitrogen
```

终止步额外加入：

```text
+ final_GRNWT
```

最后整体乘以：

```text
reward_scale = 0.001
```

也就是未缩放季节总 reward 等价于：

```text
final_GRNWT - 1.1 * total_irrigation - 1.58 * total_nitrogen
```

本任务不加入 stress relief bonus，不加入 1620 hard gate。

## guardrail

每个 checkpoint 用 DSSAT 确定性回放后评估：

1. interface pass 必须为 True；
2. 产量不低于 official expert；
3. 产量、WP_ET、PFP_N 三者至少一个不低于/超过 official expert；
4. 排序优先级：guardrail_pass、project_simple_profit、施氮少、灌溉少。

## 输出

- `benchmark_results/035_04_fqa2014_project_profit_reward_ppo_checkpoint_guardrail/evaluation/035_04_training_checkpoints.csv`
- `benchmark_results/035_04_fqa2014_project_profit_reward_ppo_checkpoint_guardrail/evaluation/035_04_checkpoint_eval_summary.csv`
- `benchmark_results/035_04_fqa2014_project_profit_reward_ppo_checkpoint_guardrail/evaluation/035_04_guardrail_ranking.csv`
- `docs/035_04_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_record.md`

## 停止线

无论结果好坏，本任务只跑这一组 50K seed0。若仍全失败，不现场加步数、不调 reward、不切算法。
