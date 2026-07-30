# 040_30 SYA lowIC 040_28 checkpoint75k 代表年份五情景日过程图

## 目的

仅做可视化审计，不训练、不改模型、不改奖励、不改输入。

用户判断 040_28 checkpoint 75k 的总体结果仍不够理想，因为平均产量没有超过 expert，水氮节省幅度也有限。因此本任务只执行一个低成本检查：抽取 3 个代表年份，绘制五情景日过程图，看 PPO 候选措施是否具有农艺可解释性。

## 固定输入

- 站点：SYA / SY
- 输入版本：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- RL 候选：`040_28_sya_lowIC_ppo_i240_swfac_guardrail_reward`
- checkpoint：75000
- 年份：
  - 2014：通过 90% expert 产量 guardrail，但仍有水分胁迫日，用于看“通过但有胁迫”的措施。
  - 2017：未通过 90% expert 产量 guardrail，用于看主要失败年。
  - 2022：产量接近/略高于 expert 且水分胁迫低，用于看相对成功年。

## 五情景

1. null
2. recorded farmer
3. DSSAT auto
4. official extension expert
5. 040_28 MaskablePPO candidate, seed0, checkpoint75000

## 绘图内容

每个年份输出一张 4×2 日过程图，保持此前五情景图样式：

1. 降雨、最高气温、最低气温
2. 土壤水分 SWTD
3. 水分胁迫 WSPD
4. 氮胁迫 NSTD
5. 灌溉事件
6. 施氮事件
7. 籽粒产量和生物量轨迹
8. 统一奖励口径下的累计奖励

注意：040_28 PPO 冻结评估 daily CSV 不保存 `SoilWat.OUT` 的 SWTD。为避免伪造数据，若没有 PPO 原始 SWTD 快照，则土壤水面板不强行填 PPO 线，只保留可追溯来源。

## 奖励绘图口径

累计奖励仅作为同图诊断，不作为训练结果重新评价。对五情景统一使用 040_28 风格的诊断口径：

```text
step_reward_unscaled =
  harvest_day ? 0.158 * final_grain_yield : 0
  - 1.1 * irrigation_mm
  - 1.58 * nitrogen_kg_ha
  + 10 * irrigation_mm * max(previous_WSPD - current_WSPD, 0)
  + 5 * nitrogen_kg_ha * max(previous_NSTD - current_NSTD, 0)
  - 50 * max(current_WSPD - 0.05, 0)

plot_reward = step_reward_unscaled * 0.001
```

该口径用于避免旧图里“common reward”和 040_28 训练奖励口径不一致。

## 停止线

- 不训练。
- 不挑新 checkpoint。
- 不根据图结果修改模型。
- 若某年份缺少必要日值或基线快照，记录失败并停止解释该年份。

