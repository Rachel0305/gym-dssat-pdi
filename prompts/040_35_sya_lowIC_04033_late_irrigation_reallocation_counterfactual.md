# 040_35 SYA lowIC 040_33 后期灌溉重分配反事实

## 目的

040_33 把 040_28 中频繁 15 mm 小灌和 40 kg/ha 小肥的问题压住了，措施形态更合理；但 2014、2017、2022 五情景图仍显示后期水分胁迫，尤其 2017 产量明显不足。

本任务不训练 PPO，不修改 reward，不修改动作空间，不修改 lowIC 输入。只做固定管理日程的 DSSAT 反事实回放：

> 在 040_33 checkpoint 25k 的原始 PPO 管理方案基础上，把一个中期 45 mm 灌溉事件后移到 DAP96、DAP103 或 DAP110，检查后期水分胁迫和产量是否改善。

## 数据与范围

- 输入数据：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 年份：2014、2017、2022
- 原始候选：040_33 `checkpoint_step=25000`
- 原始动作来源：
  `benchmark_results/040_33_sya_lowIC_ppo_i240_swfac_guardrail_coarse_actions/evaluation/040_33_checkpoint_validation_summary.csv`

## 分支

每个年份固定比较：

1. `original_replay`
   - 原样回放 040_33 checkpoint 25k 的 PPO 动作。

2. `move_DAPxx_to_96`
   - 从原始策略中选择一个 DAP31–90 的 45 mm 灌溉事件，移除它；
   - 在不违反 7 天最小灌溉间隔、不超过单次 45 mm、不超过季节总量的前提下，改放到 DAP96。

3. `move_DAPxx_to_103`
   - 同上，目标后移到 DAP103。

4. `move_DAPxx_to_110`
   - 同上，目标后移到 DAP110。

若目标 DAP 与已有灌溉事件间隔冲突，则在目标 DAP 后逐日寻找第一个合法日期；若找不到合法日期，则该分支标记失败，不临时改规则。

## 被移动事件选择规则

为了避免事后挑选，事件选择在运行前固定：

- 候选集合：DAP31–90 内灌溉量等于 45 mm 的 PPO 事件；
- 选择候选集合中 DAP 最大的事件，也就是原策略中最晚的中期 45 mm 灌溉；
- 理由：当前问题是后期水分胁迫，因此先测试“把最靠近后期的中期灌溉再往后移”是否有效，而不是从早期随机挑一个事件。

## 成功判读

本任务只回答一个问题：

> 040_33 的失败是否主要来自“水用得太早，后期没有留水”？

预注册判据：

- 若某个后移分支在 2017 上同时满足：
  - `final_grnwt` 高于 `original_replay`；
  - `swfac_days_gt_0p05` 低于 `original_replay`；
  - 总灌溉量与总施氮量不高于原始 PPO；
  则说明“后期留水/后移灌溉”是值得进入下一轮 PPO 约束或 reward 设计的方向。

- 若 2017 没有任何后移分支改善，则不应继续围绕“简单后移灌溉”调 PPO。

## 输出

- `docs/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual_record.md`
- `benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/tables/040_35_action_plan.csv`
- `benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/tables/040_35_action_edits.csv`
- `benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/tables/040_35_counterfactual_summary.csv`
- `benchmark_results/040_35_sya_lowIC_04033_late_irrigation_reallocation_counterfactual/tables/040_35_counterfactual_daily.csv`

## 禁止事项

- 不训练模型。
- 不改 lowIC 输入。
- 不改 040_33 结果。
- 不根据结果追加新的后移日期。
- 不把反事实成功直接表述为 PPO 已经学会该策略；它只能作为下一轮训练设计依据。
