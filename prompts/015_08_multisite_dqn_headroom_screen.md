# 015_08 multisite DQN headroom screen prompt

## 目标

基于已有结果，不重新训练、不重新跑 DSSAT，统一筛选当前最值得继续做正式 DQN 的站点年份，并给出下一步优先级。

## 需要统一比较的证据

- HLA2010：已知存在稳定成功策略，seed0/seed1 都能给出合理高产少氮行为。
- HLA2015：已有结果显示可解释但 seed 敏感，需要看是否作为正式训练候选。
- YC2014：已知 DQN seed0/seed1 都能追平专家产量，且 seed0 少氮更优，但 015_07 headroom probe 说明当前约束下产量已经接近平台。
- FQ2016：已有结果显示是当前最像“有可解释优化空间”的候选年之一，适合继续做正式训练复核。

## 输出要求

1. 生成一个 `multisite_dqn_headroom_screen_summary.csv`，包含：
   - site
   - year
   - scenario_status
   - current_best_label
   - yield_kg_ha
   - irrigation_mm
   - fertilizer_kg_ha
   - max_water_stress
   - max_nitrogen_stress
   - stability_flag
   - headroom_flag
   - next_action

2. 生成一个中文 Markdown 记录：
   - 为什么这些站点年份被保留或淘汰；
   - 哪些可以直接进入正式 DQN 长训练；
   - 哪些只适合作为成功案例或对照，不建议继续硬追更高产；
   - 哪些应该排在后面。

3. 如果已有证据表明某站点年份“产量平台已到、继续训练也难超越专家”，要明确写出来，不要为了继续训练而训练。

## 初步优先级

- 第一优先：FQ2016
- 第二优先：HLA2010
- 第三优先：HLA2015
- 第四优先：YC2014（保留成功案例，但不作为追求更高产的首选）

## 判断标准

- 若某站点年份仍有明确增产空间且 seed 间可复现，则进入正式 DQN。
- 若某站点年份已接近平台但可稳定少投投入，则作为成功案例保留。
- 若某站点年份只是产量追平、没有超越空间，则不再把“更高产”作为目标。
