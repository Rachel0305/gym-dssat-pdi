# 040_44 SYA lowIC PPO 策略年际响应性审计

## 背景

040_40 的 MaskablePPO 在 SYA lowIC 验证年份 2014–2023 上得到较好的综合指标，但 040_43 的五情景日过程图显示：PPO 灌溉措施在多个年份几乎完全一致，施氮措施也只有小幅变化。这削弱了“算法根据年份天气自适应决策”的解释力。

本任务不训练新模型，不改 reward，不改约束，只读取 040_40 checkpoint100k 的逐日评估输出，检查 PPO 的措施是否确实存在“固定日程化”倾向。

## 数据来源

- PPO 逐日输出：
  `benchmark_results/040_40_sya_lowIC_ppo_yield_guardrail_v3/daily_outputs/SYA/SYA_<year>_seed0_ckpt100000_daily.csv`
- 验证年份：2014–2023
- 模型：040_40 checkpoint100k
- 输入条件：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`

## 审计内容

1. 汇总每年 PPO 的灌溉和施氮事件序列。
2. 统计灌溉序列和施氮序列的唯一模式数。
3. 按 DAP1–30、DAP31–60、DAP61–90、DAP91+ 汇总各年份灌溉量。
4. 对每次灌溉事件记录：
   - 事件 DAP；
   - 灌溉量；
   - 事件前 7 天降雨；
   - 事件后 7 天降雨；
   - 事件前水分胁迫；
   - 事件后水分胁迫。
5. 输出一个判定：
   - 若 2014–2023 的灌溉序列唯一模式数为 1，则判定为 `fixed_irrigation_schedule_detected`；
   - 若唯一模式数大于 1，则判定为 `year_responsive_irrigation_schedule_detected`。

## 边界

- 本任务只能说明已训练 PPO 在验证年份上的行为是否随年份变化。
- 本任务不能单独证明 PPO 为什么这样决策，也不证明 SAC 或其他算法会更好。
- 本任务不允许根据审计结果现场修改 reward、动作约束或 checkpoint 选择。

