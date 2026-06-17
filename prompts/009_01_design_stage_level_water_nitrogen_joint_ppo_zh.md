# 009_01 阶段级水氮联合 PPO 设计文档

## 目标

设计下一阶段 PPO 框架，在保留 008 系列已验证有效的 soft-stress 阶段灌溉框架的基础上，重新引入氮肥决策。

本任务仅为设计和准备阶段。

**009_01 不进行 PPO 训练。**

## 背景

008 系列的成功框架表明，在以下条件下 PPO 能够在水分胁迫年份产生可解释的灌溉决策：

- 对站点年份进行真实水分胁迫潜力筛选；
- PPO 按 DAP 阶段决策，而非每日决策；
- S1 灌溉被封锁；
- 天气预报/胁迫信息门控灌溉机会；
- 去除硬性最低灌水量；
- soft SWFAC 惩罚使水分胁迫信号在奖励中提前可见；
- 氮肥固定为 N150。

当前成功 PPO 案例的主要科学局限是：由于氮肥固定，尚未实现真正意义上的水氮联合优化。

## 设计原则

不回到无约束的每日水氮联合 PPO。

采用：

- DAP 阶段决策 wrapper；
- 农学基础氮先验；
- PPO 控制追加氮；
- 无硬性最低灌水的天气/胁迫辅助灌溉门控；
- soft SWFAC 惩罚；
- 明确记录水分和氮肥动作的各自贡献。

## 阶段动作设计方案

采用五个 DAP 阶段：

| 阶段 | DAP | 灌溉设计 | 氮肥设计 |
|---|---:|---|---|
| S1 建苗期 | 1-20 | 封锁，上限 0 mm | 固定基础氮 50 kg ha⁻¹，PPO 追加为 0 |
| S2 营养生长前期 | 21-45 | 门控允许时 PPO 0-40 mm | 固定基础氮 50 kg ha⁻¹，PPO 追加为 0 |
| S3 中期生长 | 46-75 | 门控允许时 PPO 0-40 mm | PPO 追加氮 0-50 kg ha⁻¹ |
| S4 后期水分风险 | 76-100 | 门控允许时 PPO 0-40 mm | PPO 追加氮 0-20 kg ha⁻¹ |
| S5 终期水分风险 | 101-结束 | 门控允许时 PPO 0-40 mm | PPO 追加氮 0-20 kg ha⁻¹ |

全季氮肥范围：

- 最低：100 kg ha⁻¹
- 最高：190 kg ha⁻¹

这意味着氮肥不再固定，但早期农学供氮受保护，避免 PPO 从零开始崩塌。

## 009_02 Smoke Test 奖励函数设计

从 008_17 的 stronger soft-stress 配置出发：

- topwt_delta_coef = 0.001
- grnwt_delta_coef = 0.020
- terminal_grnwt_coef = 0.010
- water_cost = 0.050
- soft SWFAC 阈值 = 0.05
- swfac_excess_cost = 3.0
- swfac_day_cost = 1.0

仅对 PPO 控制的追加氮加入小额氮素成本：

- 候选范围：nitrogen_cost = 0.020 到 0.050

第一版 smoke test 使用：

- nitrogen_cost = 0.030

009_01 阶段不扫多个氮素成本系数。

## 必须记录的指标

现有阶段 wrapper 已记录：

- raw_stage_action_irrigation
- raw_stage_action_nitrogen
- stage_action_amir
- stage_action_anfer
- stage_nitrogen_base
- stage_nitrogen_extra_cap
- stage_extra_anfer
- season_cumulative_irrigation
- season_cumulative_n
- growth_reward
- terminal_reward

009 系列框架报告必须额外汇总：

- total_irrigation
- total_n
- total_base_n
- total_ppo_extra_n
- total_ppo_extra_irrigation
- 各阶段施氮量
- 各阶段灌溉量
- final GRNWT
- SWFAC 胁迫天数
- NSTRES 胁迫天数
- 首次灌溉 DAP
- 首次 PPO 控制施氮 DAP
- 灌溉是否打满 cap
- 氮肥是否打满 cap
- 各阶段 raw action 模式

## 009_02 Smoke Test 建议

009_01 设计确认后，运行一次小规模 smoke test：

- 站点：HLA
- 年份：2004
- seed：0
- 训练步数：5000

**成功信号：**

- 灌溉非零且未打满 cap；
- 总氮在 110-190 kg ha⁻¹ 之间；
- S3/S4/S5 的 PPO 追加氮非零，或 PPO 主动选择低氮但 NSTRES 不严重；
- NSTRES 不全季持续偏高；
- GRNWT 接近 008_15 HLA 灌溉基准，至少不崩溃；
- S1 无灌溉；
- 无不合理的早期灌溉。

**失败信号：**

- 总灌溉为 0；
- 总氮接近基础氮下限 100 kg ha⁻¹ 且 NSTRES 严重；
- 总氮每次都打满 190 kg ha⁻¹ 且无产量依据；
- 灌溉打满 cap；
- GRNWT 崩溃；
- S1 发生灌溉。

## 禁止事项

- 不修改 site-packages 奖励文件；
- 不修改 `my_data/`；
- 009_01 阶段不进行训练；
- 不覆盖 008 系列输出；
- 单次 smoke test 通过前不运行多 seed；
- 不删除现有 008 HLA/FQA 输出。
