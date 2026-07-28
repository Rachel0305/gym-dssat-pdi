# 035_02 FQA2014 linked MaskablePPO checkpoint guardrail 重训 prompt

## 背景

034_05 证明 linked 修复后 PPO 动作真实进入 DSSAT，但 FQA2014 50K 终点策略不优。

035_00 证明 linked 自由时序动作空间中存在比 PPO 50K 更优、且产量/WP_ET 可超过 official expert 的固定规则策略。

035_01 证明当前 stress-aware reward 与最终汇报指标不完全一致：reward 第一是 `noop`，simple_profit 第一是 `water_saving_n160`；PPO 50K 既不是 reward 最优，也不是指标最优。

因此下一步不应只看最终模型，也不应只按训练 reward 选模型，而应保存中间 checkpoint，并用 DSSAT 真实回放后的指标 guardrail 选择。

## 目标

训练 FQA2014 linked MaskablePPO 单 seed，并保存多个 checkpoint。每个 checkpoint 必须真实进入 DSSAT 评估，然后按外部指标 guardrail 选择候选。

## 固定设置

- 站点年份：FQA2014。
- 算法：MaskablePPO。
- seed：0。
- 总训练步数：50,000。
- checkpoint：10K, 20K, 30K, 40K, 50K。
- linked 管理必须为 `IRRIG=L, FERTI=L`。
- 动作网格、上限、7天间隔、后期禁氮、stress-aware reward 沿用 034_05。
- 不修改 reward，不扫描超参数。

## Guardrail

每个 checkpoint 用 DSSAT 评估后，计算：

- 产量；
- 灌溉总量；
- 施氮总量；
- WP_ET；
- PFP_N；
- simple_profit；
- 与 official expert 的差值；
- interface_pass。

候选通过条件：

1. `interface_pass == True`；
2. 产量不低于 official expert；
3. 产量、WP_ET、PFP_N 三个指标中至少一个超过 official expert。

候选排序：

1. 通过 guardrail 优先；
2. simple_profit 高优先；
3. 氮投入少优先；
4. 灌溉少优先。

## 停止线

本任务只跑一次 50K checkpoint 训练，不根据结果追加步数或改 reward。

若没有任何 checkpoint 通过，结论是“当前 PPO 配置在 linked 真实动作链路下未产生可用 checkpoint”，下一步应改 reward/训练机制，而不是现场调参。

若有 checkpoint 通过，也只能称为 FQA2014 seed0 候选，不能直接扩展到全站点。

