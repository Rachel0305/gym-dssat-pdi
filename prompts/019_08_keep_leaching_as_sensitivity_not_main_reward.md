# 019_08 决策记录：淋洗惩罚暂不并入正式 reward

## 背景

019_03–019_05 已确认：

- PDI/DSSAT 能输出氮淋洗相关变量；
- `cleach/NLCC` 可以被 gym-DSSAT 运行链路读取；
- reward 中加入 `leaching_cost * delta_cleach` 的技术链路是通的。

019_06–019_07 进一步测试了 FQ2016 的淋洗惩罚系数：

- 500-step smoke 中，`leaching_cost=20` 一度表现较好；
- 5K 短训练中，`leaching_cost=20` 在 2000-step checkpoint 表现较好，但到 4000/5000 steps 退化为不灌溉、只施氮；
- `leaching_cost=0` 在 5K 下反而更稳定。

## 当前决策

暂不把氮淋洗惩罚并入正式 DQN reward。

正式主线仍使用当前统一的经济型奖励：

```text
reward = max(0, final_grnwt - local_null_yield)
         - water_cost * irrigation
         - nitrogen_cost * nitrogen
```

氮淋洗相关结果暂时作为敏感性分析和环境效益扩展保留。

## 理由

1. 淋洗惩罚链路已经证明可用，但短训练显示该项会显著改变学习动态。
2. `leaching_cost=20` 存在中间 checkpoint 好结果，但最终 checkpoint 不稳定。
3. 当前导师确认的核心目标是：产量、水分利用效率、氮肥利用效率尽量同时超过 expert 和 DSSAT auto。
4. 过早把淋洗项加入正式 reward，会引入新的权重选择问题，导致主线叙事复杂化。
5. 当前更稳妥的写法是：先证明 DQN 在产量和水氮效率上能形成优势，再把氮淋洗作为环境影响敏感性讨论。

## 后续使用方式

淋洗相关代码、表格和记录保留：

- 用于回答导师关于“是否考虑氮损失/环境效益”的问题；
- 用于论文讨论部分说明：本研究已验证氮淋洗变量可以接入 reward，但正式优化目标暂聚焦于产量和水氮投入效率；
- 后续如导师要求扩展环境目标，可从 `leaching_cost=20` 的 best checkpoint + seed 复核继续。

## 下一步

回到正式 DQN 主线：

1. 不修改正式 reward；
2. 不继续长训练淋洗惩罚版本；
3. 继续五站点优化空间审计和 DQN 成功策略筛选；
4. 重点检查每个站点是否有能力在产量、水量、氮量上同时优于 expert 和 DSSAT auto。
