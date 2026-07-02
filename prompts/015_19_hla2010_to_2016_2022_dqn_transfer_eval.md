# 015_19 HLA2010 训练 DQN 迁移到 HLA2016 / HLA2022 验证

## 目的

在不重新训练的前提下，将 HLA2010 已训练好的 DQN checkpoint 直接用于 HLA2016 和 HLA2022，检查当前 DQN 策略是否能在同站点更多年份上保持可解释的水氮管理表现。

## 背景

015_17 已显示 HLA2010 → HLA2015 迁移有效；015_18 已显示 HLA2010 → HLA2007 迁移有效。现在继续扩展到候选筛选中响应较弱但更干净的年份 HLA2016 和 HLA2022，用于判断该策略是否只在高响应年份成立，还是具有更一般的同站点跨年迁移能力。

## 实验约束

- 不训练，只做 checkpoint 评估。
- 使用指定 Docker 容器：`b2fd6726c8c1`。
- 使用指定 Python：`/opt/gym_dssat_pdi/bin/python`。
- 保持现有统一 DQN 框架、奖励函数、动作空间和管理约束不变。
- 先生成 HLA2016 / HLA2022 的 null 与 DSSAT auto 基线；若无 recorded expert，则不强行构造 recorded。
- 保存日值 CSV、汇总 CSV、过程图、实验记录 MD。
- 不覆盖旧结果。

## 待评估模型

- HLA2010 seed0 checkpoint 35000 → HLA2016 / HLA2022
- HLA2010 seed1 checkpoint 25000 → HLA2016 / HLA2022

## 对照情景

- null
- DSSAT auto
- transfer DQN seed0
- transfer DQN seed1

## 主要判断指标

- GWAD/HWAM 产量
- CWAD 生物量
- 总灌溉量
- 总施氮量
- 最大水分胁迫
- 最大氮胁迫
- 累积 reward
- 管理事件是否合理

## 解释规则

如果 transfer DQN 在 2016/2022 上仍能接近或超过 DSSAT auto，并显著节水节氮，则说明 HLA2010 学到的策略具有更强的同站点跨年份泛化能力。

如果表现只在部分年份成立，则说明当前策略具有年份敏感性，后续应考虑多年训练或按年份筛选适用场景。
