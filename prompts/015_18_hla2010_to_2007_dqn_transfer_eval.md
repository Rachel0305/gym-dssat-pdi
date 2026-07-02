# 015_18 HLA2010 训练 DQN 迁移到 HLA2007 验证

## 目的

在不重新训练的前提下，将 HLA2010 已训练好的 DQN checkpoint 直接用于 HLA2007，检查 DQN 策略是否具有同站点跨年份迁移能力。

## 背景

015_17 已经显示：HLA2010 训练的 DQN checkpoint 迁移到 HLA2015 后，仍能达到接近本地 DQN 和 DSSAT auto 的产量平台，并且用水更少、不施氮。现在继续测试更强的跨年验证：HLA2010 → HLA2007。

## 实验约束

- 不训练，只做 checkpoint 评估。
- 使用指定 Docker 容器：`b2fd6726c8c1`。
- 使用指定 Python：`/opt/gym_dssat_pdi/bin/python`。
- 保持现有统一 DQN 框架、奖励函数、动作空间和管理约束不变。
- 不覆盖旧结果，输出到独立目录。
- 保存日值 CSV、汇总 CSV、过程图、实验记录 MD。

## 待评估模型

- HLA2010 seed0 checkpoint 35000 → HLA2007
- HLA2010 seed1 checkpoint 25000 → HLA2007

## 对照情景

优先使用已有 HLA2007 四情景/筛选结果中的：

- null
- recorded/expert，若已有对应结果
- DSSAT auto，若已有对应结果
- transfer DQN seed0/seed1

如果 HLA2007 没有完整四情景对照，则至少输出 transfer DQN 与 null/DSSAT auto 的可比结果，并在记录中明确说明缺失项。

## 主要判断指标

- GWAD/HWAM 产量
- CWAD 生物量
- 总灌溉量
- 总施氮量
- 最大水分胁迫
- 最大氮胁迫
- 累积 reward
- 管理事件是否合理：是否早期乱灌/乱施，是否在无优化空间时打满资源

## 预期解释

如果 HLA2010 模型在 HLA2007 上仍能明显优于 null，并接近或超过 DSSAT auto/recorded expert，同时节水节氮，则说明当前 DQN 策略具有更强的同站点跨年份泛化潜力。

如果表现明显变差，则说明当前 DQN 仍主要是特定年份策略，需要继续使用逐年训练或加入多年训练方案。
