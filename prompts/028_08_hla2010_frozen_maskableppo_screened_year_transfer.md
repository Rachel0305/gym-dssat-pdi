# 028_08 HLA2010 阶段型 MaskablePPO 固定权重迁移至已筛选年份

## 目标

复用 HLA2010 已训练的 3 个预注册 selected checkpoint，零训练迁移到 HLA2007、2015、2016、2022。回答固定策略在同站点跨年条件下能否满足导师当前汇报规则。

## 不重复原则

- HLA2010 已完成，不重评。
- 四基线复用 020_11/028_06 快照，不重跑。
- 仅新增 4 年 × 3 seed = 12 个固定权重 RL 季节。

## 冻结对象

- seed0：027_02 checkpoint 180。
- seed1：027_03 checkpoint 60。
- seed2：027_03 checkpoint 60。
- observation scaler：027_01_attempt2。
- 阶段点、动作空间、mask、网络权重和确定性评估协议均冻结。
- 目标年不得重新训练、选择 checkpoint 或修改 scaler。

## 汇报判据

每个站点年内，分别计算四基线对产量、WP_ET、PFP_N 的最大值。RL 至少一个指标严格大于相应四基线最大值时，记为 `advisor_any_metric_win=True`；其余两个指标只报告差值和百分比，不自行定义“接近”容差。

## 工程判据

- 模型评估前后 SHA256 不变。
- 12/12 季完成，0 invalid action。
- 每季 6 个阶段动作。
- 每季保存 DSSAT snapshot 与动作表。

## 停止条件

任何输入、维度、mask 或模型哈希不一致即停止；不得现场训练或改参数补救。

