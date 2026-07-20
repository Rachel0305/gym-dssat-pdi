# 028_09 YC2014 阶段型 MaskablePPO 固定权重迁移至 YC2008

## 目标

复用 YC2014 三个 selected checkpoint，在历史筛选出的 YC2008 上做零训练固定权重迁移；不得在目标年重新训练或选模。

## 冻结配置

- YC2014 observation scaler、22 维观测、DAP 7/30/45/60/80/100、9 动作与 mask 原样复用。
- seed0 checkpoint60、seed1 checkpoint60、seed2 checkpoint120。
- YC2008 treatment 1、IC=1、weather CNYC0801、soil YC99001200、cultivar ZD0985。
- 四基线复用 013_01/028_07，不重跑。

## 执行

1. YC2008 seed0 单季 smoke，核对 22 维观测、6 阶段、mask、模型哈希。
2. smoke 成功后直接复用该季，只新增 seed1/2 两季。
3. 保存 snapshot、阶段动作和完整指标。

## 导师判据

产量、WP_ET、PFP_N 中至少一项严格大于 YC2008 四基线最大值即记为单 seed 成功；其余指标只报告差距。

## 停止条件

输入 provenance、观测维数、阶段覆盖、mask 或模型哈希任一失败即停止；不得现场调参。

