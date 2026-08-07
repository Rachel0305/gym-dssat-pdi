# 046_09 SYA originIC raw forecast observation PPO

## 目的

在 046_07 证明“仅归一化已有 observation”导致近似不管理模板后，回到归一化之前的 PPO 主线，只增加天气/完美未来 7 天天气预报输入，检验天气信息本身是否改善 PPO 决策。

## 固定不变

- 输入：originIC。
- 训练年：SYA 2005–2013。
- 验证年：SYA 2014–2023。
- 算法：MaskablePPO。
- 动作：灌溉 `[0,45]` mm；施氮 `[0,80]` kg/ha。
- 训练步数：100000；checkpoint：25000、50000、75000、100000。
- reward、safety mask、7 天操作间隔、季节上限、DAP90 后禁氮均沿用 046_02/042_15 系列。

## 唯一改动

Observation 从原始 25 维扩展为 30 维：

```text
原始 25 维 + rain_today_mm + tmin_today_c + rain_past7_mm + rain_future7_mm + tmean_future7_c
```

本任务不做 z-score、fixed-scale normalization 或 clipping；新增天气变量以原始物理量进入网络。

## 停止线

如果策略仍退化为 DAP1 少量水肥/近似不管理，说明“只加 raw forecast”不足以解决 PPO 决策问题，不能把它包装成天气预报改进成功。
