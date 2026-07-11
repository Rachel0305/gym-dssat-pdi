# 019_05 FQ2016 leaching-aware reward smoke test

## 目的

在 019_03 和 019_04 已确认 `cleach/tleachd/cnox` 可从 PDI full state 读取后，做一个低成本 smoke test，验证 leaching-aware reward 能否正常运行。

## 关键约束

- 不修改 DSSAT/PDI 模板。
- 不修改既有主线训练脚本。
- 不长训练。
- 使用指定 Docker 容器 `b2fd6726c8c1`。
- 使用指定虚拟环境 `/opt/gym_dssat_pdi/bin/python`。
- 先跑 FQ2016 seed0，500 steps，checkpoint 500。

## 奖励函数

在现有 baseline-relative reward 基础上只新增一项：

```text
reward = max(0, final_grnwt - local_null_yield)
         - water_cost * irrigation
         - nitrogen_cost * nitrogen
         - leaching_cost * delta_cleach
```

其中：

- `delta_cleach = max(0, cleach_t - cleach_{t-1})`
- `cleach` 从 gym/PDI full state 读取，不从 ordinary observation 读取。
- 本轮 `leaching_cost=50` 只是 smoke 系数，用于验证链路和方向，不作为正式论文参数。

## 输出

- 每步日值 CSV，必须包含：
  - `cleach`
  - `tleachd`
  - `cnox`
  - `delta_cleach`
  - `leaching_cost_term`
  - `yield_gain`
  - `water_cost_term`
  - `nitrogen_cost_term`
  - `reward`
- checkpoint summary CSV。
- 中文实验记录 MD。

## 判断标准

如果脚本完成、日值表有 leaching 字段、reward component 数值正常、DQN 动作链路仍能输出灌溉施肥动作，则 019_05 通过。

本轮不根据 500 steps 的产量判断策略优劣。
