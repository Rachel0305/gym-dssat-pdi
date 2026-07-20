# 026_11 五站点物理深度统一观测 adapter 离线单测

## 目的

为后续多站点联合 PPO 建立固定维度、物理含义一致的土壤水分表示。该任务只实现纯函数和离线单测，不接入训练，不运行 DSSAT。

## 固定表示

公共深度带固定为：

1. 0–15 cm
2. 15–30 cm
3. 30–60 cm
4. 60–90 cm
5. 90–120 cm
6. 120–150 cm

每个深度带输出两个值：

- `sw_vwc`：该带已覆盖部分的厚度加权体积含水量；完全无覆盖时为 0。
- `coverage_fraction`：源剖面对该深度带的覆盖比例，范围 0–1。

由于显式提供 coverage，`sw_vwc=0, coverage=0` 表示缺失深层，不表示真实含水量为0。

保留原始16个非土壤特征，公共观测维度固定为 `16 + 6 + 6 = 28`。

## 边界

- 不修改环境原始 observation space。
- 不补造 HLA/YC 的90 cm以下信息。
- 不修改 reward、action space、IC、DSSAT输入或现有模型。
- 不训练任何模型，不运行 DSSAT。

## 单测

- 输入层厚必须为正，`len(sw)==len(dlayr)`。
- 厚度重叠映射正确。
- 0–150 cm 范围内的水储量 `sum(sw*overlap)` 在映射前后数值守恒。
- 覆盖率在0–1之间，无覆盖带为 `(sw=0, coverage=0)`。
- 原始16个非土壤特征逐位保留。
- 026_10五站点实际运行时层结构全部转换为28维。
- SY/HLA/YC/FQ/LC 的最大覆盖深度分别保持100/90/90/100/150 cm，不外推。

## 输出

- `src/depth_harmonized_observation.py`
- `src/test_depth_harmonized_observation_026_11.py`
- `benchmark_results/026_11/026_11_result.json`
- `benchmark_results/026_11/026_11_five_site_harmonized_profiles.csv`
- `docs/2026-07-17_026_11_depth_harmonized_observation_adapter_unit_test.md`

