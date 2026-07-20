# 026_10 五站点观测结构与土层深度审计

## 背景

026_09 在 HLA2010 seed0 smoke 的第一次策略动作之前停止：SY 冻结 PPO 模型要求 25 维观测，而 HLA 原始环境返回 24 维。初步定位为 SY 有 9 个土壤含水量层、HLA 有 8 个。该结果不是策略失败，不能进入正式跨站点评估。

## 目标

用最小成本、仅 reset 的方式审计 HLA2010、YC2014、FQ2016、LC2010 的实际原始观测维度，并与 SY2014 的 25 维训练结构和 observation scaler 对齐；同时读取各站点实际使用 SOIL.SOL 中目标剖面的层底深度。

## 严格边界

- 不训练 PPO/DQN。
- 不执行任何策略动作；每个站点最多 reset 一次并立即关闭。
- 不修改原始 MZX/WTH/SOL/CUL。
- 不复用/覆盖既有正式结果目录。
- 不用补零、复制末层或删除特征来强行适配。
- 本任务只审计，不实现跨站点适配器，不报告跨站点科学结论。

## 固定案例

- HLA2010：复用 026_09 smoke 已准备输入。
- YC2014、FQ2016：复用 020_12 已验证输入准备函数，使用 DQN-linked 管理入口，仅 reset。
- LC2010：复用 017_11 修复后的输入准备函数，使用 null 管理入口，仅 reset。
- SY2014：以 021_24 observation scaler 的 25 个标签和正式 SY 环境结构作为训练端参照，不重复运行。

## 输出

- `benchmark_results/026_10/026_10_observation_schema.csv`
- `benchmark_results/026_10/026_10_soil_profile_layers.csv`
- `benchmark_results/026_10/026_10_result.json`
- `docs/2026-07-17_026_10_five_site_observation_schema_and_soil_depth_audit.md`

## 判定

- A：所有外站均为 25 维且 9 土层，可恢复原 026_09 直接迁移。
- B：至少一个外站维度/土层数不同，但剖面深度可追溯；停止正式迁移，下一步预注册“按物理深度守恒映射”的观测 adapter。
- C：剖面深度或实际使用 soil id 无法追溯；停止，先修输入 provenance。

