# 017_10 LC PDI/gym 初始化问题诊断与抢救

## 背景

017_07 中 LC2008 在 `gym.make` 阶段超时，导致不能进入 DQN 本地训练或跨站点迁移判断。

当前检查发现一个高度可疑输入问题：

- `CNLC0801.MZX` 的 `ID_SOIL` 为 `LC99001200`
- `SOIL.SOL` 中实际 profile 为 `LC990012007`

这可能导致 PDI/DSSAT 初始化时找不到土壤 profile 或进入异常等待。

## 目标

在不修改原始输入文件的前提下，创建临时副本并测试：

1. 原始 LC2008 输入是否仍在 `gym.make` 超时；
2. 仅修正临时 MZX 中 `ID_SOIL: LC99001200 -> LC990012007` 后，是否能完成：
   - env constructor；
   - reset；
   - no-op step；
3. 若可运行，输出 LC2008 null/recorded/dssat_auto 的低成本基准结果。

## 约束

- 不训练模型。
- 不覆盖 017_07 结果。
- 不修改 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/LC` 原始输入。
- 所有测试写入新目录：
  `DSSAT_auto_validation/lc_pdi_initialization_rescue_017_10`
- 每个 gym/PDI 初始化测试必须设置超时，防止卡死。

## 成功标准

- `soil_id_fix` 临时副本可以完成 reset 和至少一步 no-op；
- 如果能完整跑完 season，则生成：
  - daily CSV；
  - events CSV；
  - summary CSV；
  - 中文实验记录。

## 判断口径

- 如果修正土壤 ID 后 LC 能跑通，则 LC 问题不是 DQN 框架问题，而是输入文件土壤 ID 不一致。
- 如果仍不能跑通，再继续查 LC 的 MZX 多 treatment、自动管理日期、WeatherMan 文件或 PDI 对 LC 文件结构的兼容性。
