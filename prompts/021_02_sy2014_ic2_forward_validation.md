# 021_02 SY2014 IC=2 输入固化与低成本前向验证

## 目的

在不训练 DQN、不修改 reward、不调整初始条件数值的前提下，验证用户修复后的
`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX`
是否确实以 treatment 2 / IC=2 运行，并建立可追溯的 SY2014 IC=2 前向证据。

## 约束

1. 只运行 `null`、`recorded`、`dssat_auto` 三个情景。
2. 使用指定 Docker 容器 `b2fd6726c8c1` 和虚拟环境 `/opt/gym_dssat_pdi/bin/python`。
3. 先做单情景 smoke test；通过后再运行三个完整生长季。
4. 不覆盖任何历史结果，不把历史 IC=0 训练结果改称 IC=2。
5. 不修改 reward、品种参数、土壤、天气或 IC=2 剖面。
6. 保存输入副本、SHA-256、运行时 PDI 快照、日值、管理事件、汇总、图和中文记录。

## 输入核查

- treatment 2 的 IC 指针必须为 2。
- `*INITIAL CONDITIONS` 必须包含 IC level 2 的六层 `SH2O/SNH4/SNO3`。
- 三个运行副本和 PDI 临时目录中的实际 `fileX` 均须再次通过上述核查。

## 情景定义

- `recorded`：保留 treatment 2 原有记录管理。
- `null`：关闭灌溉和施肥并清零 treatment 2 的报告管理行。
- `dssat_auto`：仅把 treatment 2 的灌溉和施肥管理切换为 DSSAT 自动管理。

所有情景必须保持 treatment 2、IC=2、2014 天气、相同土壤和品种参数不变。

## 执行与判定

1. 备份并哈希当前 MZX。
2. `null` 运行 20 个交互步 smoke test，检查 PDI 握手和 IC=2 输入快照。
3. 三情景分别完整前向运行，单任务顺序执行，防止 OOM。
4. 汇总 HWAM、CWAM、季节灌溉量、施氮量、最大 WSPD/NSTD、最终 DAP。
5. 输出三情景过程图和中文实验记录。
6. 只有在运行时输入确认 IC=2 且三情景完整结束后，才把 SY 当前输入状态标为已确认。

## 明确不做

- 不训练或评估 DQN checkpoint。
- 不修改站点优化结论。
- 不用该前向实验替代后续 seed 稳定性训练。
- 不重复运行已有 IC=0 历史实验。

