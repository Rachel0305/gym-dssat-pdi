# 020_12 冻结 n-step DQN 框架的 YC/FQ 跨站点验证

## 目的

把 HLA 已冻结的 n-step=5 DQN **训练框架**用于 YC2014 与 FQ2016，并在每个目标站点重新训练。该实验检验的是“同一方法能否跨站点使用”，不是直接迁移 HLA 模型权重。

## 冻结项

- 奖励：每步 `-1.0 * I - 5.0 * N`；终止时加 `max(0, GWAD_final - local_null_yield)`。
- 9 个离散动作：`I∈{0,15,30}` mm 与 `N∈{0,50,100}` kg/ha 的笛卡尔积。
- 约束：季节 `I≤120 mm`、`N≤300 kg/ha`；单次 `I≤30 mm`、`N≤100 kg/ha`；两次有效操作至少间隔 7 DAP；水氮窗口均为 DAP 1–120。
- DQN：从 `src/frozen_nstep_dqn_config_020_11.py` 读取，核心为 learning rate=1e-4、buffer=10000、learning starts=50、batch=32、gamma=0.99、n-step=5、探索率 1.0→0.05。
- checkpoint：默认每 5K 评估；以确定性评估的 `total_reward` 最大者为最佳；并列时选最早 checkpoint。
- 环境随机种子固定为 0；模型随机种子由 CLI 的 `--seed` 指定。

## 站点变量

- YC2014：使用 `multisite_new_cultivar_inputs_013/YC`、treatment 2、CNYC1401、YC99001200。
- FQ2016：使用 `multisite_new_cultivar_inputs_013/FQ` 的 treatment 2 输入结构平移到 2016、CNFQ1601、FQ99001200。
- 每个站点都必须先在同一输入包上独立运行 null，不能共享 HLA 或其他站点的 null 产量常数。
- 本轮不修改 IC 日期、土壤初始剖面、天气、品种或管理窗口，避免同时改变多个变量。

## 强制输入审计

正式创建环境前检查：

1. treatment 使用 `IC=1`；
2. simulation options 使用 `WATER=Y`、`NITRO=Y`；
3. DQN 输入使用 `IRRIG=L`、`FERTI=L`；
4. action table 恰好为冻结的 9 动作；
5. 环境 seed 为 0；
6. 目标年份天气文件、`MZCER048.CUL`、`SOIL.SOL` 均存在；
7. 保存输入哈希和冻结配置清单。

任一检查失败立即停止，不进入训练。

## 节省算力顺序

1. 先只做代码静态检查，不训练。
2. YC2014 seed0 运行 500 steps smoke。
3. FQ2016 seed0 运行 500 steps smoke。
4. 两个 smoke 均通过输入、动作传输、预算、输出和内存检查后，才串行运行 YC seed0 50K，再运行 FQ seed0 50K；禁止并行长训练。
5. 只有 seed0 产生有效候选时，才增加 seed1/seed2。

## 输出

输出到独立目录 `DSSAT_auto_validation/frozen_nstep_cross_site_020_12/`，不得覆盖旧实验。每次运行保存：

- 输入文件及 SHA256 审计；
- 本地同输入 null 日值、汇总与 PDI 快照；
- checkpoint 模型、逐 checkpoint 日值和汇总；
- 最佳 checkpoint 清单；
- 动作预算/间隔/MgmtEvent 传输审计；
- 中文实验记录。
