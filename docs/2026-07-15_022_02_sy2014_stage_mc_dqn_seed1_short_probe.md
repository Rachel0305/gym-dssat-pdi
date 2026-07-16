# 022_02 SY2014 阶段型 MC-target DQN seed1 短验证记录

## 1. 任务边界

本任务只验证 022_00 预注册的阶段型结构是否能在 SY2014（IC=2、PDI/DSSAT 4.8.0）中学到同时满足产量、水分生产率和氮肥偏生产力门槛的策略。没有修改 DSSAT 输入、IC、阶段点、动作档位、季节预算、奖励权重或判据；没有使用 demo、DQfD、PER、target bootstrap、1-step TD 或 5-step TD；没有追加 seed 或自动延长训练。

方法应准确称为：`stage-based DQN-style Q network with terminal-complete Monte Carlo targets`，不是未修改的 SB3 DQN。

## 2. 冻结设置

- 决策 DAP：1、30、50、65、85、110；其余底层 DSSAT 日全部强制 no-op。
- 动作：I∈{0,15,30} mm × N∈{0,50,100} kg/ha，共 9 档。
- 季节预算：I≤120 mm、N≤300 kg/ha；DAP110 禁止施氮。
- 观测：沿用 021_24 的固定 25 维标准化器，不新增信息。
- 回报：完整季节回报作为唯一 Q 目标，统一除以 1000 后进入 SmoothL1 loss。
- 网络：25→64→64→9，ReLU，Adam 1e-4。
- replay：uniform、容量 2000；第 5 个完整季节后开始学习；每季 6 次更新；batch 32；梯度裁剪 10。
- seed=1；60 个训练季，共 360 个阶段交互；epsilon 按季从 1.0 线性降到 0.2。
- checkpoint：15/30/45/60 季；每个只做一次确定性评估。

## 3. 训练前阶段环境验证

精确回放动作序列 `[3,4,7,1,1,0]` 全部通过：

| 检查项 | 结果 |
|---|---:|
| 阶段 DAP | 1/30/50/65/85/110，完全一致 |
| HWAM | 11202.004 kg/ha |
| 灌溉总量 | 60 mm |
| 施氮总量 | 200 kg/ha |
| 首阶段原始完整回报 G0 | 6354.004 |
| WP_ET | 2.31 kg/m3 |
| PFP_N | 56.0 kg/kg |
| 非阶段日 | 全部 no-op |
| DAP110 含氮动作 | 正确抛错，未裁剪混叠 |
| Summary.OUT 总量 | 与请求总量一致 |

回报闭合：`11202.004 - 5408 + 1620 - 60 - 5×200 = 6354.004`。

验证过程中出现四次训练前失败，均被门槛拦截且未启动训练：

1. 初版错误假设 reset 后立即处于 DAP1；
2. 发现 gym-DSSAT 的 DAP0 初始化需要多次 no-op，而非一次；
3. 错误尝试从底层原始字典重建 25 维观测，DAP1 的 9 层土壤水仍为 NaN；
4. 修正为直接复用 SB3 wrapper 实际返回的 25 维有限观测，并用底层所有有限字段验证展开顺序。

旧 022_01 逐日表显示 DAP0 初始化持续 21 个底层 transition，因此最终 wrapper 用有上限的 no-op 循环进入 DAP1。各失败均保留独立 JSON，不覆盖、不隐藏。

## 4. 正式短训练结果

| checkpoint季 | HWAM (kg/ha) | I (mm) | N (kg/ha) | WP_ET (kg/m3) | PFP_N (kg/kg) | 主判据 | 严格判据 | 动作序列 |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 15 | 11201.759 | 75 | 300 | 2.25 | 37.3 | False | False | [7,7,7,7,7,0] |
| 30 | 11201.759 | 105 | 300 | 2.24 | 37.3 | False | False | [7,7,7,7,7,2] |
| 45 | 11201.759 | 105 | 300 | 2.24 | 37.3 | False | False | [7,7,7,7,7,2] |
| 60 | 11201.759 | 105 | 300 | 2.24 | 37.3 | False | False | [7,7,7,7,7,2] |

预注册主门槛为 HWAM≥11077、WP_ET≥2.26、PFP_N≥36.9，并同时满足预算和晚期禁氮。四个 checkpoint 均通过产量和 PFP_N 门槛，但 WP_ET 只有 2.24–2.25，因此严格按预注册规则为 **0/4 主判据通过**。

训练过程共 60 季、336 次梯度更新；训练季产量范围为 8117.17–11239.91 kg/ha，I 为 30–120 mm，N 为 50–300 kg/ha。SmoothL1 loss 为 2.982–6.520，最大更新前梯度范数 4.496，小于裁剪阈值 10；最大绝对 Q 值 5.285。没有数值发散证据。

## 5. 预注册分支判定

**C_failed：停止本结构，不现场调参。**

这不是“完全没有学到高产”：四个 checkpoint 都达到约 11.2 t/ha，说明阶段结构能学到较好的早期施氮时序。失败点是没有同时学到预注册要求的资源效率。

同时发现一个需要保留的实现/学习现象：模型在 DAP1/30/50 连续选择动作 7 后已经耗尽 N300，但在 DAP65/85 仍请求动作 7；budget wrapper 将其执行为 I15/N0。即请求动作不同而实际执行动作混叠。该行为是冻结的 022_00 动作执行逻辑产生的真实结果，本轮没有事后修改动作 mask。

因此当前证据只能支持：

- 固定官方阶段空间中客观存在满足联合目标的方案（022_01）；
- 本次单 seed、60 季 MC-target Q 网络学到了稳定高产但偏高投入的策略；
- 它没有达到预注册的产量+WP_ET+PFP_N 联合门槛；
- 不能自动补 seed、延长季数或现场修改动作 mask 来把阴性结果调成阳性。

## 6. 输出文件

- `prompts/022_02_sy2014_stage_mc_dqn_seed1_short_probe.md`
- `src/run_sy2014_stage_mc_dqn_seed1_short_022_02.py`
- `benchmark_results/022_02/022_02_stage_environment_validation.json`
- `benchmark_results/022_02/022_02_validation_stage_actions.csv`
- `benchmark_results/022_02/022_02_validation_daily_values.csv`
- `benchmark_results/022_02/022_02_training_seasons.csv`
- `benchmark_results/022_02/022_02_training_stage_actions.csv`
- `benchmark_results/022_02/022_02_update_log.csv`
- `benchmark_results/022_02/022_02_checkpoint_evaluation_summary.csv`
- `benchmark_results/022_02/022_02_checkpoint_stage_actions.csv`
- `benchmark_results/022_02/022_02_checkpoint_daily_values.csv`
- `benchmark_results/022_02/checkpoints/season_015.pt`、`030.pt`、`045.pt`、`060.pt`
- `benchmark_results/022_02/022_02_training_diagnostics.png/.svg`
- `benchmark_results/022_02/022_02_result.json`
- `benchmark_results/022_02/022_02_validation_attempt_1_failure.json` 至 `attempt_4_failure.json`

## 7. 可追溯性

| 文件 | SHA256 |
|---|---|
| 022_02 prompt | `F347E7DF406350ED127E068E65FA1234BF7779CBFECB3A4EFB19DAE7D9F07344` |
| 022_02 执行脚本 | `6FAE7BE91F0F40E20D0553DA6AF75B5B635FCDC2A8B4A57A2625D7838AC5A205` |
| 022 阶段核心 | `A96F5FCC35F839F32252C19022AEAFE101FEF370EB039AB980311F603569580C` |
| 021_24 scaler | `FE1602A9F4C07253A83BDEF27D6A0A58A8F65906A701B27A2424157CC53F8369` |
| 022_01 thresholds | `7C7E0F05F3934AC1938073FD2F2D067E4F5DF7AE98E234D76F043CB3EF035E7B` |

本任务未执行 Git commit 或 push。
