# 026_05 SY2012 四基线与冻结 PPO 跨年迁移记录

## 1. 目的与边界

本实验检验在 SY2014 训练并按既定规则选出的三个阶段型 MaskablePPO checkpoint，能否在**不重新训练、不重新选模、不修改 reward/IC/scaler/动作空间/阶段点**的条件下，直接迁移到 SY2012。

迁移对象是三个固定的模型权重，而不是在 SY2012 重新执行训练与 checkpoint 选择流程。因此，本实验回答的是“这些 SY2014 候选策略能否跨年工作”，不能单独证明 PPO 训练过程已经稳定收敛。

## 2. 冻结模型与输入溯源

| seed | SY2014 checkpoint | SHA256 |
|---:|---|---|
| 0 | `benchmark_results/026_03/checkpoint_000120.zip` | `d88938a7d939cc772f5da185360298f59af86ea0b95cf411d3ccb0722ba541ab` |
| 1 | `benchmark_results/026_02/checkpoint_000060.zip` | `96d5d85e82a0e7a10b3fe68014b8f6c93ef903326d8640478c0becc6477832ec` |
| 2 | `benchmark_results/026_04/checkpoint_000240.zip` | `6add96bd8e827ec9613d5f04e18cb233f83418aa7bc07f690811acd5509e4b64` |

SY2012 使用当前权威输入链，treatment 1、year 2012、IC=1。关键输入哈希如下：

- MZX：`20b071bc49549cbf561be3aae81caa355aa0582564db524e17d68ab6a274418a`
- WTH：`4e2489653b7a72f75ce255602b2a58953e40e6424b4f27e8cf64d900361e7c68`
- SOL：`3c303d68dda19ba6f40573a3bba5464e40230ddd60c6c746f0ea6bbfeee9e644`
- CUL：`7810190a042b2fe3c53c54d43bf8080ae02dd3d1edfec202bf572e130761403c`

原始输入没有被覆盖。official expert 与 PPO 外部动作场景只修改运行目录中的副本：保持 IC=1，将 treatment 的 MI/MF 指针设为 1/1，清除副本中的 recorded 管理事件，并启用 `IRRIG=L`、`FERTI=L`。

## 3. 失败尝试与修正

所有失败均保留，且没有作为科学结果使用。

1. `026_05_attempt1_n_accounting_mismatch`：recorded 的 MgmtEvent/管理表施氮总量为 293 kg/ha，而 `Summary.OUT` 的 NICM 为 247 kg/ha。二者是不同统计口径，不能强制要求相等。修正后同时保留两套口径，PFP_N 使用 DSSAT 原生 Summary 口径。
2. `026_05_attempt2_expert_irrigation_not_executed`：仅设置 `IRRIG=L/FERTI=L` 不足以启用外部灌溉；2012 treatment 原指针 MI=0，导致 expert 请求 266.25 mm、Summary 实际为 0。修正为只在运行副本中显式设置 MI=1/MF=1。
3. `026_05_attempt3_scenario_adapter_name`：场景名称不满足既有 adapter 的命名约定，在 DSSAT 运行前终止。修正场景适配后重新执行。

## 4. SY2012 四基线

| 场景 | 产量 kg/ha | 灌溉 mm | 管理事件氮 kg/ha | Summary NICM kg/ha | WP_ET kg/m³ | PFP_N kg/kg |
|---|---:|---:|---:|---:|---:|---:|
| null | 7002 | 0 | 0 | 0 | 1.64 | 不定义 |
| recorded / farmer practice | 10166 | 0 | 293 | 247 | 2.36 | 41.2 |
| DSSAT auto | 7016 | 33 | 0 | 0 | 1.60 | 不定义 |
| official extension expert | 9609 | 266.25（Summary 266） | 300 | 300 | 2.21 | 32.0 |

按预注册规则，主要本地判据以 DSSAT auto 与 official extension expert 为对照，取二者较高值：

- 产量 ≥ 9609 kg/ha；
- WP_ET ≥ 2.21 kg/m³；
- PFP_N ≥ 32.0 kg/kg（仅对正施氮基线定义）。

recorded 作为农民实践基线单独报告，不被悄悄并入或替代主要判据。

## 5. 三个冻结 PPO 模型的零训练迁移结果

| seed | 六阶段动作序列 | 产量 kg/ha | 灌溉 mm | 施氮 kg/ha | WP_ET | PFP_N | 对 auto+expert 主判据 |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | `0,3,4,6,6,2` | 10120.19 | 45 | 300 | 2.35 | 33.7 | 通过 |
| 1 | `4,4,0,5,5,1` | 10056.05 | 105 | 200 | 2.28 | 50.3 | 通过 |
| 2 | `3,3,7,7,1,1` | 10068.63 | 60 | 300 | 2.34 | 33.6 | 通过 |

工程检查全部通过：

- `training_steps=0`；
- 三个 checkpoint 运行前后哈希不变；
- 3/3 模型均完成确定性整季评估；
- invalid action 为 0；
- 所需指标均可计算。

## 6. 判定

预注册分支判定为：`A_fixed_models_transfer`。

三个冻结模型均同时超过 SY2012 的 DSSAT auto 与 official extension expert 的产量、WP_ET 和 PFP_N 本地门槛。因此，本实验提供了**SY2014 固定 PPO 候选模型向 SY2012 的跨年迁移证据**。

但必须同时保留以下边界：

- 三个 PPO 模型产量均低于 recorded 的 10166 kg/ha，差值分别为 45.81、109.95 和 97.37 kg/ha；
- seed0/2 的 WP_ET 略低于 recorded 的 2.36，seed1 也低于 recorded；
- 仅 seed1 的 PFP_N 高于 recorded；
- 因此不能表述为“全面超过所有基线”，也不能表述为“PPO 已稳定收敛”；
- 当前可支持的准确表述是：**三个固定模型均在 SY2012 超过 DSSAT auto 和官方 expert 的联合效率门槛，但尚未全面超过 recorded/farmer practice。**

## 7. 输出文件

- `benchmark_results/026_05/026_05_result.json`
- `benchmark_results/026_05/026_05_sy2012_four_baselines.csv`
- `benchmark_results/026_05/026_05_frozen_ppo_transfer_summary.csv`
- `benchmark_results/026_05/026_05_frozen_ppo_stage_actions.csv`
- `benchmark_results/026_05/026_05_all_scenarios_summary.csv`
- `src/run_sy2012_frozen_stage_ppo_transfer_026_05.py`
- `prompts/026_05_sy2012_baselines_and_frozen_ppo_transfer.md`

## 8. 下一步边界

026_05 已经完成，不在本任务内追加训练、改 reward 或根据 SY2012 重新选择 checkpoint。若导师要求目标必须同时超过 recorded，则当前结果尚未完成该更严格目标，应另立预注册任务；不能事后修改 026_05 的成功判据。
