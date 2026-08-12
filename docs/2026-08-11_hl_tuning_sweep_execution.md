# HL PPO 单因素 kwargs 2K smoke：预注册执行记录

日期：2026-08-11。目标是诊断 HL 的训练稳定性机制，不比较或宣布最终性能。原 `054_00` 代码、配置、模型和结果均不修改或覆盖；本任务只创建隔离的 `064_00` 配置、runner 和输出。没有修改既有文件，因此不需要对既有脚本创建备份。

## 实际基线 kwargs 核验

容器 `nifty_taussig` 中，`experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml` 的 PPO 段为：`learning_rate=3e-4`、`gamma=1.0`、`gae_lambda=1.0`、`n_steps=144`、`batch_size=144`、`n_epochs=5`、`ent_coef=0.01`、`clip_range=0.2`、`net_arch=[64,64]`。`run_free_timing_stress_aware_ppo_dqn_smoke_032_00.ppo_kwargs()` 实际传入上述前七个 kwargs，并把 `net_arch` 包装在 `policy_kwargs`；因此 `n_epochs=5` 与 `[64,64]` 不只是 JSON 声明。每个 064 smoke 还将从已保存模型读取 kwargs 与 actor/critic 隐层维度，作为实际生效证据。

固定不变项：HL lowIC 输入根、2004--2013 train / 2014--2023 validation 年份、seed0、I=[0,15,30,45] 与 N=[0,40,80,120]（16 动作）、原始 046_02 observation、reward、safety、天气预报关闭，以及上述所有未被单一 override 指定的 PPO 参数。

## 固定候选及顺序

按 A、B、C 串行运行，均为 2K，保存 1K/2K checkpoint；任何候选 preflight、import 或 provenance 失败时仅停止该候选，不换环境。

|候选|唯一变化|其余有效 PPO kwargs|
|---|---|---|
|A_net32|`policy_kwargs.net_arch=[32,32]`|基线完全相同|
|B_lr1e4|`learning_rate=1e-4`|基线完全相同，包括 `[64,64]`|
|C_ent002|`ent_coef=0.02`|基线完全相同，包括 `[64,64]`|

## 2K 机制 gate 与 25K 规则

候选在 2K 只能被称作“机制通过”，须在最终 2K checkpoint 满足：十个验证年均有 daily 输出；安全动作在 16 动作网格内；request→safe→DSSAT transmission mismatch=0；全站至少三个非零动作对；至少一个 DAP1 后正动作；跨年动作签名数不为 1。`WP_ET` 若没有独立 snapshot/ETCP 同口径表，则显式标为不可用，不反推。2K 不比较优劣、不宣称 rescue 成功。

最多只有一个通过 2K 且容器资源正常的候选可进入 25K rescue；其选择顺序固定为：先满足所有机制 gate，再按 2K 最终 checkpoint 的跨年非零动作对数降序、DAP1 后正动作均值降序、平均验证产量降序。若仍相同则不挑选，停止并报告平局；任何情况下禁止 100K。

## 实际执行状态（仅 HL）

- A_net32：dry-run 通过；实际运行在任何训练或 DSSAT step 之前停止。原因是新 runner 首版将有效 YAML 写入输出 `configs/`，继承引擎随后对同一路径 `copy2` 触发 `SameFileError`。已保留隔离的两个配置文件作为失败证据；没有模型、daily 输出或结果。按预注册规则，A 不重跑、不换环境。
- B_lr1e4：实际 2K 训练与十年验证完成，运行约 90 秒，峰值采样 RSS 约 424 MB（2.6%），未见 OOM。首版 runner 在训练、checkpoint 和验证完成后才因缺少 `audits/` 目录停止；没有重训，以 no-retrain finalize 补写 audit、manifest 和 result。随后已修复新 runner 的目录创建，但未启动新的候选。
- C_ent002：未启动，原因是本轮被要求在 A/B 状态返回前停止扩展；未将其计入通过或失败。

B 的已保存模型证实有效 kwargs 为 `learning_rate=1e-4`、`gamma=1.0`、`gae_lambda=1.0`、`n_steps=144`、`batch_size=144`、`n_epochs=5`、`ent_coef=0.01`、`policy_kwargs.net_arch=[64,64]`，实际 actor/critic 隐层宽度均为 `[64,64]`。结果目录：`benchmark_results/064_00_hla_lowIC_ppo_kwargs_smoke_sweep_B_lr1e4_smoke2k/`。

|B 2K 最终 checkpoint（十年验证均值）|数值|
|---|---:|
|产量 kg/ha|6734.59|
|灌溉 mm|135.0|
|施氮 kg/ha|240.0|
|PFP-N kg/kg|28.06|
|WP_ET|不可用；2K validation summary 无 ETCP/snapshot，同口径不反推|
|非零动作对|7|
|DAP1 后正动作平均数/年|4.9|
|跨年动作签名数|3|
|off-grid / request→safe / raw→safe / safe→DSSAT mismatch|0 / 0 / 0 / 0|

B 通过 2K 机制 gate，但它只是唯一已完成的候选，不能在 A/C 不完整时宣称 sweep 最优。当前不启动 25K；需用户/主流程明确确认是否以 B 作为唯一可进入 25K rescue 的候选。仍禁止 100K 和 YC。

## 更新：A/B/C 全部 2K 状态（仍仅 HL）

在 B 完成后，已修复新 runner 的两个隔离性/收尾缺陷：有效 YAML 的复制源改为 `configs/064_00_effective_<candidate>_ppo_config.yaml`（与目标输出 `configs/` 不同），并在写 audit 前创建 `audits/` 目录；`clip_range` schedule 以字符串序列化。没有修改任何 054 文件。A 的第一次启动仍保留为训练前失败证据；其 retry 使用独立目录 `...A_net32_smoke2k_retry_after_runnerfix`，不覆盖该失败目录。

|候选|唯一有效变化（已加载模型验证）|2K状态|平均产量|I/N|PFP-N|非零动作对|DAP1后动作/年|跨年签名|网格/请求到安全/raw到安全/安全到DSSAT|机制 gate|
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
|A_net32|`net_arch=[32,32]`，actor/critic=[32,32]|retry 完成；首次训练前 SameFileError 已停止|5854.05|90/240|24.39|1|5.0|1|0/0/0/0|失败：动作对和签名退化|
|B_lr1e4|`learning_rate=1e-4`，actor/critic=[64,64]|完成；训练后无重训 finalize|6734.59|135/240|28.06|7|4.9|3|0/0/0/0|**通过**|
|C_ent002|`ent_coef=0.02`，actor/critic=[64,64]|完成|6653.70|135/240|27.72|4|5.0|1|0/0/0/0|失败：单一跨年签名|

三项均使用相同 lowIC root、train/validation 年份、seed0、16 动作、observation、reward、安全和天气关闭；每项均只训练至 2K 并保存 1K/2K checkpoint。三项 `WP_ET` 都没有独立的 ETCP/snapshot 同口径表，统一记为不可用，未反推。训练年层面的逐日动作日志并非继承 engine 的产物；本轮只记录了训练年份范围和验证年动作多样性，训练年动作多样性明确为“未导出/未估计”，不能替代验证审计。

每次容器进程约 90--100 s，采样 RSS 约 418--427 MB（约 2.5--2.6%），无 OOM。按预注册排序，B 是唯一通过全部 2K 机制 gate 的候选，因此**仅具备进入 25K rescue 的资格**；当前用户约束仍是不进入 25K、不启动 YC、不训练 100K，故本轮在此停止，不将 B 宣称为最终性能改善。
