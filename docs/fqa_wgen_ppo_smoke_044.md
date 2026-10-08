# 044 FQA WGEN PPO 2K 逐 episode 天气归档 smoke

## 执行范围与结论

按 [044 prompt](../prompt_02/044_fqa_wgen_ppo_2k_episode_archive_smoke.md)在独立目录执行了 FQA 2007 年、PPO seed 0、WGEN 训练 seed 池 1001–1080 的 2K smoke。沿用 FQA `051_00` 的 originIC、原始观测、16 动作、奖励、掩码和 PPO 超参数；只替换天气运行方式为 `CNFQ.CLI` + `WTHER=W` + PDI `rseed1_`。未改既有训练脚本、输入数据或历史结果。

[最终门槛](../results/fqa_wgen_ppo_smoke_044/attempt_05/final_gate.json)为 **PASS_SMOKE_ONLY**：请求 2,000 步，PPO 依 144 步 rollout 实际执行 2,016 步；19 个完整 episode 加 1 个停止时部分 episode，合计 20 份[实际训练天气](../results/fqa_wgen_ppo_smoke_044/attempt_05/weather_daily)覆盖全部 2,016 天。逐份复读 SHA-256、CSV 行数和六字段，每步仅一次 state 捕获，物理筛查均通过。20 份天气哈希各不相同。checkpoint 保存且在 runner 内成功载入，模型 SHA-256 见 [run_result.json](../results/fqa_wgen_ppo_smoke_044/attempt_05/run_result.json)。峰值进程树 RSS 430.91 MB，低于 1,536 MB 门槛。

每条 [episode manifest](../results/fqa_wgen_ppo_smoke_044/attempt_05/episode_manifest.csv)含 PPO/weather seed、PDI 运行时 `_rseed1`、天气文件路径/哈希/天数、FileX/CLI 身份和物理筛查结果。[逐 episode 运行时证据](../results/fqa_wgen_ppo_smoke_044/attempt_05/runtime_evidence)读取项目内临时目录的 FileX、CLI、PDI 配置：FileX 确为 `WTHER=W` 且 WSTA 为 CNFQ0701；运行时 CLI 与冻结源文件哈希相同；PDI 配置文件证明首个 bootstrap seed，后续各 episode 以 reset 后的 `_rseed1` 核对。复用 DSSAT 进程时，PDI 配置文件保留初始 seed，不能把它误读为每次 reset 的实际 seed。

## 失败尝试与修正

- 首次尝试：在 `_get_state` 捕获到 202 条 state、101 个 PPO step，归档门槛停止。错误及运行结果保留于 `results/fqa_wgen_ppo_smoke_044/` 顶层。
- `attempt_02`：证实成对 state 后半段的天气值不一致，不能机械去重；保存了 mismatch 诊断并停止。
- `attempt_03`：改为 DSSAT `step` 返回后捕获一次 state，2,016 步/2,016 天闭合，证明训练归档可行。
- `attempt_04`：加入运行时文件核查，发现复用进程后的 PDI YAML 仍记录首个 seed；第 2 个 episode 按过严门槛停止，天气与错误均保留。
- `attempt_05`：分别核对 YAML bootstrap seed 与每次 reset 后运行时 `_rseed1`，FileX/CLI 与逐日天气全部闭合。失败脚本版本在 `backups/044_attempt_*` 保存。

## 后续边界

本次固定 2007 年，仅证明 FQA 冻结 PPO 合同可以在 WGEN 下运行 2K 步并完整保存**实际使用**的天气。2005–2013 多年份训练调度、80/20 天气池气候分布、留出天气 episode、正式 100K 或多 PPO seed 尚未验证/执行。043 合同要求正式训练前至少有训练与留出 episode 的归档 smoke。YC/FQA 共同出现的 native FIELD 坐标警告仍未解决，不能因本次 PPO smoke 通过而宣称无功能影响。
