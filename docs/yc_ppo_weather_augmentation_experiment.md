# YC PPO 天气增强受控实验（004）

**最终状态：** `BLOCKED_UPSTREAM_WGEN_NOT_READY`

## 执行结论

本轮没有启动 PPO。任务规定必须先有 `003_06_yc_train_only_cli_and_wgen_pilot` 的 `PASS_YC_WGEN_PILOT` 证据，但当前项目中找不到 003_06 专属结果。最近可核验的 YC WGEN pilot 结论仍为 `BLOCKED_WGEN_NOT_READY`，train-only `.CLI` provenance 为 `BLOCKED_CLI_GENERATION`。因此本轮按前置 Gate 停止，未绕过 WGEN 或 DSSAT smoke。

这表示受控实验尚未执行，不能解释为天气增强有效或无效。

## 研究问题与冻结范围

原问题是在 PPO 算法、网络、optimizer、learning rate、gamma、GAE、clip、batch、rollout、训练步数、action、mask、observation、reward、预算、终止条件、cultivar、soil、management 和 evaluation mode 全部不变时，比较 `HISTORICAL_ONLY` 与 `HISTORICAL_PLUS_WGEN`，并在真实 2014-2023 validation 天气上评价表现。

预定数据划分保持 train weather 2004-2013、validation weather 2014-2023。Validation 不参与 WGEN 参数拟合、天气生成或模型选择。独立 test 状态为 `not_available_or_not_verified`。本轮没有冻结 PPO config 或 seed set，因为更上游的 WGEN Gate 尚未通过。

## 上游证据

| 环节 | 可核验证据 | 状态 |
|---|---|---|
| 2004-2013 天气 candidate | `docs/yc_weather_gapfill_finalize.md`；3653 天 candidate QC 通过 | `PASS_YC_WEATHER_CANDIDATE` |
| 指定的 003_06 结果 | `results/yc_train_only_cli_and_wgen_pilot/` 不存在 | 未找到，不能认定 PASS |
| 最近的旧 WGEN pilot | `docs/yc_wgen_cli_pilot.md` | `BLOCKED_WGEN_NOT_READY` |
| train-only CLI provenance | `results/yc_wgen_cli_pilot/yc_cli_provenance.json` | `BLOCKED_CLI_GENERATION` |

Candidate 通过只证明训练天气表完整并通过相应 QC，不证明 WGEN 参数文件已生成或 DSSAT 已读取随机天气。

旧 pilot 记录指出当前 YC lowIC 输入目录没有合格的 `CNYC.CLI`。发现的旧文件 SHA256 为 `58dbe11fdb3af9d34cc25644d8fb965627403778eb92da31a8f91619bd100d51`，其标注的数据窗口为 2008-01-01 至 2014-12-31，含 validation 年份，且缺少只使用 2004-2013 拟合的证据；关联运行还设置了 `random_weather=false`。因此该文件不满足本任务前置。FileX 中 8 字符 WSTA 与 CLI 的映射也尚未通过运行时确认。

机器可读核验及输入哈希见 [upstream_gate.json](/C:/Users/DELL/gym_workspace/gym_dssat_pdi_bingo/gym-dssat-pdi/results/yc_ppo_weather_augmentation/upstream_gate.json)。对应的既有证据为 [WGEN pilot 记录](/C:/Users/DELL/gym_workspace/gym_dssat_pdi_bingo/gym-dssat-pdi/docs/yc_wgen_cli_pilot.md)、[CLI provenance](/C:/Users/DELL/gym_workspace/gym_dssat_pdi_bingo/gym-dssat-pdi/results/yc_wgen_cli_pilot/yc_cli_provenance.json) 和 [weather candidate 记录](/C:/Users/DELL/gym_workspace/gym_dssat_pdi_bingo/gym-dssat-pdi/docs/yc_weather_gapfill_finalize.md)。

## 004 执行状态

| 任务项 | 状态 |
|---|---|
| Frozen synthetic weather bank | 未启动，0 套 |
| weather bank seed 与 PPO seed 分离 manifest | 未创建 |
| 50/50 historical/WGEN source sampler | 未实现、未 smoke |
| Historical-only / Historical+WGEN Arms | 均未运行 |
| PPO config、训练步数及相同 seed set 冻结 | 未执行 |
| Stage 0 integration smoke | 因 upstream Gate 阻塞，未运行 |
| Stage 1 full controlled training | 未运行 |
| 2014-2023 validation | 未运行 |
| Yield、fertilizer、irrigation、WUE、NUE、reward paired comparison | 未产生 |
| Seed success rate | 未产生；success criteria 未冻结 |
| Cherry-picking | 不适用，未启动任何 PPO run |

不得把已有 `221YCA` 历史天气场景增强结果当成本任务的 WGEN bank 或两 Arms 结果。两者天气来源和实验合同不同。

## 恢复条件

下一步先完成 `003_06_yc_train_only_cli_and_wgen_pilot`，确认 `.CLI` 仅由 2004-2013 candidate 拟合、记录工具版本和 source/output hashes、解决 FileX WSTA 映射，并完成 WGEN seed 复现与 DSSAT smoke。只有最终 Gate 为 `PASS_YC_WGEN_PILOT` 后，才进入本任务的 bank QC、50/50 source sampler smoke、共同 PPO seeds/full training 与 2014-2023 validation。

详细执行记录见 [experiment_log.md](/C:/Users/DELL/gym_workspace/gym_dssat_pdi_bingo/gym-dssat-pdi/results/yc_ppo_weather_augmentation/experiment_log.md)。当前机器可读状态见 [experiment_summary.json](/C:/Users/DELL/gym_workspace/gym_dssat_pdi_bingo/gym-dssat-pdi/results/yc_ppo_weather_augmentation/experiment_summary.json)。

四页状态汇报：[PPT](/C:/Users/DELL/gym_workspace/gym_dssat_pdi_bingo/gym-dssat-pdi/docs/yc_ppo_weather_augmentation_experiment.pptx)。包完整性和布局检查通过，逐页预览已检查；validation receipt 在 `results/yc_ppo_weather_augmentation/presentation_validation.json`。原生 PowerPoint 字体渲染未验证。
