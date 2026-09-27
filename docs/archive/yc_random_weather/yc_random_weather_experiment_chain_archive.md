# YC Random Weather Experiment Archive

本档案按实际研究顺序索引 YC 随机天气方案、WGEN、PPO、provenance 与汇报产物。每个文件的相对路径、SHA256、大小和 `FORMAL` / `DIAGNOSTIC` / `PROVISIONAL` 状态见同目录 `yc_random_weather_experiment_chain_manifest.csv` 和 `.json`。**档案不复制整个 results，也不把诊断天气冒充历史训练天气。**

## 1. Scientific motivation

起点见 `prompt_02/004_yc_ppo_weather_augmentation_experiment.md` 与最终受控实验任务 `prompt_02/004_05_yc_weather_augmentation_multi_seed_archetype_experiment.md`。方法学借鉴 Wang et al. (2025) 的 episode-level DSSAT WGEN 随机天气思路，但不是逐项复现其站点/模型设置。核心问题是：增加训练天气多样性，能否在不改 PPO architecture、reward、observation、action space 和验证协议的情况下，提高 YC PPO 的稳定性。增强变量为 `RAIN/SRAD/TMAX/TMIN`；这些是逐日天气，不是新观测特征。

## 2. Weather-generator construction

冻结拟合输入 `results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv` 覆盖 2004–2013；CLI `results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI` 由 `scripts/build_dssat_cli.py` 路线生成。两者 SHA256 分别为 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34` 与 `65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929`。`004_01` 参数方法审计独立核对 14 参数 × 12 月 = 168 个未序列化值，168/168 一致；完整独立全年生成接口当时未验证，这一工程阻断不等于参数失败。输入、代码、元数据与审计报告均列入机器索引。

## 3. WGEN validation

`003_06_06` 完成早期天气 QC / DSSAT smoke。`004_02` 转向 episode-level 验证：seeds 1001–1100 共 100 个 episode，生成 QC 100/100 通过；预指定 5 个 seed 重跑逐日一致 5/5；100 个不同 seed 的完整序列 100/100 唯一。共同比较窗为 2008-06-01 至 2008-09-14（106 天），并有 rain/temperature/SRAD/相关结构 QC。见 `docs/yc_random_weather_episode_ensemble_validation.md` 与 `results/yc_random_weather_episode_validation/004_02/summary.json`。这些是生成/QC 证据，**不等于 004_05 当时的逐日天气已永久归档**。

## 4. Weather-pool design

正式训练池 1001–1080（80 套），held-out 1081–1100（20 套），后者不回流训练。`004_05/config/training_weather_schedule_random.csv` 为 100,000 行调度，每个训练 weather seed 恰有 1,250 次选择；每个 PPO seed 的实际 episode 使用由日志/审计核实。RSEED1 以实际 episode 上下文传入 runtime，不能由 weather-seed 编号单独推断完整天气序列。004_02 的 100-episode QC 与 004_05 的上下文/使用记录属于不同证据层。

## 5. 004_05 PPO augmentation experiment

正式比较 75 mm canonical early cap 下 Historical 与 Random-weather PPO，各 8 个 seed；Observed 2014–2023 与独立 held-out WGEN 1081–1100 为既定评估域。runner、schedule、paired episode/seed 结果、训练 manifest 和中文报告在机器索引中。`multi_seed_decision.json` 的结论为 `NO_POSITIVE_SIGNAL`，预注册 paired success **0/8**；行为 archetype 仅 `POSSIBLE_SHIFT`。Observed 平均 yield/reward：historical 6575.7 kg/ha / 0.524，random-weather 6328.5 / 0.457；held-out 平均 yield/reward：7026.2 / 0.877 对 6747.2 / 0.829。**不能因为后续单个 seed5 有农业折中价值，就抹去 004_05 八 seed 整体负结果。** 单个 checkpoint 的五情景复核属于后续固定模型解释，不替代原 paired 结论。

## 6. Why weather provenance became a problem

后续 `docs/yc_random_weather_weather_domain_coverage_audit.md` 查明 schedule 覆盖完整且均衡，但 004_05 没有完整持久化 1001–1100 的实际逐日天气，因此 `weather_pool_expansion_decision = DEFER`。仅有 seed 与 runtime hash 不足以判定 observed 气候尾部是否被训练域覆盖，也不能把不同上下文的同号 seed 简化为同一全年天气。

## 7. 004_16 provenance audit

`004_16` 先追生成链：seed→`RSEED1` 和 crop-year/context 有历史记录，但独立 weather-only WGEN 路径及 `runtime_weather_sha256` 的对象当时未核实。按当时禁止 crop simulation 的边界，物化未启动；`weather_reconstruction_status = FAILED` 只表示 provenance gate 未通过，**不是 WGEN 或 004_05 训练失败**。Observed/training coverage 与 tail gap 均 `INSUFFICIENT`，不启动扩池。见 004_16 prompt、runner、decision 与中文报告。

## 8. 004_17 historical weather recovery

通过历史 runtime 源码追踪发现 `runtime_weather_sha256` 是对 PDI daily-state 天气字段按历史算法序列化后的 SHA256，**不是 raw WTH bytes**。seed1001/context2013 smoke 与历史 hash 完全匹配后才批量重放并归档。`weather_archive_manifest_verified.csv` 唯一指定 100 套 canonical daily CSV；文件完整性 100/100，历史精确核验 98/100（training 78/80、held-out 20/20）。seed1003/context2011 与 seed1036/context2009 仍为 `PROVISIONAL`，不能称为历史 exact。runtime 没有生成新的 raw WTH，归档的是实际捕获的 PDI daily-state 天气；其余 34 套诊断尝试不属于权威 100 套。DSSAT crop side-effect 输出一律 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`。恢复结论为 `PARTIALLY_VERIFIED`，未给正式 climate coverage verdict。见 004_17 chain、reconciled ledger、verified manifest、integrity ledger 与报告。

## 9. 004_18A descriptive weather summary

在 100/100 文件哈希核验后，按 verified manifest 的 80 training、20 held-out 与 004_05 正式 observed WTH/FileX（2014–2023）计算整季及 DAP 阶段统计，制作九张组会图。WGEN 统一从**最后一条 DAP=0**至季末，剔除 runtime 前置记录；其 DATE 为参考日期、crop year 取 manifest。Observed 的原 step trace 天气四列为空，故用正式 WTH + FileX 播期 + trace 最大 DAP；逐源证据见 `season_alignment_qc.csv`。这些是**描述性统计**，不是新的 crop evaluation。

FULL80 vs EXACT78 共 6 个预设敏感项；尤其 2014、2016、2019 的 observed 整季降雨在 FULL80 min/max 内，但在 EXACT78 min 以下。故 provisional 1003/1036 不能忽略，不能宣称描述性 tail identification 完全稳健。Training 整季降雨中位数 466.1 mm（P5–P95 275.4–768.3），held-out 中位数 515.2 mm，observed 198.7–709.0 mm。正式 climate coverage verdict **pending**。完整方法/表格/图见 `docs/yc_random_weather_004_18A_weather_summary_for_reporting.md`，组会短版见 `docs/yc_random_weather_weather_summary_for_presentation.md`。

## 10. Current scientific status

| 议题 | 状态 |
|---|---|
| 004_05 8-seed PPO 性能 | FORMAL；paired success 0/8，无可复现正信号 |
| 004_02 WGEN episode QC | FORMAL；100/100 QC，5/5 固定 seed 重放一致 |
| 004_17 文件完整性 | FORMAL；100/100 canonical CSV 哈希吻合 |
| 004_17 历史天气一致性 | FORMAL 98/100；PROVISIONAL 1003/1036 |
| 004_18A 气候汇报 | DIAGNOSTIC；FULL80/EXACT78 尾部敏感性存在 |
| 训练天气是否覆盖 observed / 是否应 80→160 | 未作正式判定，不扩池 |

## 11. Open issues

- 004_18B：闭合 seed1003、1036 的历史 hash / 长度 / context 差异；若无法闭合，继续保留 provisional。
- 004_19：统一可比季长/阶段口径，做正式 climate coverage audit；尤其检查 FULL80 低降雨支持是否只来自 provisional。
- 80 套是否足够、随机扩到 160 或定向补尾部，均待 004_19 的明确证据；不能由本次组会图直接决定。
- 未来 paired PPO retraining 只有天气池设计和气候 support 改善确认后才讨论；不在本档案启动。

## 12. Next planned experiments

`004_18B` resolve seed1003 / seed1036 → `004_19` formal climate coverage audit。

- 若结论 `KEEP_80`：停止天气池扩展，进入策略解释与最终评估。
- 若证据支持覆盖不足：`004_20` 先设计 expanded weather pool、验证 climate support 改善，**不训练 PPO**；随后才考虑 `004_21` 80-weather vs expanded-weather paired PPO retraining。

## 13. File index and future contract

机器索引覆盖关键 prompt、source、report、result、manifest、figure 及 verified manifest 指向的 100 套 canonical daily weather，并记录每个文件 SHA256 与状态：`yc_random_weather_experiment_chain_manifest.csv` / `.json`。其中两条 provisional weather 显式标 `PROVISIONAL`；004_16 的阻断和 004_18A 的描述性输出标 `DIAGNOSTIC`。模型 zip、逐日大 trace、临时 runtime、DSSAT 作物副产物、134 套非权威候选中的额外 34 套均不复制到本档案，也不自动纳入 Git 备份。004_05 checkpoint 的存在和哈希以其训练 manifest 为准，本档案未重新评估模型。

`.gitattributes` 只对本链的冻结 fitting CSV、CLI、QC/调度表、004_17 天气归档和 004_18A 统计表禁用 Git 文本换行转换，以保持已报告的原始文件 SHA256；不改变天气数值或实验配置。

今后 YC / HL / FQ / LC / SY 随机天气实验遵守 `docs/random_weather_archival_contract.md`：永久保存实际 runtime WTH（若产生）、canonical daily series、seed/RSEED1、crop-year/date context、站点、CLI/config hash、runtime version、episode→weather archive ID、raw 与 canonical hash；若 runtime 不输出 WTH，必须明确记 `NOT_EMITTED_BY_RUNTIME`，绝不伪造 raw WTH。`src/archive_runtime_weather.py` 是独立归档 helper，不追改历史实验。
