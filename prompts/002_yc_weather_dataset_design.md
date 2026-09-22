# 002 — YC weather dataset design and feasibility verification

## 背景与目标

001 审计已确认：`gym_dssat_pdi==0.0.5`；现有 YC/YCA 实验 `055_00` 的 `random_weather=false`，训练天气来自 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC*.WTH` 的 2004–2013 年历史年份抽样；当前 YC lowIC 输入目录无 `.CLI`；固定 seed 0 的抽样年份序列可复现，实际前三次 reset 的 `active_year` 为 2012、2010、2009。001 报告位于 `docs/yc_weather_pipeline_audit.md`，配置快照为 `results/yc_weather_audit/yc_weather_config_snapshot.json`，审计 commit 为 `523588e`。这些事实以仓库现存文件为准，先逐一核对，不要将本 Prompt 当作取代代码证据的事实来源。

本次只完成 **YC 天气增强数据集设计、天气生成路线核查和最小可行性验证**。不要开展正式 PPO 训练；不要宣称天气增强已改善 PPO。重点是让下一步实验能以独立、可复现、无数据泄漏的天气情景为输入。

## 关键执行要求

1. **检查工作区并设保护边界。** 开始时记录 `git status`、当前分支和 `HEAD`，阅读根目录 `AGENTS.md` 及相关目录指令，打开 001 审计报告与配置快照交叉核查。LC/SY 为冻结成果，禁止修改、删除、重新生成数据或触发训练；本任务不修改 HL/FQ。不得覆盖 `055_00`、旧模型、旧结果和原始 `CNYC*.WTH`。新输出仅放入 YC 专用新目录；如必须改共享代码，先提供补丁设计与受影响路径清单，未经授权不要落地。
2. **追查数据划分。** 从真实配置确认 YC/YCA 的 train、validation、test 年份及其来源；明确 2004–2013 是否仅为训练年，禁止猜测验证和测试年份。检查生长季与天气文件的对应关系，以及 `WSTA`、天气扩展名和 DSSAT 读取路径。把每项结论标记为“代码证实 / 运行证实 / 尚未证实”，附路径与行号。
3. **比较两条可行路线，而不是预设技术答案。** A：使用当前 `gym_dssat_pdi==0.0.5` 所支持的内置 WGEN/`random_weather`，核实 `.CLI` 的正确格式、如何从 YC 训练年估计参数、与试验文件的站点代码匹配规则，以及是否能够自动化/复现；B：在 Python 中使用有文献或官方实现依据的天气生成器，批量生成有效的 YC DSSAT 天气文件，再由环境在 episode 边界加载。分别评估代码改动、验证难度、随机种子控制、训练/测试隔离、输出持久化和失败风险。不能凭空构造 `.CLI` 字段或气候参数；若缺乏可验证实现，明确标记阻塞。选择有证据、最少侵入且易复现的首选方案，并说明理由。
4. **设计数据集与 manifest（清单），但不生成大型数据集。** 给出 `scenario_id`、`site`、`split`、`source_years`、`generator_name`、`generator_version`、`generator_seed`、`weather_file`、`file_sha256`、`climate_parameter_file`、`parameters_hash`、`generation_config`、`validation_status` 等字段及单位、含义和示例。明确训练数据仅用于拟合生成器；真实 validation/test 年份既不能参与拟合，也不能用来选择天气生成超参数。保留历史天气抽样作为 `baseline_weather`，另设计 `augmented_weather`；两组未来的 PPO 算法、网络、reward、observation、action space、训练步数和测试天气均应一致。
5. **区分三个随机源。** 检查 PPO 初始化/采样 RNG、历史年份抽样 RNG、天气生成 RNG 的真实代码路径；“都设置为 0”本身不等于共用一个 RNG 实例。设计独立 `ppo_seed`、`weather_selection_seed`、`weather_generation_seed` 或等效方案，保持旧 `055_00` 完全不变；新方案须能从清单复现天气和年份选择。优先通过新实验配置/外部封装实现，不要悄悄改旧实验的随机序列。
6. **建立质量验收门槛。** 至少检查降水量和太阳辐射非负、`TMAX >= TMIN`、缺失值与单位、日历/闰年、DSSAT 必需字段；并比较合成与训练年历史数据的月降水总量、雨日比例、降雨强度、连续无雨天分布、季节温度/辐射、变量相关性。建议先定义异常判据与人工审查项，不能将经验阈值描述为科学定律。对通过天气检查的小样本再运行 DSSAT 最小 smoke test，检查能否读取、是否完整完成季节、生育期和产量是否出现明显异常；不可将单个 smoke test 视为最终农业有效性证明。
7. **最小可行性验证。** 仅在技术路线已经核实、可安全隔离的前提下，使用 YC 训练年份参数生成 3–5 套合成天气，写入新的临时/实验专属目录，记录每套独立 seed 与文件 hash；固定 seed 重复生成应得到相同结果，不同 seed 的天气应有可测差异。运行极少量 DSSAT reset/单季读取 smoke test，不启动 PPO 训练，不修改原始天气。若目前没有经过核实的生成器实现，就停在可执行设计和障碍报告，不要用随意独立高斯扰动伪造“合理天气”。
8. **处理临时目录必须谨慎。** 审计遗留的 `.codex-yc-weather-pptx-build/` 与 `results/yc_weather_audit/rendered_probe/`：先检查文件归属、是否被追踪以及是否仍有依赖，列出拟清理清单；不确定或含用户内容就保留，绝不能 `git clean -fd` 或递归删除未知目录。若是明确可再生的本任务产物且符合 `AGENTS.md`，才可安全清理。
9. **明确下一步的实验证据。** 规划未来原始历史天气 vs 增强天气的公平对照：相同训练步数、相同 `ppo_seed` 集合、明确的天气 seed、相同真实未见年份测试；预先定义成功条件（产量、WUE/NUE 的具体公式、资源预算、管理次数和合理性、与基线相比的综合判据）并报告全部随机种子与成功率，不从测试集事后挑选赢家。不得预设 PPO 必然优于其他四种农业基线；四种基线留到正式农业比较阶段，不在本任务运行。
10. **交付与备份。** 在 `docs/` 写 `yc_weather_dataset_design.md`（依据、已确认事实、两条技术路线比较、选型、数据划分、manifest、质量验收、smoke test 结果、风险与 003 实施建议），同时制作 `docs/yc_weather_dataset_design.pptx`（内容与 MD 一致，清晰标注“设计/小样本验证，非训练结果”）。只使用英文小写及下划线命名 docs 文件；图片等临时产物放入 YC 专用目录。提交所需代码、报告和小样本清单前确认 `git diff` 不包含 LC/SY、HL/FQ、旧 `055_00` 及无关文件。允许按仓库要求创建**本地** commit 并记录 hash；`AGENTS.md` 要求推送审批，因此不得擅自 `git push`，请报告待用户审批的推送命令、目标远端和分支。若未实现某要求，要明确原因，不要声称已完成。

## 结束时请用中文汇报

依次汇报：① 已确认的 YC 数据划分及证据路径；② 当前可行的天气生成方案、`.CLI` 是否仍为阻塞项；③ 是否完成 3–5 套小样本与可复现/物理/统计/DSSAT smoke test；④ 新增和修改的所有路径及隔离检查；⑤ 未解决的技术阻塞；⑥ 本地 commit/GitHub push 状态；⑦ 下一份 `003` 应实施的最小工作。不得用笼统“测试通过”代替实际运行命令、数据与结果。
