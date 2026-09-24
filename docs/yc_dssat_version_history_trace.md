# DSSAT 4.8.0 / 4.8.5 历史版本证据追溯

**审计日期：** 2026-09-24
**任务：** `003_06_02_01_trace_dssat_version_history`
**范围：** 项目内当前文件与本地 Git history；未访问仓库外路径、Windows 安装位置或 container。

## 1. 结论

| 问题 | 判定 | 主要依据 |
|---|---|---|
| 历史 DSSAT 4.8.0 | `CONFIRMED` | YC 2014 PDI 临时快照的 `Summary.OUT` 首行直接写出 `DSSAT Cropping System Model Ver. 4.8.0.024`，时间为 2026-07-02 08:37:47 |
| 历史 DSSAT 4.8.5 | `SUPPORTED` | 2026-06-27 的明确实验总结记载 Windows DSSAT 4.8.5 的 HLA 2004 `CNHL0404` 结果；本轮在该 CNHL0404 文件夹未找到可核对的 4.8.5 原始输出版本头 |
| 历史 4.8.0 vs 4.8.5 对照 | `SUPPORTED` | 文档及作图脚本记载 Windows 4.8.5 与 PDI/Gym 4.8.0 的对照；但结论警告不要把它当作纯版本 A/B 因果实验 |
| 历史 Gym-DSSAT 与 `/opt/dssat_pdi/run_dssat` 的联系 | `CONFIRMED_FOR_SAVED_2026-07-02_RUN` | 同一 YC 2014 运行目录中的 `env_args.json` 配置该 launcher，配对 PDI snapshot 的 `Summary.OUT` 记载 4.8.0.024 |
| 当前 active runtime DSSAT 版本 | `UNKNOWN` | 2026-07 的文件不能证明 2026-09 的活动容器/安装没有变化；`AGENTS.md` 禁止访问仓库外 runtime |

版本字符串分属不同软件：历史诊断记载的 `gym_dssat_pdi 0.0.5`、`requirements.txt` 声明的 `0.0.9` 都是 Python package 版本，不是 DSSAT 模型版本。当前安装的 `gym_dssat_pdi` 与当前 DSSAT runtime 版本仍未知。

用户关于“主 Gym-DSSAT 环境历史使用 DSSAT 4.8.0”的回忆受到直接历史输出支持，至少可确认到 2026-07-02 的 YC/PDI 运行。仓库也支持曾有 Windows DSSAT 4.8.5 实验记录；但“4.8.5 在另一台电脑”目前仍只由本任务引用的用户回忆提供，项目内没有找到可独立确认物理电脑身份的记录。

## 2. 搜索方法与范围

对 `docs/`、`results/`、`src/`、`prompt_02/`、`prompts/`、`references/`、`scripts/`、`backups/` 和相关 `DSSAT_auto_validation/` 子目录进行了当前树检索；覆盖 Markdown、Python、JSON、YAML、TXT、LOG、OUT、CSV 等文本文件。对 `DSSAT_auto_validation/` 的大量结果输出使用版本头定向检索，避免打印或复制大体积原始结果。

检索词包括：`4.8.0`、`4.8.5`、`DSSAT 4.8`、`DSSAT4.8`、`DSSAT48`、`DSSAT480`、`DSSAT485`、`dscsm048`、`dscsm048.exe`、`run_dssat`、`DSSAT version`、`version comparison`、`版本对照`、`版本比较`、`不同版本`、`另一台电脑`、`second computer`、`simulation difference`、`result difference`。

对 Git 本地 refs 使用 `git log --all`、`-S` pickaxe、commit-message 搜索、`--name-status` 和删除文件名检查。没有切换 commit，没有恢复文件，没有改写历史。任务提示中的用户回忆与列举的搜索词只当成待查线索，不当作独立版本证据。

## 3. 当前树中的 DSSAT 4.8.0 证据

### 直接版本头

文件：

```text
DSSAT_auto_validation/yc2014_station_level3_true_model_transfer_016_04/runs/2014/seed0/null/pdi_tmp_snapshot_eval/Summary.OUT
```

第 1 行为：

```text
*SUMMARY : fileX.MZ   CNYC ... DSSAT Cropping System Model Ver. 4.8.0.024 -stable JUL 02, 2026; 08:37:47
```

这是 `.OUT` 内 DSSAT 自报版本头，级别为 `DIRECT_VERSION_EVIDENCE`。上下文是 YC/CNYC、weather year 2014、seed 0、null case，目录位于 `pdi_tmp_snapshot_eval`。

### 与 Gym-DSSAT 历史运行链的连接

同一运行目录的 `env_args.json` 记录：

```json
"run_dssat_location": "/opt/dssat_pdi/run_dssat"
```

文件还记录 `CNYC2014_null.MZX`、PDI log 路径和辅助输入。相关 runner `src/run_yc2014_station_level3_true_model_transfer_016_04.py` 将同一 launcher 写入 env args；该源码于 Git commit `080c1dff7177e02cee60dcaced8f56d4e9ae5af5`（2026-07-05）加入。由此可把这份历史 `Summary.OUT` 与当时的 Gym-DSSAT/PDI 调用链连接起来。

注意：`.OUT` 与 `env_args.json` 本身未被 Git 跟踪，是本机当前项目目录中的历史产物；它们证明的是保存下来的那次运行，不是 2026-09 当前活动 runtime。该记录没有同时给出该次运行安装的 `gym_dssat_pdi` package 版本。

## 4. DSSAT 4.8.5 及比较记录

### 明确实验记录

[2026-06-27 验证总结](2026-06-27_gym_dssat_windows_dssat_validation_summary.md#L41-L42) 记录 HLA 2004 `CNHL0404`：Windows DSSAT 4.8.5 可完整成熟，HWAM 约 2038 kg/ha；同一命名输入的 PDI/Gym DSSAT 4.8.0 记录为早熟、HWAM=0。总结指出不可直接把 Windows 4.8.5 与 PDI 4.8.0 混比。按本任务证据标准，这是带具体站点/试验/结果的显式实验记录，评级 `DIRECT_VERSION_EVIDENCE`，总体状态仍记 `SUPPORTED`，因为本轮没有在该 Windows run 文件夹找到对应的 4.8.5 原始 `.OUT` 版本头。

该比较的字段如下：

| 项目 | 仓库记录 |
|---|---|
| purpose | 排查 HLA 2004 `CNHL0404` 的不同成熟/产量结果，判断 FileX、初始条件还是模型/runtime 链路差异 |
| machine/runtime | Windows standalone DSSAT 4.8.5 对 PDI/Gym DSSAT 4.8.0；Windows 安装线索在 `DSSBatch.v48`，路径为 `C:\DSSAT48\Maize`、可执行名为 `DSCSM048.EXE` |
| date/context | 2026-06-27 文档所总结的 HLA 2004 IC=1 单年诊断；文档没有给出物理电脑标识 |
| compared_against | PDI/Gym DSSAT 4.8.0 的 `CNHL0404` 结果；随后文档还讨论 Windows 4.8.0 null 输出 |
| what_outputs_were_compared | 作物成熟/生育进程和最终 HWAM；其他 HLA 多年图脚本还展示 `WSPD/NSTD/CWAD/GWAD` 日序列 |
| conclusion_if_recorded | 4.8.5 与 4.8.0 输出不可直接混比；后续以 Windows 4.8.0 与 PDI/Gym 4.8.0 做版本一致性检查 |

同日 Git commit `7bc4024b4e7e3d171e659e2dddbf6c8a16cc26b5`（`docs: summarize gym dssat validation checks`）新增该总结、`src/plot_windows_vs_gym_high_contrast_all_years.py` 和诊断/对照脚本。作图脚本将 Windows 一侧标为 DSSAT 4.8.5、Gym/PDI 一侧标为 4.8.0。后续 `src/compare_hla_windows480_vs_gym_pdi_ic0_2004_2023.py` 的说明写明：新的 Windows 4.8.0 standalone 输出替换此前的 Windows 4.8.5 对照侧。

`src/diagnose_pdi_field_coordinates_ic0_null.py:176-180` 还把 Windows DSSAT 4.8.5 rerun 写为目的/预期判读。检查其代码可见该函数准备复跑包并写 README；这一段计划文字本身不证明手动 Windows 复跑已完成，故不作为实际结果的依据。

### Windows 可执行文件的间接证据

本机文件 `DSSAT_auto_validation/HLA_2004/run_CNHL0404/DSSBatch.v48` 包含 `C:\DSSAT48\Maize` 和命令 `C:\DSSAT48\DSCSM048.EXE MZCER048 B DSSBatch.v48`。这支持该文件属于 DSSAT 4.8 家族 Windows 安装，等级 `STRONG_INDIRECT_EVIDENCE`；但 `DSSAT48` 和 `DSCSM048.EXE` 不能区分 DSSAT 4.8.0 与 4.8.5。精确的 4.8.5 结论来自上述实验报告，不是从文件名推算。

## 5. Git history 结果

- `7bc4024b4e7e3d171e659e2dddbf6c8a16cc26b5`（2026-06-27）是当前 4.8.5 与 4.8.0 对照文档/脚本的首个 Git 记录。
- `DSSAT 4.8.0` 内容的 pickaxe 还命中后续实验说明和 prompts；这些主要是模型配置/试验记录，不都包含版本头。
- `DSSAT480` 同时存在于目录命名、文件路径和文档中。单独命中该字符串只能视为 `AMBIGUOUS`，除非与明确实验记录或 `.OUT` 头结合。
- 对 `/opt/dssat_pdi/run_dssat` 的 Git 历史检索命中 2026-06-06 的 `src/ppo_safe_rendering.py` 和 2026-07-05 的 YC2014 runner。它证明项目有这条历史配置，但不能证明当前容器继续沿用。
- 本轮 Git 删除文件名扫描没有发现相关 DSSAT 版本/比较文件被删除；`4.8.5` pickaxe 的相关版本记录显示为 2026-06-27 的新增文件。Git commit message 未找到独立命名的 4.8.0-vs-4.8.5 比较实验，具体内容存在于提交文件正文。
- 当前 `.OUT`、`env_args.json` 和 `DSSBatch.v48` 是本地树证据而非 Git 版本历史文件；Git history 没有保存这些原始内容的历史快照。

完整匹配行与查询记录见 `results/yc_wgen_cli_pilot/003_06_02_01/`。

## 6. 版本字段分离与连续性

| 字段 | 结论 |
|---|---|
| `historical_dssat_version` | `4.8.0.024`，2026-07-02 YC/PDI 运行直接确认；4.8.0.024 属于 DSSAT 4.8.0 系列 |
| `historical_gym_dssat_runtime_version` | 该保存的 PDI/Gym 运行使用 DSSAT 4.8.0.024，证据等级 `DIRECT_VERSION_EVIDENCE` |
| `historical_gym_dssat_package_version` | 本次 016_04 run 未找到同目录 package 元数据；另一份较早诊断记录 `0.0.5`，只能作为独立历史包记录 |
| `requirements.txt` 声明 | `gym-dssat-pdi @ file:///home/gymdev/gym_dssat_pdi/gym-dssat-pdi-0.0.9`，不等同于已安装版本 |
| `current_active_runtime_version` | `UNKNOWN`；当前活动 runtime 与 7 月快照之间是否连续，仓库没有证明 |

## 7. 尚未确认与下一步

1. 2026-09 当前 container 中 DSSAT 实际版本及 `/opt/dssat_pdi/run_dssat` 是否仍指向同一 executable。
2. 2026-07-02 016_04 run 所安装的 `gym_dssat_pdi` package 版本。
3. Windows DSSAT 4.8.5 对应的原始 `.OUT` 版本头，以及该安装是否确属用户记忆中的“另一台电脑”。
4. 4.8.0 与 4.8.5 差异是否在相同输入、相同初始条件、相同管理和可复现机器环境下单独控制验证。现有报告没有支持这种纯版本因果结论。

建议先通过项目允许的方式取得当前 runtime 的只读版本证据并放入项目内；如需确认 4.8.5，再索取已有 Windows run 的原始输出头、安装版本页面/元数据与机器/日期 provenance。不要从 `DSSAT480`、`DSCSM048.EXE` 或目录名推版本，也不要把历史版本自动外推为当前版本。

## 8. 改动、Git 与禁止事项核验

- 新增本报告、PPT、版本检索日志、版本头摘录、summary JSON 与 PPT 构建脚本。
- 未修改任何 `.WTH`、`.CLI`、frozen weather candidate、模型/训练代码或其他站点文件；没有创建 `CNYC.CLI`，没有运行 DSSAT simulation / WGEN / WeatherMan / PPO。
- 审计基线分支：`codex/sya-forecast-freeze-2026-08-16`；审计开始 HEAD：`cffee7a6766e7a8c07933e3ce466ba8ee30a2fbf`。提交时仅 stage 本任务新增产物；既有工作区改动保持不动。
- Local commit message：`chore: trace DSSAT version history`。不执行 `git push`；GitHub backup pending explicit user approval。

## 9. 证据产物

- `results/yc_wgen_cli_pilot/003_06_02_01/dssat_version_search_current_tree.txt`
- `results/yc_wgen_cli_pilot/003_06_02_01/dssat_version_search_git_history.txt`
- `results/yc_wgen_cli_pilot/003_06_02_01/dssat_version_output_header_evidence.txt`
- `results/yc_wgen_cli_pilot/003_06_02_01/dssat_version_evidence_summary.json`
