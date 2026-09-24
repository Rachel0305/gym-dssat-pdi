# YC DSSAT Runtime、版本与 WGEN 能力审计

**审计日期：** 2026-09-24
**任务：** `003_06_02_verify_dssat_runtime_version`
**范围：** 仅审计项目内配置、源码与历史记录；未运行 DSSAT、WGEN、WeatherMan、CLI 拟合或 PPO。

## 1. 结论摘要

当前 Gym-DSSAT 安装版本、DSSAT 底层可执行文件及版本、当前 runtime 中 WeatherMan 是否存在，均不能从本轮允许访问的证据确认。仓库历史执行记录支持项目曾通过 Linux Docker 容器 `nifty_taussig` 运行；这不是当前容器状态的实时证明。

Gym-DSSAT 源码层有 `random_weather` 接口：打开后将 FileX 天气模式设为 `W`，并向 PDI 交互传递随机种子。但活动 runtime 中的 WGEN 是否可运行、是否读取预期 `.CLI` 尚未被验证。接口没有从日天气拟合 `.CLI` 的逻辑；活动 runtime 是否安装了 WeatherMan 或其他拟合工具同样未知。

本轮没有访问候选天气文件，没有创建 `CNYC.CLI`，没有修改 PPO 或其他站点文件。

## 2. Runtime 架构

| 项目 | 审计结果 | 证据等级 |
|---|---|---|
| 项目执行记录 | Windows 项目目录通过 Docker 容器 `nifty_taussig` 调用 Linux Python `/opt/gym_dssat_pdi/bin/python` | `SUPPORTED_BUT_NOT_FULLY_CONFIRMED`，来自仓库历史记录 |
| 容器当前是否运行 | 未探测 | `UNKNOWN_DUE_TO_ACCESS_RESTRICTION` |
| 容器镜像及其版本 | 仓库未找到 Dockerfile / compose 配置；没有读取活动镜像元数据 | 未确认 |
| 历史调用链 | `Windows checkout -> Docker/Linux nifty_taussig -> /opt/gym_dssat_pdi/bin/python -> Gym-DSSAT -> /opt/dssat_pdi/run_dssat` | 仅为记录过的架构；最末端模型二进制未知 |

依据包括 [src/audit_220_yc_feedback_controller_site_year.py](../src/audit_220_yc_feedback_controller_site_year.py) 中记录的容器命令，以及 [2026-09-09 YC controller audit](2026-09-09_yc_feedback_controller_demo_audit.md)。当前 runtime、Docker daemon 和容器内部路径没有探测。按项目 `AGENTS.md`，文件系统访问限制在仓库内；`nifty_taussig`、`/opt/gym_dssat_pdi/`、`/opt/dssat_pdi/` 不在可检查的项目目录范围内。本轮没有运行一个会越过该限制的命令。

## 3. `gym_dssat_pdi` 版本

**当前已安装版本：`UNKNOWN_DUE_TO_ACCESS_RESTRICTION`。**

- [requirements.txt](../requirements.txt#L9) 声明 `gym-dssat-pdi @ file:///home/gymdev/gym_dssat_pdi/gym-dssat-pdi-0.0.9`。这是依赖声明，不证明当前安装结果。
- [历史天气诊断](../results/yc_weather_audit/yc_weather_reset_diagnostic.json) 的 `package_info` 记录版本 `0.0.5` 及容器内包路径 `/opt/gym_dssat_pdi/lib/python3.10/site-packages/gym_dssat_pdi/`。这是旧运行时记录，不是本轮实时查询。
- 两条证据不一致，因此不能把 `0.0.9` 或 `0.0.5` 宣布为当前版本。未对外部 runtime 运行 `pip show`、`pip list` 或 import 版本探测。

## 4. DSSAT executable 与版本

项目配置的 Gym-DSSAT/PDI 启动器为 `run_dssat`，配置路径 `/opt/dssat_pdi/run_dssat`（[src/ppo_safe_rendering.py](../src/ppo_safe_rendering.py#L341)；旧渲染记录也保存了该路径）。这只是项目传给 wrapper 的 launcher 配置，不等同于确认了 DSSAT 模型程序本体。

[references/dssat_pdi.py](../references/dssat_pdi.py#L34) 接收 `run_dssat_location` 并用 `shutil.which` 检查；其 `_launch_client` 在临时目录构造 `/usr/bin/env {run_dssat_location} C fileX.MZX {experiment_number}` 并调用 `subprocess.Popen`（:259-270）。

| 字段 | 结果 |
|---|---|
| `dssat_executable` | 底层 executable：`UNKNOWN_DUE_TO_ACCESS_RESTRICTION`；项目只确认配置了 `run_dssat` 启动器 |
| `dssat_executable_path` | 底层实际解析路径：未知；配置的启动器路径为 `/opt/dssat_pdi/run_dssat`，未在当前 runtime 验证 |
| `dssat_version` | `UNKNOWN` |
| `dssat_version_status` | `UNKNOWN` |

没有取得版本输出、模拟结果头、安装元数据、镜像元数据或 runtime 文档。没有根据 `DSSAT480`、`dscsm048` 等名字推断版本，也没有把用户记忆中的 `4.8.0` 当作证据。

## 5. WeatherMan 与两类 WGEN 能力

### WeatherMan

`weatherman_status: UNKNOWN_DUE_TO_ACCESS_RESTRICTION`。没有枚举当前 Linux runtime 的程序、包或目录，因此既不能说发现了 WeatherMan，也不能说它没有安装。仓库文档描述过离线参数估计工作流，但文档不是活动 runtime 的安装清单。

### 使用已有 `.CLI` 生成随机天气

wrapper 接口层支持请求此路径：`references/dssat_pdi.py:91-92` 将 `random_weather=True` 映射为天气模式 `W`；:294、:486-504 处理并传递 `rseed1`。然而活动 DSSAT runtime 是否支持 WGEN、是否解析了目标 `.CLI`、是否生成了合规天气，均为 `UNKNOWN_DUE_TO_ACCESS_RESTRICTION`。源码接口能力不代表本轮已有成功运行证据。

当前项目 YC 渲染配置仍是 `random_weather=False`，辅助输入只有 cultivar、单年 `.WTH` 和 soil（[src/ppo_safe_rendering.py](../src/ppo_safe_rendering.py#L336) 与 :340-341）；历史诊断记录相同设置。因此目前这条 YC 链路没有启用 WGEN，也没有通过该配置传入 `.CLI`。

### 从历史日天气拟合/创建新的 `.CLI`

Gym-DSSAT wrapper 没有 `.CLI` 参数拟合器。它复制显式给定的 auxiliary files 到临时运行目录（`references/dssat_pdi.py:336-341`）；源码中的 cotton `.CLI` 默认项是既有输入文件路径（:74-78），不代表生成工具。仓库搜索未发现从日天气拟合新 `.CLI` 的实现。活动 runtime 是否另有 WeatherMan 或其他估计程序：`UNKNOWN_DUE_TO_ACCESS_RESTRICTION`。

所以“WGEN 消费已有 `.CLI`”与“参数工具从历史天气创建 `.CLI`”是两个独立能力，本轮均没有对活动 runtime 做端到端确认。

## 6. `.CLI` 示例与当前调用逻辑

仓库中存在其他站点格式样例，例如 `benchmark_results/027_07_site_specific_stage_maskable_ppo/LC/readiness/baseline_runs/dssat_auto/input/CNLC.CLI`。本轮仅查看了文件头作为格式存在性参考；没有复制、改名、抽取气候参数或用于 YC。

Gym-DSSAT 只会复制 `auxiliary_file_paths` 中调用者显式提供的文件，并以 basename 放进环境临时目录。是否被 DSSAT 的 WGEN 使用，还取决于天气模式、station 匹配和活动 runtime 行为；源码不能证明本轮活动环境中的具体解析路径。

## 7. 尚未解决的问题

1. 当前容器/镜像是否仍为 `nifty_taussig`，以及当前安装的 `gym_dssat_pdi` 版本。
2. `/opt/dssat_pdi/run_dssat` 在当前容器中的真实文件、其调用的底层 DSSAT executable 及绝对路径。
3. DSSAT 版本输出或其他强版本证据。
4. 当前 runtime 是否安装 WeatherMan、是否有替代 `.CLI` 参数拟合器。
5. 活动 DSSAT 是否支持 WGEN、`.CLI` 寻址规则与 YC station code 的匹配方式，以及随机种子能否复现天气序列。

## 8. 建议的下一步

先由用户授权一个严格限于当前项目所用容器的 runtime 只读探测，或将以下命令输出/文本证据复制到项目内再审计：

1. 容器标识和镜像摘要；容器内 `python -m pip show gym-dssat-pdi` 与 import 文件路径。
2. `command -v run_dssat`、该 launcher 的文本/包归属，以及其最终调用的 DSSAT executable 路径。
3. DSSAT 自报版本、安装元数据或一份已存在模拟输出头；不要为了取证启动新作物模拟。
4. 仅针对容器内已知应用目录检查 WeatherMan、WGEN 和参数估计工具，并区分 GUI 工具与 Linux runtime。

拿到这些证据后，再决定 train-only `.CLI` 的来源。若 Linux runtime 不含参数估计工具，优先使用经过版本确认的官方 WeatherMan 流程或用户明确批准的其他工具；只使用冻结训练期 2004–2013 输入，并保留参数来源、版本、日期范围和输出哈希。该步骤不属于本轮，本报告不生成 `.CLI`。

## 9. 输出文件与改动范围

- 中文报告：`docs/yc_dssat_runtime_version_audit.md`
- 中文任务记录：`docs/yc_dssat_runtime_version_audit.pptx`
- 文字证据：`results/yc_wgen_cli_pilot/003_06_02/`
- 冻结天气候选：未读取、未修改、未重建、未重新评价。
- PPO、reward、action/observation space、LC/SY/HL/FQ：未修改；没有训练或模拟。

审计开始时分支为 `codex/sya-forecast-freeze-2026-08-16`，HEAD 为 `88267bfb3704a2519551858718252f14c74fa560`。工作区有多个与本任务无关的既有改动/未跟踪产物；本任务会只提交新增审计文件，不暂存或覆盖这些内容。

计划本地提交信息：`chore: verify YC DSSAT runtime and WGEN capability`。不会执行 `git push`；GitHub backup pending explicit user approval。
仓库规则默认忽略 `.pptx`；本任务按要求只显式暂存了这份审计 PPT。

## 10. 证据文件

- `results/yc_wgen_cli_pilot/003_06_02/runtime_environment.txt`
- `results/yc_wgen_cli_pilot/003_06_02/gym_dssat_pdi_version.txt`
- `results/yc_wgen_cli_pilot/003_06_02/dssat_executable_info.txt`
- `results/yc_wgen_cli_pilot/003_06_02/dssat_version_evidence.txt`
- `results/yc_wgen_cli_pilot/003_06_02/weatherman_wgen_capability.txt`
- `results/yc_wgen_cli_pilot/003_06_02/cli_usage_search.txt`
- `results/yc_wgen_cli_pilot/003_06_02/audit_summary.json`
