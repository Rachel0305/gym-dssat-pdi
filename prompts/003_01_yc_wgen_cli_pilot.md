# 003_01 YC WGEN / `.CLI` pilot：打通真实随机天气生成链路（不启动 PPO）

## 任务背景

本项目当前只处理 **YC 站（Yucheng / YCA）**。请先阅读仓库根目录 `AGENTS.md`、现有 `docs/yc_weather_pipeline_audit.md`、`docs/yc_weather_dataset_design.md`、相关配置与脚本，再执行本任务。

已确认事实：

- 当前 `gym_dssat_pdi` 版本为 `0.0.5`。
- 当前 YC 基线实验（`055_00`）没有启用 WGEN / `random_weather`。
- 当前 episode 间天气变化来自对 YC/YCA 训练年份 `2004-2013` 历史 `.WTH` 的抽样。
- 当前 validation 为 `2014-2023`。
- 当前没有已经证明“从未用于调参/模型选择”的独立 test 年份。
- 当前 YC lowIC 输入目录没有 `.CLI`。
- 当前 `ppo_seed=0`，天气年份抽样也使用 seed 0，二者尚未形成独立 seed contract。
- 001/002 已完成审计和方案设计；本任务不重复做大范围审计。
- LC、SY 为已冻结成功站点，**禁止修改**；HL、FQ 本任务也不处理。

当前研究目的不是修改 PPO 算法，而是参考已有农业强化学习中的随机天气增强思路，为 YC 建立一个**有来源、可复现、可验证**的天气生成链路，然后再决定是否进入 PPO 增强训练。

---

# 本任务唯一主目标

**使用 YC 训练期历史天气（原则上仅 `2004-2013`）建立并验证 YC 专属的 DSSAT WGEN / `.CLI` 随机天气生成链路，生成 3–5 个真正新的 pilot 天气 realization，完成 seed 复现、合理性检查和 DSSAT 单季 smoke test。**

本任务结束时只回答：

> “YC 是否已经具备可信、可复现、可被 Gym-DSSAT/DSSAT 正常使用的随机天气生成能力？”

**本任务禁止启动正式 PPO 训练。**

---

# 关键执行要求

## 1. 先处理数据划分与信息泄漏问题

1.1 核查 repo 中 YC/YCA 的全部可用真实天气年份，以及这些年份过去是否已经用于训练、validation、超参数选择、模型挑选或人工反复比较。

1.2 保持以下已确认划分不被静默改写：

```text
train = 2004-2013
validation = 2014-2023
```

1.3 搜索是否存在尚未使用过的额外真实年份，可作为未来 independent test 候选。

1.4 **禁止**仅把已经反复用于模型选择的 validation 年份重新命名为 `test`，然后宣称是独立测试集。

1.5 如果当前确实没有未使用真实年份，则在报告中明确写：

```text
independent_test_status = not_available_or_not_verified
```

这不阻塞本次 `.CLI` / WGEN pilot，但必须记录为后续正式论文实验设计的限制。

1.6 `.CLI` / 天气生成器参数估计原则上只允许使用 `train=2004-2013`。若任何官方工具强制读取其他年份，必须停下来说明原因，不能默认混入 validation。

---

## 2. 核查 YC 的 WSTA、`.WTH` 完整性和 WGEN 输入条件

2.1 找到 YC 实际运行使用的：

- experiment / FileX；
- `WSTA`；
- `CNYC*.WTH`；
- soil file；
- 当前 Gym-DSSAT 启动参数；
- 相关临时输入目录。

2.2 记录 `WSTA` 与 `.CLI` 命名/匹配要求，**以当前安装的 DSSAT / gym-DSSAT 源码和官方说明为准**，不要凭记忆推断。

2.3 对 `2004-2013` 训练天气做最低限度完整性检查：

- 日期连续性；
- 必要字段是否存在；
- `RAIN >= 0`；
- `TMAX >= TMIN`；
- solar radiation 非负且不存在明显格式错误；
- 缺失值、重复日期、异常 sentinel；
- 每年实际记录天数及 DSSAT 年份编码是否一致。

2.4 将检查结果保存为机器可读 artifact，而不是只写在 Markdown 中。

---

## 3. 优先建立“有官方来源”的 YC `.CLI`

本任务**优先采用 DSSAT 原生 WGEN 路线**。

3.1 首先调查本机/容器/安装目录中是否存在能够从历史逐日天气计算 WGEN 参数或创建 `.CLI` 的 DSSAT 官方工具、WeatherMan 组件、可调用 executable、脚本、源代码或现有 pipeline。

3.2 检查当前版本 `.CLI` 的真实格式与参数含义。允许参考：

- 当前 DSSAT 安装文件；
- DSSAT 官方用户指南；
- DSSAT 官方源码/示例；
- gym-DSSAT 当前版本源码与文档。

3.3 **严禁**根据月平均值“猜一个 `.CLI`”，严禁手工拼接未经验证的 WGEN 参数，严禁把其他站点（尤其默认 Florida）的 `.CLI` 改名后当作 YC 参数。

3.4 如果可以可靠自动化，则：

- 只使用 YC `2004-2013` 训练天气估计参数；
- 生成 YC 专属 `.CLI`；
- 保存参数来源、工具版本、执行命令、输入文件 hash、输出 `.CLI` hash；
- 将生成过程写成可重复执行的脚本或命令记录。

3.5 如果官方参数估计只能通过 WeatherMan GUI 完成，而 Codex 无法可靠自动操作，则**不要伪造结果**。请：

- 明确判定 `BLOCKED_CLI_GENERATION`；
- 找出最短的官方操作路径；
- 列出需要用户手动完成的最小步骤和输入文件；
- 同时继续调查是否存在官方可脚本化替代路径。

只有在 `.CLI` 来源可信后，才能继续后面的 WGEN pilot。

---

## 4. 建立独立的 weather seed contract

无论 WGEN 的具体调用接口如何，都必须把天气随机性与 PPO 随机性分开。

至少建立并记录以下字段：

```text
ppo_seed
weather_generation_seed
weather_source
weather_parameter_artifact
weather_realization_id
```

本任务不训练 PPO，因此 `ppo_seed` 只作为保留字段，不参与 pilot 天气生成。

建议 pilot 使用 3–5 个明确 seed，例如：

```text
weather_generation_seed = [101, 102, 103, 104, 105]
```

如果当前 WGEN API 对 seed 有特殊范围或行为，以实测为准。

必须证明：

1. 相同 `.CLI` + 相同 `weather_generation_seed` → 生成结果可复现；
2. 相同 `.CLI` + 不同 `weather_generation_seed` → 生成结果存在可测差异；
3. 复现证据必须基于真实生成的逐日天气 artifact、临时 WTH、可导出的天气表，或等价的可审计序列；不能只凭“程序没有报错”。

如果 gym-DSSAT/WGEN 不直接导出 `.WTH`，请追踪运行时临时文件或从 DSSAT 可访问输出中提取实际逐日天气序列。若确实无法获得完整序列，必须在报告中说明，并设计当前条件下最强的可复现证据；不得伪造 hash。

---

## 5. 生成 3–5 个真正新的 YC WGEN pilot realization

只有第 3 节 `.CLI` 来源可信后才执行。

要求：

5.1 pilot 必须来自 YC 专属 WGEN 参数，不得复用 217YCA deterministic rain-window scenario bank 充当“新随机天气”。

5.2 不要求现在生成几百或几千套天气；本轮只做 3–5 套，目的为技术验证。

5.3 为每套 realization 建立 manifest，至少包含：

```text
site
weather_realization_id
weather_generation_seed
weather_source
cli_path_or_id
cli_sha256
source_train_years
generation_timestamp
generator_name
generator_version
weather_artifact_path
weather_artifact_sha256
```

如果某字段无法获得，写 `null` 并解释原因，不能编造。

---

## 6. Weather quality gate：天气合理性检查

对 pilot 天气做两级检查。

### 6.1 硬性物理/格式 gate

至少检查：

- 降水非负；
- `TMAX >= TMIN`；
- solar radiation 非负；
- 日期连续；
- DSSAT 能正确解析；
- 无 NaN/缺失/sentinel 泄漏；
- 无明显不可能值或格式错位。

任何硬性 gate 失败，则本任务不得判定 WGEN ready。

### 6.2 与训练期历史天气的统计 sanity check

由于本轮只有 3–5 个 realization，**不要做夸张的统计显著性结论**。只做描述性 sanity check，至少比较：

- 月降水量；
- 年/生长季累计降水；
- rainy days；
- maximum dry spell length；
- `TMAX` / `TMIN` 月均值与标准差；
- solar radiation 月均值与标准差；
- 必要时检查 wet/dry day 条件下的温度/辐射差异。

输出表格/图，标记明显脱离 `2004-2013` 历史范围的情况，但不要因为一个 pilot 超出历史 min/max 就自动判错；需要结合生成机制和物理合理性解释。

---

## 7. DSSAT / Gym-DSSAT smoke test

天气 gate 通过后，对每个 pilot realization 至少完成：

1. Gym-DSSAT 环境能够创建/reset；
2. DSSAT 能读取 YC `.CLI` / 随机天气；
3. 完成至少一个完整生长季模拟或等价的最小可证明运行；
4. 无致命 `WARNING.OUT` / parser error / weather station mismatch；
5. 记录 planting/termination date、最终 yield（如果 smoke run 能得到）、以及关键日志路径。

这里的 yield **只用于发现异常或崩溃，不做 PPO 性能结论**。

不得在这一轮启动 SB3 PPO 正式训练，也不得为了 smoke test 改 reward/action/observation。

---

## 8. 明确 PASS / BLOCKED 判定

最终必须给出下面二者之一，不能含糊：

### `PASS_WGEN_READY`

只有同时满足以下条件才允许：

- `.CLI` provenance 清楚且只来自允许的 train 数据；
- 3–5 个新随机天气 realization 成功生成；
- 相同 seed 可复现；
- 不同 seed 有差异；
- 硬性天气 gate 通过；
- DSSAT/Gym-DSSAT smoke test 通过。

### `BLOCKED_WGEN_NOT_READY`

如果 `.CLI` 无法可信生成、seed 无法控制、天气无法导出/验证、或 DSSAT smoke 失败，则使用此结论，并列出**最小阻塞项**及下一步，不要绕过问题开始 PPO。

---

# 禁止事项

1. **禁止修改 LC、SY。**
2. **禁止修改 HL、FQ。**
3. 禁止覆盖、删除或重写现有 YC `055_00` baseline 结果。
4. 禁止修改 PPO 算法结构。
5. 禁止修改 reward、action space、observation space、训练步数来“顺便优化”。
6. 禁止正式 PPO 训练。
7. 禁止使用 validation `2014-2023` 拟合 `.CLI`，除非发现当前数据事实与记录冲突；如冲突先报告。
8. 禁止把 217YCA deterministic rain-window scenario bank 冒充 WGEN 随机天气。
9. 禁止使用默认 Florida `.CLI` 作为 YC 气候参数。
10. 禁止伪造 `.CLI`、天气 hash、seed 复现结果或 DSSAT smoke 证据。
11. 不要擅自删除 001/002 遗留的未追踪目录；只做 cleanup inventory。若确认是本任务创建的临时构建目录，可在报告中说明后清理。

---

# 推荐新增/输出文件

文件名可以根据 repo 实际结构微调，但应保持清楚、可追踪。

```text
prompts/003_01_yc_wgen_cli_pilot.md

docs/yc_wgen_cli_pilot.md
docs/yc_wgen_cli_pilot.pptx

scripts/prepare_yc_wgen_cli.py          # 仅在确有可靠自动化方法时创建
scripts/pilot_yc_wgen_weather.py
scripts/validate_yc_weather_pilot.py

results/yc_wgen_cli_pilot/
    split_audit.json
    train_weather_qc.csv
    yc_cli_provenance.json
    yc_weather_manifest.csv
    seed_reproducibility.json
    weather_quality_summary.csv
    dssat_smoke_summary.csv
    logs/
    weather_artifacts/
```

若 `.CLI` 应放在 DSSAT input 目录中，请同时在 `results/yc_wgen_cli_pilot/yc_cli_provenance.json` 记录其真实路径和 hash；不要只保存一个无上下文副本。

---

# `docs/yc_wgen_cli_pilot.md` 必须回答的问题

请用“证据 → 判断”的方式写，不要只描述做了什么。

至少回答：

1. YC 真实天气最终有哪些年份？有没有真正未使用的 independent test 候选？
2. `.CLI` 是如何得到的？用了哪些年份？什么工具/版本？是否完全可复现？
3. YC 的 `WSTA` 与 `.CLI` / WGEN 链路如何匹配？
4. `weather_generation_seed` 如何控制？是否与未来 `ppo_seed` 解耦？
5. 相同 seed 是否得到相同天气？证据是什么？
6. 不同 seed 是否得到不同天气？差异有多大？
7. 3–5 套天气是否通过物理、格式和统计 sanity check？
8. DSSAT/Gym-DSSAT smoke 是否全部通过？
9. 最终结论是 `PASS_WGEN_READY` 还是 `BLOCKED_WGEN_NOT_READY`？
10. 如果 PASS，下一轮 004 最小任务是什么？如果 BLOCKED，唯一/主要阻塞是什么？

报告必须清楚区分：

```text
observed historical weather
WGEN parameter artifact (.CLI)
stochastic weather realization
DSSAT smoke result
PPO result
```

本任务没有 PPO result，禁止把任何 smoke yield 描述为 PPO 表现。

---

# PPT 任务记录要求

生成：

```text
docs/yc_wgen_cli_pilot.pptx
```

建议 7–10 页，至少包括：

1. 任务背景与 001/002 已知事实；
2. train/validation/test 状态；
3. YC `.CLI` provenance；
4. WGEN + seed contract；
5. pilot realization 与复现结果；
6. 天气质量检查；
7. DSSAT smoke test；
8. PASS/BLOCKED 结论与下一步。

PPT 只记录事实和证据，不要把尚未运行的 PPO 写成结果。

---

# Git / GitHub 要求

1. 执行前记录：

```bash
git status --short
git branch --show-current
git rev-parse --short HEAD
```

2. 不要把 001/002 的无关临时文件误提交进本任务 commit。

3. 完成本任务后进行本地 commit，建议 message：

```text
chore: validate YC WGEN weather pilot
```

4. 按项目 `AGENTS.md` 的 GitHub 权限规则执行。若规则要求用户确认 push，则**不要擅自 push**；在最终报告中给出当前 branch、commit hash 和建议的精确 `git push` 命令，等待用户确认。

5. 若规则与本 prompt 有冲突，以 `AGENTS.md` 和用户最新明确指令为准。

---

# 执行原则

- 先证据，后修改。
- 优先最小改动，不重构无关代码。
- 所有生成天气必须有 provenance、seed 和 hash/等价可复现证据。
- 遇到官方工具不可自动化时，宁可报告阻塞，也不要自己“猜参数”。
- 本轮的成功标准不是“生成了一个文件”，而是**可信地证明 YC 随机天气生成链路已经可以用于下一阶段实验**。

最终回复用户时，请用简洁编号总结：

1. test split 状态；
2. `.CLI` 是否成功；
3. 生成了多少个真正的新 WGEN realization；
4. seed 复现是否通过；
5. 天气质量 gate 是否通过；
6. DSSAT smoke 是否通过；
7. `PASS_WGEN_READY` / `BLOCKED_WGEN_NOT_READY`；
8. 新增/修改文件；
9. Git commit / push 状态；
10. 下一步建议。
