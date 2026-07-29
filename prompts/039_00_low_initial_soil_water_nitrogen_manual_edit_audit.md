# 039_00 低初始土壤水分/氮条件手动修改与生效审计 prompt

## 背景

导师指出：灌溉和施肥应该通过影响土壤水分/氮胁迫来帮助强化学习算法学习。如果当前初始土壤水分和氮条件过于充足，许多站点年份中 WSPD/NSTD 变化很弱，PPO/DQN 的水肥动作就很难通过胁迫过程获得清晰反馈。

因此，本任务准备测试：

> 降低初始土壤水分与初始土壤无机氮后，DSSAT 输出中的水分胁迫/氮胁迫是否明显增强，并且灌溉/施肥动作是否能更清楚地改变胁迫过程。

本任务只做输入修改生效审计，不训练 PPO/DQN。

## 数据源

原始数据源：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/
```

该目录包含五个站点：

```text
FQ/
HL/
LC/
SY/
YC/
```

每个站点的初始条件主要在 `.MZX` 管理文件的 `*INITIAL CONDITIONS` 段中，例如：

```text
@C  ICBL  SH2O  SNH4  SNO3
```

其中：

- `ICBL`：土层深度；
- `SH2O`：该层初始土壤水分；
- `SNH4`：该层初始铵态氮；
- `SNO3`：该层初始硝态氮。

## 非常重要的边界

不要直接覆盖原始目录。

原始目录必须保持只读意义：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/
```

建议用户手动复制一份派生目录：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/
```

然后只在派生目录中修改 `.MZX` 初始条件字段。

这样后续审计可以明确比较：

```text
原始输入 multisite_new_cultivar_inputs_013
vs
低初始水氮输入 multisite_new_cultivar_inputs_013_lowIC_manual
```

## 用户手动修改任务

用户先手动复制并修改：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/
```

建议需要检查和修改的站点管理文件包括：

```text
FQ/CNFQ0801.MZX
HL/CNHL1001.MZX
LC/CNLC0801.MZX
SY/CNSY1201.MZX
YC/CNYC0801.MZX
```

注意：HL 目录里还存在：

```text
HL/CNHL0701_corrected_IC123.MZX
```

后续审计脚本需要识别当前训练/渲染流程实际使用哪一个 `.MZX`，不能仅凭文件名假设。

手动修改原则：

1. 只修改派生目录；
2. 只修改 `*INITIAL CONDITIONS` 段；
3. 不修改天气、土壤剖面、品种、管理日期、施肥灌溉历史；
4. 每个站点保留相同土层结构；
5. 降低 `SH2O`、`SNH4`、`SNO3`，但不要改乱列宽和字段顺序；
6. 修改完成后不要立刻训练。

## Codex 后续代码任务

用户手动修改完成后，Codex 编写审计脚本：

```text
src/audit_low_initial_soil_water_nitrogen_inputs_039_00.py
```

脚本目标：

1. 读取原始目录和 lowIC 派生目录；
2. 找到每个站点实际可用的 `.MZX`；
3. 解析 `*INITIAL CONDITIONS` 段；
4. 输出每个站点、每个土层的：
   - 原始 `SH2O/SNH4/SNO3`
   - lowIC `SH2O/SNH4/SNO3`
   - 绝对变化
   - 相对变化
5. 检查 IC=1 是否仍然存在于 treatment 行；
6. 检查土层数和 `ICBL` 是否一致；
7. 若发现派生目录缺文件、列错位、土层不一致、IC 被关掉，则直接报错，不进入 DSSAT smoke。

## DSSAT smoke 任务

通过输入审计后，选择一个站点一年进行 smoke。

优先候选：

```text
FQ2014
```

原因：

- FQ 已有 037_07/037_08 可信基线和图表；
- 后续比较更容易；
- 不先碰五站点全量，避免算力浪费。

smoke 比较：

```text
原始输入 + null
低IC输入 + null
原始输入 + official expert
低IC输入 + official expert
```

可选再加入：

```text
原始输入 + recorded farmer
低IC输入 + recorded farmer
原始输入 + DSSAT auto
低IC输入 + DSSAT auto
```

但第一轮优先 null/expert 即可。

## 输出文件

脚本应输出到：

```text
benchmark_results/039_00_low_initial_soil_water_nitrogen_input_audit/
```

至少包含：

```text
tables/039_00_ic_profile_comparison.csv
tables/039_00_ic_profile_issues.csv
tables/039_00_smoke_stress_summary.csv
figures/039_00_fq2014_original_vs_lowIC_stress_smoke.png
039_00_result.json
```

实验记录：

```text
docs/039_00_low_initial_soil_water_nitrogen_input_audit_record.md
```

## 判定标准

### A 分支：可以继续

满足：

1. lowIC 派生目录存在；
2. 五站点 `.MZX` 可解析；
3. IC=1 未被破坏；
4. `ICBL` 土层结构一致；
5. `SH2O/SNH4/SNO3` 确实降低；
6. FQ2014 smoke 中 lowIC 的 WSPD 或 NSTD 相比原始输入明显增强；
7. expert 或管理情景对胁迫有可见缓解。

则进入 039_01：低 IC 条件下单站点 PPO smoke。

### B 分支：输入修改未生效

如果 `.MZX` 数值改变了，但 DSSAT 输出胁迫几乎不变，需要检查：

- 渲染流程是否实际使用了该 `.MZX`；
- 是否被别的模板覆盖；
- IC 是否被实验 treatment 行忽略；
- DSSAT 是否读取了别的初始条件来源。

不得直接训练。

### C 分支：输入文件错误

如果发现：

- IC=0；
- 土层错位；
- 字段列错；
- 缺少站点文件；
- 低 IC 目录结构不完整；

则停止，先修输入，不训练。

## 禁止事项

- 禁止直接修改原始输入目录；
- 禁止跳过输入审计直接训练；
- 禁止在五站点全量上直接重跑；
- 禁止把 lowIC 结果和原始 IC 结果混在一起不标注；
- 禁止把本任务结果表述为 PPO 改进成功。

## 本任务完成后再决定

只有当 039_00 证明 lowIC 真的会增强胁迫、且管理动作能改变胁迫后，才允许进入：

```text
039_01_lowIC_single_site_free_timing_ppo_smoke
```

