# 003_06_05_02：正式重生成 corrected `CNYC.CLI` 并恢复完整 WGEN seed pilot

## 一、任务定位

当前 YC 随机天气增强项目已经完成：

```text
2004–2013 frozen weather candidate ✅
        ↓
reproducible CLI generator ✅
        ↓
initial CNYC.CLI candidate ✅
        ↓
WGENIN 5010 root cause identified ✅
        ↓
isolated parse-fixed CLI runtime smoke PASS ✅
        ↓
regenerate official corrected CNYC.CLI
        ↓
resume full WGEN seed pilot   ← CURRENT TASK
```

已知上一轮 `003_06_05_01` 已确认：

```text
runtime_version = DSSAT 4.8.0.024
WGENIN parse contract = I6 + 14*(1X,F5.0)
root_cause = XDMN/XWMN/NAMN were over-width
parameter_values_changed = NO
weather_seed=101 single smoke = PASS
WGENIN 5010 = RESOLVED
```

隔离修复候选：

```text
results/yc_wgen_cli_pilot/003_06_05_01/candidate/CNYC_parsefix_01.CLI
```

SHA256：

```text
65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929
```

该文件已通过：

```text
12/12 monthly row static parse
23 regression tests
seed 101 runtime smoke
120 daily weather states captured
```

本任务不再重新诊断 `WGENIN 5010`。

---

# 二、本任务目标

本轮分为两个连续 gate：

## Gate A：正式重生成 corrected `CNYC.CLI`

使用已经修复的：

```text
scripts/build_dssat_cli.py
```

重新从冻结天气输入生成一个新的正式 corrected CLI。

目的：

> 后续所有实验统一使用由 corrected formatter 正式生成的 `CNYC.CLI`，避免误用旧的 93-column candidate。

## Gate B：恢复完整 WGEN seed pilot

在 Gate A 通过后，运行：

```text
101a
101b
102
103
104
105
```

验证：

```text
same seed reproducibility
different seed diversity
basic weather physical sanity
```

---

# 三、冻结天气输入

唯一允许用于 CLI 重新生成的输入：

```text
results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv
```

冻结 SHA256：

```text
4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34
```

必须重新核验：

```text
hash
row count = 3653
date range = 2004-01-01 ~ 2013-12-31
```

禁止使用：

```text
2014–2023
```

进行任何 fitting 或 parameter regeneration。

---

# 四、正式 corrected CLI 输出位置

不要覆盖历史错误文件：

```text
results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI
```

该文件必须作为历史错误证据保留。

正式 corrected CLI 请输出到新的稳定位置，例如：

```text
results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI
```

并生成：

```text
final_cli_metadata.json
final_cli_generation_log.txt
final_cli_schema_check.json
```

---

# 五、正式 corrected CLI 的生成要求

使用当前已修复 formatter：

```text
I6 + 14*(1X,F5.0)
```

重新从 frozen 2004–2013 weather 生成。

必须确认：

```text
parameter values identical to parse-fixed candidate
format corrected
12 monthly WGEN rows
90 chars each
15 total columns including MTH
no NaN
no Inf
no malformed rows
```

必须对比：

```text
formal corrected CLI
vs
003_06_05_01 parsefix candidate
```

要求：

```text
all WGEN parameter values identical
serialization equivalent
```

如果两者完全字节一致：

记录：

```text
formal_cli_matches_parsefix_candidate = YES
```

如果 hash 不一致但数值完全一致：

必须说明差异原因，例如：

```text
header
metadata
line endings
non-WGEN formatting
```

不得静默接受。

---

# 六、正式 corrected CLI 冻结

Gate A 通过后：

记录新的：

```text
official_corrected_cli_path
official_corrected_cli_sha256
```

并将其状态设为：

```text
CORRECTED_CLI_READY_FOR_SEED_PILOT
```

后续 101a/101b/102–105 全部只能使用这一正式 corrected CLI。

禁止混用：

```text
003_06_04 old CLI
003_06_05_01 temporary runtime copy
```

---

# 七、seed pilot 设计

运行：

```text
weather_seed = 101  # run A
weather_seed = 101  # run B
weather_seed = 102
weather_seed = 103
weather_seed = 104
weather_seed = 105
```

本轮：

```text
ppo_seed = NOT_APPLICABLE
```

不得运行 PPO training。

---

# 八、固定所有非 weather_seed 条件

必须固定：

```text
site = YC
crop
soil
management
FileX
cultivar
simulation start
simulation length
random_weather=True
WTHER=W
WSTA=CNYC0801
CNYC.CLI
runtime
wrapper config
```

除了：

```text
weather_seed
```

之外不得改变其他输入。

---

# 九、weather seed 与 PPO seed 分离

本轮必须继续明确：

```text
weather_seed != ppo_seed
```

当前：

```text
ppo_seed = NOT_APPLICABLE
```

检查代码接口是否存在：

```text
SEED_COUPLING_FOUND
```

如果 `weather_seed` 仍被某个通用 `seed` 参数隐式覆盖：

必须记录并停止进入未来 PPO 阶段。

但不应影响本轮 isolated WGEN seed pilot，只要可以明确控制 WGEN 的 `rseed1`。

---

# 十、same-seed reproducibility

对：

```text
101a
101b
```

必须捕获逐日：

```text
RAIN
SRAD
TMAX
TMIN
```

及：

```text
DATE / DOY
```

要求：

```text
same seed
→ same daily sequence
```

生成：

```text
results/yc_wgen_cli_pilot/003_06_05_02/reproducibility_check.json
```

至少记录：

```text
seed
run_a_hash
run_b_hash
identical
row_count
variables_compared
first_difference_if_any
```

如果不一致：

状态：

```text
FAIL_SEED_REPRODUCIBILITY
```

并停止后续正式 PPO 设计。

---

# 十一、different-seed diversity

比较：

```text
101
102
103
104
105
```

生成：

```text
seed_diversity_check.json
```

至少报告：

```text
weather sequence hash
pairwise identical/not-identical
RAIN differing rows
SRAD differing rows
TMAX differing rows
TMIN differing rows
```

最低要求：

```text
not all seeds identical
```

理想要求：

```text
all 5 seeds distinct
```

如果多个不同 seed 产生完全相同天气：

标记：

```text
FAIL_SEED_DIVERSITY
```

并追踪：

```text
rseed1 propagation
reset behavior
DSSAT internal seed consumption
```

---

# 十二、天气输出保存

统一保存：

```text
results/yc_wgen_cli_pilot/003_06_05_02/generated_weather/
```

建议：

```text
seed_101_run_a.csv
seed_101_run_b.csv
seed_102.csv
seed_103.csv
seed_104.csv
seed_105.csv
```

字段至少：

```text
DATE
DOY
RAIN
SRAD
TMAX
TMIN
```

如果 runtime 提供其他字段，可保留，但不要影响核心比较。

---

# 十三、基础物理 QC

每个 seed 检查：

```text
RAIN >= 0
SRAD >= 0
TMAX >= TMIN
finite values
no malformed dates
no duplicate daily rows
```

并计算：

```text
rain_total
rainy_days
max_daily_rain
longest_dry_spell
mean_TMAX
mean_TMIN
mean_SRAD
min_TMIN
max_TMAX
max_SRAD
```

如果运行窗口是 120 天：

必须称为：

```text
simulation-period statistics
```

不得称为：

```text
annual statistics
```

---

# 十四、初步气候 sanity check

只使用冻结训练天气：

```text
2004–2013
```

作为参考范围。

本轮只做初步 sanity：

```text
生成天气是否出现明显物理异常？
是否全程无雨？
是否所有 seed 一模一样？
是否出现异常极端值？
```

不要为了更贴近历史而调整 CLI。

---

# 十五、本轮不处理 wet-day threshold sensitivity

当前正式 corrected CLI 继续使用：

```text
RAIN > 0.0 mm
```

作为拟合定义。

本轮不生成：

```text
0.254 mm threshold alternative CLI
```

也不做敏感性分析。

只有完整 seed pilot 成功后，才单独评估：

```text
RAIN > 0
vs
RAIN >= 0.254 mm
```

是否对生成气候产生实质影响。

---

# 十六、运行时错误处理

如果任意 seed 再出现：

```text
WGENIN
CLI lookup
parse error
seed interface error
```

必须记录具体错误。

禁止：

```text
自动归因于版本不同
```

如果 101a 成功但后续 seed 失败：

报告：

```text
seed-specific runtime failure
```

并保留所有成功与失败证据。

---

# 十七、成功判据

只有以下全部满足才标记：

```text
WGEN_SEED_PILOT_PASS
```

要求：

1. 正式 corrected CLI 成功重生成；
2. corrected CLI static schema PASS；
3. 101a runtime PASS；
4. 101b runtime PASS；
5. 101a == 101b daily weather；
6. 102–105 runtime PASS；
7. different seeds 产生不同天气；
8. 基础物理 QC PASS；
9. 无 validation leakage；
10. weather_seed 与 ppo_seed 概念分离。

---

# 十八、如果 Gate A 失败

如果正式 corrected CLI 与 parsefix candidate 不一致且原因不明：

停止 seed pilot。

状态：

```text
FAIL_FORMAL_CLI_REGENERATION
```

必须解释：

```text
difference
affected lines
affected values
formatter state
```

不得继续拿临时 parsefix 文件跑完整实验。

---

# 十九、如果 seed pilot 成功

下一阶段为：

```text
003_06_06_yc_wgen_weather_qc_and_dssat_smoke
```

重点：

```text
1. 更完整的随机天气统计 QC
2. 与 2004–2013 train climate 的分布比较
3. single-season DSSAT crop-output smoke
4. 检查产量、作物生长、水氮状态是否合理
```

仍然不要直接进入正式 PPO 大训练。

---

# 二十、允许的代码修改

允许：

```text
scripts/build_dssat_cli.py
scripts/run_yc_wgen_seed_pilot.py
seed pilot analysis code
tests
```

但仅限：

```text
formal corrected CLI regeneration
seed pilot execution
weather capture
QC / analysis
```

禁止修改：

```text
PPO reward
action space
observation
baseline definitions
LC/SY/HL/FQ
weather fitting formulas
wet-day threshold
```

---

# 二十一、测试要求

至少运行：

```text
tests/test_build_dssat_cli.py
tests/test_yc_wgen_seed_pilot.py
```

以及本轮新增测试。

必须报告：

```text
test count
pass/fail
```

如有 regression：

不得继续完整 seed pilot，先修复。

---

# 二十二、结果目录

全部本轮结果：

```text
results/yc_wgen_cli_pilot/003_06_05_02/
```

建议：

```text
final/
generated_weather/
runtime/
validation/
reproducibility_check.json
seed_diversity_check.json
weather_summary_by_seed.csv
physical_sanity_check.json
experiment_log.md
```

---

# 二十三、中文 Markdown 报告

生成：

```text
docs/yc_corrected_cli_and_full_seed_pilot.md
```

至少包括：

```text
1. task scope
2. formal corrected CLI regeneration
3. corrected CLI provenance
4. formatter regression status
5. seed design
6. same-seed reproducibility
7. different-seed diversity
8. physical weather QC
9. runtime compatibility
10. remaining issues
11. readiness decision
12. next stage
13. files changed
14. git status
```

正文使用中文。

---

# 二十四、PPT 任务记录

生成：

```text
docs/yc_corrected_cli_and_full_seed_pilot.pptx
```

建议 6–8 页：

```text
1. 从 parse fix 到正式 corrected CLI
2. corrected CLI provenance
3. seed pilot design
4. 101 reproducibility
5. 102–105 diversity
6. physical QC
7. final pilot decision
8. next stage
```

---

# 二十五、Codex prompt 保存

保存：

```text
prompts/003_06_05_02_finalize_cli_and_resume_seed_pilot.md
```

---

# 二十六、Git 与 GitHub

完成后：

1. `git diff`
2. 只 stage 本任务文件
3. local commit
4. commit message：

```text
test: finalize YC CLI and complete WGEN seed pilot
```

5. 不清理无关工作区改动
6. 未经用户明确批准：

```text
git push = NO
```

记录：

```text
GitHub backup pending explicit user approval
```

---

# 二十七、最终终端摘要

必须直接返回：

```text
=== YC CORRECTED CLI + FULL WGEN SEED PILOT SUMMARY ===

frozen_weather_sha256:
frozen_weather_hash_verified:

old_broken_cli_preserved:
old_broken_cli_sha256:

formal_corrected_cli:
formal_corrected_cli_sha256:
matches_parsefix_candidate:
formal_cli_status:

runtime_version:

seeds_run:
repeated_seed:

same_seed_reproducible:
seed_101_run_a_hash:
seed_101_run_b_hash:

different_seeds_distinct:
distinct_weather_sequences:

weather_seed_interface:
seed_coupling_status:

physical_sanity_status:
weather_summary_path:

runtime_errors:
runtime_warnings:

wgen_seed_pilot_status:

validation_data_used_for_fitting: NO
weather_candidate_modified: NO
wet_day_definition_changed: NO
ppo_training_run: NO
other_sites_modified: NO

recommended_next_step:

report_md:
report_pptx:
results_directory:

git_commit:
git_push: NO
github_backup_status:
```

成功时：

```text
formal_cli_status: CORRECTED_CLI_READY
same_seed_reproducible: YES
different_seeds_distinct: YES
physical_sanity_status: PASS
wgen_seed_pilot_status: WGEN_SEED_PILOT_PASS
recommended_next_step: 003_06_06_yc_wgen_weather_qc_and_dssat_smoke
```

---

# 二十八、完成标准

本任务完成必须满足：

1. 旧错误 CLI 保留；
2. frozen weather hash 复核；
3. 使用 corrected formatter 正式重生成 CLI；
4. 正式 CLI 与 parsefix candidate 数值一致；
5. 正式 CLI schema PASS；
6. 101a/101b/102–105 全部运行；
7. same-seed reproducibility 检查完成；
8. different-seed diversity 检查完成；
9. 基础物理 QC 完成；
10. weather_seed / ppo_seed 区分清楚；
11. 不使用 validation fitting；
12. 不修改 wet-day definition；
13. 不运行 PPO；
14. 不修改其他站点；
15. 生成中文 Markdown + PPT；
16. 保存完整结果；
17. local git commit；
18. 未经用户批准不执行 `git push`。
