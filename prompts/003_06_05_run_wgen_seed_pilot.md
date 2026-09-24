# 003_06_05：运行 YC WGEN seed pilot 与 runtime compatibility smoke

## 一、任务定位

当前 YC 随机天气增强项目已经完成：

```text
2004–2013 frozen weather candidate ✅
        ↓
reproducible CNYC.CLI generator ✅
        ↓
CNYC.CLI candidate ✅
        ↓
WGEN seed pilot + runtime compatibility smoke   ← CURRENT TASK
        ↓
DSSAT single-season smoke
        ↓
formal PPO weather-augmentation experiment
```

当前已经生成：

```text
results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI
```

SHA256：

```text
5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0
```

CLI 状态：

```text
CANDIDATE_READY_FOR_WGEN_SMOKE
```

本任务的核心目标不是继续考古 DSSAT / WeatherMan 版本，而是：

> 让当前实际 Gym-DSSAT / DSSAT runtime 直接读取这个 `CNYC.CLI`，验证接口兼容性、随机天气生成能力和 seed 行为。

---

# 二、关键原则：兼容性以实际运行结果为准

本任务明确采用以下原则：

> 不要求生成 `CNYC.CLI` 时参考的 DSSAT/WGEN 文档版本与当前 runtime 的版本号机械一致。

只要：

```text
CLI format
field meaning
units
station lookup
WGEN runtime behavior
```

能够与当前实际 Gym-DSSAT / DSSAT runtime 对接，并通过 runtime smoke，则视为兼容。

只有当实际运行出现：

```text
parse error
missing field
schema mismatch
station lookup failure
WGEN parameter interpretation mismatch
```

等具体问题时，才回到对应字段或格式修复。

禁止仅因为：

```text
current DSSAT exact version unknown
```

或：

```text
reference source version differs
```

就中止本任务。

---

# 三、输入保护

本任务必须使用：

```text
results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI
```

并在运行前验证 SHA256：

```text
5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0
```

不得修改该文件。

如果需要调整：

```text
random_weather
auxiliary_file_paths
weather_seed
isolated test configuration
```

应在新的 pilot 配置、脚本或临时副本中完成。

禁止修改：

```text
results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv
```

禁止重新生成 `CNYC.CLI`，除非 runtime 给出明确、具体的兼容性错误，并且修复得到用户后续批准。

---

# 四、本轮不再处理 wet-day threshold 分歧

当前 `CNYC.CLI` 使用：

```text
RAIN > 0.0 mm
```

作为 wet-day 拟合定义。

已知资料文字中另存在：

```text
0.01 inch ≈ 0.254 mm
```

描述差异。

本轮：

> 不因为这一差异阻止 WGEN smoke。

先验证当前 candidate 是否能被 runtime 正确读取和使用。

wet-day threshold sensitivity analysis 留到 runtime smoke 通过后单独进行。

---

# 五、测试对象

当前只测试：

```text
YC
```

不得修改或测试：

```text
LC
SY
HL
FQ
```

不得运行正式 PPO training。

本轮测试对象是：

```text
CNYC.CLI
+
Gym-DSSAT / DSSAT runtime
+
internal WGEN
+
weather_seed
```

---

# 六、首先确认最小运行链

基于现有 wrapper 和 YC 配置，建立一个隔离的 WGEN pilot。

必须明确记录实际调用链，例如：

```text
pilot script
  -> Gym-DSSAT environment
  -> auxiliary_file_paths includes CNYC.CLI
  -> random_weather=True
  -> weather mode W
  -> weather_seed / rseed1
  -> DSSAT runtime
```

不要修改正式 PPO pipeline。

优先新增 isolated pilot script，例如：

```text
scripts/run_yc_wgen_seed_pilot.py
```

具体文件名可按仓库现有习惯调整。

---

# 七、确认 `CNYC.CLI` 被实际传入 runtime

必须提供运行证据证明：

```text
CNYC.CLI
```

确实进入 DSSAT 临时运行目录，而不是仅在 Python 配置中声明。

至少记录：

```text
source CLI path
source SHA256
runtime copied basename
runtime working directory if observable
station/file lookup evidence
```

如果 runtime 临时目录受现有边界限制无法读取：

使用当前项目允许的日志、wrapper instrumentation 或现有 debug 输出证明传递过程。

允许为 pilot 增加最小只读/日志型 instrumentation。

禁止改变 DSSAT 数值逻辑。

---

# 八、seed pilot 设计

正式使用以下：

```text
weather_seed = 101
weather_seed = 102
weather_seed = 103
weather_seed = 104
weather_seed = 105
```

为了检查 reproducibility：

至少选择一个 seed 重复运行两次。

推荐：

```text
101a
101b
102
103
104
105
```

如果运行成本很低，也可以所有 seed 各重复两次。

---

# 九、必须验证 same-seed reproducibility

对于相同 seed，例如：

```text
101a
101b
```

必须检查：

> 在固定所有其他条件时，WGEN 产生的天气序列是否逐日完全一致。

理想判据：

```text
same seed
→ identical generated daily weather
```

如果无法直接导出完整天气序列，则寻找最接近的可靠证据，例如：

```text
generated weather file
DSSAT weather trace
runtime daily weather variables
DSSAT input echo
PDI daily weather observations
```

优先验证：

```text
RAIN
SRAD
TMAX
TMIN
```

不要仅比较最终 yield 来判断天气是否相同。

如果当前 runtime 无法暴露完整天气序列：

明确记录：

```text
WEATHER_SEQUENCE_NOT_DIRECTLY_OBSERVABLE
```

并说明使用了什么替代证据。

---

# 十、必须验证 different-seed diversity

对于：

```text
101
102
103
104
105
```

检查：

```text
different seed
→ meaningfully different generated weather
```

至少比较：

```text
daily RAIN
daily SRAD
daily TMAX
daily TMIN
```

或 runtime 能提供的等效天气输出。

不能仅因为最终产量不同就推断天气不同。

---

# 十一、天气输出捕获

如果当前 Gym-DSSAT / DSSAT runtime 能保存 WGEN 生成的 daily weather：

将其保存到：

```text
results/yc_wgen_cli_pilot/003_06_05/generated_weather/
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

字段至少包括：

```text
DATE / DOY
RAIN
SRAD
TMAX
TMIN
```

如果 WGEN 不直接输出独立 weather file：

可从 PDI/runtime daily stream 捕获同等信息。

必须在报告中写清：

```text
weather_capture_method
```

---

# 十二、最小 runtime compatibility smoke

本轮首先回答：

```text
Can current DSSAT runtime read and use CNYC.CLI?
```

状态只允许使用类似：

```text
PASS
FAIL_CLI_PARSE
FAIL_CLI_LOOKUP
FAIL_REQUIRED_FIELD
FAIL_WGEN_RUNTIME
FAIL_SEED_INTERFACE
BLOCKED_BY_RUNTIME_ACCESS
OTHER_SPECIFIC_FAILURE
```

禁止使用泛化：

```text
version mismatch suspected
```

除非有明确 runtime error 支持。

---

# 十三、不要把 parser warning 自动当失败

如果 runtime 对：

```text
ANGA
ANGB
REFHT
WNDHT
GSST
GSDU
range QC fields
```

中的 `-99` 产生 warning：

先判断：

```text
warning only
```

还是：

```text
fatal / affects WGEN
```

只有实际阻止 WGEN 或改变必需参数解释时，才判定 candidate 不兼容。

禁止因为存在可选字段 warning 就自动重建 CLI。

---

# 十四、基础天气 QC

如果成功得到生成天气，分别对每个 seed 计算：

```text
annual / season rainfall
rainy days
maximum daily rainfall
longest dry spell
mean TMAX
mean TMIN
mean SRAD
minimum TMIN
maximum TMAX
maximum SRAD
```

如果生成窗口不是完整 calendar year，则明确使用：

```text
simulation-season statistics
```

不要错误称为 annual。

---

# 十五、与训练气候做初步 sanity check

仅使用冻结训练天气：

```text
2004–2013
```

作为参考。

比较生成天气是否出现明显不合理情况，例如：

```text
negative RAIN
negative SRAD
TMAX < TMIN
extreme impossible values
all-zero rainfall
identical weather across all different seeds
```

可以比较：

```text
monthly climatology
RAIN totals
rainy days
TMAX/TMIN range
SRAD range
```

但本轮只是：

```text
sanity check
```

不是正式统计验证，也不要为了“贴近历史”修改 CLI 参数。

---

# 十六、same-seed 判定

必须产生机器可读结果：

```text
results/yc_wgen_cli_pilot/003_06_05/reproducibility_check.json
```

至少记录：

```text
seed
run_a_hash
run_b_hash
identical
variables_compared
row_count
```

如果天气数据可完整捕获，建议对规范化后的 daily weather 表计算 SHA256。

要求：

```text
seed 101 run A == seed 101 run B
```

若不一致：

必须停止进入正式 PPO，并调查：

```text
seed propagation
hidden stochasticity
runtime reset behavior
```

---

# 十七、different-seed 判定

生成：

```text
results/yc_wgen_cli_pilot/003_06_05/seed_diversity_check.json
```

至少报告：

```text
pairwise identical/not-identical
weather hashes
RAIN differences
TMAX differences
TMIN differences
SRAD differences
```

最低要求：

```text
not all seeds identical
```

理想要求：

```text
all 5 seeds produce distinct weather sequences
```

---

# 十八、明确区分 seed 类型

本任务必须明确区分：

```text
weather_seed
```

与：

```text
ppo_seed
```

本轮不运行 PPO，因此：

```text
ppo_seed = NOT_APPLICABLE
```

不得重新使用同一个变量同时代表：

```text
agent stochasticity
weather stochasticity
```

如果现有接口仍将二者绑定：

必须记录为：

```text
SEED_COUPLING_FOUND
```

并在下一步正式 PPO 前修复。

---

# 十九、本轮允许的代码修改

允许：

```text
新增 isolated pilot script
新增 weather capture/logging
新增 seed configuration
新增结果分析脚本
新增 tests
```

允许对 wrapper 做：

```text
最小、可逆、仅为暴露 weather_seed 或捕获天气的改动
```

但必须：

1. 不改变默认 historical-weather 行为；
2. 不改变 reward；
3. 不改变 action space；
4. 不改变 observation semantics；
5. 不影响其他站点；
6. 有测试覆盖；
7. 明确记录 diff。

如果无需改 wrapper，则不要改。

---

# 二十、本轮禁止事项

禁止：

1. 正式 PPO training；
2. 修改 PPO reward；
3. 修改 action space；
4. 修改 observation space；
5. 修改 LC/SY/HL/FQ；
6. 修改 frozen weather candidate；
7. 重新拟合 `CNYC.CLI`；
8. 使用 2014–2023 validation weather 拟合参数；
9. 为了让生成天气“更好看”手调 WGEN 参数；
10. 因版本号不同自动判失败；
11. 因 optional metadata warning 自动判失败；
12. `git push`。

---

# 二十一、若 runtime smoke 失败

必须根据实际错误分类。

例如：

```text
CNYC.CLI not found
```

则调查：

```text
auxiliary_file_paths
basename copying
working directory
station lookup
```

如果：

```text
parse error on field X
```

则只修复：

```text
field X format/schema
```

如果：

```text
WGEN runtime rejects -99 optional metadata
```

则确认具体字段和 requiredness。

如果：

```text
weather seed ignored
```

则追踪：

```text
weather_seed -> rseed1 -> PDI -> DSSAT
```

禁止失败后直接回到“大版本不一致”解释。

---

# 二十二、成功 gate

只有以下全部满足，才允许进入下一阶段：

```text
1. current runtime successfully reads CNYC.CLI
2. internal WGEN starts without fatal error
3. same weather_seed reproducible
4. different weather_seed produces different weather
5. generated weather passes basic physical sanity checks
6. no validation leakage
7. weather_seed concept is distinguishable from ppo_seed
```

成功状态：

```text
WGEN_SEED_PILOT_PASS
```

---

# 二十三、下一阶段

如果本任务成功：

下一阶段应为：

```text
003_06_06_yc_wgen_weather_qc_and_dssat_smoke
```

重点：

```text
更完整的生成天气统计 QC
single-season DSSAT smoke
crop output sanity
```

仍然不要立刻进入大规模 PPO。

如果本轮已经不可避免地执行了一个最小 DSSAT season 才能触发 WGEN：

必须明确区分：

```text
runtime-trigger smoke
```

与：

```text
正式 DSSAT agronomic evaluation
```

后者留给下一阶段。

---

# 二十四、结果目录

所有结果保存到：

```text
results/yc_wgen_cli_pilot/003_06_05/
```

建议：

```text
results/yc_wgen_cli_pilot/003_06_05/
├── generated_weather/
├── runtime/
│   ├── cli_copy_evidence.txt
│   ├── runtime_log.txt
│   └── wgen_status.json
├── reproducibility_check.json
├── seed_diversity_check.json
├── weather_summary_by_seed.csv
├── physical_sanity_check.json
└── experiment_log.md
```

---

# 二十五、Markdown 记录

生成：

```text
docs/yc_wgen_seed_pilot.md
```

正文中文。

至少包括：

```text
1. task scope
2. candidate CLI provenance
3. runtime integration
4. seed design
5. weather capture method
6. same-seed reproducibility
7. different-seed diversity
8. physical sanity check
9. runtime warnings/errors
10. compatibility decision
11. remaining issues
12. next stage
13. files changed
14. git status
```

---

# 二十六、PPT 任务记录

生成中文 PPT：

```text
docs/yc_wgen_seed_pilot.pptx
```

建议 6–8 页：

```text
1. 当前项目位置
2. CNYC.CLI -> WGEN runtime integration
3. seed pilot design
4. same-seed reproducibility
5. different-seed diversity
6. weather sanity
7. runtime compatibility result
8. next step
```

如果 smoke 失败：

展示具体错误，不做模糊版本归因。

---

# 二十七、Codex prompt 记录

本任务 prompt 保存到：

```text
prompts/003_06_05_run_wgen_seed_pilot.md
```

---

# 二十八、Git 与 GitHub

完成后：

1. 检查 `git diff`；
2. 只 stage 本任务产生/明确修改的文件；
3. local commit；
4. commit message：

```text
test: run YC WGEN seed pilot
```

5. 记录 GitHub backup 状态；
6. **未经用户明确批准不得执行 `git push`**。

记录：

```text
GitHub backup pending explicit user approval
```

---

# 二十九、最终终端摘要

任务结束后必须直接返回：

```text
=== YC WGEN SEED PILOT SUMMARY ===

cli_path:
cli_sha256:
cli_hash_verified:

runtime_cli_integration:
runtime_compatibility_status:

weather_seed_interface:
ppo_seed_used: NO
seed_coupling_status:

seeds_tested:
repeated_seed:

weather_capture_method:

same_seed_reproducible:
same_seed_weather_hash:

different_seeds_distinct:
distinct_weather_sequences:

physical_sanity_status:

runtime_warnings:
runtime_errors:

wgen_seed_pilot_status:

validation_data_used_for_fitting: NO
weather_candidate_modified: NO
cli_modified: NO
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
wgen_seed_pilot_status: WGEN_SEED_PILOT_PASS
```

失败时必须使用具体原因，例如：

```text
FAIL_CLI_LOOKUP
FAIL_CLI_PARSE
FAIL_WGEN_RUNTIME
FAIL_SEED_REPRODUCIBILITY
FAIL_SEED_DIVERSITY
FAIL_PHYSICAL_SANITY
```

禁止只写：

```text
possible version mismatch
```

---

# 三十、完成标准

本任务只有在以下内容全部完成后才算完成：

1. `CNYC.CLI` hash 已验证；
2. candidate 已实际传入当前 runtime；
3. `random_weather=True` 已在隔离 pilot 中启用；
4. weather seed 101–105 已测试；
5. 至少一个 seed 重跑；
6. same-seed reproducibility 已检查；
7. different-seed diversity 已检查；
8. 生成天气做了基本物理 QC；
9. runtime compatibility 有明确 PASS/FAIL；
10. 没有因为版本号不同而提前中止；
11. 不修改 CLI candidate；
12. 不修改 frozen weather；
13. 不正式训练 PPO；
14. 不修改其他站点；
15. 生成中文 Markdown + PPT；
16. 保存完整 experiment log；
17. local git commit；
18. 未经用户批准不执行 `git push`。
