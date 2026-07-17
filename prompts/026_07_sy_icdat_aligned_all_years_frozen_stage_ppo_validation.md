# 026_07 SY ICDAT 年份对齐后的全权威年份冻结阶段型 PPO 验证

## 1. 背景与授权

026_06 审计确认，当前权威 `CNSY1201.MZX` 中只有 2012、2014、2015 三个正式 treatment。原文件的 IC=1 行为 `ICDAT=14099`，因此：

- SY2012：`SDATE=12092, ICDAT=14099, PDATE=12119`，年份不一致；
- SY2014：`SDATE=14091, ICDAT=14099, PDATE=14111`，日期链有效；
- SY2015：`SDATE=15091, ICDAT=14099, PDATE=15108`，DSSAT 报 `Y4KDOY` 并停止。

用户已明确批准进行最小日期对齐。本任务只允许在新建运行目录的 MZX 副本中，将所选 treatment 指向的 IC 行的 `ICDAT` 年份改为目标年，同时保持年内日序 099 不变：

- 2012：`14099 -> 12099`；
- 2014：保持 `14099`；
- 2015：`14099 -> 15099`。

原始 MZX、IC 指针、IC 分层剖面、SDATE、PDATE、水氮初值、品种参数和天气文件均不得修改。

## 2. 科学问题

固定 SY2014 已选出的三个 MaskablePPO checkpoint，在完成上述唯一输入溯源修正后，验证它们能否不经训练地迁移到 SY2012、SY2014、SY2015，并相对各年 DSSAT auto 与官方推广 expert 同时达到：

- 产量不低于两者较高值；
- WP_ET 不低于两者较高值；
- PFP_N 不低于两者中正施氮基线的较高值。

recorded 单独报告，不纳入上述主判据。

## 3. 冻结内容

- seed0：`benchmark_results/026_03/checkpoint_000120.zip`；
- seed1：`benchmark_results/026_02/checkpoint_000060.zip`；
- seed2：`benchmark_results/026_04/checkpoint_000240.zip`；
- 25 维 scaler、六个阶段点、九动作、预算 mask、DAP>=90 禁氮规则全部冻结；
- 训练步数严格为 0；
- 不根据目标年份结果重新选 checkpoint；
- 不修改 reward、PPO 超参数、动作空间、IC 值或 DSSAT 原始输入。

## 4. 输入修正的硬约束

每次准备运行目录后必须：

1. 读取运行副本中所选 IC 行；
2. 只替换该行的五位 ICDAT；
3. 验证恰好替换一行，且 IC 指针和其余字段逐字不变；
4. 保存原始权威 MZX SHA256、运行副本修改前/后 SHA256、修改前/后 IC 行；
5. expert/PPO 对管理段进行二次改写后，再验证 ICDAT 仍为目标值；
6. 任务结束后重新核对原始权威 MZX SHA256 未变化。

对于 expert/PPO 的外部动作场景，运行副本中的 reported irrigation/fertilizer application 表必须整体重绘为“仅目标 treatment 一条安全零值占位行”。这是外部动作环境既有的管理表清理步骤，不改变 IC、SDATE/PDATE 或实际执行的 expert/PPO 动作。不得只清理目标 treatment 而留下其他 treatment 的 `IROP=-99` 占位行；026_07 第一次正式尝试已证明 DSSAT 会解析该无效非目标行并报 `IPIRR`。第一次失败目录必须单独保留。

任何验证失败立即进入 C 分支，不得静默回退到旧 ICDAT，也不得改用 IC=0。

## 5. 执行顺序

1. 新目录中只运行 SY2015 null smoke；
2. smoke 必须确认 DSSAT 正常到季末、产量有限、Summary.OUT 可匹配、ICDAT 已从 14099 改为 15099、原始 MZX 哈希未变；
3. smoke 通过后，按 2012、2014、2015 串行执行；
4. 每年均重新运行 null、recorded、DSSAT auto、official extension expert 四基线；
5. 每年依次评估三个冻结 PPO，CPU-only，不并行；
6. 保存逐年和总汇 CSV/JSON，不复用 026_05 的旧 2012 数值作为正式修正后证据。

## 6. 判定

- 每年 3 个模型中至少 2 个通过本地主判据，记该年通过；
- 三年全部通过且所有工程检查通过：`A_SY_all_years_transfer_after_icdat_alignment`；
- 工程检查通过但至少一年不足 2/3：`B_partial_year_transfer_after_icdat_alignment`；
- 日期、哈希、基线、Summary、mask、数值或执行失败：`C_input_or_execution_blocked`。

无论分支如何，不得把结果写成“超过所有基线”；recorded 必须单列。

## 7. 输出与停止规则

- smoke：`benchmark_results/026_07_smoke_2015_aligned_null/`；
- 第一次正式失败证据：`benchmark_results/026_07/`；
- 修正管理表渲染后的正式结果：`benchmark_results/026_07_attempt2/`；
- 记录：`docs/2026-07-17_026_07_sy_icdat_aligned_all_years_frozen_stage_ppo_validation.md`。

禁止覆盖 026_05/026_06，禁止启动其他站点，直到 026_07 完整记录、失败信息和 Git 本地提交完成。
