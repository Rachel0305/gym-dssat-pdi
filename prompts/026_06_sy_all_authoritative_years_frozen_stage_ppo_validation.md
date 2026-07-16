# 026_06 SY 全部权威年份冻结阶段型 PPO 验证

## 1. 目的

固定 026_02/026_03/026_04 在 SY2014 产生并按既定 reward 规则选出的三个 MaskablePPO checkpoint，检验它们在 SY 当前权威 MZX 中全部正式 treatment 年份上的表现。

本任务是固定模型的跨年验证，不在目标年份重新训练、重新选择 checkpoint 或调整管理方案。

## 2. 年份范围审计

当前 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY1201.MZX` 只包含：

- treatment 1：2012，IC=1；
- treatment 2：2014，IC=2；
- treatment 3：2015，IC=1。

因此本任务的“SY 全部可用年份”严格定义为 2012、2014、2015。仅有 WTH、没有权威 treatment/IC 的 2001–2011、2013、2016–2023 不运行，不得通过复制其他年份 IC 来补齐。

执行前必须额外审计每个 treatment 的 `SDATE <= ICDAT <= PDATE` 且三者年份一致。2026-07-17 第一次 SY2015 null smoke 发现：treatment 3 虽指向 IC=1，但该 IC 的 `ICDAT=14099`，而 2015 的 `SDATE=15091`、`PDATE=15108`，DSSAT 报 `Y4KDOY` 后停止。同期还发现 treatment 1（2012）使用同一 `ICDAT=14099`，虽然旧运行能够终止，但日期不属于2012季节。未经用户明确批准，不得在运行副本中把 ICDAT 静默改成12099/15099，也不得把2015改成IC=0。

## 3. 冻结模型

- seed0：`benchmark_results/026_03/checkpoint_000120.zip`；
- seed1：`benchmark_results/026_02/checkpoint_000060.zip`；
- seed2：`benchmark_results/026_04/checkpoint_000240.zip`。

运行前后核对模型 SHA256。模型参数、25维 scaler、六阶段点、九动作、预算 mask、DAP>=90 禁氮规则全部冻结。

## 4. 复用与新运行

- 2012：复用已通过一致性检查的 026_05 四基线和三个冻结模型结果，不重复 DSSAT 调用；
- 2014：重新按统一口径运行四基线和三个冻结模型，用作训练年参照；
- 2015：重新按统一口径运行四基线和三个冻结模型，作为新增独立年份验证。

全部新 DSSAT 调用串行执行，CPU-only，不并行，防止 OOM。

若日期审计不通过，2012的026_05只保留为“旧输入原样运行证据”，不升级为日期溯源完整的正式跨年证据；026_06判为输入阻塞并停止。

## 5. 每年四基线

1. null；
2. recorded/farmer practice；
3. DSSAT auto；
4. official extension expert fixed DAP。

记录产量、灌溉、管理事件施氮、Summary NICM、ETCP、WP_ET 和 PFP_N。MgmtEvent 与 Summary 的氮口径分别保存，不强制相等。

对于 expert/PPO 外部动作，只修改运行目录中的 MZX 副本：保持该 treatment 原 IC 指针不变，设置 MI=1/MF=1，清除该 treatment 的 recorded 管理事件并启用 `IRRIG=L/FERTI=L`。禁止修改权威原始 MZX/WTH/SOL/CUL。

## 6. 每年本地判据

以该年 DSSAT auto 与 official extension expert 为主要比较对象：

- yield ≥ 两者较高值；
- WP_ET ≥ 两者较高值；
- PFP_N ≥ 两者中所有正施氮基线的较高值；
- recorded 单独报告，不替代主要判据；
- PFP_N 对零施氮场景不定义，不得设为无穷大。

每年 2/3 以上冻结模型同时通过三项，记为该年 `local_primary_pass`。

## 7. 汇总判定

### A_SY_all_years_transfer

- 三个年份工程检查全部通过；
- 2012复用证据哈希/来源完整；
- 2014、2015均完成四基线和三模型评估；
- 2012、2014、2015每年均至少2/3模型通过本地主要判据。

### B_partial_year_transfer

工程检查通过，但至少一个年份少于2/3模型通过。

### C_input_or_execution_blocked

treatment/IC/WTH、模型哈希、四基线、Summary匹配、mask或数值检查失败。

## 8. 硬边界

- 训练步数严格为0；
- 不根据2012/2015结果重新选SY2014 checkpoint；
- 不改reward、IC、scaler、动作空间、阶段点和预算；
- 不新增seed；
- 不把只超过auto/expert写成超过recorded；
- 不启动其他站点，直到SY汇总和失败记录完成；
- 所有失败尝试单独保留，不覆盖。

## 9. 输出

输出目录：`benchmark_results/026_06/`

至少包括：

- treatment/年份/输入哈希审计；
- 每年四基线CSV；
- 每年三冻结模型summary与六阶段动作CSV；
- 三年总汇总CSV；
- 模型-年份通过矩阵；
- JSON判定；
- `docs/2026-07-17_026_06_sy_all_authoritative_years_frozen_stage_ppo_validation.md`。
