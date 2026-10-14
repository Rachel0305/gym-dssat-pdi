# YC 外部天气增强 004_24A：拟合期方法冻结

## 结论

```yaml
fitting_only_data_gate: PASS
dap_alignment_status: NOT_VERIFIED_USE_CALENDAR_ALIGNMENT
augmentation_method: MULTIVARIATE_MOVING_BLOCK_BOOTSTRAP
method_freeze_status: FROZEN
observed_data_used_for_tuning: NO
candidate_count_frozen: 20
ppo_training_status: NOT_STARTED
next_step: 004_24B_GENERATE_AND_VALIDATE_20_CANDIDATES
```

本轮只使用 YC 2004–2013 fitting weather 冻结方法。没有读取 2014–2023 天气、004_19/004_20 的 observed 对比结果或任何 observed 汇总；没有运行 WGEN、DSSAT、Gym-DSSAT 或 PPO；没有生成正式 candidate weather。

## 数据门

唯一输入为 [yc_wgen_fitting_weather_2004_2013.csv](../results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv)，SHA256：`4b8ffe9e881d0a0743921b78b9c0e0ebfb1d2d645c5aa9737948b2b088ed7b34`。文件包含完整的 2004–2013 日序列，共 3,653 行；DATE/YEAR/DOY 一致，逐年日期连续且无缺日、重复日或非有限值。RAIN、SRAD 非负，TMAX 不低于 TMIN。

`DATE/YEAR/DOY` 是全年日历日期，不是可验证的 crop DAP。为避免把全年日序列误标为作物生育进程，本方案不使用 DAP；按完整目标日历年生成序列，以日历 phase 匹配 donor blocks。CSV 本身没有单位列；变量单位标签沿用冻结 fitting-weather / DSSAT WTH 约定（RAIN 为 mm/day，SRAD 为 MJ m⁻² day⁻¹，TMAX/TMIN 为 °C），本轮不做任何单位换算。该标注不是从 CSV 内嵌元数据独立验证所得，004_24B 需确认序列化接口不改变单位。

数据 QA、输入 hash 及只读边界记录见 `results/yc_external_weather_augmentation_004_24A/fitting_source_qa.csv` 和 `analysis_manifest.json`。

## 拟合期诊断

### 持续性与事件长度

湿日定义冻结为 `RAIN >= 0.1 mm`；干日为其补集。拟合期 pooled SRAD P5/P10 分别为 `2.46823896` 和 `4.5568 MJ m⁻² day⁻¹`，高温日定义为 `TMAX > 32.7°C`（fitting P90），凉日定义为 `TMIN < -6.48°C`（fitting P10）。

2004–2013 干 spell 的 pooled P90 为 `18` 天（年等权 P90 为 `18.78` 天），最大为 `78` 天；湿 spell P90 为 `3` 天、最大 `6` 天；低 SRAD spell P90 为 `2` 天、最大 `5` 天；高温 spell 年等权 P90 为 `5.6` 天、最大 `17` 天；凉 spell 年等权 P90 为 `6.5` 天、最大 `18` 天。

TMAX/TMIN 在 lag 14 的 pooled ACF 分别约 `0.844/0.872`，SRAD lag 14 约 `0.399`。因此原示例上限 14 天不足以保留温度持续性；扩展候选长度至 21–56 天进行 fitting-only 对照。完整 ACF 与 spell 表在 `extended_block_diagnostics_v3/`。

### Block length 选择

比较候选：`3, 5, 7, 10, 14, 21, 28, 35, 42, 56` 天。冻结规则为：选择最短 block，使其长度不小于拟合期各年 dry-spell P90 的年等权 P90，且其理论块内 ACF 保留率 `max(0, 1 - lag/L)` 在 lag 7 对 SRAD、TMAX、TMIN 均至少为 75%。

- 21 天满足干 spell 长度条件，但 lag-7 保留率只有 66.7%，不通过。
- 28 天 lag-7 保留率为 75%，且大于 18.78 天；它是最短通过候选。
- 在 28 天下，拟合期 dry spells 中约 `5.09%` 长于 block；wet、low-SRAD、hot、cool spell 均未超过 28 天。
- 35–56 天能进一步减少理论 ACF 损失，但会减少每年重组块数；按“满足结构门槛的最短块”规则不选更长值。

因此冻结 `block_length_days = 28`。这不是声称能无损保留 lag 10/14 的相关性；它们仍在 004_24B 报告，但不作为 block 长度通过门槛。

### Calendar phase 与 donor 可用性

`dap_alignment_status = NOT_VERIFIED_USE_CALENDAR_ALIGNMENT`。target 是 004_24B 生成前 manifest 中明确记录的完整日历年（1 月 1 日至 12 月 31 日，365/366 天）。这个目标年份只用于输出日期和闰年长度，不参与 donor 选择；未记录运行时上下文则停止生成。

每个 target block 与 donor block 的中点，在循环 365 日 calendar phase 上相差不得超过 7 天。donor block 必须逐日连续且不能跨 source calendar-year 边界；可以跨月。闰年保留 2 月 29 日，phase 映射中 2 月 29 日处于 2 月 28 日与 3 月 1 日之间。最后不足 28 天的尾块，使用长度完全相等的 donor block，并应用同一中点 phase 规则。

按 365/366 日真实 target 分块及尾块长度核算，28 天 block 每个 target block 至少有 `80` 个合格 donor，median 为 `150`。v1 donor 诊断曾错误地把全年每个日期都按完整 28 日块计数；该结果保留但不用于决策。v3 已按 target block 位置和实际尾块长度重算，作为冻结依据。

## 004_24B 固定算法

1. 根据预先记录的 target runtime calendar year 建立完整 365/366 日目标日期序列。
2. 从 2004–2013 fitting 数据枚举长度为 28 日的连续 donor blocks；不允许 source-year 跨界。
3. 按目标/ donor block midpoint circular phase 距离 `<= 7 days` 筛选 donor。
4. 从合格 donor 中有放回均匀抽取，逐行整体搬运 `RAIN/SRAD/TMAX/TMIN`，并按 target 日期重新序列化；不得独立抽变量。
5. 每次抽取记录 candidate ID、target date、source year/date、block ID/start/end、seed。
6. 最后不足 28 日时抽取相同长度 donor tail；不平滑、不插值、不裁剪、不改原始值。
7. 生成固定 20 套。candidate master seed 为 `240424`，`candidate_i` 使用 `240424 + i`（i 从 1 开始），见 `candidate_seed_schedule.csv`。抽样有放回；任何 candidate QA 失败都保留失败证据，不换 seed 补抽。

## 冻结的 QA 门槛

### Candidate-level

- 365/366 日长度正确，日期连续且唯一；无缺失、NaN 或 sentinel。
- 全部行满足 `RAIN >= 0`、`SRAD >= 0`、`TMAX >= TMIN`。
- candidate 的每个四变量 daily tuple 必须精确对应 fitting 源文件的一行，来源可追溯率 100%。
- 所有已登记年度指标须在 fitting-year 3×IQR Tukey fence 内；数值见 `extended_block_diagnostics_v3/candidate_level_annual_metric_fences.csv`。
- 每个变量每套天气最多允许 1 个 seam 超过拟合期相邻日差分 P99；超过即 candidate FAIL。

### Pool-level

- 20 套的每个登记年度指标均值必须落入 `pool_qa_bootstrap_reference.csv` 的 2.5–97.5%区间。该区间以完整 fitting calendar year 为 resampling unit，等概率有放回抽取 20 个年份，5,000 次，固定 seed `20260927`；不把逐日值当独立样本。
- 低 SRAD 日比例必须落入拟合年 cluster-bootstrap 区间：低于 fitting P5 为 `0.04460736–0.05666526`，低于 fitting P10 为 `0.09225541–0.10872213`。同时报告每套的 min/P1/P5/P10、low-SRAD spell、最小滚动 3/5/7 日均值。
- 每个变量 seam 超阈值总数不超过全部 pool seams 的 2%。绝对变化 P99 阈值分别为：RAIN `58.09 mm`、SRAD `19.29667275`、TMAX `9.5058°C`、TMIN `9.0°C`。降雨 wet-state transition 作为描述指标报告。
- 不允许完全重复 candidate；若任意一对在对齐日期上至少 95% 的四变量 tuple 完全相同，则 pool FAIL。
- 10 个 fitting source years 都必须出现在 donor provenance 中，任一 source year 不得贡献 pool donor rows 的 20% 以上；同一 donor block 允许重复，但不得超过总抽取 block 数的 5%。
- 候选失败不得靠重抽、换 seed 或筛选“更像 observed”的子集补救；全 20 套共同决定 pilot 是否通过。

Annual metric fences 与 pool bootstrap reference 覆盖降雨总量、湿日、降雨强度、dry/wet persistence、Rx1/Rx5、SRAD 低尾、温度极值/热凉 spell、DTR、自相关和主要变量联合关系。诊断 CSV 保留每个量的 fitting-derived 数值。

## 004_24B 的 observed 外部验证边界

只有在 20 套候选、seed、生成 provenance、weather hashes 及内部 QA 全部冻结后，才可读取 2014–2023。届时仅做一次 descriptive external validation：比较 candidate、旧 WGEN 和 observed 的支持位置及低 SRAD 尾部等，不拟合或选择参数，不改变 block length/phase/seam/QA 阈值，不重抽候选，也不挑选子集。若外部结果不理想，作为方法限制记录，另开独立编号修订。

## 解释边界

Multivariate moving-block bootstrap 能保留并重组 2004–2013 中真实出现的日天气向量、block 内连续结构和变量联合关系，但不会产生超出 fitting sample 单日支持的新极端。它检验的是天气增强机制是否可追溯且避免旧 WGEN 丢失 fitting 低尾，不是保证覆盖 2014–2023 所有极端、证明 20 套足够 PPO，或预测 PPO 一定改善。

## 文件索引

- 方法冻结：`results/yc_external_weather_augmentation_004_24A/method_freeze.yaml`
- 机器判定：`results/yc_external_weather_augmentation_004_24A/decision.json`
- seed 清单：`results/yc_external_weather_augmentation_004_24A/candidate_seed_schedule.csv`
- 原始候选 3–14 日诊断：`results/yc_external_weather_augmentation_004_24A/`
- 修正并扩展至 56 日的 fitting-only 诊断：`results/yc_external_weather_augmentation_004_24A/extended_block_diagnostics_v3/`
- Runner：`src/run_004_24A_yc_external_weather_augmentation_fitting_only_method_freeze.py`

v1/v2 donor availability 中间诊断没有被删除；v3 按实际 target block 与 tail 长度重算并用于冻结。没有生成 synthetic weather，也没有运行 DSSAT/PPO。
