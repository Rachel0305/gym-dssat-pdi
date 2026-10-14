# 004_24B YC 外部天气增强：20 套候选生成与验证

## 1. 冻结配置

- 004_24A frozen config：`results/yc_external_weather_augmentation_004_24A/method_freeze.yaml`，SHA256 `33ff590debe0f8019359b4c6284ef63949a0ee055fa09b90618fb68e9215544e`。
- fitting source：`results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv`，SHA256 `4b8ffe9e881d0a0743921b78b9c0e0ebfb1d2d645c5aa9737948b2b088ed7b34`，2004–2013 共3653日。
- 方法：多变量 moving-block bootstrap，完整 `[RAIN, SRAD, TMAX, TMIN]` 向量；块长28日；calendar midpoint ±7日；源块不跨年；尾块按剩余日数；原样拼接，不平滑、不插值、不裁剪。
- 目标日期上下文：复用004_23B已核验的YC 2004 measured-weather FileX（CNYC0401，SDATE 04165，PDATE 04169，WTHER=M）；完整目标日历年为2004闰年。上下文只控制日期/闰年及代表候选I/O，不参与donor选择。
- Seed：240425–240444；候选数固定20；Python `random.Random(seed)`按时间顺序逐块等概率有放回抽样，未重抽。

## 2. 生成

- 实际生成并保留 `20/20` 套；没有替补或删除。逐日源年、源日期、块边界、目标日期和选择信息见 `candidates/candidate_NNN/provenance.csv`。
- 生成值保存在 `weather.csv`；DSSAT 文件保存在 `weather.WTH`；逐套 SHA256 在 `manifests/candidate_hash_manifest.csv`。

## 3. 单位与序列化

- 004_24A冻结的变量单位沿用fitting/WTH约定；ChinaFLUX降雨按日累积mm，SRAD按已记录的0.0018积分转换为MJ m⁻² d⁻¹，candidate阶段无二次换算；TMAX/TMIN保持°C。
- 004_23B输入WTH曾byte-identical进入runtime，字段无scale变换；candidate WTH按YC模板列序SRAD/TMAX/TMIN/RAIN、一位小数序列化，CSV↔WTH最大误差和容差见各candidate `generation_metadata.json`。

## 4. Candidate 与 pool 内部 QA

- 内部QA：`FAIL`；candidate PASS/FAIL `6/14`；pool `FAIL`。
- Candidate gate拆分：结构/物理/provenance/序列化全通过=`True`；指标fence失败 `8/20`；seam失败 `12/20`；overall失败14/20。
- 候选年度指标3×IQR、20套均值cluster-bootstrap、低SRAD P5/P10比例、seam、来源年覆盖、重复块和候选重复均严格使用004_24A冻结阈值；结果分别在 `internal_qa/` 下。失败candidate仍完整保留。
- pool冻结指标未通过项：seasonal_rain_total, rx5day, srad_min, srad_p10, max_low_srad_spell_days, tmax_mean, tmin_min, diurnal_range_mean, srad_tmax_spearman, acf_srad_lag7, acf_tmin_lag3；seam检查 `seam_pool_2pct_pass=False`，违规比例 `{'RAIN': 0.015384615384615385, 'SRAD': 0.03076923076923077, 'TMAX': 0.11538461538461539, 'TMIN': 0.057692307692307696}`。候选重复数 `0`。
- `internal_weather_qa_decision.json` 在读取Observed之前写入并冻结；不根据Observed结果改候选或阈值。

## 5. candidate_001 runtime I/O

- status：`PASS`；runtime复制WTH byte-identical：`True`；random_weather=False；WTHER=M。
- 固定映射按004_23B marker证据使用 source date = runtime YRDOY−1；匹配日数 `102`，重复runtime日期 `0`，逐日字段容差 `1e-5`。crop过程输出均为 `NON_SCIENTIFIC_RUNTIME_SIDE_EFFECT`。

## 6. Observed 描述性外部 QA

- QA解锁：`True`；Observed 2014–2023仅在内部QA、candidate hashes及candidate_001 runtime I/O证据写入后读取。Observed没有用于调参、筛选、替补或重抽。
- Observed日数 `3652`，10个年份的WTH哈希及来源见 `external_qa_2014_2023/observed_source_manifest.csv`。逐日分布、逐年天气指标和support对照见对应CSV。
- 新候选pool是否保留fitting日尺度支持，按每变量年际范围在 `realization_support_comparison.csv` 中描述；它不要求20套覆盖所有Observed极端。旧WGEN为episode-season capture，本报告不将其与全年candidate作时长不匹配的直接池化；旧WGEN低SRAD缺口引用004_20正式诊断。

### 描述性分布摘录（realization中位数）

| 指标 | fitting 2004–2013 | 新候选20套 | Observed 2014–2023 |
|---|---:|---:|---:|
| 年总降雨 (mm) | 636.2 | 710.9 | 535.1 |
| wet-day fraction | 0.162 | 0.161 | 0.119 |
| 最大连续无雨日 | 43.5 | 43.0 | 44.5 |
| SRAD season minimum | 0.155 | 0.052 | 0.650 |
| SRAD season P5 | 2.606 | 2.357 | 3.490 |
| mean Tmax (°C) | 19.411 | 19.221 | 20.234 |
| mean Tmin (°C) | 8.299 | 8.232 | 8.732 |
| Spearman SRAD–RAIN | -0.225 | -0.238 | -0.241 |
| Spearman SRAD–Tmax | 0.635 | 0.642 | 0.690 |
| RAIN lag-1 ACF | 0.100 | 0.124 | 0.094 |
| wet/dry lag-1 ACF | 0.274 | 0.257 | 0.133 |
| SRAD lag-1 ACF | 0.595 | 0.580 | 0.648 |

- 20候选池出现 fitting 低SRAD支持：逐日池 P5=`2.315`，fitting逐日 P5=`2.468` MJ m⁻² d⁻¹；候选 realization 的季节SRAD P5中位数=`2.357`，fitting=`2.606`。结合004_20中旧WGEN作物季在fitting P5以下为0日，这说明低尾在新候选中重新出现；但其频率/持续性仍须服从冻结QA，且内部pool gate当前失败，不能据此判方法通过。
- 全年候选与fitting/Observed的总体分布并非处处重合：candidate年降雨中位数高于两参照，SRAD–Tmax关联中位数也高于fitting与Observed；这些为描述性差异，不代表显著性或因果。详细范围及所有固定依赖/持久性指标见 `metrics_by_realization.csv` 和 `realization_support_comparison.csv`。

## 7. 限制

- moving-block bootstrap不能生成超出2004–2013 donor日支持的新单日四变量组合；统计相似不等于天气因果，也不证明20套足够PPO训练。
- 工程尝试记录：首次预检因SHA文本大小写比较误报停止，改为大小写无关核验后通过；runtime attempt_001因复用目录中已有输入被保护逻辑拒绝，attempt_002因缺少同目录CUL/SOL未启动，attempt_003完成唯一一次episode并通过。各次原始stderr与attempt记录均保留在 `runtime_io_candidate_001/`。
- observed汇总首次解锁后处理因runner漏接candidate编号触发 `KeyError`；异常证据保存在 `logs/external_attempt_001_failure.json`。只修复行标签join后，对同一20套固定candidate和同一Observed源重新计算描述统计，没有重生成、重抽、替换或修改任何阈值。
- 代表candidate只执行了一次no-op runtime I/O smoke；其crop/yield/stress输出不作为科学结果。未训练PPO、未运行WGEN。

## 8. 最终判定

- `augmentation_pilot_status = FAIL`
- `weather_generation_method_status = NOT_VALIDATED`
- `next_step = STOP_AND_REVISE_METHOD_IN_NEW_TASK`
- PPO training：`NOT_STARTED`；git commit/push：`NOT_PERFORMED`。
