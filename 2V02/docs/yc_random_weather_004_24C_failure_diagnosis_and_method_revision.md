# 004_24C YC 外部天气增强失败诊断与 fitting-only 方法修订

## 1. 范围与数据隔离

- 本任务只读2004–2013 fitting weather、004_24A冻结诊断，以及004_24B的20套candidate、provenance、candidate hash和internal QA。没有读取Observed天气值，也没有使用004_24B observed-derived summary。
- `observed_data_used_for_method_revision = NO`。未运行WGEN、DSSAT、Gym-DSSAT或PPO。全部LOYO逐日序列均标记 `DEVELOPMENT_ONLY_NOT_FOR_PPO`。

## 2. 004_24B失败概况

- 20套候选中6套通过、14套失败；结构、物理、provenance、序列化为20/20通过；指标fence失败8套、seam失败12套；pool gate失败。候选没有重抽或删改。
- pool失败指标：seasonal_rain_total, rx5day, srad_min, srad_p10, max_low_srad_spell_days, tmax_mean, tmin_min, diurnal_range_mean, srad_tmax_spearman, acf_srad_lag7, acf_tmin_lag3。冻结区间原样保留，没有放宽门槛。

## 3. Seam归因

| 变量 | fitting相邻日P99阈值 | candidate block内部违规率 | candidate seam违规率 | 判断 |
|---|---:|---:|---:|---|
| RAIN | 58.0900 | 1.151% | 1.538% | 集中于seam |
| SRAD | 19.2967 | 1.094% | 3.077% | 集中于seam |
| TMAX | 9.5058 | 1.136% | 11.538% | 集中于seam |
| TMIN | 9.0000 | 0.767% | 5.769% | 集中于seam |

- 违规是否集中于block边界：`YES`。具体分变量计数、率和fitting delta分布见 `seam_decomposition_by_variable.csv`。

## 4. Metric-fence与降雨偏高

- pool均值越界项共11项，逐项候选值、冻结bootstrap区间及可检验归因列在 `metric_fence_failure_attribution.csv`。这些归因是候选机制，不作因果断言。
- 8/20候选至少有一项candidate-level fence失败；失败计数（候选可在多项重复计数）为：acf_tmin_lag1=1, acf_tmin_lag3=1, rx1day=2, srad_lt_fit_p05_fraction=2, srad_p05=5。
- 降雨分解：候选年总雨量均值/中位数为 **729.92/710.95 mm**，wet-day数均值 **59.95日**，湿日强度均值 **12.25 mm/day**；按fitting wet-day P95（45.48 mm）定义的重雨贡献均值为 **267.80 mm/年（总雨量34.45%）**。冻结区间显示wet-day数及wet-day强度总体仍在范围内，但年度总雨量高于上界。
- 年总雨量与wet-day数相关较弱（r=0.228），与湿日强度和重雨总量相关较强（r=0.901、0.900）。供体年2005贡献14.56% donor rows、19.83%总雨量、22.49%重雨；仍低于预注册的20% source-row上限，最大单个donor block复用占比仅1.07%。这支持“强降雨金额/抽样变异值得继续诊断”，不支持归因于wet day过多或单一block重复；均为描述性关联，不作因果断言。详见 `candidate_rainfall_decomposition.csv`、`donor_year_rain_contribution.csv`、`donor_block_reuse_and_rain.csv`。

## 5. Low-SRAD分解

- 低SRAD日均来自fitting donor tuple：`True`；候选池低于fitting P5阈值 `2.46823896` 的日数为405。
- candidate低SRAD spell数=301；跨越seam的spell=1；触及seam的spell=57；最长spell=5日。低尾确实保留，但最小值、P10、spell长度与lag-7 ACF等pool统计仍有冻结gate失败。

## 6. LOYO设计与方法

- 十折LOYO：每次留出一个2004–2013完整calendar year，donor仅用其余9年；每折每个方法20个配对development seeds，共200个realization/method。Leap year保留366日，沿用冻结365日phase映射。
- M0：原uniform 28日block、midpoint ±7日、有放回、原样拼接。M1：同长度/phase；按四变量delta的donor-fold fitting P95尺度和wet-state转移surprisal构成transition score，取phase-eligible中最低分20%，再均匀随机抽样。没有nearest-neighbor确定性选择、平滑或改值。
- 设计、种子、LOYO重复数及门槛在 `analysis_plan.json` 中先冻结；LOYO逐日天气和source provenance都标注development-only。

## 7. LOYO结果

- M0 TMAX/TMIN seam违规率：9.000% / 4.692%。
- M1 TMAX/TMIN seam违规率：0.423% / 0.077%。
- 四变量seam违规率（M0→M1）：RAIN 1.000%→0.385%；SRAD 1.385%→0.077%；TMAX 9.000%→0.423%；TMIN 4.692%→0.077%。M1四项均下降，且TMAX/TMIN降幅均超过50%，满足seam gate。
- 其余门槛：candidate-level metric fence为 141/9200 个检查失败；逐fold pool区间为 62/460 失败；persistence/dependence 非劣性为 5/21 项通过。annual metric shift、diversity、structural gates通过。
- M1各gate：`{"annual_metric_shift": true, "candidate_metric_fence": false, "diversity": true, "persistence_dependence": false, "pool_interval": false, "seam": true, "structural": true}`。逐metric LOYO paired error、bootstrap区间、各fold pool checks见 `development/`。

## 8. 方法选择与冻结

- `method_revision_status = NOT_RESOLVED`；selected method=`None`。
- `method_revision_freeze.yaml` 若为 `NOT_FROZEN`，仅记录被评估方案和保留的独立validation seed schedule，不授权生成下一轮正式candidate。没有满足所有门槛时，按任务规定停止，不继续搜索block length、phase window或donor balancing。

## 9. 限制与下一步

- LOYO仅有10个fitting年份；它检验拟合期内部的伪未见年份泛化，不代替独立气候时期validation。指标关联与候选机制分析均属描述性。
- 下一validation预留candidate数=20，seed见 `next_validation_seed_schedule.csv`；本任务未使用这些seed生成天气。
- `ppo_training_status = NOT_STARTED`；`next_step = STOP_AND_RECONSIDER_GENERATOR_FAMILY`。
