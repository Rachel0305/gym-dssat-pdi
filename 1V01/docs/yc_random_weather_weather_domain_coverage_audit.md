# YC WGEN 天气域覆盖审计

## 审计范围

本审计只读取已有文件：004_05 的随机天气训练 schedule、训练 episode summary、seed5 observed/held-out evaluation manifest 与 episode summary、已保存 WTH 快照，以及 003_06_06 的既有 WGEN QC。没有运行 PPO、DSSAT 或 WGEN，没有改动天气、模型或配置。

可复现脚本：`src/audit_yc_wgen_weather_domain_coverage.py`

机器可读结果：

- `results/yc_random_weather_ppo/weather_domain_coverage_audit/audit_summary.json`
- `results/yc_random_weather_ppo/weather_domain_coverage_audit/training_weather_seed_coverage.csv`
- `results/yc_random_weather_ppo/weather_domain_coverage_audit/training_seed_log_inventory.csv`
- `results/yc_random_weather_ppo/weather_domain_coverage_audit/weather_artifact_completeness.csv`

## 主要发现

### 已证实：schedule 覆盖完整且均匀

- 004_05 随机天气 schedule 有 100,000 行，包含 WGEN seed 1001–1080 共 80 个 seed。
- 每个 seed 在完整 schedule 中恰好出现 1,250 次。
- seed5 的已有训练记录包括 962 个 episode，实际覆盖全部 80 个 WGEN seed；每个 seed 出现 12 或 13 次。
- 004_05 seed3–7 可用的训练 episode logs 记录了相同的 170 个 runtime weather hashes，天气输入序列可配对复现。004_05 路径下 seed0–2 没有同格式 episode summary，因此该项按可用记录报告，不外推到缺失日志。
- seed5 held-out evaluation manifest 有 20 个 realization（1081–1100），episode summary 有 20 个不同 runtime weather hashes。

这些事实说明随机数 seed 的分配和运行记录覆盖齐全；它们本身不说明气候条件分布覆盖充分。

### 未证实：观测气候尾部是否落在训练天气支持范围内

逐日天气值的留存不足以计算完整覆盖关系：

- seed5 训练目录目前只有 10 个 WTH 快照，而不是 80 个训练 realization 的逐日天气序列。
- held-out 20 个 realization 均有 runtime weather hash，但目录只留存 1 个 WTH 快照；该快照的文件 SHA256 不匹配这 20 个 episode hashes 中的任何一个。
- seed5 observed weather 有 2014–2023 共 10 个逐日 WTH 文件。
- 现有 003_06_06 WGEN QC 是 `PASS_WITH_NOTES`，但范围为 WGEN seeds 101–105 的既有 pilot 和 2004–2013 训练气象窗口，不是 004_05 的 1001–1100 realization 集合，不能用它代替本审计的天气池覆盖统计。
- 训练 episode summary 保存天气 seed 与 runtime weather SHA256，没有保存对应 80 个 realization 的逐日 SRAD/TMAX/TMIN/RAIN 数列。因此不能从哈希、seed 编号或 crop outcome 推断降雨尾部、干旱持续期、热旱复合或阶段性覆盖。

## 判定

```text
weather_seed_schedule_coverage = COMPLETE_AND_BALANCED
weather_series_archive_coverage = INSUFFICIENT
observed_vs_training_climate_support = INSUFFICIENT
weather_pool_expansion_decision = DEFER
```

当前不能在以下选项中作有证据支持的选择：不扩天气、随机扩到 160、定向补极端天气、或修改 WGEN 分布。特别是，不能把“seed 1001–1080 均匀出现”当成“关键气候尾部已被覆盖”。

## 下一步所需证据

若要完成 coverage audit，需要恢复或单独物化 004_05 已使用训练 seed 1001–1080 与 held-out seed 1081–1100 的逐日天气文件，并保留 frozen CLI/参数哈希、seed、crop-year context、WTH SHA256 的对应关系。该步骤应保持 held-out 集不参与拟合或定向补样；可只做天气生成与统计，不需要运行 PPO 或 DSSAT。之后按整季、DAP 0–30/31–60/61–90/>90 或实际 phenological stage 计算降雨、干旱段、TMAX/TMIN、SRAD 及复合极端，再检查每个观测年份在训练分布中的位置。

在这些天气序列可核验之前，seed5 可继续作为当前阶段候选冻结；不建议据此直接启动 80→160 训练。
