# 028_07 缺失 official expert 六季补齐

## 依据

028_06 已确认 40 个四基线单元中 34 个快照可复用，仅以下 6 个 official expert 缺失：YC2008、FQ2013、FQ2014、FQ2019、FQ2020、FQ2023。

## 严格范围

- 只运行上述 6 季。
- 不重跑 null、recorded、DSSAT auto。
- 不训练 RL。
- 不修改 reward、IC、品种参数、天气文件或原始输入。
- 使用 018_03 已冻结的华北黄淮/汾渭夏玉米 official extension expert 固定 DAP 方案。
- YC2008 使用 treatment 1；FQ 使用 2008 treatment 2 weather-year 派生链路。

## 执行顺序

1. 输入 smoke：只准备到新任务目录，核对站点、年份、experiment number、MZX/WTH 引用和管理链接，不启动 DSSAT。
2. smoke 通过后串行运行 6 季，每次只开一个环境，限制 BLAS/OpenMP 线程为 1。
3. 每季保存完整 DSSAT snapshot、逐日轨迹、实际管理事件、汇总指标和关键文件哈希。
4. 核对实际灌溉/施肥总量与冻结调度一致；失败时停止，不自动修改调度。

## 冻结调度

华北黄淮/汾渭夏玉米 DAP 7/30/45/60/80/100；使用 018_03 推文中值换算及单次灌溉不超过 50 mm 的拆分规则。

## 输出

- `benchmark_results/028_07_missing_official_expert_six_season_completion/`
- `docs/2026-07-18_028_07_missing_official_expert_six_season_completion.md`

## 判定

本任务只判定基线是否成功生成并与冻结调度一致，不判定 RL 成败。

