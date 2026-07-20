# 026_08 其余四站点冻结阶段型 PPO 验证就绪审计

## 1. 目的

在 SY 026_07 完成后，为 HLA、YC、FQ、LC 的冻结 SY2014 阶段型 PPO 跨站点验证建立可追溯输入清单。该任务只审计，不调用 DSSAT、不训练、不评估模型。

## 2. 预注册验证集

只使用已经进入五站点正式证据链、且存在已验证 adapter/input 的站点年份：

- HLA：2007、2010、2015、2016、2022（020_11）；
- YC：2014（020_12/020_13）；
- FQ：2016（020_12/020_13 的 shifted-input adapter）；
- LC：2010（017_11 fixed-input adapter 与 021_01）；
- SY：2012、2014、2015 已由 026_07 完成，只登记，不重跑。

其他只有天气文件、没有正式 treatment/IC 或没有完成输入修复的年份不自动纳入。

## 3. 必查内容

逐站点年份确认：

1. MZX/WTH/SOL/CUL 或已验证 prepared input 目录存在；
2. 既有四/五情景证据中至少存在 null、DSSAT auto、official expert；
3. 对应输入 adapter 明确：HLA prepared input、YC linked input、FQ shifted input、LC fixed input；
4. 三个冻结 PPO、SY2014 scaler 和 SHA256 完整；
5. 后续执行训练步数必须为 0；
6. 后续每个站点先跑一个 null/stage smoke，再串行评估三个模型；
7. reward 总值不是跨站点科学比较指标，主比较只使用当地基线的产量、WP_ET、PFP_N；
8. recorded 独立报告。

## 4. 停止条件

任一站点年份缺少输入、adapter 或基线来源，标记 blocked，不得用相邻年份天气/IC 临时拼接。审计不得启动正式模型评估。

## 5. 输出

- `benchmark_results/026_08/026_08_case_readiness.csv`；
- `benchmark_results/026_08/026_08_result.json`；
- `docs/2026-07-17_026_08_four_site_frozen_stage_ppo_readiness_audit.md`。
