# 028_01 五站点全部年份范围与可行性证据注册表审计记录

## 结论先行

本任务完成了零训练、零 DSSAT 的站点—年份证据注册。当前**只有 SY** 做过最新阶段型 MaskablePPO 的固定权重同站跨年验证；HLA、YC、FQ、LC 均未完成。现有 SY 结果采用旧的 auto+expert local-primary 口径，不能当作已经同时超过 null、recorded、auto、official expert 的产量、WP_ET 与 PFP_N。

当前尚无任何站点—年份按 028 新定义完成“同阶段、同动作、同预算、同时严格超过四基线三指标”的可行性认证。因此本任务不允许直接进入全量调参或训练；下一步必须先完成 028_02 四基线复算和 deterministic strict-feasibility certification。

## 执行边界

- RL `learn()` 调用：0；
- DSSAT 调用：0；
- 原始输入修改：0；
- 旧结果覆盖：0；
- 只扫描 WTH 文件名、读取已有记录并哈希关键证据。

## 注册表规模

- 总行数：104；
- `A_authoritative_treatment`: 16
- `B_derived_weather_replay`: 13
- `C_weather_only`: 75
- 028 严格可行性已认证：0；
- 关键哈希源缺失：无。

Tier A 是原始/正式 treatment 或正式单年锚点；Tier B 是明确派生天气回放；Tier C 只有 WTH，不得因为天气文件存在就视为正式训练年份。

## 当前最新 PPO 跨年证据

| 站点 | 训练锚点 | 最新 PPO 同年状态 | 固定权重同站跨年 | 028 严格结论 |
|---|---:|---|---|---|
| SY | 2014 | 三 seed | 已完成 2012/2014/2015，旧 local-primary 为 3/3、3/3、2/3 | 未按四基线三指标严格认证；2012 三模型产量均低于 recorded |
| HLA | 2010 | 旧 primary 1/3 | 未启动 | 未认证 |
| YC | 2014 | 旧 primary 2/3 | 未启动 | 未认证 |
| FQ | 2016 | seed0 旧 primary 失败后停止 | 未启动 | 未认证；2016 为派生天气回放 |
| LC | 2010 | seed0 旧 primary 失败后停止 | 未启动 | 未认证；仅 PFP_N 有单项优势 |

## 重要 provenance 纠正

1. HLA 原始多 treatment MZX 的年份为 2007、2009、2011；HLA2010 是另一个正式单年锚点。后续正式 prepared-adapter 集中的 2015、2016、2022，以及所用 2007 变体，不能统称为原始权威 treatment。
2. FQ 原始 treatment 为 2007、2008、2010；FQ2016 及 2013、2014、2019、2020、2023 是由 FQ2008 风格模板结合目标年天气生成的派生候选。
3. YC 原始 treatment 为 2008、2014；2006、2009、2015、2018 是旧 DQN 天气回放证据。
4. LC 原始 treatment 为 2008、2009、2010、2011，但运行依赖 soil ID/IC 日期 adapter；只有 2010 目前有明确优化空间证据。
5. SY 原始正式 treatment 为 2012、2014、2015；其余 WTH-only 年份不进入正式结论。

## 输出文件

- `benchmark_results/028_01_all_site_year_scope_registry/028_01_station_year_registry.csv`
- `benchmark_results/028_01_all_site_year_scope_registry/028_01_current_crossyear_evidence.csv`
- `benchmark_results/028_01_all_site_year_scope_registry/028_01_source_file_hashes.csv`
- `benchmark_results/028_01_all_site_year_scope_registry/028_01_result.json`

## 下一步

执行 028_02：先对 Tier A 逐年补齐/复算四基线，再用与未来 RL 完全相同的阶段、动作、预算和 mask 做 deterministic strict-feasibility certification。Tier B 单列执行，不能与 Tier A 合并宣称。只有通过认证的年份才进入统一 reward 单测和小型全局调参。
