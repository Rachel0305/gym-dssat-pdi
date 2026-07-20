# 026_12 五站点固定 DAP 决策点物候兼容性审计

## 问题

SY 阶段型 PPO 使用固定 DAP 1/30/50/65/85/110。即使026_11已经统一输入维度，不同生态站点的相同DAP也可能对应不同生育阶段。若直接联合训练，会把不同农学语义的决策混在一起。

## 方法

- 固定 HLA2010、YC2014、FQ2016、LC2010、SY2014。
- 每站点使用已验证输入副本，各跑1个全季零动作前向。
- 不训练、不加载模型；只记录每日 DAP、ISTAGE、VSTAGE、TOPWT、GRNWT。
- 提取 DAP 1/30/50/65/85/110 的跨站点快照。
- 原始输入哈希执行前后必须不变。

## 预注册判定

- A：五站点都到达六个固定DAP，且每个DAP的跨站点 VSTAGE 极差不超过2.0；可以保留固定DAP联合训练设计。
- B：任一固定DAP缺失，或任一DAP的VSTAGE极差超过2.0；不得把SY绝对DAP直接用于联合训练，下一步改为DSSAT物候触发并单测。
- C：输入或DSSAT执行失败；先修 provenance。

此阈值只用于判断“固定DAP能否代表近似一致物候”，不声称 VSTAGE 差2.0具有普适农学等价性。

## 输出

- `benchmark_results/026_12/026_12_daily_phenology.csv`
- `benchmark_results/026_12/026_12_fixed_dap_snapshots.csv`
- `benchmark_results/026_12/026_12_cross_site_vstage_spread.csv`
- `benchmark_results/026_12/026_12_result.json`
- `docs/2026-07-17_026_12_five_site_fixed_dap_phenology_compatibility_audit.md`

