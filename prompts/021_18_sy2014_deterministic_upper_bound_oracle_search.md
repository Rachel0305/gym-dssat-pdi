# 021_18 SY2014 确定性上界 / oracle 调度搜索

## 目的

在不训练 DQN、不修改 reward、不修改 IC、不修改 DSSAT 输入的前提下，回答一个先于算法优化的问题：

> 在当前 SY2014、IC=2、DQN 可执行动作与 I≤120 mm/N≤300 kg ha-1 约束内，是否客观存在同时具有高产和较高水氮利用效率的水氮时序？

若不存在，则继续调 DQN 没有意义；若存在，则把高质量时序保存为后续 demonstration-guided DQN 的示范数据，但不能把人工搜索结果冒充 DQN 结果。

## 冻结条件

- 使用 `021_14` 已验证的 SY2014 IC=2 输入和 PDI/DSSAT 4.8.0 环境。
- 只做确定性前向模拟，不训练模型。
- 单次动作仍来自 DQN 的离散档位：灌溉 0/15/30 mm，施氮 0/50/100 kg ha-1。
- 决策窗口 DAP 1–120；共享操作间隔至少 7 d。
- 季节预算 I≤120 mm、N≤300 kg ha-1。
- 不覆盖 `021_00`–`021_17` 的任何结果。

## 同输入比较基准

- null：5408 kg ha-1，I0/N0。
- recorded：9613 kg ha-1，I0/N293。
- DSSAT auto：5498 kg ha-1，I66/N0。
- 官方推广 expert：11077 kg ha-1，I266.1/N300。
- 现有确定性高产回放候选：约 11199 kg ha-1，I120/N300。

以上产量比较只使用同一套 SY2014 IC=2 输入；旧输入结果不得混用。

## 预注册成功条件

在 I≤120/N≤300 且动作/间隔合法的前提下：

1. `HWAM ≥ 11077 kg ha-1`（不低于同输入官方推广 expert 与 auto 的较高者）；
2. `WP_ET` 不低于两套基准中的较高者；
3. 对 N>0 的候选，`PFP_N ≥ 36.923 kg grain kg-1 N`（不低于官方推广 expert）；
4. `IWP_gross`、`NUtE`、`PNB_N` 同时报告，但不得把 N=0 的 PFP 解释为无穷大。

其中第 2–4 项使用 DSSAT `Summary.OUT` 的同口径指标：YPEM、YPIM、YPNAM、YPNUM，并保存重新计算交叉核查。

## 分阶段执行

### A. smoke

回放 `021_15` 的 scaled 25K 高产时序：

- I：DAP 1=30；8/22/29/42/56/79 各 15 mm；
- N：DAP 1=50、8=100、15=100、29=50 kg ha-1。

要求：HWAM 与 11199 kg ha-1 相差不超过 2 kg ha-1，Summary.OUT 的 IRCM/NICM 分别为 120/300，所有动作均被执行。

### B. 固定灌溉、扫描施氮时序和总量

- 灌溉固定为 smoke 的 I120 时序。
- 施氮时序：early / mid / late。
- 总氮：150/200/250/300 kg ha-1，使用 50/100 kg ha-1 离散档位，组合动作间隔合法。
- 共 12 次确定性运行。

### C. 剪枝水量扫描

- 仅选择 B 阶段中最多 3 个达到或最接近预注册成功条件的施氮方案。
- 对每个方案测试 I90/I60/I30 三套由 smoke 时序删减得到的合法灌溉方案。
- 最多 9 次运行；不做全组合暴力搜索。

### D. 阈值边界细化（在 C 完成后追加登记）

C 阶段结果显示 I90 同时通过产量和 PFP，但 WP_ET=2.24 略低于官方 expert 的 2.26；I60 的 WP_ET=2.28，但产量低于 11077。由于成功边界被夹在 I60–I90 之间，追加一个不改变阈值的低成本细化：

- 只测试两套 I75 时序：从 I90 分别删去早期 DAP8 或晚期 DAP79 的 15 mm；
- 分别与 C 阶段的 mid-N200/N250/N300 组合；
- 共 6 次前向模拟；不继续扩展更多水量档位。

这属于依据预注册阈值做的边界定位，必须在记录中标注为 adaptive refinement，不能冒充原始预注册网格。

## 输出

保存到 `benchmark_results/021_18/`：

- 每个候选的 schedule、daily CSV、requested actions、MgmtEvent、PlantGro、Summary.OUT 快照；
- `021_18_candidate_summary.csv`；
- `021_18_baseline_metrics.csv`；
- `021_18_success_candidates.csv`；
- `021_18_search_manifest.json`；
- 必要的失败记录。

实验记录：

- `docs/2026-07-15_021_18_sy2014_deterministic_upper_bound_oracle_search.md`

## 停止规则

- smoke 不通过：停止，不进入搜索。
- 输入、IC、动作执行总量或指标解析不一致：停止并记录。
- 不因搜索结果临时修改成功阈值。
- 本任务不启动 DQN demonstration learning；只决定它是否值得作为下一步。
