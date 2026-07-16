# 022_10 SY2014 DAP65 action1 vs action7 受控DSSAT对照

## 1. 目的

022_09发现三个离线MC-Q checkpoint在DAP65均把action7（I15/N100）排在action1（I15/N0）之前，但分组经验target受前期管理历史混杂。本任务固定全部前后缀，只交换DAP65动作，以确定该具体状态下的因果差异。

## 2. 设计

基础方案为022_01已验证的 `W60_critical__N200_early`：

| DAP | 1 | 30 | 50 | 65 | 85 | 110 |
|---|---:|---:|---:|---:|---:|---:|
| Control | 3 | 4 | 7 | 1 | 1 | 0 |
| Treatment | 3 | 4 | 7 | 7 | 1 | 0 |

唯一差异为DAP65 action1换成action7。DAP65之前累计I30/N200，剩余I90/N100，因此两个动作均完整执行，没有预算裁剪或动作别名。两组最终灌溉同为I60，施氮分别为N200和N300。

## 3. 实现检查

全部通过：

- Control复现022_01产量，误差<1 kg/ha；
- 两组均完整运行6个阶段；
- 非DAP65动作完全一致；
- DAP65前预算为I30/N200；
- DAP65执行量分别为I15/N0与I15/N100；
- 季节灌溉均为I60；
- 季节施氮为N200与N300；
- Summary.OUT匹配误差均<1。

## 4. 结果

| 指标 | Control action1 | Treatment action7 | Treatment−Control |
|---|---:|---:|---:|
| HWAM (kg/ha) | 11202.004 | 11202.111 | +0.106 |
| CWAM (kg/ha) | 20111.039 | 20190.853 | +79.814 |
| 灌溉 (mm) | 60 | 60 | 0 |
| 施氮 (kg/ha) | 200 | 300 | +100 |
| ET (mm) | 484.8 | 484.9 | +0.1 |
| WP-ET (kg/m³) | 2.31 | 2.31 | 0 |
| PFP-N (kg/kg) | 56.0 | 37.3 | −18.7 |
| G0 raw | 6354.004 | 5854.111 | −499.894 |
| 主判据 | 通过 | 通过 | — |
| 严格判据 | 通过 | 未通过 | — |

额外N100只增产0.106 kg/ha，远不足抵消reward中的500成本；新增生物量79.8 kg/ha没有转化为可观的籽粒产量增益。

## 5. 判定

按预注册规则判为 **C：Treatment籽粒产量略高，但reward和氮效率显著下降**。

从当前研究目标（产量与水氮效率联合优化）看，Control action1在该固定状态占优：

- 籽粒产量实际等同；
- 少施氮100 kg/ha；
- PFP-N高18.7 kg/kg；
- reward高499.894；
- 严格判据由不通过变为通过。

因此022_09中三个offline checkpoint在该状态偏好action7，是被受控实验确认的真实关键排序错误，而不是仅由历史状态混杂造成的表面现象。

## 6. 对下一步的影响

1. `022_11` seed1 warm-start草案继续暂停；当前初始化来源没有解决关键病灶。
2. 不能简单把action1硬编码为所有DAP65状态的答案；本次因果结论只适用于该固定前缀状态。
3. 下一步应构建多个不同前缀下的DAP65 action1/action7受控配对，形成训练/留出均可验证的反事实排序数据。
4. 只有跨多个固定前缀均确认额外N100效率不足，才适合预注册pairwise ranking约束或保守Q学习；不能只用一个配对现场修网络。

## 7. 输出

- `prompts/022_10_sy2014_dap65_action1_vs_action7_controlled_dssat.md`
- `src/run_sy2014_dap65_action_swap_controlled_022_10.py`
- `benchmark_results/022_10/022_10_controlled_summary.csv`
- `benchmark_results/022_10/022_10_controlled_stage_actions.csv`
- `benchmark_results/022_10/022_10_controlled_daily_values.csv`
- `benchmark_results/022_10/022_10_treatment_minus_control.csv`
- `benchmark_results/022_10/022_10_dap65_action_swap.png`
- `benchmark_results/022_10/022_10_dap65_action_swap.svg`
- `benchmark_results/022_10/022_10_result.json`

此前Docker权限阻塞记录保留在：

- `benchmark_results/022_10_failed_attempt_1_docker_permission/`
- `docs/2026-07-16_022_10_sy2014_dap65_action_swap_controlled_dssat_partial.md`
