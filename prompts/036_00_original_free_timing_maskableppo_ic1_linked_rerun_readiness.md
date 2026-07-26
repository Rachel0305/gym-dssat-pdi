# 036_00 原自由时序 MaskablePPO：IC=1 + linked 动作接口正式重跑前审计

## 背景

之前自由时序 PPO 结果需要整体回到主线重跑，原因不是算法目标改变，而是后来发现两个底层问题：

1. 输入渲染链中 IC 需要确认使用 `IC=1`；
2. 动态 RL 动作必须在 DSSAT linked management 下执行，即 `IRRIG=L, FERTI=L`，否则 Python 侧动作可能不会真实进入 DSSAT。

034 系列已修复 linked action 接口；033 系列已检查 multisite 输入包 IC=1。但正式重跑前需要把“原训练配置 + IC=1 + linked 接口”放在同一个 036_00 前置审计里锁定，避免把 035_04/035_06 的 reward 探索线混进正式主线。

## 本任务性质

本任务不训练 PPO，不做参数调整，只做 readiness/smoke。

## 原配置来源

以 032_22 五站点 half-split stress-aware MaskablePPO 批量训练为原正式配置来源：

- 脚本：`src/run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py`
- 配置：`experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml`

## 必须保持不变的训练框架

- 每日环境运行；
- agent 每天观察；
- 动作档位：灌溉 `{0,15,30,45}`，施氮 `{0,40,80,120}`；
- 最小操作间隔：灌溉 7 天，施氮 7 天；
- 单季灌溉上限：160 mm；
- 单季施氮上限：250 kg/ha；
- DAP90 后禁氮；
- 原 stress-aware reward：

```text
harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001
yield_coef = 0.158
water_cost = 1.1
nitrogen_cost = 1.58
water_stress_relief_coef = 10.0
nitrogen_stress_relief_coef = 5.0
reward_scale = 0.001
```

- 原 PPO 超参数；
- 原训练步数：100K；
- 原 checkpoint：25K / 50K / 75K / 100K。

## 本任务检查项

1. 配置检查：确认上述动作约束、reward、PPO 超参数来自原配置；
2. IC/输入源检查：对 five-site half-split 中所有有天气数据的年份，确认渲染输入来自 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013` 且 treatment 1 为 `IC=1`；
3. linked 动作 smoke：五站点各选一个年份，强制 DAP1 执行 I45/N80，其余 no-op，确认：
   - safe action 总量与 DSSAT Summary 总量一致；
   - OVERVIEW 显示 `IRRIG=L, FERTI=L`；
   - null/no-op 对照仍为 I0/N0。

## 输出

- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_config_check.csv`
- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_ic_source_audit.csv`
- `benchmark_results/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness/evaluation/036_00_linked_forced_action_smoke.csv`
- `docs/036_00_original_free_timing_maskableppo_ic1_linked_rerun_readiness_record.md`

## 通过条件

全部三类检查均通过，才允许进入 036_01 正式重跑。

## 停止线

若任一检查失败，不启动正式训练；先修复输入/接口/配置问题。
