# 040_01 SYA lowIC 自由时序 DQN 对照任务书

## 任务目的

在 040_00 已完成 SYA lowIC MaskablePPO 后，按导师建议补一个 DQN 对照。

本任务只回答：

> 在同一 lowIC 输入、同一年份划分、同一动作空间、同一安全约束和同一奖励函数下，DQN 是否比 040_00 的 MaskablePPO 更能学出高质量自由时序水氮策略？

## 固定边界

相对 040_00，本任务只改变算法：

- `MaskablePPO` → `DQN`

以下保持不变：

- 输入数据根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 训练/验证年份划分：沿用 `032_21_half_split_years.csv`
- 训练年份：SYA 2005–2013
- 验证年份：SYA 2014–2023
- 训练步数：100,000 timesteps
- checkpoint：25k / 50k / 75k / 100k
- seed：0
- 动作空间：灌溉 `{0,15,30,45}` mm；施氮 `{0,40,80,120}` kg/ha
- 管理约束：单季上限、7天最小操作间隔、DAP90后禁氮等沿用 032/040 主线配置
- 奖励函数：沿用 `harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001`

## DQN 与 Mask 的处理

SB3 原生 DQN 不支持像 MaskablePPO 那样在训练采样时直接传入 action mask。

本任务采用保守处理：

1. 训练阶段：DQN 与环境交互，非法动作由现有安全层转换/兜底；
2. 评估阶段：使用 DQN 的 Q 值，并在贪心选择时显式屏蔽非法动作，即 masked-greedy evaluation。

因此，本任务是“当前框架下的 DQN 对照”，不是完整 MaskableDQN 算法。

## 执行纪律

- 不调 DQN 超参数；
- 不改 reward；
- 不改动作档位；
- 不改年份划分；
- 不现场追加训练步数；
- 如果失败，只记录失败原因。

## 推荐运行命令

```bash
cd /workspace/src
python run_sya_lowIC_free_timing_dqn_040_01.py --dry-run
python run_sya_lowIC_free_timing_dqn_040_01.py
```

## 主要输出

- `benchmark_results/040_01_sya_lowIC_free_timing_dqn/`
- `docs/040_01_sya_lowIC_free_timing_dqn_record.md`

