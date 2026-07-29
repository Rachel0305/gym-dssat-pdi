# 040_02 SYA lowIC 严格 MaskableDQN 框架任务书

## 任务目的

040_01 使用 SB3 DQN 做了快速对照，但 SB3 DQN 训练阶段不原生支持 action mask。040_02 改为项目本地实现的严格 MaskableDQN，使 mask 真正参与：

- epsilon 随机探索；
- greedy action selection；
- replay buffer；
- next-state target max。

本任务回答：

> 在同一 SYA lowIC 输入、同一自由时序动作空间、同一安全约束和同一奖励函数下，严格 MaskableDQN 是否比 040_00 MaskablePPO / 040_01 普通 DQN 更合适？

## 与 040_01 的唯一核心差异

- `SB3 DQN + masked-greedy evaluation`
- 改为
- `StrictMaskableDQN + mask-aware training + mask-aware evaluation`

## 固定配置

- 输入数据根目录：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual`
- 站点：SYA
- 训练年份：2005–2013
- 验证年份：2014–2023
- 默认训练步数：100,000
- checkpoint：25k / 50k / 75k / 100k
- seed：0
- 动作空间：灌溉 `{0,15,30,45}` mm；施氮 `{0,40,80,120}` kg/ha
- 管理约束：沿用 032/040 主线
- 奖励函数：沿用 `harvest_yield_minus_water_nitrogen_cost_plus_stress_relief_scaled_0p001`
- DQN 超参数：读取 `config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml` 中的 `dqn` 块

## 执行纪律

1. 先跑单元测试：

```bash
python test_strict_maskable_dqn_040.py
```

2. 再跑 dry-run：

```bash
python run_sya_lowIC_strict_maskable_dqn_040_02.py --dry-run
```

3. 如果要正式跑：

```bash
python run_sya_lowIC_strict_maskable_dqn_040_02.py
```

4. 如果只想先做小 smoke：

```bash
python run_sya_lowIC_strict_maskable_dqn_040_02.py --timesteps 2000 --checkpoint-steps 1000,2000 --suffix smoke2k
```

注意：带 `--timesteps` 的运行是工程 smoke，不等同于正式 100k 结果。

## 输出

- `benchmark_results/040_02_sya_lowIC_strict_maskable_dqn/`
- `docs/040_02_sya_lowIC_strict_maskable_dqn_record.md`

