# 017_01 FQ2016 baseline-relative DQN seed1 稳定性复核记录

## 目的

在不改变算法、奖励函数、动作空间、预算约束的前提下，对 FQ2016 补做 seed1 训练，判断 seed0 中出现的 FQ2016 DQN 策略是否具有跨 seed 复现迹象。

本轮不是新奖励函数，不是调参，不是跨站点迁移；只是 FQ2016 同一 baseline-relative DQN 框架下的 seed1 复核。

## 执行信息

| 项目 | 内容 |
|---|---|
| Prompt | `prompts/017_01_fq2016_baseline_relative_dqn_seed1_validation.md` |
| Script | `src/run_fq2016_baseline_relative_dqn_checkpoint_015_14.py` |
| Docker/env | `b2fd6726c8c1` + `/opt/gym_dssat_pdi/bin/python` |
| Site-year | FQ2016 |
| Seed | 1 |
| Timesteps | 50000 |
| Checkpoint interval | 5000 |
| Output dir | `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps` |

## 固定框架

奖励函数：

```text
reward_t = -1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

固定参数：

| 项目 | 值 |
|---|---:|
| FQ2016 null baseline | 7066 kg/ha |
| water cost | 1.0 |
| nitrogen cost | 5.0 |
| irrigation budget | 120 mm |
| nitrogen budget | 300 kg/ha |
| single irrigation cap | 30 mm |
| single nitrogen cap | 100 kg/ha |
| minimum interval | 7 days |
| action space | I in 0/15/30, N in 0/50/100 |

基准情景：

| scenario | GWAD kg/ha | I mm | N kg/ha |
|---|---:|---:|---:|
| null | 7066 | 0 | 0 |
| recorded_shifted | 7933 | 75 | 144 |
| DSSAT auto | 8012 | 59.9 | 0 |

## seed1 checkpoint 结果

| checkpoint | I mm | N kg/ha | GWAD kg/ha | CWAD kg/ha | max water stress | max N stress | total reward | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5000 | 120 | 300 | 8012 | 14093 | 0.000 | 0.012 | -673.6 | 最高产量之一，但水氮用满且 reward 低 |
| 10000 | 120 | 300 | 8012 | 14095 | 0.000 | 0.012 | -673.6 | 同上 |
| 15000 | 30 | 300 | 7066 | 13148 | 0.657 | 0.012 | -1529.9 | 低产且高氮，不合适 |
| 20000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.1 | 退回 null |
| 25000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.1 | 退回 null |
| 30000 | 60 | 0 | 7995 | 14078 | 0.050 | 0.012 | 869.2 | best reward，接近 auto 且省水省氮 |
| 35000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.1 | 退回 null |
| 40000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.1 | 退回 null |
| 45000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.1 | 退回 null |
| 50000 | 0 | 0 | 7066 | 13148 | 0.657 | 0.012 | 0.1 | 退回 null |

## seed0 vs seed1 对比

| seed | best reward checkpoint | I mm | N kg/ha | GWAD kg/ha | total reward | 与 auto 关系 | 与 recorded 关系 |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 25000 | 60 | 0 | 7779 | 652.6 | 低 233 kg/ha，水量几乎相同，少 N | 低 154 kg/ha，少水 15 mm，少 N 144 kg/ha |
| 1 | 30000 | 60 | 0 | 7995 | 869.2 | 低 17 kg/ha，水量几乎相同，少 N | 高 62 kg/ha，少水 15 mm，少 N 144 kg/ha |

## 关键判断

1. FQ2016 的 best reward 策略在 seed0 和 seed1 上都指向 `I60/N0`，说明“节水、零氮、接近高产”的策略结构具有一定跨 seed 一致性。
2. seed1 的 `I60/N0/GWAD7995` 几乎追平 DSSAT auto 的 8012 kg/ha，同时比 recorded 少 15 mm 水、少 144 kg/ha N。
3. seed0 的同类策略产量为 7779 kg/ha，低于 auto 和 recorded，但仍明显优于 null。
4. 训练后期会退回 null，说明 FQ 与 HLA/YC 一样，需要 checkpoint selection，不能只取 final checkpoint。
5. FQ2016 当前可以进入下一阶段：整理四情景过程图，并基于已有 FQ2016 checkpoint 做同站点跨年份迁移复核。

## 文件

| 文件 | 路径 |
|---|---|
| summary | `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/checkpoint_summary.csv` |
| daily | `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/dqn_eval_daily.csv` |
| figure | `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/figures/fq2016_baseline_relative_seed1_50000steps_checkpoint_summary.png` |
| models | `DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/models/` |

