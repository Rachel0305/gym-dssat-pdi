# 015_14 FQ2016 baseline-relative DQN checkpoint training

## 目的

将当前统一 baseline-relative DQN 框架应用到封丘站 FQ2016，判断该站点年份是否也能产生“高产 + 节水节氮 + 可解释操作”的策略。

本轮只先做 seed0 50K checkpoint；如果 seed0 结果有价值，再继续 seed1 稳定性复核。

## 背景基准

来自 014_01 FQ 全年份筛选：

| 情景 | GWAD kg/ha | I mm | N kg/ha | 说明 |
|---|---:|---:|---:|---|
| null | 7066 | 0 | 0 | 有水分胁迫，存在优化空间 |
| recorded_shifted | 7933 | 75 | 144 | FQ2008 管理平移 |
| DSSAT auto | 8012 | 59.9 | 0 | 自动灌溉有效，自动施肥未触发 |

旧 014_01 DQN 5K 能到 8012，但用 N300；因此需要在 baseline-relative reward 下重新检查。

## 固定设置

奖励函数：

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

FQ2016 null baseline:

```text
GWAD_null = 7066 kg/ha
```

训练设置：

- 算法：DQN
- year：FQ2016
- seed：0
- timesteps：50K
- checkpoint interval：5K
- 动作空间：9 actions
  - irrigation: 0 / 15 / 30 mm
  - nitrogen: 0 / 50 / 100 kg/ha
- 预算：
  - I <= 120 mm
  - N <= 300 kg/ha
- 单次上限：
  - I <= 30 mm
  - N <= 100 kg/ha
- 最小操作间隔：7 days
- 输入：沿用 014_01 FQ2016 shifted MZX 和已校准品种参数。

## 执行命令

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_fq2016_baseline_relative_dqn_checkpoint_015_14.py --timesteps 50000 --seed 0 --checkpoint-interval 5000"
```

## 输出目录

```text
DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed0_50000steps/
```

## 判定标准

优先看 checkpoint，而不是最终模型。

若出现：

```text
GWAD >= 8012, I <= 59.9, N <= 0
```

则可认为全面超过或追平 auto 且资源更优。

若出现：

```text
GWAD >= 7933, I <= 75, N < 144
```

则可认为超过 recorded_shifted 且资源更优。

若只能靠 N300 达到高产，则说明 FQ2016 在当前奖励下不是理想成功案例。

