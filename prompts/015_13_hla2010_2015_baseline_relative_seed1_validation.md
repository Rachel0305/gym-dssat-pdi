# 015_13 HLA2010/HLA2015 baseline-relative DQN seed1 stability validation

## 目的

在 015_12 已完成 HLA2010/HLA2015 seed0 50K checkpoint 的基础上，补做 seed1 稳定性复核。

这一步的目标不是继续调参，而是判断当前 baseline-relative DQN 框架能否跨 seed 复现：

- HLA2010 是否稳定学到 `I120/N0/GWAD≈7854` 这一类策略；
- HLA2015 是否稳定学到 `I75-90/N0/GWAD≈7653` 这一类策略；
- 如果 seed1 不稳定，则不能把 seed0 结果直接作为正式成功结论。

## 固定设置

- 算法：DQN
- 奖励函数：

```text
reward_t = - 1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

- 每个站点年份使用自己的 null 产量作为 baseline，但公式、成本、动作空间、预算和算法保持一致。
- 年份：HLA2010、HLA2015
- Seed：1
- Timesteps：50K
- Checkpoint interval：5K
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
- 保持 IC=1 和已更新品种参数，不改变输入模板。

## 执行顺序

先跑 HLA2010 seed1；完成并确认输出正常后，再跑 HLA2015 seed1。

```bash
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_baseline_relative_dqn_checkpoint_015_12.py --year 2010 --timesteps 50000 --seed 1 --checkpoint-interval 5000"
docker exec b2fd6726c8c1 bash -lc "cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_hla_baseline_relative_dqn_checkpoint_015_12.py --year 2015 --timesteps 50000 --seed 1 --checkpoint-interval 5000"
```

## 输出

输出仍放在 015_12 目录下，但 seed 子目录不同：

```text
DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/
DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2015/baseline_relative_seed1_50000steps/
```

需要记录：

- checkpoint_summary.csv
- dqn_eval_daily.csv
- checkpoint diagnostic figure
- docs 实验记录

## 判定标准

### HLA2010

seed1 如果也出现高 reward checkpoint：

```text
GWAD≈7854, I≈120, N≈0
```

则 HLA2010 可认为在该框架下有跨 seed 稳定性。

### HLA2015

seed1 如果也出现高 reward checkpoint：

```text
GWAD≈7650, I明显低于 auto, N≈0
```

则 HLA2015 可认为是更接近“超越 auto + 节水”的主案例。

### 如果不复现

若 seed1 退化为 null、I120/N300、或 reward/yield 明显不稳定，则结论必须写成：

```text
seed0 显示潜力，但尚未跨 seed 稳定。
```

