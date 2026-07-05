# 017_01 FQ2016 baseline-relative DQN seed1 稳定性复核

## 目的

在已经确认 HLA、YC、FQ 当前 baseline-relative DQN 主线奖励函数一致后，对 FQ2016 补做 seed1 稳定性验证。

本轮不是更换奖励函数，不是调参，不是扩展新站点；只是在 FQ2016 同一训练框架下，把 seed 从 0 改为 1，检查 FQ2016 是否具备跨 seed 可复现性。

## 固定框架

- 算法：DQN
- 站点年份：FQ2016
- reward：

```text
reward_t = -1.0 * I_t - 5.0 * N_t
reward_T += max(0, GWAD_final - GWAD_null_site_year)
```

- null baseline：FQ2016 null = 7066 kg/ha
- 动作空间：9-action
  - irrigation: 0 / 15 / 30 mm
  - nitrogen: 0 / 50 / 100 kg/ha
- 总预算：I <= 120 mm, N <= 300 kg/ha
- 单次上限：I <= 30 mm, N <= 100 kg/ha
- 最小操作间隔：7 days
- timesteps：50000
- checkpoint interval：5000
- seed：1

## 执行要求

1. 必须使用指定 Docker 容器和虚拟环境：

```text
docker exec b2fd6726c8c1 /opt/gym_dssat_pdi/bin/python ...
```

2. 不覆盖 seed0 结果。
3. 不覆盖旧的 `015_14` 记录 MD。
4. 保存：
   - `checkpoint_summary.csv`
   - `dqn_eval_daily.csv`
   - checkpoint 模型 zip
   - checkpoint 诊断图
   - 本轮 seed1 专属实验记录 MD
5. 运行后和 seed0 对比：
   - best reward checkpoint
   - best yield checkpoint
   - 是否达到/接近 DSSAT auto
   - 是否比 recorded/expert 更省水氮
   - 是否存在跨 seed 稳定性

## 预期输出目录

```text
DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps
```

## 本轮判断口径

如果 seed1 也能找到与 seed0 类似的高产或高 reward checkpoint，则 FQ2016 可进入“同站点跨年份迁移/四情景图组整理”阶段。

如果 seed1 退化到 null 或明显低产，则 FQ2016 暂时只能作为有优化潜力但稳定性不足的候选站点年份。

