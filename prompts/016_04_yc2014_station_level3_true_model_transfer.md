# 016_04 YC2014 第3层：真模型跨年份、跨 seed 迁移验证

## 目标

把 YC 从上一轮“动作策略回放”推进到真正的“模型迁移”：

1. 使用 YC2014 训练得到的 DQN 模型 checkpoint；
2. 不在目标年份重新训练；
3. 将模型迁移到 YC 2000-2023 的天气年；
4. 判断同一站点内部最多能跨多少年仍保持可解释的水氮管理效果；
5. 对比 seed0 / seed1 是否稳定。

## 背景

016_03 已经确认：

- FQ 可以做真模型迁移，因为 FQ2016 checkpoint zip 已保存；
- YC 上一轮只能做动作回放，因为当时误以为 YC 没有保存模型；
- 复查后发现 `yc2014_baseline_relative_dqn_015_10` 其实已经保存了 YC2014 seed0 的 baseline-relative DQN checkpoint zip；
- 但 seed1 尚未训练保存，因此本轮需要先补 seed1。

## 输入

YC2014 seed0 已有模型：

```text
DSSAT_auto_validation/yc2014_baseline_relative_dqn_015_10/seed0/dqn_baseline_relative_checkpoint/models/
```

需要补跑 seed1：

```bash
/opt/gym_dssat_pdi/bin/python src/run_yc2014_baseline_relative_dqn_015_10.py --seed 1 --timesteps 50000 --checkpoint
```

必须在指定 Docker 容器中运行：

```text
b2fd6726c8c1
```

Python 环境：

```text
/opt/gym_dssat_pdi/bin/python
```

## 迁移方法

1. 使用 YC2014 treatment 2 作为训练年份模板。
2. 将天气站和日期平移到目标年份：

- `CNYC1401` -> `CNYCyy01`
- `CNYC2014` -> `CNYCyyyy`
- `Sim2014` -> `Simyyyy`
- `14DDD` -> `yyDDD`
- 初始条件日期保留同一 DAP 口径，将 `08153` 平移为 `yy153`

3. 目标年份管理方式设为 `IRRIG=L / FERTI=L`，由 DQN 动态决策。
4. 每个 seed 从训练年 checkpoint summary 中选择 `total_reward` 最高的 checkpoint 作为主迁移模型。
5. 如果需要保留参考，可以额外记录最高产量 checkpoint，但本轮主判定使用最高 reward checkpoint。

## 输出

```text
DSSAT_auto_validation/yc2014_station_level3_true_model_transfer_016_04/
  yc2014_true_model_transfer_summary.csv
  yc2014_true_model_transfer_daily.csv
  yc2014_transfer_success_by_year.csv
  figures/
docs/2026-07-02_016_04_yc2014_station_level3_true_model_transfer_record.md
```

## 判定口径

本轮只评价“同站点内部泛化”，不评价跨站点泛化。

一个目标年份暂定为迁移成功，需要同时满足：

- DQN 产量达到 DSSAT auto 的 98% 或以上；
- DQN 不比 DSSAT auto 多用水；
- DQN 最终产量明显高于 null；
- 管理动作不是完全无操作导致的偶然接近。

如果 auto 本身没有用水，则 DQN 要想被判为“节水成功”必须也不能明显多用水；这种年份通常更难证明 DQN 有实际价值。

