# 016_09 YC2014 seed1 checkpoint 验证

## 目标

在已经确认 YC2014 seed0 存在明显优化空间的基础上，复核 seed1 是否也呈现相同规律：

1. 是否也能明显优于 null；
2. 是否也会出现“最高产 checkpoint”和“最高奖励 checkpoint”分离；
3. 是否也存在中后期漂移，说明 YC2014 适合 checkpoint 选优而不是最终步模型。

## 已知前情

- `016_08` 已完成 seed0 baseline-relative checkpoint refresh；
- seed0 结果表明：
  - 5K 可达高产平台 `9418 kg/ha`，但高投入；
  - 25K 奖励最高，但更偏节氮；
  - 长训练存在漂移。

## 本轮约束

- 不覆盖旧结果；
- 只新增 seed1 结果；
- 不改奖励公式、不改动作空间、不改预算；
- 只验证稳定性，不额外扩站点。

## 奖励与约束

- 奖励：
  - 非终止步：`reward = - 1.0 * irrigation - 5.0 * nitrogen`
  - 终止步：`+ max(0, GWAD_final - GWAD_null_site_year)`
- null baseline：由脚本自动复算本地 YC2014 null
- 预算：
  - 灌溉总量上限 `120 mm`
  - 施氮总量上限 `300 kg/ha`
- 动作：
  - 灌溉 `{0, 15, 30}`
  - 施氮 `{0, 50, 100}`
  - 共 9 个离散联合动作

## 执行命令

```bash
docker exec b2fd6726c8c1 bash -lc 'cd /workspace && /opt/gym_dssat_pdi/bin/python src/run_yc2014_baseline_relative_checkpoint_refresh_016_08.py --seed 1 --timesteps 50000 --checkpoint --checkpoint-interval 5000'
```

## 输出目录

- `DSSAT_auto_validation/yc2014_baseline_relative_checkpoint_refresh_016_08/seed1/`
- `docs/2026-07-05_016_08_yc2014_baseline_relative_checkpoint_refresh_record.md`（由脚本更新单次摘要）

## 运行后检查

重点检查：

1. `final_grain_kg_ha` 是否明显高于 null baseline；
2. 最佳产量 checkpoint 与最佳奖励 checkpoint 是否分离；
3. 是否也出现 30K 以后漂移或退化；
4. 与 seed0 对比时，规律是否一致。

## 预期输出

如果 seed1 也复现相同结构，就可以把 YC2014 定位为：

- 有优化空间；
- 可跨 seed 复现；
- 适合 checkpoint 选优；
- 可作为下一阶段正式图表与导师讨论样例。
