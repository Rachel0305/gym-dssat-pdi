# 018_10 HLA/YC/FQ seed 复核记录

## 说明

- 这次没有盲目重跑三个站点。
- 经核查，YC2014 与 HLA2010 的 seed1 结果早已存在；FQ2016 的 seed1 也已有完整 50K 结果，仅缺统一归档。
- 因此本轮工作的核心是：核实旧结果、补充 FQ 的 smoke 证据、统一形成正式复核口径。

## seed 最佳 checkpoint 表

| site | station | year | seed | checkpoint_step | action_irrigation_total | action_fertilizer_total | final_grain_kg_ha | final_biomass_kg_ha | max_water_stress | max_nitrogen_stress | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FQ | Fengqiu | 2016 | 0 | 25000 | 60 | 0 | 7779 | 13861 | 0.369 | 0.012 | 652.609 |
| FQ | Fengqiu | 2016 | 1 | 30000 | 60 | 0 | 7995 | 14078 | 0.050 | 0.012 | 869.176 |
| HLA | Hailun | 2010 | 0 | 35000 | 120 | 0 | 7854 | 20886 | 0.416 | 0.062 | 777.665 |
| HLA | Hailun | 2010 | 1 | 25000 | 60 | 0 | 7573 | 20255 | 0.751 | 0.173 | 557.022 |
| YC | Yucheng | 2014 | 0 | 5000 | 120 | 250 | 9418 | 20513 | 0 | 0.013 | 8047.855 |
| YC | Yucheng | 2014 | 1 | 10000 | 120 | 300 | 9418 | 20493 | 0 | 0.013 | 7797.902 |

## 站点复核结论

| site | station | year | seed0_checkpoint | seed0_gwad | seed0_irrigation | seed0_nitrogen | seed1_checkpoint | seed1_gwad | seed1_irrigation | seed1_nitrogen | recheck_status | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLA | Hailun | 2010 | 35000 | 7854 | 120 | 0 | 25000 | 7573 | 60 | 0 | resource_stable_but_yield_not_stable | HLA2010 seed0 可在 N=0 条件下追平高产，但 seed1 最优点只有约 7573 kg/ha，说明跨 seed 产量不稳定。 |
| YC | Yucheng | 2014 | 5000 | 9418 | 120 | 250 | 10000 | 9418 | 120 | 300 | stable_yield_but_nitrogen_not_stable | YC2014 两个 seed 的最佳产量一致，但 seed1 需要 300 kg/ha 氮、seed0 只需 250 kg/ha，说明产量稳定而节氮方向不稳定。 |
| FQ | Fengqiu | 2016 | 25000 | 7779 | 60 | 0 | 30000 | 7995 | 60 | 0 | seed1_success_but_seed0_not_reproduced | FQ2016 历史上用于四情景图的代表性 DQN 本来就是 seed1@ckpt30000=7995；本次统一复核发现 seed0 本地 checkpoint 最优仅约 7779，说明当前只确认了 seed1 成功，尚未形成跨 seed 稳定性。 |
