# 019_07 FQ2016 leaching_cost=20 5K 短训练记录

## 结论先行

本轮跑了 FQ2016、seed0、5000 steps 的两个对照：`leaching_cost=0` 和 `leaching_cost=20`，每 1000 steps 评估一次。

最重要的结果：`leaching_cost=20` 在 2000-step checkpoint 表现很好（GWAD=8012 kg/ha，I=90 mm，N=300 kg/ha，final_cleach=0），但继续训练到 4000/5000 steps 后退化为不灌溉、只施氮，产量降到 7106 kg/ha。因此它不能简单取 final checkpoint；如果继续这条线，必须使用 best checkpoint 选择和 seed 稳定性复核。

`leaching_cost=0` 在 4000/5000 steps 达到同样 GWAD=8012 kg/ha，I=120 mm，N=300 kg/ha，final_cleach=0。也就是说，在本轮 5K 下，无淋洗惩罚对照并不差，甚至更稳定；`leaching_cost=20` 的优势主要出现在中间 checkpoint 的节水 30 mm，而不是最终 checkpoint。

这一轮的直接判断：淋洗惩罚链路有效，但 `leaching_cost=20` 仍不是可以直接放大的正式配置。下一步若继续，应该围绕 `best checkpoint` 和 `0 vs 20` 的多 seed 复核，而不是盲目加训练步数。

## 各 checkpoint 结果

| run_label | checkpoint_step | leaching_cost | final_grain_kg_ha | final_biomass_kg_ha | action_irrigation_total | action_fertilizer_total | final_cleach | soilni_final_NLCC | sum_leaching_cost_term | max_water_stress | max_nitrogen_stress | total_reward | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lc0_5k | 1000.000 | 0.000 | 7970.000 | 13805.000 | 120.000 | 300.000 | 14.141 | 14.140 | 0.000 | 0.000 | 0.012 | -716.462 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc0 |
| lc0_5k | 2000.000 | 0.000 | 7988.000 | 14027.000 | 120.000 | 300.000 | 1.932 | 1.930 | 0.000 | 0.000 | 0.012 | -698.021 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc0 |
| lc0_5k | 3000.000 | 0.000 | 7980.000 | 14000.000 | 90.000 | 300.000 | 3.941 | 3.940 | 0.000 | 0.000 | 0.012 | -675.926 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc0 |
| lc0_5k | 4000.000 | 0.000 | 8012.000 | 14033.000 | 120.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | -673.578 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc0 |
| lc0_5k | 5000.000 | 0.000 | 8012.000 | 14093.000 | 120.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | -673.578 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc0 |
| lc20_5k | 1000.000 | 20.000 | 7751.000 | 13830.000 | 30.000 | 300.000 | 3.193 | 3.190 | 63.866 | 0.482 | 0.012 | -908.981 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc20 |
| lc20_5k | 2000.000 | 20.000 | 8012.000 | 14093.000 | 90.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | -643.578 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc20 |
| lc20_5k | 3000.000 | 20.000 | 8012.000 | 14090.000 | 120.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | -673.578 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc20 |
| lc20_5k | 4000.000 | 20.000 | 7106.000 | 13188.000 | 0.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.653 | 0.012 | -1459.553 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc20 |
| lc20_5k | 5000.000 | 20.000 | 7106.000 | 13188.000 | 0.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.653 | 0.012 | -1459.808 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc20 |

## 各组按产量选出的 best checkpoint

| run_label | checkpoint_step | leaching_cost | final_grain_kg_ha | final_biomass_kg_ha | action_irrigation_total | action_fertilizer_total | final_cleach | soilni_final_NLCC | sum_leaching_cost_term | max_water_stress | max_nitrogen_stress | total_reward | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lc0_5k | 4000.000 | 0.000 | 8012.000 | 14033.000 | 120.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | -673.578 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc0 |
| lc20_5k | 2000.000 | 20.000 | 8012.000 | 14093.000 | 90.000 | 300.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | -643.578 | DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05\seed0_5000steps_lc20 |

## 文件

- 汇总表：`DSSAT_auto_validation\fq2016_leaching_cost20_5k_019_07\019_07_fq2016_leaching_cost20_5k_summary.csv`
- 来源目录：`DSSAT_auto_validation\fq2016_leaching_aware_reward_smoke_019_05`

## 下一步建议

1. 不建议直接把 `leaching_cost=20` 继续训练到更长步数，因为 5K 已经出现后期退化。
2. 如果导师希望纳入淋洗惩罚，建议使用 `best checkpoint` 逻辑，并先做 seed1 复核。
3. 如果当前主线目标仍是产量和水氮效率优先，FQ2016 这一轮暂时更支持把 `leaching_cost=0` 作为稳定对照，把淋洗惩罚作为敏感性/环境效益扩展。
