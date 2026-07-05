# 015_10 YC2014 baseline-relative reward DQN recheck

## 目的

用和 HLA baseline-relative 成功线一致的奖励公式复核 YC2014，确认跨站点正式 reward 是否可以统一。

## 奖励

`reward_t = - 1.0 * I_t - 5.0 * N_t`

`reward_T += max(0, GWAD_final - GWAD_null_site_year)`

每个站点年份使用自己的 null baseline；本轮 YC2014 的 null baseline 来自同一输入包的 null 情景。

## 结果

| item | value |
| --- | ---: |
| null_baseline_yield | 7825.000 |
| final_grain_kg_ha | 8657.000 |
| final_biomass_kg_ha | 18828.000 |
| action_irrigation_total | 75.000 |
| action_fertilizer_total | 0.000 |
| max_water_stress | 0.000 |
| max_nitrogen_stress | 0.451 |
| total_reward | 756.584 |

## 判断

- 如果该结果仍接近专家产量且水氮投入较低，说明 YC2014 在 baseline-relative 统一奖励下仍可保留为成功案例。
- 如果该结果明显退化，则说明旧 YC 成功依赖 delta_grnwt 奖励，需要单独说明。