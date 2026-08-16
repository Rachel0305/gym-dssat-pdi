# E2 跨 seed 5K 稳定性与 WP_ET 核验

本报告复用既有冻结 checkpoint，不重新训练。

## 每个 seed 的 E2 与 no-forecast 配对

|   seed |   mean_yield_delta |   mean_pfp_n_delta |   mean_wp_et_delta |   yield_win_years |   pfp_n_win_years |   wp_et_win_years |
|-------:|-------------------:|-------------------:|-------------------:|------------------:|------------------:|------------------:|
|      0 |            60.8138 |               0.24 |              0.013 |                 4 |                 4 |                 5 |
|      1 |            23.7306 |               0.1  |             -0.069 |                 6 |                 5 |                 1 |
|      2 |            23.0854 |              -1.5  |             -0.031 |                 6 |                 5 |                 1 |

## pooled

- E2 平均 WP_ET：2.0443；no-forecast：2.0733。
- E2 加权 WP_ET：2.0439；no-forecast：2.0710。
- E2 平均产量胜出 seed：3/3；PFP-N：2/3；WP_ET：1/3。

## 稳定性门禁

- 5K 天气响应通过：2/3。
- 5K 动作多样性通过：3/3。
- 5K 天气响应和动作多样性同时 3/3：False。

## 结论

E2 是有希望的 forecast 候选：部分 seed 的产量/PFP-N 优于 no-forecast，且 5K 动作多样性通过。但 WP_ET 没有跨 seed 稳定占优，天气响应也只有 2/3 seed 通过，因此不能宣称 E2 已经全面、稳定优于 no-forecast。

- 逐年配对：`154E2_paired_by_year.csv`
- 每 seed 汇总：`154E2_paired_summary_by_seed.csv`
- WP_ET 闭合核验：`154E2_wp_et_reproducibility_check.json`
