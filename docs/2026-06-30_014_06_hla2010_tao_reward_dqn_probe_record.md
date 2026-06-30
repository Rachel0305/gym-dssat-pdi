# 014_06 HLA2010 Tao et al. 奖励函数 DQN 探针记录

## 目的

测试 Tao et al. 2023 RF1 economic profit 奖励函数的简化版，是否能缓解 HLA2010 当前 economic DQN 的 no-op 退化。

## 奖励函数

```text
非终止日: r_t = -0.79 * N_t - 1.1 * W_t
终止日:   r_t = 0.158 * GRNWT_final - 0.79 * N_t - 1.1 * W_t
```

暂不包含硝态氮淋失项，因此是 Tao RF1 的简化版。

## 当前结果

| timesteps | seed | final_grnwt | final_topwt | action_irrigation_total | action_nitrogen_total | max_water_stress | max_nitrogen_stress | eval_reward_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200.000 | 0.000 | 6970.795 | 19367.311 | 0.000 | 200.000 | 0.919 | 0.016 | 943.386 |
| 5000.000 | 0.000 | 6956.454 | 19344.495 | 0.000 | 0.000 | 0.919 | 0.157 | 1099.120 |

## 文件

- 汇总：`DSSAT_auto_validation/HLA_2004/hla2010_tao_reward_dqn_probe_014_06/014_06_hla2010_tao_reward_summary.csv`
- 输出目录：`DSSAT_auto_validation/HLA_2004/hla2010_tao_reward_dqn_probe_014_06`
