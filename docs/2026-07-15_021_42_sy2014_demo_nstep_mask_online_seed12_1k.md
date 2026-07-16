# 021_42 SY2014 demo n-step屏蔽在线seed1/2复核记录

## 范围

固定同一个021_34 seed0离线起点，只改变在线action RNG和sampling RNG；这不是不同网络初始化复现。

## 结果

| online_seed | checkpoint | yield_kg_ha | irrigation_mm | nitrogen_kg_ha | late_n_after_dap90_kg_ha | expert_efficiency_gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| 1 | 250 | 11175.0 | 75.0 | 300.0 | 0.0 | True |
| 1 | 500 | 11170.0 | 105.0 | 300.0 | 0.0 | True |
| 1 | 750 | 11154.0 | 120.0 | 300.0 | 0.0 | True |
| 1 | 1000 | 10787.0 | 90.0 | 150.0 | 0.0 | False |
| 2 | 0 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| 2 | 250 | 11175.0 | 90.0 | 300.0 | 0.0 | True |
| 2 | 500 | 11170.0 | 105.0 | 300.0 | 0.0 | True |
| 2 | 750 | 10787.0 | 90.0 | 150.0 | 0.0 | False |
| 2 | 1000 | 10787.0 | 90.0 | 150.0 | 0.0 | False |

| online_seed | sample_seed | pass_count | final_pass | minimum_yield | success |
| --- | --- | --- | --- | --- | --- |
| 1 | 21036 | 3 | False | 10787.0 | False |
| 2 | 21037 | 2 | False | 10787.0 | False |

预注册分支：**C**。两个额外在线seed均失败，seed0正结果未复现。

未启动5K。
