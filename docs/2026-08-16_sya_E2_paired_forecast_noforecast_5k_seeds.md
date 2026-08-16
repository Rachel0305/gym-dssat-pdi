# E2 5K forecast / no-forecast 配对验证

| seed | forecast yield | no-forecast yield | delta | yield wins | forecast PFP_N | no-forecast PFP_N | PFP delta | PFP wins | I delta | N delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10070.19 | 10009.38 | 60.81 | 4/10 | 41.959 | 41.706 | 0.253 | 4/10 | 0.0 | 0.0 |
| 1 | 10039.33 | 10015.59 | 23.73 | 6/10 | 41.831 | 41.732 | 0.099 | 6/10 | -13.5 | 0.0 |
| 2 | 9988.66 | 9965.57 | 23.09 | 6/10 | 41.619 | 43.123 | -1.503 | 5/10 | 0.0 | 8.0 |

- 三个seed×年份总体平均产量差：`35.88` kg/ha。
- 三个seed×年份总体PFP_N差：`-0.384`。
- 产量为正的seed数：`3/3`；PFP_N为正的seed数：`2/3`。
- WP_ET 因缺少有效 ETCP 保持 unavailable。
- 该配对比较只支持当前5K checkpoint，不外推到10K或更长训练。
