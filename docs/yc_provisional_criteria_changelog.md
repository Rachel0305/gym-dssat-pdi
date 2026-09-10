# YC V1 provisional criteria changelog

日期：2026-09-10  
适用实验：`yc_v1_three_seed_confirmation`  
状态：正式三种子运行前冻结；后续结果不得回溯修改阈值。

| 指标 | 初始/旧标准 | 本次冻结标准 | 冻结理由与证据边界 |
|---|---|---|---|
| 3-seed mean Yield | 未定义 | Augmented 3-seed mean Yield ≥ official/extension expert mean 的 98% | 评价平均产量，同时允许跨年/跨种子随机性；比较对象固定为同一 YC lowIC 验证集。 |
| 每个 seed mean Yield | 未定义 | 每个 seed mean Yield ≥ official/extension expert mean 的 95% | 防止平均值被单个高产种子掩盖，作为种子稳健性约束。 |
| 3-seed mean irrigation | 未定义 | ≤ official/extension expert mean 的 110% | 允许有限水量代价，但限制以水换产的不可接受偏移。 |
| 3-seed mean N | 未定义 | ≤ official/extension expert mean 的 100% | 不允许相对专家基线增加平均氮投入。 |
| 3-seed mean PFP-N | 未定义 | ≥ official/extension expert mean 的 100% | 要求氮利用效率不低于专家基线。 |
| 3-seed mean WP_ET | 未定义 | ≥ official/extension expert mean 的 95% | 要求蒸散水分生产率不出现实质性退化；WP_ET 只能使用 Summary.OUT/ETCP 精确回放值。 |

说明：

1. 本表是“初始冻结标准”，不是根据正式结果调参后的标准；正式结果只接受或拒绝这些标准。
2. official/extension expert 与 recorded farmer template 均从冻结的 `055_02` baseline summary 读取；Farmer 仅作为多目标参照，不替换 expert 作为判定基线。
3. Yield、灌溉、氮、PFP-N 与 WP_ET 的统计粒度为 2014–2023 验证年份；PFP-N 和 WP_ET 的定义、缺失值处理与来源写入正式确认报告。
