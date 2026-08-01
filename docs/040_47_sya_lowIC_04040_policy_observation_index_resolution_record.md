# 040_47_sya_lowIC_04040_policy_observation_index_resolution 记录

## 任务目的

解决 040_46 遗留的 `observation_names_unresolved_run_runtime_probe`：
确认 25 维原始观测向量里，哪些下标对应哪个命名变量、未命名的尾部维度是什么、
以及哪些通道在跨年匹配步数上真的会变化（而不是同一份天气无关的常数）。

## 边界

- 不训练，不改 reward/mask/action，不重跑 040_40。
- 只回放冻结的 040_40 checkpoint100000 策略。

## 结论先说

- observation 总维度：`25`；
  命名变量数：`17`；未命名维度数：
  `8`。
- 命名变量里能唯一解析到具体下标的数量：`9/17`。
- 命名且已解析下标的变量中，跨年在匹配步数上有真实变化（非退化常数）的数量：`8`。
- 未命名维度中仍检测到跨年信号的数量：`14`（下标：[4, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24]）
- DAP 下标核验：`{'dap_index_resolved': True, 'resolved_dap_raw_index': 1, 'all_years_increment_by_one_each_step': False, 'n_years_checked': 10}`

## 变量名 -> 原始下标 解析表

| name       |   resolved_index | status                                           |
|:-----------|-----------------:|:-------------------------------------------------|
| cumsumfert |                0 | unique_match                                     |
| dap        |                1 | unique_match                                     |
| dtt        |                2 | unique_match                                     |
| ep         |                3 | unique_match                                     |
| grnwt      |              nan | no_consistent_match_check_scaling_or_derived_var |
| istage     |                5 | unique_match                                     |
| nstres     |                6 | unique_match                                     |
| rtdep      |                7 | unique_match                                     |
| srad       |                8 | unique_match                                     |
| sw         |                9 | unique_match                                     |
| swfac      |              nan | no_consistent_match_check_scaling_or_derived_var |
| tmax       |              nan | no_consistent_match_check_scaling_or_derived_var |
| topwt      |              nan | no_consistent_match_check_scaling_or_derived_var |
| totir      |              nan | no_consistent_match_check_scaling_or_derived_var |
| vstage     |              nan | no_consistent_match_check_scaling_or_derived_var |
| wtdep      |              nan | no_consistent_match_check_scaling_or_derived_var |
| xlai       |              nan | no_consistent_match_check_scaling_or_derived_var |

## 每个原始下标的跨年变异性

|   raw_index |   n_steps_with_multiple_years |   max_cross_year_std_at_any_step |   step_index_of_max_std |   overall_std_across_all_rows | carries_cross_year_signal   |
|------------:|------------------------------:|---------------------------------:|------------------------:|------------------------------:|:----------------------------|
|           0 |                           142 |                       99.8666    |                      53 |                  109.939      | True                        |
|           1 |                           142 |                        0         |                       0 |                   39.8935     | False                       |
|           2 |                           142 |                        5.09758   |                      13 |                    4.91996    | True                        |
|           3 |                           142 |                        1.95496   |                      97 |                    1.83266    | True                        |
|           4 |                           142 |                     1480.61      |                     140 |                 3031.72       | True                        |
|           5 |                           142 |                        4.21637   |                      16 |                    2.15346    | True                        |
|           6 |                           142 |                        0.114326  |                     141 |                    0.00804471 | True                        |
|           7 |                           142 |                        7.96621   |                      51 |                   35.9979     | True                        |
|           8 |                           142 |                        9.90524   |                      18 |                    6.92645    | True                        |
|           9 |                           142 |                        0.0714183 |                      90 |                    0.0424699  | True                        |
|          10 |                           142 |                        0.0655128 |                      90 |                    0.0415357  | True                        |
|          11 |                           142 |                        0.0628675 |                     140 |                    0.0427264  | True                        |
|          12 |                           142 |                        0.0783614 |                     122 |                    0.0544709  | True                        |
|          13 |                           142 |                        0.0903888 |                     141 |                    0.0509147  | True                        |
|          14 |                           142 |                        0.0924521 |                     141 |                    0.0575421  | True                        |
|          15 |                           142 |                        0.0814651 |                     141 |                    0.0595506  | True                        |
|          16 |                           142 |                        0.0999828 |                     141 |                    0.0630064  | True                        |
|          17 |                           142 |                        0.113873  |                     141 |                    0.0591247  | True                        |
|          18 |                           142 |                        0         |                       0 |                    0          | False                       |
|          19 |                           142 |                        6.78269   |                      13 |                    4.62449    | True                        |
|          20 |                           142 |                     2892.05      |                     141 |                 5722.14       | True                        |
|          21 |                           142 |                        0         |                       0 |                   69.6227     | False                       |
|          22 |                           142 |                        1.12599   |                     130 |                    8.00215    | True                        |
|          23 |                           142 |                        5.93978   |                     102 |                    8.74398    | True                        |
|          24 |                           142 |                        0.48722   |                      92 |                    1.15236    | True                        |

## 解释边界

- 本任务只做只读诊断，不对 mask 几何或 reward 做任何调整。
- 若某个已解析下标 `carries_cross_year_signal=False`，只能说明该变量在本checkpoint的
  验证年份匹配步数上退化为常数，不能反推 policy 训练全程都看不到该信号。
- 若未命名维度里发现有跨年信号的通道，需要回到 wrapper 源码确认它具体是什么工程特征
  （例如 safety_state 剩余额度、mask 标志位等），本表不猜测具体含义。
- 这个结果决定的是任务2（observation 疑点）是否已解决，不直接决定要不要做 mask 几何消融；
  mask 几何消融是否值得做，取决于本表是否排除了“看不见天气”这个假设。
