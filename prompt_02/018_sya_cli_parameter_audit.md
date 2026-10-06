# SYA CLI 参数拟合与审计任务

## 背景

已完成 SYA train-only fill 敏感性审查。

当前结论：

-   SYA 原始 WTH 存在 462 项非降水缺口；
-   原 full-period monthly mean fill 使用了包含验证期的信息；
-   已构造仅使用 2005--2013 训练期资料的 train-only fill 候选序列；
-   train-only fill 与现有结果基本一致：
    -   CLI 参数最大均值变化 0.2668；
    -   标准差参数最大相对变化 3.91%；
    -   降水与 Gamma 参数不变；
    -   候选 QC 无违规。

当前允许： - 进入独立 CLI 参数审计。

当前禁止： - 运行 WGEN； - 生成随机天气； - 运行 DSSAT； - 运行 PPO。

## 任务目标

使用 SYA train-only fill 候选天气序列，完成 DSSAT CLI 参数计算与审计。

本任务只验证： - CLI 参数是否完整； - 参数是否满足物理范围； -
参数是否可作为下一阶段 WGEN 输入候选。

## 输入要求

固定：

-   站点：SYA
-   时间范围：2005--2013
-   使用 train-only fill 候选序列：
    `results/sya_train_only_fill_sensitivity/candidate_weather_2005_2013.csv`

禁止使用： - 原 full-period fill 结果作为正式拟合输入； - 2014--2023
验证期资料。

保持： - 输入数据哈希记录； - 候选文件不覆盖原 WTH。

## CLI 参数计算

使用当前项目 CLI 参数计算流程：

`scripts/build_dssat_cli.py`

固定：

-   湿日定义： `RAIN > 0.0`

计算：

14 × 12 CLI 参数，包括：

-   湿/干日 SRAD 均值和标准差；
-   湿/干日 TMAX 均值和标准差；
-   TMIN 月均值和标准差；
-   降水统计；
-   湿日数量；
-   干后湿概率；
-   Gamma 参数。

## 审计重点

重点检查：

1.  12个月参数完整性；
2.  参数物理范围；
3.  CLI 序列化格式；
4.  2005 年集中填补月份：
    -   1月；
    -   2月；
    -   3月；
    -   4月；
    -   5月。

重点比较：

-   湿/干条件统计；
-   Gamma 参数；
-   降水参数；
-   标准差参数。

## 输出

生成：

1.  `docs/sya_cli_parameter_audit_018.md`

内容包括：

-   输入说明；
-   参数计算方法；
-   QC结果；
-   参数异常检查；
-   是否允许进入 WGEN。

2.  `results/sya_cli_parameter_audit_018/`

包含：

-   candidate CLI 文件；
-   parameter QC JSON；
-   参数审计结果。

## 判定要求

如果通过：

输出：

`READY_FOR_WGEN_REVIEW`

如果存在问题：

明确：

-   阻塞原因；
-   是否需要重新调整填补方案。

## 禁止事项

禁止：

-   修改原始 WTH；
-   修改 baseline 配置；
-   覆盖 weather_clean；
-   运行 WGEN；
-   生成 synthetic weather；
-   运行 DSSAT/PPO。

完成后停止，等待下一步授权。
