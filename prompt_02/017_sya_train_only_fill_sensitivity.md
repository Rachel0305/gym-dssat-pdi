# SYA train-only fill 敏感性审查任务

## 背景

已完成 SYA/LCA WGEN 前置方法学复审。

当前状态：

-   LCA：
    -   CLI 参数审计完成；
    -   因 1 月降水 Gamma 参数样本不足，维持
        `BLOCKED_BY_JANUARY_GAMMA_SAMPLE`；
    -   本任务不涉及 LCA。
-   SYA：
    -   训练期 2005--2013 WTH 四变量 QC 已通过；
    -   462 个 SRAD/TMAX/TMIN 缺口已确认来自历史站点×月份均值填补；
    -   当前问题：
        1.  历史填补使用全时期数据，包含验证期信息；
        2.  2005 年存在集中填补；
        3.  需要评估训练期独立填补对 WGEN 参数的影响。

## 任务目标

在不修改原始 WTH 的情况下，构造 SYA 训练期独立填补候选方案，并评估其对
CLI 参数拟合的影响。

本任务只做敏感性分析。

禁止运行： - WGEN； - DSSAT； - PPO。

禁止修改： - 原始 WTH； - baseline 配置； - 已有审计结果。

## 执行要求

### 1. 固定训练范围

使用： - 站点：SYA - 训练年份：2005--2013

验证年份： - 2014--2023

禁止使用验证期数据参与候选填补。

### 2. 构造 train-only fill 候选序列

保持原始缺测位置不变。

对于 SRAD、TMAX、TMIN：

重新计算： - 仅使用 2005--2013； - 站点×月份平均值。

生成候选填补结果。

不要覆盖原始 WTH 或 weather_clean 数据。

### 3. 比较两套天气序列

比较：

A. 当前正式 WTH： - full-period monthly mean fill

B. 候选序列： - train-only monthly mean fill

比较：

-   填补值差异；
-   SRAD/TMAX/TMIN 最大绝对和相对差异；
-   CLI 14×12 参数差异；
-   湿/干条件统计变化；
-   Gamma 参数变化。

重点关注： - 2005 年连续填补影响月份； - 湿日/干日条件统计。

### 4. 敏感性判断

判断：

A. 与现有结果基本一致，可作为 WGEN 输入候选；

B. 参数变化明显，需要进一步处理；

C. 不适合进入 WGEN。

不要自行设置未经说明的 DSSAT 官方阈值。

## 输出文件

生成：

1.  `docs/sya_train_only_fill_sensitivity_review.md`

包括： - 问题说明； - 方法； - 两套填补方案比较； - CLI 参数影响； -
最终建议。

2.  `results/sya_train_only_fill_sensitivity/`

包含： - `fill_difference.csv` - `cli_parameter_comparison.csv` -
`sensitivity_summary.json`

## 额外要求

-   保留输入文件 SHA256；
-   明确记录：
    -   原始 WTH 未修改；
    -   未运行 WGEN；
    -   未生成随机天气；
    -   未运行 DSSAT/PPO。

完成后停止，等待下一步授权。
