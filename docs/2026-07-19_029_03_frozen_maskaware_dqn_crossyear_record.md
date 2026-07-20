# 029_03 冻结 mask-aware DQN 跨年验证实验记录

状态：`completed`

## 目的与边界

在不改变 DSSAT 输入、IC、阶段窗口、动作空间、动态 mask、预算、reward、观测/scaler 和评价口径的前提下，把 029_02 五个站点训练锚点的 DQN checkpoint 冻结迁移到其余 12 个筛选年份。验证年不训练、不调参、不重新选模型。

## 输入与复用

- 锚点：FQ2016、HLA2010、LC2010、SY2014、YC2014；
- 迁移年：FQ2013/2014/2019/2020/2023，HLA2007/2015/2016/2022，SY2012/2015，YC2008；
- 每年复用 null、recorded farmer、DSSAT auto、official expert 四基线；
- 每个站点固定复用本地锚点 seed0/1/2 的冻结 DQN 权重。

## 执行

脚本：`src/run_five_site_frozen_maskaware_dqn_crossyear_029_03.py`

结果目录：`benchmark_results/029_03_frozen_maskaware_dqn_crossyear/`

共执行 12 年 × 3 seed = 36 个冻结评价季。SY 使用既有、已批准的 ICDAT 对齐适配；未修改源 DSSAT 输入或 IC。

## 工程检查

- 36/36 评价季齐全；
- validation training steps 全部为 0；
- 非法动作尝试全部为 0；
- 评价前后模型 SHA256 不变；
- 所有预期站点年与 seed 均存在。

## 结果

|站点年|DQN winner seed|至少 2/3 seed|
|---|---:|---|
|FQ2013|0/3|否|
|FQ2014|1/3|否|
|FQ2019|3/3|是|
|FQ2020|1/3|否|
|FQ2023|2/3|是|
|HLA2007|0/3|否|
|HLA2015|1/3|否|
|HLA2016|2/3|是|
|HLA2022|0/3|否|
|SY2012|1/3|否|
|SY2015|2/3|是|
|YC2008|1/3|否|

加上五个训练锚点后，DQN 在 17 年中共有 8 年达到至少 2/3 seed。

## 失败与修复记录

汇总阶段第一次读取 CSV 时，pandas 把场景字符串 `null` 解析为缺失值，导致场景完整性检查失败。修复为读入后显式把该字段的缺失值恢复为字面量 `null`，仅重新汇总，没有重跑 DSSAT，原始输出全部保留。

## 输出

- `029_03_dqn_frozen_crossyear_seed_summary.csv`
- `029_03_dqn_frozen_crossyear_stage_actions.csv`
- `029_03_dqn_frozen_crossyear_year_matrix.csv`
- `029_03_result.json`

## 结论

冻结 DQN 具有部分同站跨年迁移能力，但迁移稳定性存在明显站点差异；单独依靠训练锚点结果不能代表跨年表现。

