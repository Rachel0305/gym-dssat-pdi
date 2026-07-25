# 033 系列 multisite 输入源与 IC=1 阶段说明

## 为什么做 033 系列

用户确认正式训练输入不应再使用旧 `my_data/UFGA8201-*.jinja2` 链条，而应使用：

```text
DSSAT_auto_validation/multisite_new_cultivar_inputs_013/
```

033 系列的目的不是调参，而是修正并锁定输入链条：

1. 使用 multisite 输入包中的站点模板、天气、土壤、品种；
2. 渲染后的 treatment 1 必须启用 `IC=1, MI=1, MF=1`；
3. 渲染后的 WSTA 必须与目标年 `.WTH` 文件一致；
4. 在通过渲染审计和环境 smoke 后，才允许正式训练。

## 关键任务

- `033_00`：初始土壤水分敏感性审计，确认 IC 是否启用会影响 WSPD/产量响应。
- `033_01`：旧输入链 IC 因子审计，确认旧链存在 IC=0 问题；该结论只用于解释旧结果，不代表新 multisite 输入包。
- `033_04`：使用正确 multisite 输入源重跑五站点 half-split MaskablePPO。
- `033_05`：033_04 前置渲染审计，确认所有目标站点-年份均满足 IC/WSTA/源文件要求。

## 当前 033_04 固定配置

- 算法：MaskablePPO；
- 站点：FQA、HLA、LCA、SYA、YCA；
- 年份切分：沿用 `032_21_half_split_years.csv`；
- 输入源：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013`；
- 每站点训练步数：100,000 timesteps；
- checkpoint：25K、50K、75K、100K；
- seed：0；
- reward、动作空间、约束、训练参数沿用 032_22；
- 不与旧四情景基线做正式比较，因为旧基线使用旧输入链。

## 文件上传边界

本阶段 Git 备份轻量、可复盘文件：

- prompt；
- docs 记录；
- 源代码；
- CSV/JSON 汇总结果；
- 小型审计表。

不上传：

- 模型 `.zip`；
- tensorboard；
- rendered_inputs；
- DSSAT 运行日志；
- 临时/缓存文件。

模型文件仅保留在本地输出目录中；CSV 库存表记录了模型路径和 sha256，可用于本地追溯。

## 当前解释边界

033_04 是修正输入链后的新 PPO 主实验结果。它回答“正确 multisite 输入源与 IC=1 下，原 032_22 PPO 框架能否完成五站点 half-split 训练/验证”。它还不能回答“是否优于四情景最高值”，因为四情景基线也需要用同一输入源重跑。
