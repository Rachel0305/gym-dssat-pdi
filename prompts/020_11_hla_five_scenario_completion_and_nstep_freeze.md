# 020_11 HLA 五情景补齐与 n-step 框架冻结

## 历史核对

- 020_06/020_07/020_08 已完成 HLA2010 `n_steps=5` 的 seed1/seed2/seed0 训练，按“总奖励最大、并列取最早”分别选中 10K、20K、30K。
- 020_09/020_10 已完成三个模型向 HLA2015、2007、2016、2022 的站内跨年迁移。
- 旧文件中的 `expert_2007_shifted` 实际是 2007 年农民/试验田管理记录平移，不是推文里的官方推广 expert；本轮必须纠正名称和口径。
- 018_03 只计算了 HLA2010 的推文 expert，2007/2015/2016/2022 尚缺。

## 正式五情景定义

1. Null。
2. Recorded farmer practice：2007 记录本身或其 DAP 平移版本。
3. DSSAT auto。
4. Official extension expert：依据《玉米大豆水肥一体化单产提升技术方案》表 1 中值、固定 DAP 的实现。
5. n-step DQN：HLA2010 训练的 seed0/seed1/seed2 三个已选 checkpoint。

概念上为五类情景；每个年份的长表为 7 行，因为 DQN 情景包含 3 个 seed。

## 执行原则

- 年份：HLA 2007、2010、2015、2016、2022。
- 使用各年份当前 IC=1、同品种、同土壤、同天气输入；新增情景不得修改输入基线。
- 推文 expert 必须在每个年份重新前向回放，不能复制 HLA2010 的产量或胁迫。
- 农民记录统一按 DAP0 施氮 165 kg/ha、DAP49/70/95 各灌溉 10 mm 回放。执行中发现旧 HLA2007 baseline 使用 ICDAT=07121、SH2O≈0.30、SDATE=07121，而当前 n-step 输入使用 ICDAT/SDATE=07125、SH2O≈0.26；因此旧 2007 数字只保留为输入不一致审计证据，不再硬拼入新五情景。2010/2015 仍做数值复现核对。
- 为保证同输入，五年 null、recorded farmer、DSSAT auto、official extension expert 四条基线全部在当前输入包上重新前向回放；DSSAT auto 复用对应年份的自动灌溉管理模板，但天气、土壤、品种与初始条件保持当前输入。
- 新回放仅为确定性 DSSAT 前向模拟，不训练。
- 所有训练/模拟命令必须在 Docker `b2fd6726c8c1`、`/opt/gym_dssat_pdi/bin/python` 中执行。

## 输出

- 五年 × 七行的总汇总表。
- 统一日值表、管理事件表、调度表、输入哈希审计表。
- 每年一张包含 4 条基线和 3 条 DQN seed 的高对比度过程图；同时输出 PNG/SVG/PDF。
- 图中包含降雨、水/氮胁迫、灌溉、施氮、籽粒/生物量和统一累积奖励。
- 冻结配置 JSON：奖励、动作表、预算、单次上限、最小间隔、DQN 超参数、`n_steps=5`、选中 checkpoint、输入开关和代码哈希。
- 中文实验记录，明确 recorded 与 official expert 的区别。

## QA

- 每个年份必须包含 `null / recorded_farmer / dssat_auto / extension_expert / nstep_seed0 / nstep_seed1 / nstep_seed2`。
- 每年降雨不能全为 0；推文 expert 的降雨从同年 null/WTH 对齐，不使用 018_03 中错误的全零 `rain_obs`。
- 推文 expert 的实际管理总量以 `MgmtEvent.OUT` 为准；动作指令总量另列。
- 任何输入哈希不一致、情景缺失、异常提前结束或既有 recorded 结果明显不一致都必须停止并记录。
- 不覆盖 018_03、020_06—020_10 的旧结果。
