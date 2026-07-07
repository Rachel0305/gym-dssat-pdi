# 017_04 FQ2019 确定性上界扫描

## 目的

FQ2016 seed1 checkpoint30000 迁移到 FQ2019 后，DQN 比 null 有增产，但明显低于 recorded 和 DSSAT auto。  
本实验不训练模型，只做确定性 DSSAT forward，判断 FQ2019 是：

1. 当前 DQN 没学好，但约束内存在更好调度；
2. 当前 I≤120、N≤300 约束本身追不上 DSSAT auto；
3. FQ2019 不适合作为封丘站主案例。

## 实验原则

- 不训练 DQN/PPO；
- 不修改奖励函数；
- 不覆盖旧结果；
- 使用 FQ2019 同一套输入；
- 先用小型人工调度网格做上界诊断，节省算力。

## 扫描设计

### 当前 DQN 约束内

灌溉：

- I0
- I60 early：DAP 35/55 各 30 mm
- I60 late：DAP 75/95 各 30 mm
- I90 mid：DAP 35/55/75 各 30 mm
- I120 even：DAP 35/55/75/95 各 30 mm
- I120 late：DAP 65/80/95/110 各 30 mm

施氮：

- N0
- N100 early：DAP 10 施 100 kg/ha
- N200 split：DAP 10/45 各 100 kg/ha
- N300 split：DAP 10/45/65 各 100 kg/ha

共 24 个约束内组合。

### 额外参考点

- I180 even + N0/N200/N300

这些只用于判断是否水预算限制导致追不上 auto，不作为当前 DQN 约束内策略。

## 输出

保存到：

`DSSAT_auto_validation/fq2019_deterministic_upper_bound_scan_017_04/`

输出：

- summary CSV
- daily CSV
- management event CSV
- 产量响应图
- 水氮胁迫诊断图
- 实验记录 MD

