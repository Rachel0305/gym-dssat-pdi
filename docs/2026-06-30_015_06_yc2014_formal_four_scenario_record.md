# 015_06 YC2014 正式四情景对照与 best checkpoint 汇总记录

## 结论

YC2014 的正式四情景材料已补齐降雨、管理事件、产量/生物量和统一 reward proxy。DQN 的主图代表采用 seed0 best checkpoint，因为它在与专家策略相同 grain yield 下使用更少氮肥；seed1 作为跨 seed 稳定性验证保留在汇总表中。

## 关键数字

- recorded expert: grain=9418.0 kg/ha, I=120.0 mm, N=374.0 kg/ha, proxy=7427.9
- DQN seed0: checkpoint=5000, grain=9418.0 kg/ha, I=120.0 mm, N=250.0 kg/ha, proxy=8048.0
- DQN seed1: checkpoint=10000, grain=9418.0 kg/ha, I=120.0 mm, N=300.0 kg/ha, proxy=7798.0

解释口径：如果只看产量，DQN 与专家策略持平；如果看同一经济 proxy，DQN seed0 因少用氮而优于专家策略。这不是声称 DQN 已经找到更高产策略，而是说明它已经找到更高资源效率的策略。

## 输出文件

- 主图：`DSSAT_auto_validation\yc2014_formal_four_scenario_015_06\seed0_seed1_best\figures\yc2014_formal_four_scenario.png`
- 日值总表：`DSSAT_auto_validation\yc2014_formal_four_scenario_015_06\seed0_seed1_best\015_06_yc2014_formal_four_scenario_daily.csv`
- 汇总表：`DSSAT_auto_validation\yc2014_formal_four_scenario_015_06\seed0_seed1_best\015_06_yc2014_formal_four_scenario_summary.csv`
- 管理事件表：`DSSAT_auto_validation\yc2014_formal_four_scenario_015_06\seed0_seed1_best\015_06_yc2014_formal_four_scenario_management_events.csv`

## 下一步

015_07 不直接加长训练，而是先做低成本 headroom 诊断：在当前动作空间、预算和操作窗口内，是否存在超过 recorded expert 9418 kg/ha 的调度组合。如果没有，继续训练也不可能证明产量超越；应换年份/站点或调整研究问题。如果有，再围绕这些候选调度继续训练或扩大动作空间。