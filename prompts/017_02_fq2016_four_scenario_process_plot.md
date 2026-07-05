# 017_02 FQ2016 四情景过程图与日值表整理

## 目的

在不新增训练、不修改奖励函数、不覆盖旧结果的前提下，把 FQ2016 当前已有的 baseline-relative DQN 结果整理成可汇报的四情景过程图和对应数据表。

## 情景

1. `null`：无灌溉、无施肥。
2. `recorded_shifted`：FQ2008 记录管理平移到 2016 年。
3. `dssat_auto`：DSSAT 原生自动管理尝试。
4. `dqn_seed1_best_reward`：读取 `017_01` 中 FQ2016 seed1 50K 训练的 best-reward checkpoint，即 checkpoint 30000，不重新训练。

## 执行要求

- 使用指定 Docker 容器 `b2fd6726c8c1`。
- 使用指定虚拟环境 `/opt/gym_dssat_pdi/bin/python`。
- 只做 forward / 读取已有结果 / 绘图，不进行 DQN 或 PPO 训练。
- 输出目录必须新建为 `DSSAT_auto_validation/fq2016_four_scenario_process_017_02`，不能覆盖旧结果。
- 输出以下文件：
  - 四情景日值 CSV；
  - 管理事件 CSV；
  - 汇总 CSV；
  - 降雨、水分胁迫、氮胁迫、灌溉施肥事件、产量/生物量、累积奖励代理值的过程图；
  - 中文实验记录 MD。

## 绘图要求

- 使用 Python/matplotlib。
- 保持和前面 YC/HLA 过程图一致的可读风格。
- 横坐标统一为 DAP。
- 降雨用灰色柱状图。
- 情景颜色和线型：
  - null：黑色实线；
  - recorded_shifted：红色虚线；
  - dssat_auto：深黄色实线；
  - DQN：绿色实线。
- 灌溉画竖线，施肥画三角点。
- 图和日值表必须一一对应。

## 判断重点

- FQ2016 的 DQN best-reward 策略是否确实是 `I60/N0`；
- 是否接近或超过 recorded / DSSAT auto；
- 这个结果是否与 `017_01` 的 seed1 复核结论一致；
- 是否能够作为后续 FQ 跨年份迁移前的站点内代表图。
