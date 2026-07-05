# 015_22 FQ2016 同框架 baseline-relative DQN 低成本 smoke

目标：

- 不做跨站点直接搬模型。
- 改为在 `FQ2016` 上，使用与当前成功线一致的 `baseline-relative DQN` 框架重新训练。
- 先做低成本 smoke，确认：
  - 训练链路正常；
  - 奖励、动作、预算约束都能正常记录；
  - 不会一开始就 OOM 或跑坏；
  - 检查点结果是否有基本可读性。

统一框架要求：

- 奖励：
  - 终止时 `max(0, GWAD_final - GWAD_null_site_year)`
  - 减去 `water_cost * irrigation + nitrogen_cost * nitrogen`
- 成本系数：
  - water_cost = 1.0
  - nitrogen_cost = 5.0
- 动作：
  - 9-action 离散动作
  - I ∈ {0,15,30}
  - N ∈ {0,50,100}
- 约束：
  - I <= 120 mm
  - N <= 300 kg/ha
  - 最小操作间隔 7 天

执行要求：

- 必须在 Docker 容器 `b2fd6726c8c1` 中运行。
- 必须使用虚拟环境 `/opt/gym_dssat_pdi/bin/python`。
- 只跑 smoke：
  - seed0
  - timesteps = 5000
  - checkpoint_interval = 1000
- 保存：
  - checkpoint summary csv
  - daily csv
  - 图
  - 中文实验记录

本次只回答一个问题：

- `FQ2016` 在这套统一 DQN 框架下，是否值得继续做正式长训练。
