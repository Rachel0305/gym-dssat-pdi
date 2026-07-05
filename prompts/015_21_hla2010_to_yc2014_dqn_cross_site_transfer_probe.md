# 015_21 HLA2010 训练 DQN 模型跨站点迁移到 YC2014 的低成本 probe

目标：

- 不重新训练，只拿已经训练好的 `HLA2010 baseline-relative DQN` checkpoint，
  直接迁移评估到 `YC2014`。
- 先做低成本 smoke test，确认：
  - 动作空间是否兼容；
  - 环境是否能正常 reset / step；
  - `gym_dssat/PDI` 链路是否能完整跑完；
  - 输出的水分/氮胁迫、灌溉、施肥、产量是否可读。
- smoke 成功后，再补做 seed1，形成完整 probe 结论：
  - 这套 `2010 训练模型` 是只在 HLA 内有效，还是对别的站点也有可迁移性。

执行约束：

- 必须在指定 Docker 容器 `b2fd6726c8c1` 中运行。
- 必须使用指定虚拟环境：`/opt/gym_dssat_pdi/bin/python`。
- 先 smoke，后 full，禁止一上来就开大训练。
- 不覆盖旧结果，单独输出到新的 `015_21` 目录。
- 保存：
  - 每日值 CSV；
  - 汇总 CSV；
  - 管理事件 CSV；
  - 图；
  - 中文实验记录 md。

本次不是训练任务，而是迁移评估任务：

- 训练来源：`HLA2010 baseline-relative DQN`
- 测试目标：`YC2014`
- 先测：
  - train seed0 / checkpoint 35000
- 若 smoke 正常，再补：
  - train seed1 / checkpoint 25000

判定重点：

1. 模型文件能否在 YC2014 环境直接加载并产生动作；
2. 迁移后是否出现“全程 no-op / 全程打满 / 立即崩溃”这类无意义行为；
3. 迁移产量相对 `YC2014` 的：
   - null
   - recorded
   - dssat_auto
   - local dqn_best
   分别是什么水平；
4. 若迁移结果接近本地 `YC2014 dqn_best` 或至少优于 `null` 且动作合理，
   说明 HLA 学到的策略有跨站点结构信息，而不只是记住 HLA。
