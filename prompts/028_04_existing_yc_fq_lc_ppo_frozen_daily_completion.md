# 028_04 YC/FQ/LC 现有 PPO 冻结日值证据补齐

## 目标

复用 027_07 已训练并按预注册 reward 选择的阶段型 MaskablePPO checkpoint，补齐
当前 PPO 的完整 DSSAT 日值快照和五情景过程图。不得调用 `learn()`，不得重新选模，
不得覆盖 027_07 结果。

## 固定模型

- YC2014 seed0 checkpoint60；
- YC2014 seed2 checkpoint120；
- LC2010 seed0 checkpoint240；
- FQ2016 seed0 checkpoint60，作为没有单项领先的阴性对照保留。

YC seed1 没有任何指标严格领先，不作为主图候选，但保留在三 seed 汇总表中；本任务
不删除、不重写其结果。

## 执行约束

1. 模型路径与 SHA-256 必须和 027_07 选中表一致；
2. 每个模型只做一次 `deterministic=True` 冻结季节评估；
3. `learn_calls=0`；
4. 保存 Summary、PlantGro、Weather、SoilWat、SoilNi、MgmtEvent 等原始输出；
5. 终值产量、水氮投入、WP_ET、PFP_N 与 027_07 在源文件精度容差内一致；
6. 复用 027_07 已保存的四基线快照，禁止重跑四基线；
7. 图沿用 027_05 样式，生成 PNG、SVG、逐日 CSV、终值 CSV、动作 CSV；
8. 串行运行、CPU 单线程，防止 OOM。

## 判定边界

本任务只补证据，不改变科学判定。YC seed0/2 和 LC seed0 的单项领先状态沿用
027_07；FQ seed0 保留为阴性结果。“接近”仍只报告数值差，不自行定义阈值。

