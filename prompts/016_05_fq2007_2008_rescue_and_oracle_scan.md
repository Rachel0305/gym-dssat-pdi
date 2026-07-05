# 016_05 FQ2007/2008 抢救与确定性上界扫描 prompt

## 目标

在不训练强化学习模型的前提下，先处理 FQ2007/FQ2008 在多年迁移实验中被排除的问题，再决定是否把它们纳入后续确定性上界扫描。

当前观察：

- `016_03` 中 FQ2007/FQ2008 的 baseline 为空。
- PDI/DSSAT 日志出现：
  `Initial conditions date is defined prior to the start of simulation.`
- 初步检查发现，多年迁移生成的 FQ 文件中，`ICDAT` / `SDATE` 等日期仍有 `07152`，说明旧模板迁移可能没有把 2007/2008 相关日期完整平移到目标年份。

## 原则

1. 先做低成本 smoke test，不直接训练。
2. 不覆盖旧结果，所有新结果写入独立目录。
3. 使用指定容器和虚拟环境：
   - Docker container: `b2fd6726c8c1`
   - Python: `/opt/gym_dssat_pdi/bin/python`
4. 记录每个尝试：
   - 输入 MZX
   - env_args
   - PDI 日志
   - gym post-state CSV
   - PlantGro 输出摘要
   - 实验记录 MD

## 诊断设计

对 FQ2007 和 FQ2008 分别运行 null smoke：

1. `old_shift_exp2`
   - 复用旧的 FQ2008 treatment 2 迁移逻辑；
   - 使用 `experiment_number=2`；
   - 用于确认原始失败。

2. `old_shift_exp1`
   - 同一个旧迁移文件；
   - 改用 `experiment_number=1`；
   - 用于判断是否是 treatment 指针导致。

3. `full_date_shift_exp2`
   - 对源文件中的 `07xxx` 和 `08xxx` 日期都平移为目标年份 `yyxxx`；
   - 保持使用 `experiment_number=2`；
   - 用于判断是否是日期迁移不完整导致。

## 判定标准

如果 `full_date_shift_exp2` 能完整产生有限的最终 `GWAD/CWAD`，则：

- FQ2007/FQ2008 不是不可用年份；
- 旧实验排除它们的原因是输入文件迁移问题；
- 后续 016_05 确定性上界扫描可以把这两个年份纳入，但必须使用修正后的日期迁移逻辑。

如果仍失败，则暂时不把 FQ2007/FQ2008 纳入主扫描，先记录失败原因。

## 后续 016_05 扫描方向

完成抢救后，再进行低成本确定性上界扫描：

- 不训练 DQN/PPO；
- 用人工确定性水氮时序组合检验在当前站点年份下是否存在“高产 + 节水节氮”的可达策略；
- 优先小规模代表年份，避免一次性大量 DSSAT 运行。

